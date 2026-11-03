"""Relación entre mercados de un mismo partido (para combinadas tipo "Crear Apuesta").

Si un equipo remata mucho, también suele tener más tiros a puerta, más corners y marcar
más, y sus delanteros rematan más. Multiplicar probabilidades como si fueran independientes
daría resultados erróneos, así que se mide esa relación con partidos reales:

1. Cada estadística de cada equipo y partido se pasa a "puntuación normal": cuánto se salió
   de lo esperado para ese equipo y rival (misma escala para goles, corners o faltas).
2. La correlación entre esas puntuaciones da la matriz que usa la app para simular
   partidos completos (cópula gaussiana): cada simulación respeta la probabilidad de cada
   mercado por separado y además su relación con los demás.
3. Cada jugador se relaciona con la estadística de su equipo (sus remates con los del
   equipo, sus faltas recibidas con las faltas del rival...).

    python correlaciones.py      # calcula, valida y guarda en calibracion.json
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

from estadisticas import CALIBRACION, ModeloConteo, preparar_equipos
from fotmob import DIR

STATS = ["goles", "corners", "remates", "tiros_puerta", "faltas", "amarillas", "entradas"]
ORDEN = [f"{s}_{lado}" for s in STATS for lado in ("L", "V")]
# Estadística de equipo con la que se relaciona cada estadística de jugador.
REF_JUGADOR = {"remates": ("remates", False), "tiros_puerta": ("tiros_puerta", False),
               "entradas": ("entradas", False), "faltas": ("faltas", False),
               "faltas_recibidas": ("faltas", True)}  # True = la del rival
rng = np.random.default_rng(7)


def cdf(y, mu, k):
    return poisson.cdf(y, mu) if k is None else nbinom.cdf(y, k, k / (k + mu))


def puntuacion_normal(y, mu, k):
    """Transformación integral aleatorizada: lleva un conteo a una normal estándar."""
    lo, hi = cdf(y - 1, mu, k), cdf(y, mu, k)
    u = lo + rng.uniform(size=len(y)) * (hi - lo)
    return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))


def medias_equipo(eqp, corte):
    """Media y dispersión ajustadas para cada fila equipo-partido y estadística."""
    modelos = {}
    for s in STATS:
        m = ModeloConteo().fit(eqp, s, corte)
        modelos[s] = m
        eqp[f"mu_{s}"] = [m.media(r.equipo_id, r.rival_id, bool(r.local), r.competicion)
                          for r in eqp.itertuples()]
    return modelos


def calcular(eq, ju, cal_jugador):
    corte = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    eqp = preparar_equipos(eq)
    modelos = medias_equipo(eqp, corte)
    eqp = eqp.dropna(subset=[f"mu_{s}" for s in STATS])
    for s in STATS:
        eqp[f"z_{s}"] = puntuacion_normal(eqp[s].to_numpy(), eqp[f"mu_{s}"].to_numpy(), modelos[s].k)

    loc = eqp[eqp["local"] == 1].set_index("partido_id")
    vis = eqp[eqp["local"] == 0].set_index("partido_id")
    comunes = loc.index.intersection(vis.index)
    Z = pd.DataFrame({f"{s}_{l}": d.loc[comunes, f"z_{s}"] for s in STATS for l, d in (("L", loc), ("V", vis))})[ORDEN]
    R = np.corrcoef(Z.to_numpy().T)

    # Jugadores titulares que jugaron al menos 60 minutos.
    ju = ju[(ju["titular"] == 1) & (ju["minutos"] >= 60) & (ju["posicion"] != "POR")].copy()
    base = eqp.set_index(["partido_id", "equipo_id"])
    rho = {}
    for s, (ref, rival) in REF_JUGADOR.items():
        clave_eq = ju["rival_id"] if rival else ju["equipo_id"]
        idx = list(zip(ju["partido_id"], clave_eq))
        ok = [i in base.index for i in idx]
        d = ju[ok].copy()
        fila = base.loc[[i for i, o in zip(idx, ok) if o]]
        d["mu_eq"], d["y_eq"], d["z_eq"] = fila[f"mu_{ref}"].to_numpy(), fila[ref].to_numpy(), fila[f"z_{ref}"].to_numpy()
        # Parte habitual del jugador en la temporada (sin decaimiento: solo para medir la relación).
        d["exp_eq"] = d["y_eq"] * d["minutos"] / 90
        parte = d.groupby("jugador_id")[s].sum() / d.groupby("jugador_id")["exp_eq"].sum().clip(lower=1e-6)
        d["mu_j"] = d["jugador_id"].map(parte).clip(lower=1e-3) * d["mu_eq"] * d["minutos"] / 90
        k = cal_jugador.get(s, {}).get("k")
        d["z_j"] = puntuacion_normal(d[s].to_numpy(), d["mu_j"].to_numpy(), k)
        rho[s] = round(float(np.corrcoef(d["z_j"], d["z_eq"])[0, 1]), 3)
    return R, rho, Z, eqp, modelos


def validar(eqp, modelos, R, n_sim=3000):
    """¿Acierta la simulación la probabilidad de que ocurran dos cosas a la vez?

    Para partidos de 2026-27 compara la frecuencia real de pares de sucesos con la que
    predice la simulación y con la que saldría multiplicando (independencia).
    """
    pares = [("goles_L", 2, "tiros_puerta_L", 5), ("remates_L", 14, "tiros_puerta_L", 5),
             ("corners_L", 6, "remates_L", 14), ("goles_L", 2, "goles_V", 2),
             ("goles_V", 1, "tiros_puerta_V", 4), ("faltas_L", 12, "amarillas_L", 3),
             ("corners_L", 6, "corners_V", 5), ("remates_L", 14, "remates_V", 12)]
    L = np.linalg.cholesky(R)
    prueba = eqp[(eqp["fecha"] >= "2026-08-01") & (eqp["local"] == 1)]
    vis = eqp[eqp["local"] == 0].set_index("partido_id")
    tot = {p: [0.0, 0.0, 0.0] for p in pares}
    n = 0
    for _, r in prueba.iterrows():
        if r["partido_id"] not in vis.index:
            continue
        v = vis.loc[r["partido_id"]]
        Zs = rng.standard_normal((n_sim, len(ORDEN))) @ L.T
        sims, real, marg = {}, {}, {}
        for j, nombre in enumerate(ORDEN):
            s, lado = nombre.rsplit("_", 1)
            fila = r if lado == "L" else v
            mu, k = fila[f"mu_{s}"], modelos[s].k
            u = norm.cdf(Zs[:, j])
            sims[nombre] = poisson.ppf(u, mu) if k is None else nbinom.ppf(u, k, k / (k + mu))
            real[nombre] = fila[s]
            marg[nombre] = (mu, k)
        n += 1
        for p in pares:
            a, na, b, nb = p
            pa = 1 - cdf(na - 1, *marg[a])
            pb = 1 - cdf(nb - 1, *marg[b])
            tot[p][0] += float(real[a] >= na and real[b] >= nb)
            tot[p][1] += float(np.mean((sims[a] >= na) & (sims[b] >= nb)))
            tot[p][2] += pa * pb
    print(f"\nValidación con {n} partidos de 2026-27: probabilidad de que ocurran las dos cosas")
    print(f"{'suceso A':<22}{'suceso B':<24}{'real':>7}{'simulación':>12}{'independiente':>15}")
    for (a, na, b, nb), (re, si, ind) in tot.items():
        print(f"{a + ' ' + str(na) + '+':<22}{b + ' ' + str(nb) + '+':<24}{100 * re / n:>6.1f}%{100 * si / n:>11.1f}%{100 * ind / n:>14.1f}%")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    eq = pd.read_csv(DIR / "equipos.csv")
    ju = pd.read_csv(DIR / "jugadores.csv")
    cal = json.loads(CALIBRACION.read_text())
    R, rho, Z, eqp, modelos = calcular(eq, ju, cal.get("jugador", {}))
    print("Correlaciones más fuertes entre estadísticas de un mismo partido:")
    pares = sorted(((abs(R[i, j]), ORDEN[i], ORDEN[j], R[i, j]) for i in range(len(ORDEN)) for j in range(i)), reverse=True)
    for _, a, b, r in pares[:12]:
        print(f"  {a:<16} {b:<16} {r:+.2f}")
    print("Relación de cada jugador con su equipo:", rho)
    validar(eqp, modelos, R)
    cal["correlacion"] = {"orden": ORDEN, "R": [[round(float(x), 3) for x in fila] for fila in R], "rho_jugador": rho}
    CALIBRACION.write_text(json.dumps(cal, indent=1))
    print(f"\nGuardado en {CALIBRACION.name}")


if __name__ == "__main__":
    main()
