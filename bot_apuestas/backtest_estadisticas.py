"""Prueba histórica y calibración de corners, remates, faltas, tarjetas y jugadores.

1. Cada semana desde diciembre de 2025 entrena solo con lo anterior y predice los partidos
   de La Liga y Champions de esa semana (jugadores: los titulares; la previsión es "si es
   titular").
2. Ajusta la calibración con las predicciones hasta junio de 2026 y la comprueba con las de
   esta temporada, que no ha usado para ajustarla.
3. Guarda en calibracion.json la calibración ajustada con todo, que es la que usa la app.

    python backtest_estadisticas.py
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

from estadisticas import CALIBRACION, STATS_EQUIPO, STATS_JUGADOR, ModeloEstadisticas, calibrar, preparar_equipos
from fotmob import DIR

DESDE, CORTE_VALIDACION = "2025-12-01", "2026-08-01"
LINEAS = {
    "equipo": {"corners": [4, 5, 6], "remates": [10, 12, 14], "tiros_puerta": [3, 4, 5], "faltas": [10, 12, 14],
               "amarillas": [2, 3], "entradas": [15, 18]},
    "total": {"corners": [9, 10, 11], "remates": [22, 25], "tiros_puerta": [8, 9], "faltas": [22, 25],
              "amarillas": [4, 5], "entradas": [30, 35]},
    "jugador": {"remates": [1, 2, 3], "tiros_puerta": [1, 2], "entradas": [1, 2, 3],
                "faltas_recibidas": [1, 2, 3], "faltas": [1, 2]},
}


def predicciones(eq, ju) -> pd.DataFrame:
    """Medias previstas (sin calibrar) y valor real, semana a semana."""
    eqp = preparar_equipos(eq)
    ju = ju.copy()
    ju["fecha"] = pd.to_datetime(ju["fecha"])
    prueba = eqp[(eqp["fecha"] >= DESDE) & eqp["competicion"].isin(["La Liga", "Champions"]) & (eqp["local"] == 1)]
    filas = []
    for semana, jornada in prueba.groupby(prueba["fecha"].dt.to_period("W-MON")):
        m = ModeloEstadisticas(calibracion={}).fit(eq, ju, semana.start_time)
        for _, p in jornada.iterrows():
            l, v, pid = p["equipo_id"], p["rival_id"], p["partido_id"]
            real = eqp[eqp["partido_id"] == pid].set_index("equipo_id")
            if v not in real.index:
                continue
            for s, (ml, mv, k) in m.equipo_partido(l, v, p["competicion"]).items():
                for lado, mu, eq_id in (("local", ml, l), ("visitante", mv, v)):
                    if pd.notna(real.at[eq_id, s]):
                        filas.append([p["fecha"], pid, "equipo", s, lado, mu, k, real.at[eq_id, s],
                                      m.equipo[s].referencia(p["competicion"])])
            for _, j in ju[(ju["partido_id"] == pid) & (ju["titular"] == 1)].iterrows():
                local = j["equipo_id"] == l
                prev = {r["id"]: r for r in m.jugadores_partido(j["equipo_id"], v if local else l, local,
                                                                semana.start_time - pd.Timedelta(days=120),
                                                                competicion=p["competicion"])}
                r = prev.get(j["jugador_id"])
                if r:
                    for s in STATS_JUGADOR:
                        filas.append([p["fecha"], pid, "jugador", s, j["jugador_id"], r[s], None, j[s], np.nan])
        print(f"  semana {semana.start_time.date()}: {len(filas)} predicciones", flush=True)
    return pd.DataFrame(filas, columns=["fecha", "partido_id", "tipo", "stat", "lado", "mu", "k", "y", "ref"])


def nll(y, mu, k):
    if k is None:
        return -poisson.logpmf(y, mu).sum()
    return -nbinom.logpmf(y, k, k / (k + mu)).sum()


def ajustar(pred: pd.DataFrame) -> dict:
    """Para cada estadística: cuánto acercar las medias a la media (s) y dispersión real (k)."""
    cal = {"equipo": {}, "jugador": {}}
    for (tipo, s), d in pred.groupby(["tipo", "stat"]):
        y, mu = d["y"].to_numpy().round(), d["mu"].to_numpy().clip(1e-3)
        m = float(mu.mean())
        ref = d["ref"].fillna(m).to_numpy()  # equipos: nivel medio de esa semana

        def f(x):
            return nll(y, ref * (mu / ref) ** x[0], np.exp(x[1]))

        sol = minimize(f, [0.9, np.log(10)], method="L-BFGS-B", bounds=[(0.2, 1.3), (np.log(0.3), np.log(1000))])
        cal[tipo][s] = {"m": round(m, 4), "s": round(float(sol.x[0]), 4), "k": round(float(np.exp(sol.x[1])), 3)}
    return cal


def probs_lineas(pred: pd.DataFrame, cal: dict | None) -> pd.DataFrame:
    """Probabilidad de cada línea (más de N) con o sin calibración, y si se cumplió."""
    def dist(mu, k, n=90):
        g = np.arange(n)
        return poisson.pmf(g, mu) if k is None or (isinstance(k, float) and np.isnan(k)) else nbinom.pmf(g, k, k / (k + mu))

    filas = []
    for (tipo, s), d in pred.groupby(["tipo", "stat"]):
        c = (cal or {}).get(tipo, {}).get(s)
        for _, r in d.iterrows():
            mu = calibrar(r["mu"], c, None if np.isnan(r["ref"]) else r["ref"])
            k = c["k"] if c else r["k"]
            p = dist(mu, k)
            for n in LINEAS[tipo][s]:
                filas.append([tipo, s, n, 1 - p[:n].sum(), r["y"] >= n])
        if tipo == "equipo":  # totales del partido
            ancho = d.pivot_table(index="partido_id", columns="lado", values=["mu", "y", "ref"], aggfunc="first").dropna()
            k0 = d["k"].iloc[0]
            for _, r in ancho.iterrows():
                k = c["k"] if c else k0
                ref = r[("ref", "local")]
                pt = np.convolve(dist(calibrar(r[("mu", "local")], c, ref), k),
                                 dist(calibrar(r[("mu", "visitante")], c, ref), k))
                for n in LINEAS["total"][s]:
                    filas.append(["total", s, n, 1 - pt[:n].sum(), r[("y", "local")] + r[("y", "visitante")] >= n])
    out = pd.DataFrame(filas, columns=["familia", "stat", "linea", "p", "y"])
    out["y"] = out["y"].astype(float)
    return out


def resumen(lineas: pd.DataFrame, titulo: str):
    print(f"\n--- {titulo} ---")
    filas = []
    for (fam, s, n), d in lineas.groupby(["familia", "stat", "linea"]):
        p = d["p"].clip(1e-4, 1 - 1e-4)
        b = np.clip(d["y"].mean(), 1e-4, 1 - 1e-4)
        ll = -(d["y"] * np.log(p) + (1 - d["y"]) * np.log(1 - p)).mean()
        ll0 = -(d["y"] * np.log(b) + (1 - d["y"]) * np.log(1 - b)).mean()
        filas.append([f"{fam} {s}", f"{n}+", len(d), f"{100 * d['p'].mean():.1f}%", f"{100 * d['y'].mean():.1f}%",
                      f"{100 * (1 - ll / ll0):+.1f}%"])
    print(pd.DataFrame(filas, columns=["mercado", "línea", "n", "prevista", "real", "mejora"]).to_string(index=False))
    lineas = lineas.assign(tramo=pd.cut(lineas["p"], np.linspace(0, 1, 11)))
    for fam in ("jugador", "equipo", "total"):
        d = lineas[lineas["familia"] == fam]
        t = d.groupby("tramo", observed=True).agg(prevista=("p", "mean"), real=("y", "mean"))
        print(f"Calibración {fam}: " + "  ".join(f"{100 * a:.0f}→{100 * b:.0f}" for a, b in t.itertuples(index=False)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    eq = pd.read_csv(DIR / "equipos.csv")
    ju = pd.read_csv(DIR / "jugadores.csv")
    pred = predicciones(eq, ju)
    pred["k"] = pred["k"].astype(object).where(pred["k"].notna(), None)

    ajuste, valida = pred[pred["fecha"] < CORTE_VALIDACION], pred[pred["fecha"] >= CORTE_VALIDACION]
    cal = ajustar(ajuste)
    print(f"\nValidación: {valida['partido_id'].nunique()} partidos de 2026-27 que no se usaron para calibrar")
    print("(mejora = cuánto mejor que usar siempre la frecuencia media; prevista vs real = calibración)")
    resumen(probs_lineas(valida, None), "SIN calibrar")
    resumen(probs_lineas(valida, cal), "CALIBRADO")

    final = ajustar(pred)
    CALIBRACION.write_text(json.dumps(final, indent=1))
    print(f"\nCalibración final (todas las predicciones) guardada en {CALIBRACION.name}")


if __name__ == "__main__":
    main()
