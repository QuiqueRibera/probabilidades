"""Modelos de corners, remates, tiros a puerta, faltas, tarjetas y entradas.

Equipo: cada estadística tiene su propio modelo "a favor / en contra" por equipo
    log μ = base + ventaja_local + ofensiva_equipo + concesión_rival
con decaimiento temporal, y una binomial negativa para la dispersión (los corners y
las faltas varían más de lo que daría una Poisson). Los cruces de Champions conectan
las ligas, así que equipos de ligas distintas quedan en la misma escala.

Jugador: lo que se espera del equipo en ese partido × la parte que suele hacer ese
jugador cuando está en el campo × minutos esperados. La parte del jugador se encoge
hacia la media de su posición cuando hay pocos minutos de muestra.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

STATS_EQUIPO = ["corners", "remates", "tiros_puerta", "faltas", "amarillas", "entradas"]
STATS_JUGADOR = ["remates", "tiros_puerta", "entradas", "faltas_recibidas", "faltas"]
CALIBRACION = Path(__file__).parent / "calibracion.json"


def cargar_calibracion() -> dict:
    """Corrección fuera de muestra que calcula backtest_estadisticas.py.

    Sin ella el modelo es demasiado confiado: sus medias se separan más de la cuenta entre
    partidos/jugadores y la variación real es mayor que la de una Poisson.
    """
    return json.loads(CALIBRACION.read_text()) if CALIBRACION.exists() else {}


def calibrar(mu: float, c: dict | None, ref: float | None = None) -> float:
    """Acerca la media a la media general (exponente s < 1).

    ref es el nivel medio actual; si no se da, se usa el del periodo de calibración.
    """
    if not c or mu <= 0:
        return mu
    ref = ref or c["m"]
    return ref * (mu / ref) ** c["s"]


def preparar_equipos(eq: pd.DataFrame) -> pd.DataFrame:
    """Añade las faltas recibidas (= faltas que comete el rival) a cada fila de equipo."""
    eq = eq.copy()
    eq["fecha"] = pd.to_datetime(eq["fecha"])
    rival = eq[["partido_id", "equipo_id", "faltas"]].rename(columns={"equipo_id": "rival_id", "faltas": "faltas_recibidas"})
    return eq.merge(rival, on=["partido_id", "rival_id"], how="left")


class ModeloConteo:
    """Media de una estadística por equipo y partido + dispersión (binomial negativa)."""

    def __init__(self, xi=0.003, ridge=3.0):
        self.xi, self.ridge = xi, ridge

    def fit(self, eq: pd.DataFrame, stat: str, corte) -> "ModeloConteo":
        df = eq[(eq["fecha"] < corte) & eq[stat].notna()]
        self.equipos = sorted(set(df["equipo_id"]) | set(df["rival_id"]))
        idx = {t: i for i, t in enumerate(self.equipos)}
        n = len(self.equipos)
        # Cada competición tiene su "ambiente" (en Champions se remata más y se pita menos).
        self.comps = sorted(df["competicion"].unique())
        nc = len(self.comps)
        e, r = df["equipo_id"].map(idx).to_numpy(), df["rival_id"].map(idx).to_numpy()
        co = df["competicion"].map({c: i for i, c in enumerate(self.comps)}).to_numpy()
        loc, y = df["local"].to_numpy(float), df[stat].to_numpy(float)
        w = np.exp(-self.xi * (corte - df["fecha"]).dt.days.to_numpy())

        def nll(p):
            a, d, g, c, h = p[:n], p[n:2 * n], p[2 * n:2 * n + nc], p[-2], p[-1]
            mu = np.exp(c + h * loc + a[e] + d[r] + g[co])
            res = w * (mu - y)
            f = np.sum(w * (mu - y * np.log(mu))) + self.ridge * (a @ a + d @ d + g @ g)
            grad = np.concatenate([np.bincount(e, res, n) + 2 * self.ridge * a,
                                   np.bincount(r, res, n) + 2 * self.ridge * d,
                                   np.bincount(co, res, nc) + 2 * self.ridge * g,
                                   [res.sum(), (res * loc).sum()]])
            return f, grad

        x0 = np.zeros(2 * n + nc + 2)
        x0[-2] = np.log(max(np.average(y, weights=w), 0.1))
        sol = minimize(nll, x0, jac=True, method="L-BFGS-B")
        self.a = dict(zip(self.equipos, sol.x[:n]))
        self.d = dict(zip(self.equipos, sol.x[n:2 * n]))
        self.g = dict(zip(self.comps, sol.x[2 * n:2 * n + nc]))
        self.c, self.h = sol.x[-2], sol.x[-1]

        g_co = sol.x[2 * n:2 * n + nc][co]
        mu = np.exp(self.c + self.h * loc + sol.x[e] + sol.x[n + r] + g_co)
        # El nivel general cambia de una temporada a otra (p. ej. se remata más): se corrige con
        # lo que ha pasado en las últimas 6 semanas, con más peso cuanto más partidos haya.
        recientes = (corte - df["fecha"]).dt.days.to_numpy() <= 42
        nr = recientes.sum()
        if nr >= 20:
            self.c += nr / (nr + 150) * np.log(y[recientes].sum() / mu[recientes].sum())
            mu = np.exp(self.c + self.h * loc + sol.x[e] + sol.x[n + r] + g_co)

        # Dispersión por momentos: Var = μ + μ²/k
        exceso = np.sum(w * ((y - mu) ** 2 - mu)) / np.sum(w * mu ** 2)
        self.k = 1 / exceso if exceso > 1e-3 else None  # None = Poisson
        return self

    def referencia(self, competicion="La Liga") -> float:
        """Media de un partido típico de esa competición."""
        return float(np.exp(self.c + self.h / 2 + self.g.get(competicion, 0.0)))

    def media(self, equipo, rival, local: bool, competicion="La Liga") -> float | None:
        if equipo not in self.a or rival not in self.a:
            return None
        return float(np.exp(self.c + self.h * local + self.a[equipo] + self.d[rival] + self.g.get(competicion, 0.0)))


def prob_al_menos(n: int, mu: float, k: float | None = None) -> float:
    """P(X ≥ n) con Poisson (k=None) o binomial negativa de media μ y forma k."""
    if k is None:
        return float(poisson.sf(n - 1, mu))
    return float(nbinom.sf(n - 1, k, k / (k + mu)))


class ModeloEstadisticas:
    def __init__(self, xi_jugador=0.004, muestra_previa=2.0, calibracion: dict | None = None):
        self.xi_jugador = xi_jugador          # vida media ≈ 6 meses: los roles cambian rápido
        self.muestra_previa = muestra_previa  # partidos "ficticios" de media de su posición
        self.cal = cargar_calibracion() if calibracion is None else calibracion

    def fit(self, eq: pd.DataFrame, ju: pd.DataFrame, corte) -> "ModeloEstadisticas":
        corte = pd.Timestamp(corte)
        eq = preparar_equipos(eq)
        self.equipo = {s: ModeloConteo().fit(eq, s, corte) for s in STATS_EQUIPO}

        ju = ju.copy()
        ju["fecha"] = pd.to_datetime(ju["fecha"])
        ju = ju[ju["fecha"] < corte]
        base = eq[["partido_id", "equipo_id"] + STATS_JUGADOR].rename(columns={s: f"eq_{s}" for s in STATS_JUGADOR})
        ju = ju.merge(base, on=["partido_id", "equipo_id"], how="inner")
        ju["w"] = np.exp(-self.xi_jugador * (corte - ju["fecha"]).dt.days)
        ju["frac"] = ju["minutos"] / 90

        # Parte media de cada posición (por 90 minutos) = punto de partida de cada jugador.
        self.parte_posicion = {}
        for s in STATS_JUGADOR:
            g = ju.groupby("posicion").apply(lambda d: (d["w"] * d[s]).sum() / (d["w"] * d[f"eq_{s}"] * d["frac"]).sum(),
                                             include_groups=False)
            self.parte_posicion[s] = g.to_dict()
        media_equipo = {s: eq[s].mean() for s in STATS_JUGADOR}

        filas = []
        for (jid, pos), d in ju.groupby(["jugador_id", "posicion"]):
            ultimo = d.loc[d["fecha"].idxmax()]
            f = {"jugador_id": jid, "jugador": ultimo["jugador"], "posicion": pos,
                 "equipo_id": ultimo["equipo_id"], "ultimo_partido": ultimo["fecha"]}
            for s in STATS_JUGADOR:
                alfa = self.muestra_previa * media_equipo[s]
                prior = self.parte_posicion[s].get(pos, 0.08)
                num = (d["w"] * d[s]).sum() + alfa * prior
                den = (d["w"] * d[f"eq_{s}"] * d["frac"]).sum() + alfa
                f[f"parte_{s}"] = num / den
            titular = d[d["titular"] == 1].sort_values("fecha").tail(8)
            f["min_titular"] = titular["minutos"].mean() if len(titular) else (90 if pos == "POR" else 75)
            ult = d.sort_values("fecha").tail(5)
            f["titular_ultimos5"] = int(ult["titular"].sum())
            f["partidos"] = len(d)
            f["minutos_total"] = d["minutos"].sum()
            filas.append(f)
        self.jugadores = pd.DataFrame(filas)
        # Un jugador puede salir con dos posiciones: nos quedamos con la de más minutos.
        self.jugadores = (self.jugadores.sort_values("minutos_total", ascending=False)
                          .drop_duplicates("jugador_id").set_index("jugador_id"))
        return self

    def equipo_partido(self, local, visitante, competicion="La Liga") -> dict:
        """{stat: (media_local, media_visitante, k)} para el partido."""
        res = {}
        for s, m in self.equipo.items():
            ml, mv = m.media(local, visitante, True, competicion), m.media(visitante, local, False, competicion)
            if ml is not None and mv is not None:
                c = self.cal.get("equipo", {}).get(s)
                ref = m.referencia(competicion)
                res[s] = (calibrar(ml, c, ref), calibrar(mv, c, ref), c["k"] if c else m.k)
        return res

    def k_jugador(self) -> dict:
        """Dispersión de cada estadística de jugador (None = Poisson)."""
        return {s: (self.cal.get("jugador", {}).get(s) or {}).get("k") for s in STATS_JUGADOR}

    def medias_equipo_jugador(self, equipo, rival, local: bool, competicion="La Liga") -> dict:
        """Total esperado del equipo para cada estadística de jugador."""
        res = {}
        for s in STATS_JUGADOR:
            if s == "faltas_recibidas":  # las faltas que recibes son las que comete el rival
                res[s] = self.equipo["faltas"].media(rival, equipo, not local, competicion)
            else:
                res[s] = self.equipo[s].media(equipo, rival, local, competicion)
        return res

    def jugadores_partido(self, equipo, rival, local: bool, desde, ids_titulares=(), ids_bajas=(),
                          competicion="La Liga") -> list[dict]:
        """Previsión de cada jugador de la plantilla si es titular."""
        medias = self.medias_equipo_jugador(equipo, rival, local, competicion)
        if any(v is None for v in medias.values()):
            return []
        plantilla = self.jugadores[(self.jugadores["equipo_id"] == equipo) &
                                   (self.jugadores["ultimo_partido"] >= pd.Timestamp(desde))]
        res = []
        for jid, j in plantilla.iterrows():
            if jid in ids_bajas or j["posicion"] == "POR":
                continue
            # Sin once probable publicado: titular probable = titular en 3 de los últimos 5.
            titular = jid in ids_titulares if ids_titulares else j["titular_ultimos5"] >= 3
            minutos = min(j["min_titular"], 90)
            res.append({
                "id": int(jid), "nombre": j["jugador"], "pos": j["posicion"],
                "titular": bool(titular), "titular_ultimos5": int(j["titular_ultimos5"]),
                "minutos": round(float(minutos)),
                **{s: round(float(calibrar(j[f"parte_{s}"] * medias[s] * minutos / 90,
                                           self.cal.get("jugador", {}).get(s))), 3) for s in STATS_JUGADOR},
            })
        return sorted(res, key=lambda r: (not r["titular"], -r["remates"]))
