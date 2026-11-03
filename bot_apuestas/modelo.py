"""Modelo Dixon-Coles con decaimiento temporal, entrenado sobre goles + xG.

Cada equipo tiene un parámetro de ataque y otro de defensa. Los goles esperados son
    log λ_local     = μ + ventaja_local + ataque_local + defensa_visitante
    log λ_visitante = μ + ataque_visitante + defensa_local
Los partidos antiguos pesan menos (exp(-xi·días)) y, cuando hay xG, el objetivo mezcla
goles reales y xG, que predice mejor el rendimiento futuro que el marcador.
"""
import difflib
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAX_GOLES = 10

# Fuerza relativa de cada liga respecto a La Liga, en unidades de log-goles.
# Solo se usa en partidos entre equipos de ligas distintas (Champions). Es una estimación
# aproximada a partir del coeficiente UEFA: ajústala si ves sesgos.
FUERZA_LIGA = {
    "E0": 0.08, "SP1": 0.0, "I1": -0.03, "D1": -0.04, "F1": -0.10,
    "P1": -0.20, "N1": -0.24, "B1": -0.28, "T1": -0.30, "SC0": -0.38, "G1": -0.35,
}

ALIAS = {
    "barca": "Barcelona", "fc barcelona": "Barcelona", "madrid": "Real Madrid",
    "atletico": "Ath Madrid", "atletico de madrid": "Ath Madrid", "atleti": "Ath Madrid",
    "athletic": "Ath Bilbao", "athletic club": "Ath Bilbao", "real sociedad": "Sociedad",
    "rayo": "Vallecano", "rayo vallecano": "Vallecano", "espanyol": "Espanol",
    "deportivo": "La Coruna", "depor": "La Coruna", "racing": "Santander",
    "bayern": "Bayern Munich", "psg": "Paris SG", "paris": "Paris SG",
    "inter": "Inter", "milan": "Milan", "juve": "Juventus", "dortmund": "Dortmund",
    "leverkusen": "Leverkusen", "manchester city": "Man City", "man city": "Man City",
    "manchester united": "Man United", "liverpool": "Liverpool", "arsenal": "Arsenal",
    "chelsea": "Chelsea", "tottenham": "Tottenham", "newcastle": "Newcastle",
    "benfica": "Benfica", "sporting": "Sp Lisbon", "porto": "Porto",
    "psv": "PSV Eindhoven", "ajax": "Ajax", "brujas": "Club Brugge", "brugge": "Club Brugge",
    "galatasaray": "Galatasaray", "celtic": "Celtic", "olympiacos": "Olympiakos",
    # Nombres de FotMob
    "athletic club": "Ath Bilbao", "atletico madrid": "Ath Madrid", "celta vigo": "Celta",
    "deportivo a coruna": "La Coruna", "deportivo alaves": "Alaves", "racing santander": "Santander",
    "real betis": "Betis", "bayern munchen": "Bayern Munich", "paris saint-germain": "Paris SG",
    "sporting cp": "Sp Lisbon", "borussia dortmund": "Dortmund", "bayer leverkusen": "Leverkusen",
    "eintracht frankfurt": "Ein Frankfurt", "ac milan": "Milan", "newcastle united": "Newcastle",
    "tottenham hotspur": "Tottenham", "internazionale": "Inter", "inter milan": "Inter",
    "psv": "PSV Eindhoven", "club brugge kv": "Club Brugge", "fc porto": "Porto",
    "sl benfica": "Benfica", "olympique marseille": "Marseille", "olympique lyonnais": "Lyon",
    "aek athens": "AEK", "paok": "PAOK", "fenerbahce": "Fenerbahce", "besiktas": "Besiktas",
    "as monaco": "Monaco", "losc lille": "Lille", "rb leipzig": "RB Leipzig", "vfb stuttgart": "Stuttgart",
}


def _sin_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower().strip()) if unicodedata.category(c) != "Mn")


def tau(x, y, lh, la, rho):
    """Corrección de Dixon-Coles para marcadores bajos (0-0, 1-0, 0-1, 1-1)."""
    t = np.ones(np.broadcast(x, y, lh, la).shape)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def matriz_marcadores(lh: float, la: float, rho: float) -> np.ndarray:
    """P[i, j] = probabilidad de que el marcador final sea i (local) - j (visitante)."""
    g = np.arange(MAX_GOLES + 1)
    P = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    H, A = np.indices(P.shape)
    P *= tau(H, A, lh, la, rho)
    return P / P.sum()


class DixonColes:
    def __init__(self, xi=0.0019, peso_xg=0.6, ridge=2.0):
        self.xi = xi            # 0.0019/día ≈ un partido de hace un año pesa la mitad
        self.peso_xg = peso_xg  # peso del xG frente a los goles cuando hay xG
        self.ridge = ridge      # encoge hacia la media a equipos con pocos datos

    def fit(self, df: pd.DataFrame, fecha_corte) -> "DixonColes":
        df = df[df["Date"] < pd.Timestamp(fecha_corte)]
        self.equipos = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        idx = {t: i for i, t in enumerate(self.equipos)}
        n = len(self.equipos)
        h = df["HomeTeam"].map(idx).to_numpy()
        a = df["AwayTeam"].map(idx).to_numpy()
        dias = (pd.Timestamp(fecha_corte) - df["Date"]).dt.days.to_numpy()
        w = np.exp(-self.xi * dias)

        gh, ga = df["FTHG"].to_numpy(float), df["FTAG"].to_numpy(float)
        xgh, xga = df["HxG"].to_numpy(float), df["AxG"].to_numpy(float)
        yh = np.where(np.isnan(xgh), gh, self.peso_xg * xgh + (1 - self.peso_xg) * gh)
        ya = np.where(np.isnan(xga), ga, self.peso_xg * xga + (1 - self.peso_xg) * ga)

        def nll(p):
            att, de, mu, home = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            lh = np.exp(mu + home + att[h] + de[a])
            la = np.exp(mu + att[a] + de[h])
            f = np.sum(w * (lh - yh * np.log(lh))) + np.sum(w * (la - ya * np.log(la)))
            f += self.ridge * (att @ att + de @ de)
            rh, ra = w * (lh - yh), w * (la - ya)
            g_att = np.bincount(h, rh, n) + np.bincount(a, ra, n) + 2 * self.ridge * att
            g_de = np.bincount(a, rh, n) + np.bincount(h, ra, n) + 2 * self.ridge * de
            return f, np.concatenate([g_att, g_de, [rh.sum() + ra.sum(), rh.sum()]])

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = np.log(max(np.average(ya, weights=w), 0.1))
        sol = minimize(nll, x0, jac=True, method="L-BFGS-B")
        self.ataque = dict(zip(self.equipos, sol.x[:n]))
        self.defensa = dict(zip(self.equipos, sol.x[n:2 * n]))
        self.mu, self.local = sol.x[2 * n], sol.x[2 * n + 1]

        # rho se ajusta sobre marcadores reales (enteros), con los λ ya fijados.
        lh = np.exp(self.mu + self.local + sol.x[h] + sol.x[n + a])
        la = np.exp(self.mu + sol.x[a] + sol.x[n + h])
        gh_i, ga_i = gh.astype(int), ga.astype(int)

        def nll_rho(rho):
            t = tau(gh_i, ga_i, lh, la, rho)
            return -np.sum(w * np.log(np.clip(t, 1e-10, None)))

        self.rho = minimize_scalar(nll_rho, bounds=(-0.25, 0.25), method="bounded").x
        self.n_partidos = len(df)
        return self

    def goles_esperados(self, local: str, visitante: str) -> tuple[float, float]:
        lh = np.exp(self.mu + self.local + self.ataque[local] + self.defensa[visitante])
        la = np.exp(self.mu + self.ataque[visitante] + self.defensa[local])
        return float(lh), float(la)


class ModeloEuropa:
    """Un Dixon-Coles por liga + ajuste de fuerza entre ligas para cruces de Champions."""

    def __init__(self, **kw):
        self.kw = kw
        self.modelos: dict[str, DixonColes] = {}
        self.liga_de: dict[str, str] = {}

    def fit(self, df: pd.DataFrame, fecha_corte) -> "ModeloEuropa":
        df = df[df["Date"] < pd.Timestamp(fecha_corte)]
        for liga, sub in df.groupby("Div"):
            if len(sub) < 100:
                continue
            self.modelos[liga] = DixonColes(**self.kw).fit(sub, fecha_corte)
        # La liga de un equipo es la de su último partido (por ascensos/descensos).
        ultimos = pd.concat([df[["Date", "HomeTeam", "Div"]].rename(columns={"HomeTeam": "eq"}),
                             df[["Date", "AwayTeam", "Div"]].rename(columns={"AwayTeam": "eq"})])
        self.liga_de = ultimos.sort_values("Date").groupby("eq")["Div"].last().to_dict()
        self.liga_de = {e: l for e, l in self.liga_de.items() if l in self.modelos}
        return self

    def buscar_equipo(self, nombre: str) -> str | None:
        if nombre in self.liga_de:
            return nombre
        clave = _sin_acentos(nombre)
        if clave in ALIAS and ALIAS[clave] in self.liga_de:
            return ALIAS[clave]
        normal = {_sin_acentos(e): e for e in self.liga_de}
        cerca = difflib.get_close_matches(clave, list(normal), n=1, cutoff=0.75)
        return normal[cerca[0]] if cerca else None

    def prediccion(self, local: str, visitante: str, neutral=False):
        """Devuelve (λ_local, λ_visitante, rho) o None si algún equipo no está en los datos."""
        l1, l2 = self.liga_de.get(local), self.liga_de.get(visitante)
        if l1 is None or l2 is None:
            return None
        m1, m2 = self.modelos[l1], self.modelos[l2]
        if l1 == l2:
            lh, la = m1.goles_esperados(local, visitante)
            if neutral:
                f = np.exp(-m1.local / 2)
                lh, la = lh * f, la / f
            return lh, la, m1.rho
        # Cruce entre ligas: cada equipo se mide respecto a la media de su liga y se
        # corrige con la fuerza relativa de esa liga.
        mu = (m1.mu + m2.mu) / 2
        home = 0 if neutral else (m1.local + m2.local) / 2
        s1, s2 = FUERZA_LIGA.get(l1, -0.4), FUERZA_LIGA.get(l2, -0.4)
        lh = np.exp(mu + home + m1.ataque[local] + s1 + m2.defensa[visitante] - s2)
        la = np.exp(mu + m2.ataque[visitante] + s2 + m1.defensa[local] - s1)
        return float(lh), float(la), (m1.rho + m2.rho) / 2
