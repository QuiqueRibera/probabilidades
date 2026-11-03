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
    "E1": -0.30, "SP2": -0.45, "D2": -0.42, "I2": -0.50, "F2": -0.52,
    "AUT": -0.42, "SWZ": -0.45, "DNK": -0.45, "POL": -0.50, "NOR": -0.52, "SWE": -0.55,
    "ROU": -0.60, "IRL": -0.80,
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
    # Rivales de Europa League y Conference League (ligas "extra" de football-data)
    "fc kobenhavn": "FC Copenhagen", "agf": "Aarhus", "jagiellonia bialystok": "Jagiellonia",
    "heart of midlothian": "Hearts", "universitatea craiova": "Univ. Craiova", "fcsb": "FCSB",
    "cfr cluj": "CFR Cluj", "legia warszawa": "Legia", "rakow czestochowa": "Rakow",
    "rapid wien": "SK Rapid", "sk rapid": "SK Rapid", "red bull salzburg": "Salzburg",
    "bsc young boys": "Young Boys", "fc basel": "Basel", "shamrock rovers": "Shamrock Rovers",
}


def _sin_acentos(s: str) -> str:
    # ø, ł, æ... no se descomponen en letra + acento: se cambian a mano.
    s = s.lower().strip().translate(str.maketrans({"ø": "o", "ł": "l", "æ": "ae", "ß": "ss", "đ": "d", "ı": "i"}))
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


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
        self.mapa: dict[str, str] = {}  # nombre FotMob -> nombre football-data
        self.extra: dict[str, tuple] = {}  # equipos sin liga en football-data: (ataque, defensa, partidos)

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

    def aprender_nombres(self, fotmob_eq: pd.DataFrame, df: pd.DataFrame, minimo=3) -> "ModeloEuropa":
        """Empareja los nombres de FotMob con los de football-data por los partidos ya jugados.

        Cada equipo de FotMob se asigna al de football-data con el que comparte más partidos
        (misma fecha y mismo marcador), así no hay que mantener alias a mano para cada liga.
        """
        loc = fotmob_eq[fotmob_eq["local"] == 1][["partido_id", "fecha", "equipo", "goles"]]
        vis = fotmob_eq[fotmob_eq["local"] == 0][["partido_id", "equipo", "goles"]]
        f = loc.merge(vis, on="partido_id", suffixes=("_l", "_v")).dropna(subset=["goles_l", "goles_v"])
        f["Date"] = pd.to_datetime(f["fecha"])
        cruce = f.merge(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]],
                        left_on=["Date", "goles_l", "goles_v"], right_on=["Date", "FTHG", "FTAG"])
        votos = pd.concat([cruce[["equipo_l", "HomeTeam"]].set_axis(["fm", "fd"], axis=1),
                           cruce[["equipo_v", "AwayTeam"]].set_axis(["fm", "fd"], axis=1)]).value_counts()
        for fm, grupo in votos.groupby(level="fm"):
            grupo = grupo.sort_values(ascending=False)
            mejor, n1 = grupo.index[0][1], grupo.iloc[0]
            n2 = grupo.iloc[1] if len(grupo) > 1 else 0
            if n1 >= minimo and n1 >= 2 * n2 and mejor in self.liga_de:
                self.mapa[fm] = mejor
        return self

    def estimar_desconocidos(self, fotmob_eq: pd.DataFrame, prior=-0.45, peso_prior=3.0) -> "ModeloEuropa":
        """Fuerza de los equipos sin liga en football-data (Chequia, Croacia, Serbia...).

        Se estima con sus partidos europeos de FotMob contra rivales que sí están en el modelo:
        un ataque y una defensa (con la fuerza de su liga ya incluida) a partir de xG y goles.
        Se parte de un equipo medio de una liga modesta, que pesa como peso_prior partidos.
        """
        rival = fotmob_eq[["partido_id", "equipo_id", "goles", "xg"]].rename(
            columns={"equipo_id": "rival_id", "goles": "goles_c", "xg": "xg_c"})
        lado = fotmob_eq.merge(rival, on=["partido_id", "rival_id"]).dropna(subset=["goles", "goles_c"])
        ms = list(self.modelos.values())
        mu, home = np.mean([m.mu for m in ms]), np.mean([m.local for m in ms])
        base = peso_prior * np.exp(mu + home / 2)
        mezcla = lambda xg, g: g if pd.isna(xg) else 0.6 * xg + 0.4 * g
        for nombre, g in lado.groupby("equipo"):
            if self.buscar_equipo(nombre):
                continue
            obs_f = esp_f = obs_c = esp_c = 0.0
            n = 0
            for r in g.itertuples():
                o = self.buscar_equipo(r.rival)
                if o is None or o in self.extra:
                    continue
                lo = self.liga_de[o]
                mo, so = self.modelos[lo], FUERZA_LIGA.get(lo, -0.4)
                obs_f += mezcla(r.xg, r.goles)
                esp_f += np.exp(mu + (home if r.local else 0) + mo.defensa[o] - so)
                obs_c += mezcla(r.xg_c, r.goles_c)
                esp_c += np.exp(mu + (0 if r.local else home) + mo.ataque[o] + so)
                n += 1
            ataque = np.log((obs_f + base * np.exp(prior)) / (esp_f + base))
            defensa = np.log((obs_c + base * np.exp(-prior)) / (esp_c + base))
            self.extra[nombre] = (float(ataque), float(defensa), n)
        return self

    def buscar_equipo(self, nombre: str) -> str | None:
        if nombre in self.mapa:
            return self.mapa[nombre]
        if nombre in self.liga_de or nombre in self.extra:
            return nombre
        clave = _sin_acentos(nombre)
        if clave in ALIAS and ALIAS[clave] in self.liga_de:
            return ALIAS[clave]
        # Por parecido de nombre, solo entre equipos que no tengan ya su pareja de FotMob
        # (si no, "Pafos FC" acabaría siendo "Paris FC").
        ocupados = set(self.mapa.values())
        normal = {_sin_acentos(e): e for e in self.liga_de if e not in ocupados}
        cerca = difflib.get_close_matches(clave, list(normal), n=1, cutoff=0.75)
        return normal[cerca[0]] if cerca else None

    def prediccion(self, local: str, visitante: str, neutral=False):
        """Devuelve (λ_local, λ_visitante, rho) o None si algún equipo no está en los datos."""
        l1, l2 = self.liga_de.get(local), self.liga_de.get(visitante)
        if local in self.extra or visitante in self.extra:
            return self._prediccion_extra(local, visitante, neutral)
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

    def _prediccion_extra(self, local, visitante, neutral):
        """Cruce en el que al menos un equipo tiene la fuerza estimada (estimar_desconocidos)."""
        fuerzas, ms = [], []
        for e in (local, visitante):
            if e in self.extra:
                fuerzas.append(self.extra[e][:2])
            elif e in self.liga_de:
                l = self.liga_de[e]
                m, s = self.modelos[l], FUERZA_LIGA.get(l, -0.4)
                fuerzas.append((m.ataque[e] + s, m.defensa[e] - s))
                ms.append(m)
            else:
                return None
        ms = ms or list(self.modelos.values())
        mu, home = np.mean([m.mu for m in ms]), 0 if neutral else np.mean([m.local for m in ms])
        (a1, d1), (a2, d2) = fuerzas
        return float(np.exp(mu + home + a1 + d2)), float(np.exp(mu + a2 + d1)), float(np.mean([m.rho for m in ms]))
