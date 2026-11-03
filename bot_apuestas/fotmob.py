"""Estadísticas por jugador y por equipo de cada partido (FotMob).

Los partidos terminados se guardan en dos CSV compactos (datos_fotmob/) y no se vuelven a
descargar, así que tras la primera carga solo se piden los partidos nuevos.
"""
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://www.fotmob.com/api/data"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
DIR = Path(__file__).parent / "datos_fotmob"
PAUSA = 0.8  # segundos entre peticiones, para no saturar a FotMob

COMPETICIONES = {87: "La Liga", 42: "Champions", 47: "Premier League", 54: "Bundesliga",
                 55: "Serie A", 53: "Ligue 1", 140: "Segunda", 48: "Championship",
                 146: "2. Bundesliga", 86: "Serie B", 110: "Ligue 2", 73: "Europa League",
                 10216: "Conference League"}
# Temporadas a cargar por competición.
TEMPORADAS = {87: ["2025/2026", "2026/2027"], 42: ["2025/2026", "2026/2027"],
              **{c: ["2026/2027"] for c in COMPETICIONES if c not in (87, 42)}}

POSICION = {0: "POR", 1: "DEF", 2: "MED", 3: "DEL"}
STATS_JUGADOR = {"Minutes played": "minutos", "Total shots": "remates", "Shots on target": "tiros_puerta",
                 "Tackles": "entradas", "Was fouled": "faltas_recibidas", "Fouls committed": "faltas"}
STATS_EQUIPO = {"corners": "corners", "total_shots": "remates", "ShotsOnTarget": "tiros_puerta",
                "fouls": "faltas", "yellow_cards": "amarillas", "matchstats.headers.tackles": "entradas",
                "expected_goals": "xg"}

_sesion = requests.Session()
_sesion.headers.update(UA)


def _get(ruta, **params):
    for intento in range(3):
        try:
            r = _sesion.get(f"{API}/{ruta}", params=params, timeout=30)
            if r.status_code == 200:
                time.sleep(PAUSA)
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(5 * (intento + 1))
    raise RuntimeError(f"FotMob no responde: {ruta} {params}")


def partidos_competicion(liga_id: int, temporada: str | None = None) -> list[dict]:
    params = {"id": liga_id}
    if temporada:
        params["season"] = temporada
    datos = _get("leagues", **params)
    res = []
    for m in datos["fixtures"]["allMatches"]:
        st = m["status"]
        if st.get("cancelled"):
            continue
        res.append({"id": int(m["id"]), "liga_id": liga_id, "fecha": st.get("utcTime"),
                    "local": m["home"]["name"], "local_id": int(m["home"]["id"]),
                    "visitante": m["away"]["name"], "visitante_id": int(m["away"]["id"]),
                    "terminado": bool(st.get("finished")), "empezado": bool(st.get("started"))})
    return res


def _num(v):
    if isinstance(v, str):
        v = v.split(" ")[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def extraer_partido(partido: dict) -> tuple[list[dict], list[dict]]:
    """Filas de equipo y de jugador de un partido terminado."""
    det = _get("matchDetails", matchId=partido["id"])
    c = det.get("content") or {}
    base = {"partido_id": partido["id"], "fecha": partido["fecha"][:10],
            "competicion": COMPETICIONES[partido["liga_id"]]}
    lados = [(partido["local_id"], partido["local"], partido["visitante_id"], partido["visitante"], 1),
             (partido["visitante_id"], partido["visitante"], partido["local_id"], partido["local"], 0)]

    equipos = [dict(base, equipo_id=e, equipo=n, rival_id=ri, rival=rn, local=l) for e, n, ri, rn, l in lados]
    try:
        for bloque in c["stats"]["Periods"]["All"]["stats"]:
            for s in bloque["stats"]:
                col = STATS_EQUIPO.get(s.get("key"))
                if col and s["stats"] and s["stats"][0] is not None:
                    equipos[0][col], equipos[1][col] = _num(s["stats"][0]), _num(s["stats"][1])
    except (KeyError, TypeError):
        pass
    for eq, gol in zip(equipos, _goles(det)):
        eq["goles"] = gol

    titulares = set()
    for lado in ("homeTeam", "awayTeam"):
        for j in ((c.get("lineup") or {}).get(lado) or {}).get("starters", []):
            titulares.add(j.get("id"))

    jugadores = []
    rival = {partido["local_id"]: lados[0][2:4], partido["visitante_id"]: lados[1][2:4]}
    for p in (c.get("playerStats") or {}).values():
        valores = {}
        for bloque in p.get("stats", []):
            for k, v in bloque.get("stats", {}).items():
                if k in STATS_JUGADOR:
                    valores[STATS_JUGADOR[k]] = _num(v.get("stat", {}).get("value"))
        if not valores.get("minutos"):
            continue
        eq_id = int(p["teamId"])
        fila = dict(base, jugador_id=int(p["id"]), jugador=p["name"], equipo_id=eq_id, equipo=p.get("teamName"),
                    rival_id=rival[eq_id][0], rival=rival[eq_id][1], local=int(eq_id == partido["local_id"]),
                    posicion=POSICION.get(p.get("usualPosition"), "?") if not p.get("isGoalkeeper") else "POR",
                    titular=int(p["id"] in titulares))
        fila.update({col: valores.get(col, 0.0) for col in STATS_JUGADOR.values()})
        jugadores.append(fila)
    return equipos, jugadores


def _goles(det):
    try:
        teams = det["header"]["teams"]
        return [int(teams[0]["score"]), int(teams[1]["score"])]
    except (KeyError, TypeError, ValueError, IndexError):
        return [None, None]


def actualizar(competiciones=TEMPORADAS, verbose=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga los partidos terminados que aún no estén en la caché."""
    DIR.mkdir(exist_ok=True)
    f_eq, f_ju = DIR / "equipos.csv", DIR / "jugadores.csv"
    eq = pd.read_csv(f_eq) if f_eq.exists() else pd.DataFrame()
    ju = pd.read_csv(f_ju) if f_ju.exists() else pd.DataFrame()
    hechos = set(eq["partido_id"]) if len(eq) else set()

    pendientes = []
    for liga, temporadas in competiciones.items():
        for t in temporadas:
            pendientes += [m for m in partidos_competicion(liga, t) if m["terminado"] and m["id"] not in hechos]
    if verbose:
        print(f"FotMob: {len(pendientes)} partidos nuevos que descargar")

    nuevos_eq, nuevos_ju = [], []
    for i, m in enumerate(pendientes, 1):
        try:
            e, j = extraer_partido(m)
        except RuntimeError as err:
            print(f"  aviso: {err}")
            continue
        nuevos_eq += e
        nuevos_ju += j
        if i % 50 == 0 or i == len(pendientes):
            # Guardado parcial: si se corta la descarga no se pierde lo bajado.
            eq = pd.concat([eq, pd.DataFrame(nuevos_eq)], ignore_index=True)
            ju = pd.concat([ju, pd.DataFrame(nuevos_ju)], ignore_index=True)
            nuevos_eq, nuevos_ju = [], []
            eq.to_csv(f_eq, index=False)
            ju.to_csv(f_ju, index=False)
            if verbose:
                print(f"  {i}/{len(pendientes)}")
    return eq, ju


def proximos_partidos(dias=8, ligas=tuple(COMPETICIONES)) -> list[dict]:
    """Partidos de los próximos días de todas las competiciones, con alineación probable y bajas."""
    ahora = datetime.now(timezone.utc)
    res = []
    for liga in ligas:
        for m in partidos_competicion(liga):
            if m["empezado"] or not m["fecha"]:
                continue
            fecha = datetime.fromisoformat(m["fecha"].replace("Z", "+00:00"))
            if fecha > ahora + timedelta(days=dias):
                continue
            det = _get("matchDetails", matchId=m["id"])
            lineup = (det.get("content") or {}).get("lineup") or {}
            m["competicion"] = COMPETICIONES[liga]
            # FotMob: 'lastStarting11' = repite el último once, 'predicted' = su previsión,
            # cualquier otro valor = alineación oficial.
            tipo = lineup.get("lineupType")
            m["alineacion"] = {None: "habituales", "lastStarting11": "ultimo_once", "predicted": "prevista"}.get(tipo, "confirmada")
            for lado, clave in (("homeTeam", "local"), ("awayTeam", "visitante")):
                t = lineup.get(lado) or {}
                m[f"titulares_{clave}"] = [j["id"] for j in t.get("starters", [])]
                m[f"bajas_{clave}"] = [{"id": j["id"], "nombre": j["name"],
                                        "motivo": (j.get("unavailability") or {}).get("type", "")}
                                       for j in t.get("unavailable", [])]
            res.append(m)
    return sorted(res, key=lambda m: m["fecha"])
