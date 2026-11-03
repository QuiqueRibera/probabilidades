"""Descarga y caché de resultados, estadísticas y cuotas de bet365 (football-data.co.uk)."""
import io
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.football-data.co.uk"
CACHE = Path(__file__).parent / "cache"

# Ligas con datos en football-data.co.uk. Las extranjeras sirven para los rivales de Champions.
LIGAS = {
    "SP1": "La Liga",
    "E0": "Premier League",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "B1": "Bélgica",
    "T1": "Turquía",
    "SC0": "Escocia",
    "G1": "Grecia",
}

COLUMNAS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HxG", "AxG", "HST", "AST",
    "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5", "AHh", "B365AHH", "B365AHA",
]


def temporada(fecha: date) -> str:
    """Código de temporada de football-data: 2026-27 -> '2627'."""
    y = fecha.year if fecha.month >= 7 else fecha.year - 1
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def temporadas_recientes(n: int, hoy: date) -> list[str]:
    actual = int(temporada(hoy)[:2])
    return [f"{(actual - i) % 100:02d}{(actual - i + 1) % 100:02d}" for i in range(n - 1, -1, -1)]


def _descargar(url: str, destino: Path, max_horas: float) -> bytes:
    if destino.exists() and time.time() - destino.stat().st_mtime < max_horas * 3600:
        return destino.read_bytes()
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "bot-apuestas/1.0"})
        r.raise_for_status()
    except requests.RequestException:
        if destino.exists():  # sin conexión: tirar de la caché aunque esté vieja
            return destino.read_bytes()
        raise
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.content)
    return r.content


def _leer_csv(contenido: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(contenido), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo leer el CSV")


def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=COLUMNAS).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    for c in COLUMNAS[5:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])


def cargar_resultados(ligas=("SP1",), n_temporadas=3, hoy: date | None = None) -> pd.DataFrame:
    hoy = hoy or date.today()
    actual = temporada(hoy)
    partes = []
    for liga in ligas:
        for t in temporadas_recientes(n_temporadas, hoy):
            # Las temporadas cerradas no cambian; la actual se refresca cada 6 h.
            horas = 6 if t == actual else 24 * 365
            try:
                raw = _descargar(f"{BASE}/mmz4281/{t}/{liga}.csv", CACHE / f"{t}_{liga}.csv", horas)
            except requests.RequestException as e:
                print(f"  aviso: no se pudo descargar {liga} {t}: {e}")
                continue
            df = _normalizar(_leer_csv(raw))
            df["Div"] = liga
            df["temporada"] = t
            partes.append(df)
    res = pd.concat(partes, ignore_index=True)
    return res.dropna(subset=["FTHG", "FTAG"]).sort_values("Date").reset_index(drop=True)


def cargar_proximos() -> pd.DataFrame:
    """Próxima jornada de las ligas con cuotas de bet365 (se actualiza los jueves/viernes)."""
    raw = _descargar(f"{BASE}/fixtures.csv", CACHE / "fixtures.csv", 6)
    return _normalizar(_leer_csv(raw))
