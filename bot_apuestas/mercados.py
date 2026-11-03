"""Mercados de apuestas como condiciones sobre el marcador final, y cuotas de bet365.

Toda selección es una condición sobre (goles_local, goles_visitante), así que la
probabilidad de cualquier combinación del mismo partido ("Crear Apuesta" en bet365) se
calcula exactamente sumando la matriz de marcadores, respetando la correlación entre
mercados (p. ej. "gana el Barça" y "más de 2.5" no son independientes).
"""
import numpy as np
from scipy.optimize import brentq, minimize

from modelo import MAX_GOLES, matriz_marcadores

H, A = np.indices((MAX_GOLES + 1, MAX_GOLES + 1))

SELECCIONES = {
    "1": ("Gana {L}", H > A),
    "X": ("Empate", H == A),
    "2": ("Gana {V}", H < A),
    "1X": ("{L} o empate", H >= A),
    "X2": ("Empate o {V}", H <= A),
    "12": ("No hay empate", H != A),
    "O1.5": ("Más de 1.5 goles", H + A >= 2),
    "U1.5": ("Menos de 1.5 goles", H + A <= 1),
    "O2.5": ("Más de 2.5 goles", H + A >= 3),
    "U2.5": ("Menos de 2.5 goles", H + A <= 2),
    "O3.5": ("Más de 3.5 goles", H + A >= 4),
    "U3.5": ("Menos de 3.5 goles", H + A <= 3),
    "BTTS": ("Ambos marcan: Sí", (H > 0) & (A > 0)),
    "NOBTTS": ("Ambos marcan: No", (H == 0) | (A == 0)),
    "L_O0.5": ("{L} marca", H >= 1),
    "V_O0.5": ("{V} marca", A >= 1),
    "L_O1.5": ("{L} marca 2 o más", H >= 2),
    "V_O1.5": ("{V} marca 2 o más", A >= 2),
    "L-1.5": ("{L} gana por 2 o más", H - A >= 2),
    "V-1.5": ("{V} gana por 2 o más", A - H >= 2),
    "L_CS": ("{L} portería a cero", A == 0),
    "V_CS": ("{V} portería a cero", H == 0),
}

# Selecciones que se usan para construir combinadas (evita patas casi idénticas).
PATAS_COMBINADA = ["1", "X", "2", "1X", "X2", "O1.5", "O2.5", "U2.5", "U3.5", "BTTS",
                   "NOBTTS", "L_O1.5", "V_O1.5", "L-1.5", "V-1.5", "L_CS", "V_CS"]


def mascara(clave: str) -> np.ndarray:
    """'1+O2.5' -> marcadores en que gana el local y hay 3 goles o más."""
    m = np.ones_like(H, dtype=bool)
    for parte in clave.split("+"):
        m &= SELECCIONES[parte][1]
    return m


def probabilidad(P: np.ndarray, clave: str) -> float:
    return float(P[mascara(clave)].sum())


def etiqueta(clave: str, local: str, visitante: str) -> str:
    return " + ".join(SELECCIONES[p][0].format(L=local, V=visitante) for p in clave.split("+"))


# --- Cuotas -----------------------------------------------------------------------------

def quitar_margen(cuotas: list[float]) -> list[float]:
    """Probabilidades 'justas' de un mercado completo (método potencia).

    Las casas inflan más las cuotas de los no favoritos; el método potencia corrige ese
    sesgo mejor que repartir el margen a partes iguales.
    """
    p = np.array([1 / c for c in cuotas])
    if p.sum() <= 1:
        return list(p / p.sum())
    k = brentq(lambda k: np.sum(p ** k) - 1, 1, 10)
    return list(p ** k)


def cuotas_bet365(fila) -> dict[str, float]:
    """Cuotas de bet365 de una fila de football-data (1X2, más/menos 2.5 y hándicap)."""
    c = {}
    for clave, col in [("1", "B365H"), ("X", "B365D"), ("2", "B365A"),
                       ("O2.5", "B365>2.5"), ("U2.5", "B365<2.5")]:
        if fila.get(col) == fila.get(col) and fila.get(col):  # descarta NaN
            c[clave] = float(fila[col])
    ah = fila.get("AHh")
    if ah == -1.5 and fila.get("B365AHH") == fila.get("B365AHH"):
        c["L-1.5"] = float(fila["B365AHH"])
    if ah == 1.5 and fila.get("B365AHA") == fila.get("B365AHA"):
        c["V-1.5"] = float(fila["B365AHA"])
    return c


def probabilidades_mercado(cuotas: dict[str, float]) -> dict[str, float]:
    """Probabilidades sin margen para los mercados completos de los que hay cuota."""
    res = {}
    for grupo in (["1", "X", "2"], ["O2.5", "U2.5"], ["O1.5", "U1.5"], ["O3.5", "U3.5"],
                  ["BTTS", "NOBTTS"]):
        if all(k in cuotas for k in grupo):
            res.update(zip(grupo, quitar_margen([cuotas[k] for k in grupo])))
    return res


def ajustar_a_mercado(lh, la, rho, objetivos: dict[str, float]):
    """Busca los goles esperados cuya matriz reproduce las probabilidades objetivo.

    Así el resultado combina modelo y mercado y sigue siendo una única distribución de
    marcadores coherente, de la que salen todos los mercados y combinaciones.
    """
    if not objetivos:
        return lh, la

    def error(x):
        P = matriz_marcadores(np.exp(x[0]), np.exp(x[1]), rho)
        return sum((probabilidad(P, k) - t) ** 2 for k, t in objetivos.items())

    sol = minimize(error, [np.log(lh), np.log(la)], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-9})
    return float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))
