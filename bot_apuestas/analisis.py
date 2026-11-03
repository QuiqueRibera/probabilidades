"""Análisis por partido y generación de apuestas simples y combinadas."""
from dataclasses import dataclass, field
from itertools import combinations, product
from math import prod

from mercados import (PATAS_COMBINADA, SELECCIONES, ajustar_a_mercado, etiqueta,
                      mascara, matriz_marcadores, probabilidad, probabilidades_mercado)


@dataclass
class Partido:
    fecha: str
    competicion: str
    local: str
    visitante: str
    cuotas: dict[str, float]
    lh_modelo: float | None
    la_modelo: float | None
    lh: float
    la: float
    rho: float
    p_mercado: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.P = matriz_marcadores(self.lh, self.la, self.rho)
        self.P_modelo = (matriz_marcadores(self.lh_modelo, self.la_modelo, self.rho)
                         if self.lh_modelo is not None else None)

    @property
    def nombre(self):
        return f"{self.local} - {self.visitante}"

    def prob(self, clave):
        return probabilidad(self.P, clave)

    def prob_modelo(self, clave):
        return probabilidad(self.P_modelo, clave) if self.P_modelo is not None else None

    def marcadores_probables(self, n=5):
        idx = sorted(((self.P[i, j], i, j) for i in range(6) for j in range(6)), reverse=True)
        return [(f"{i}-{j}", p) for p, i, j in idx[:n]]


@dataclass
class Apuesta:
    patas: list[tuple[Partido, str]]  # (partido, clave de selección; '+' = Crear Apuesta)
    prob: float
    prob_modelo: float | None
    cuota: float | None  # cuota bet365 si se conoce

    @property
    def cuota_justa(self):
        return 1 / self.prob if self.prob > 0 else float("inf")

    @property
    def ev(self):
        """Valor esperado por unidad apostada (0.05 = +5 %)."""
        return self.prob * self.cuota - 1 if self.cuota else None

    def descripcion(self):
        return " | ".join(f"{p.nombre}: {etiqueta(k, p.local, p.visitante)}" for p, k in self.patas)


def analizar_partido(modelo, fecha, competicion, local, visitante, cuotas, peso_modelo=0.2,
                     neutral=False) -> Partido | None:
    loc, vis = modelo.buscar_equipo(local), modelo.buscar_equipo(visitante)
    pred = modelo.prediccion(loc, vis, neutral) if loc and vis else None
    p_mercado = probabilidades_mercado(cuotas)
    if pred is None:
        if not p_mercado:
            return None
        # Equipo sin datos: solo se puede usar lo que dice el mercado.
        lh, la = ajustar_a_mercado(1.4, 1.1, -0.05, p_mercado)
        return Partido(fecha, competicion, local, visitante, cuotas, None, None, lh, la, -0.05, p_mercado)

    lh_m, la_m, rho = pred
    P_m = matriz_marcadores(lh_m, la_m, rho)
    objetivos = {k: peso_modelo * probabilidad(P_m, k) + (1 - peso_modelo) * p for k, p in p_mercado.items()}
    # Sin cuota de resultado o de goles, ese aspecto lo fija el modelo (si no, quedaría libre).
    for k in ("1", "2", "O2.5"):
        objetivos.setdefault(k, probabilidad(P_m, k))
    lh, la = ajustar_a_mercado(lh_m, la_m, rho, objetivos)
    return Partido(fecha, competicion, loc or local, vis or visitante, cuotas, lh_m, la_m, lh, la, rho, p_mercado)


def simples(partidos: list[Partido]) -> list[Apuesta]:
    return [Apuesta([(p, k)], p.prob(k), p.prob_modelo(k), p.cuotas.get(k))
            for p in partidos for k in SELECCIONES]


def crear_apuesta(partido: Partido, max_patas=3, cuota_min=1.5) -> list[Apuesta]:
    """Combinaciones dentro del mismo partido con probabilidad exacta (correlación incluida)."""
    # Primero las combinaciones de las que has metido la cuota de bet365 a mano.
    res = [Apuesta([(partido, k)], partido.prob(k), partido.prob_modelo(k), c)
           for k, c in partido.cuotas.items() if "+" in k]
    vistas = {mascara(a.patas[0][1]).tobytes() for a in res}
    for n in range(2, max_patas + 1):
        for patas in combinations(PATAS_COMBINADA, n):
            clave = "+".join(patas)
            p = partido.prob(clave)
            if p < 0.03 or 1 / p < cuota_min:
                continue
            # Cada pata debe aportar algo: si quitarla apenas cambia la probabilidad, sobra.
            if any(partido.prob("+".join(q for q in patas if q != x)) - p < 0.02 for x in patas):
                continue
            # Dos combinaciones que cubren los mismos marcadores son la misma apuesta.
            huella = mascara(clave).tobytes()
            if huella in vistas:
                continue
            vistas.add(huella)
            res.append(Apuesta([(partido, clave)], p, partido.prob_modelo(clave), partido.cuotas.get(clave)))
    return sorted(res, key=lambda a: (a.cuota is None, -a.prob))


def combinadas(partidos: list[Partido], max_patas=3, prob_min_pata=0.45, patas_por_partido=4) -> list[Apuesta]:
    """Combinadas de varios partidos con cuota real de bet365 (partidos independientes)."""
    candidatas = []
    for p in partidos:
        opciones = [(p, k) for k in p.cuotas if "+" not in k and p.prob(k) >= prob_min_pata]
        opciones.sort(key=lambda o: -(o[0].prob(o[1]) * o[0].cuotas[o[1]]))  # mejor valor primero
        if opciones:
            candidatas.append(opciones[:patas_por_partido])
    res = []
    for n in range(2, max_patas + 1):
        for grupo in combinations(candidatas, n):
            for patas in product(*grupo):
                pm = [p.prob_modelo(k) for p, k in patas]
                res.append(Apuesta(list(patas), prod(p.prob(k) for p, k in patas),
                                   prod(pm) if None not in pm else None,
                                   prod(p.cuotas[k] for p, k in patas)))
    return res
