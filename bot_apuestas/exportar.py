"""Genera docs/data.json para la app del móvil.

    python exportar.py            # actualiza datos, entrena modelos y escribe docs/data.json

Es lo que ejecuta GitHub Actions dos veces al día.
"""
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import fotmob
from analisis import analizar_partido
from datos import LIGAS, cargar_proximos, cargar_resultados
from estadisticas import ModeloEstadisticas
from mercados import cuotas_bet365
from modelo import ModeloEuropa

AQUI = Path(__file__).parent
SALIDA = AQUI / "docs" / "data.json"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ahora = datetime.now(timezone.utc)
    hoy = pd.Timestamp(ahora.date())

    print("1/5 Estadísticas de partidos (FotMob)")
    eq, ju = fotmob.actualizar()
    print("2/5 Resultados y cuotas bet365 (football-data)")
    resultados = cargar_resultados(list(LIGAS), n_temporadas=3)
    cuotas = cargar_proximos()
    print("3/5 Entrenando modelos")
    goles = ModeloEuropa().fit(resultados, hoy + pd.Timedelta(days=1))
    stats = ModeloEstadisticas().fit(eq, ju, hoy + pd.Timedelta(days=1))
    print("4/5 Próximos partidos")
    proximos = fotmob.proximos_partidos(dias=8)

    salida = []
    for m in proximos:
        loc_fd, vis_fd = goles.buscar_equipo(m["local"]), goles.buscar_equipo(m["visitante"])
        if loc_fd is None or vis_fd is None:
            print(f"  aviso: sin modelo de goles para {m['local']} / {m['visitante']}")
        fila = cuotas[(cuotas["HomeTeam"] == loc_fd) & (cuotas["AwayTeam"] == vis_fd)]
        c = cuotas_bet365(fila.iloc[0]) if len(fila) else {}
        p = analizar_partido(goles, m["fecha"], m["competicion"], m["local"], m["visitante"], c)

        desde = hoy - pd.Timedelta(days=90)
        jug = {}
        for lado, eq_id, rival, es_local in (("local", m["local_id"], m["visitante_id"], True),
                                             ("visitante", m["visitante_id"], m["local_id"], False)):
            jug[lado] = stats.jugadores_partido(eq_id, rival, es_local, desde, set(m[f"titulares_{lado}"]),
                                                {b["id"] for b in m[f"bajas_{lado}"]}, m["competicion"])
        salida.append({
            "id": m["id"], "fecha": m["fecha"], "competicion": m["competicion"],
            "local": m["local"], "visitante": m["visitante"],
            "alineacion": m["alineacion"],
            "goles": None if p is None else {
                "lh": round(p.lh, 3), "la": round(p.la, 3), "rho": round(p.rho, 4),
                "lh_modelo": None if p.lh_modelo is None else round(p.lh_modelo, 3),
                "la_modelo": None if p.la_modelo is None else round(p.la_modelo, 3),
                "matriz": [[round(float(x), 5) for x in fila_p] for fila_p in p.P],
                "cuotas": c,
            },
            "equipo": {s: [round(a, 3), round(b, 3), None if k is None else round(k, 2)]
                       for s, (a, b, k) in stats.equipo_partido(m["local_id"], m["visitante_id"], m["competicion"]).items()},
            "jugadores": jug,
            "bajas": {"local": m["bajas_local"], "visitante": m["bajas_visitante"]},
        })

    print(f"5/5 Escribiendo {SALIDA} ({len(salida)} partidos)")
    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(json.dumps({"generado": ahora.isoformat(timespec="minutes"), "k_jugador": stats.k_jugador(),
                                  "correlacion": stats.cal.get("correlacion"),
                                  "partidos": salida},
                                 ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
