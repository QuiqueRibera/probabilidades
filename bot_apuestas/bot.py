"""Bot de análisis de apuestas de fútbol (La Liga y Champions) con cuotas de bet365.

Uso:
    python bot.py                              # próxima jornada de La Liga
    python bot.py --equipos Barcelona "Real Madrid"
    python bot.py --manual cuotas_manual.csv   # añade Champions u otros mercados a mano
"""
import argparse
import html
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from analisis import analizar_partido, combinadas, crear_apuesta, simples
from datos import LIGAS, cargar_proximos, cargar_resultados
from mercados import SELECCIONES, cuotas_bet365, etiqueta
from modelo import ModeloEuropa

AQUI = Path(__file__).parent


def leer_manual(ruta: Path) -> dict:
    """Cuotas introducidas a mano: fecha,competicion,local,visitante,mercado,cuota."""
    if not ruta.exists():
        return {}
    df = pd.read_csv(ruta, comment="#", skipinitialspace=True, dtype=str).dropna(how="all")
    partidos = {}
    for _, f in df.iterrows():
        alias = {"BTTS_SI": "BTTS", "BTTS_NO": "NOBTTS"}
        clave = "+".join(alias.get(c, c) for c in f["mercado"].strip().upper().replace(" ", "").split("+"))
        if any(c not in SELECCIONES for c in clave.split("+")):
            print(f"  aviso: mercado desconocido '{f['mercado']}' (ver README)")
            continue
        k = (f["local"].strip(), f["visitante"].strip())
        p = partidos.setdefault(k, {"fecha": f["fecha"].strip(), "competicion": f["competicion"].strip(), "cuotas": {}})
        p["cuotas"][clave] = float(f["cuota"])
    return partidos


def pct(p):
    return "  -  " if p is None else f"{100 * p:5.1f}%"


def cuota(c):
    return "  -  " if not c else f"{c:5.2f}"


def ev(a):
    return "   -  " if a.ev is None else f"{100 * a.ev:+5.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ligas", nargs="+", default=["SP1"], help="ligas de la jornada a analizar (SP1=La Liga)")
    ap.add_argument("--equipos", nargs="+", help="solo partidos de estos equipos")
    ap.add_argument("--manual", default=str(AQUI / "cuotas_manual.csv"), help="CSV de cuotas a mano")
    ap.add_argument("--peso-modelo", type=float, default=0.2,
                    help="0 = solo mercado, 1 = solo modelo (por defecto 0.2; ver backtest.py)")
    ap.add_argument("--max-patas", type=int, default=3)
    ap.add_argument("--cuota-min", type=float, default=1.8, help="cuota mínima de las combinadas")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--fecha-corte", help="AAAA-MM-DD: entrena solo con datos anteriores (para pruebas)")
    ap.add_argument("--incluir-pasados", action="store_true", help="analiza partidos ya jugados del fixtures.csv")
    ap.add_argument("--html", default=str(AQUI / "informe.html"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    hoy = date.today()
    print("Descargando datos (resultados, xG y cuotas bet365)...")
    proximos = cargar_proximos()
    proximos = proximos[proximos["Div"].isin(args.ligas)]
    if not args.incluir_pasados:
        proximos = proximos[proximos["Date"] >= pd.Timestamp(hoy)]
    manual = leer_manual(Path(args.manual))

    if args.fecha_corte:
        corte = pd.Timestamp(args.fecha_corte)
    elif args.incluir_pasados and len(proximos):
        corte = proximos["Date"].min()  # para no usar los resultados de los partidos analizados
    else:
        corte = pd.Timestamp(hoy) + pd.Timedelta(days=1)

    resultados = cargar_resultados(list(LIGAS), n_temporadas=3, hoy=hoy)
    print(f"Entrenando modelo con {len(resultados)} partidos de {resultados['Div'].nunique()} ligas "
          f"(datos hasta {corte.date()})...")
    modelo = ModeloEuropa().fit(resultados, corte)

    partidos = []
    for _, f in proximos.iterrows():
        c = cuotas_bet365(f)
        c.update(manual.pop((f["HomeTeam"], f["AwayTeam"]), {}).get("cuotas", {}))
        p = analizar_partido(modelo, f"{f['Date']:%d/%m} {f['Time'] if isinstance(f['Time'], str) else ''}".strip(),
                             LIGAS.get(f["Div"], f["Div"]), f["HomeTeam"], f["AwayTeam"], c, args.peso_modelo)
        if p:
            partidos.append(p)
    for (loc, vis), m in manual.items():
        p = analizar_partido(modelo, m["fecha"], m["competicion"], loc, vis, m["cuotas"], args.peso_modelo)
        if p:
            partidos.append(p)
        else:
            print(f"  aviso: sin datos ni cuotas suficientes para {loc} - {vis}")

    if args.equipos:
        buscados = {modelo.buscar_equipo(e) or e for e in args.equipos}
        partidos = [p for p in partidos if {p.local, p.visitante} & buscados]

    if not partidos:
        print("\nNo hay partidos que analizar. football-data.co.uk publica la siguiente jornada "
              "hacia el jueves; mientras tanto puedes añadir partidos en cuotas_manual.csv "
              "o probar con --incluir-pasados.")
        return

    informe = generar_informe(partidos, args)
    Path(args.html).write_text(informe, encoding="utf-8")
    print(f"\nInforme HTML: {args.html}")


def generar_informe(partidos, args) -> str:
    secciones = []
    texto = []

    def tabla(titulo, cabecera, filas, nota=""):
        texto.append(f"\n=== {titulo} ===")
        if nota:
            texto.append(nota)
        if not filas:
            texto.append("(ninguna apuesta cumple los filtros)")
            secciones.append(f"<h2>{html.escape(titulo)}</h2><p class=nota>Ninguna apuesta cumple los filtros.</p>")
            return
        anchos = [max(len(str(x)) for x in col) for col in zip(cabecera, *filas)] if filas else []
        for fila in [cabecera, *filas]:
            texto.append("  ".join(str(x).ljust(w) for x, w in zip(fila, anchos)))
        th = "".join(f"<th>{html.escape(c)}</th>" for c in cabecera)
        trs = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in f) + "</tr>" for f in filas)
        secciones.append(f"<h2>{html.escape(titulo)}</h2>" + (f"<p class=nota>{html.escape(nota)}</p>" if nota else "")
                         + f"<div class=scroll><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>")

    # 1. Resumen por partido
    filas = []
    for p in partidos:
        top = ", ".join(f"{m} ({100 * q:.0f}%)" for m, q in p.marcadores_probables(3))
        filas.append([p.fecha, p.competicion, p.nombre, f"{p.lh:.2f} - {p.la:.2f}",
                      pct(p.prob("1")), pct(p.prob("X")), pct(p.prob("2")),
                      pct(p.prob("O2.5")), pct(p.prob("BTTS")), top])
    tabla("Partidos", ["Fecha", "Comp.", "Partido", "Goles esp.", "1", "X", "2", "+2.5", "Ambos", "Marcadores más probables"],
          filas, "Probabilidad final = modelo estadístico combinado con el mercado (bet365 sin margen).")

    # 2. Apuestas simples con cuota bet365
    s = [a for a in simples(partidos) if a.cuota]
    s.sort(key=lambda a: -a.ev)
    tabla("Apuestas simples con cuota bet365 (ordenadas por valor)",
          ["Apuesta", "Prob.", "Prob. modelo", "Cuota justa", "Cuota bet365", "Valor esperado"],
          [[a.descripcion(), pct(a.prob), pct(a.prob_modelo), cuota(a.cuota_justa), cuota(a.cuota), ev(a)]
           for a in s[:args.top * 2]],
          "Valor esperado > 0 = la cuota paga más de lo que indica la probabilidad estimada.")

    # 3. Combinadas de varios partidos
    comb = [a for a in combinadas(partidos, args.max_patas) if a.cuota >= args.cuota_min]
    seguras = sorted(comb, key=lambda a: -a.prob)[:args.top]
    valor = sorted(comb, key=lambda a: -a.ev)[:args.top]
    cab = ["Combinada", "Prob. acierto", "Cuota justa", "Cuota bet365", "Valor esperado"]
    fila = lambda a: [a.descripcion(), pct(a.prob), cuota(a.cuota_justa), cuota(a.cuota), ev(a)]
    tabla(f"Combinadas más probables (cuota ≥ {args.cuota_min})", cab, [fila(a) for a in seguras])
    tabla("Combinadas con más valor", cab, [fila(a) for a in valor],
          "Cada pata añade el margen de bet365: una combinada casi siempre tiene peor valor que sus simples.")

    # 4. Crear Apuesta (mismo partido)
    for p in partidos:
        ca = crear_apuesta(p, args.max_patas, cuota_min=1.5)[:args.top]
        tabla(f"Crear Apuesta: {p.nombre}", ["Combinación", "Prob. acierto", "Prob. modelo", "Cuota justa", "Cuota bet365", "Valor esperado"],
              [[etiqueta(a.patas[0][1], p.local, p.visitante), pct(a.prob), pct(a.prob_modelo),
                cuota(a.cuota_justa), cuota(a.cuota), ev(a)] for a in ca],
              "Compara la cuota justa con la que te dé bet365 en 'Crear Apuesta': solo hay valor si la suya es mayor.")

    print("\n".join(texto))
    return PLANTILLA.replace("{{CONTENIDO}}", "\n".join(secciones)).replace("{{FECHA}}", date.today().strftime("%d/%m/%Y"))


PLANTILLA = """<!doctype html><html lang=es><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Informe de apuestas</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--card:#fff;--accent:#0b6e4f}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--muted:#9a998f;--line:#2e2d2a;--card:#1c1c1a;--accent:#4cc79a}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;padding:24px 16px}
main{max-width:1100px;margin:auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 6px}
.nota{color:var(--muted);margin:0 0 8px}.aviso{border-left:3px solid var(--accent);padding:8px 12px;background:var(--card)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--card)}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;font-variant-numeric:tabular-nums}
th{font-weight:600;color:var(--muted)}
</style></head><body><main>
<h1>Informe de apuestas</h1><p class=nota>Generado el {{FECHA}} · modelo Dixon-Coles (goles + xG) combinado con cuotas de bet365</p>
<p class=aviso>Las probabilidades son estimaciones. Ningún modelo gana al mercado de forma fiable y el margen de la casa
hace que la mayoría de apuestas pierdan dinero a largo plazo. Apuesta solo lo que estés dispuesto a perder.</p>
{{CONTENIDO}}
</main></body></html>"""


if __name__ == "__main__":
    main()
