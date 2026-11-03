"""Prueba histórica: ¿qué tal acierta el bot partidos que todavía no había visto?

Para cada semana de las temporadas de prueba entrena el modelo solo con los partidos
anteriores, predice esa jornada y compara con el resultado real y con bet365.

    python backtest.py                 # La Liga 2025-26 y lo que va de 2026-27
"""
import sys

import numpy as np
import pandas as pd

from datos import cargar_resultados
from mercados import matriz_marcadores, probabilidad, quitar_margen
from modelo import DixonColes


def main(liga="SP1", temporadas=("2526", "2627")):
    sys.stdout.reconfigure(encoding="utf-8")
    df = cargar_resultados([liga], n_temporadas=5)
    prueba = df[df["temporada"].isin(temporadas)].dropna(subset=["B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"])
    filas = []
    for semana, jornada in prueba.groupby(prueba["Date"].dt.to_period("W-MON")):
        m = DixonColes().fit(df, semana.start_time)
        for _, f in jornada.iterrows():
            if f["HomeTeam"] not in m.ataque or f["AwayTeam"] not in m.ataque:
                continue
            P = matriz_marcadores(*m.goles_esperados(f["HomeTeam"], f["AwayTeam"]), m.rho)
            mh, md, ma = quitar_margen([f["B365H"], f["B365D"], f["B365A"]])
            mo, _ = quitar_margen([f["B365>2.5"], f["B365<2.5"]])
            filas.append({
                "mod_1": probabilidad(P, "1"), "mod_X": probabilidad(P, "X"), "mod_2": probabilidad(P, "2"),
                "mod_O": probabilidad(P, "O2.5"), "mer_1": mh, "mer_X": md, "mer_2": ma, "mer_O": mo,
                "c_1": f["B365H"], "c_X": f["B365D"], "c_2": f["B365A"], "c_O": f["B365>2.5"], "c_U": f["B365<2.5"],
                "res": "1" if f["FTHG"] > f["FTAG"] else "X" if f["FTHG"] == f["FTAG"] else "2",
                "over": int(f["FTHG"] + f["FTAG"] >= 3),
            })
    r = pd.DataFrame(filas)
    print(f"Partidos evaluados: {len(r)} ({liga}, temporadas {', '.join(temporadas)})\n")

    print("Precisión según el peso del modelo frente al mercado (menor = mejor)")
    print("peso_modelo  logloss 1X2  Brier 1X2  logloss +/-2.5")
    for w in (0.0, 0.2, 0.35, 0.5, 0.75, 1.0):
        p = {k: w * r[f"mod_{k}"] + (1 - w) * r[f"mer_{k}"] for k in ("1", "X", "2", "O")}
        p_res = np.select([r["res"] == "1", r["res"] == "X"], [p["1"], p["X"]], p["2"])
        brier = sum((p[k] - (r["res"] == k)) ** 2 for k in ("1", "X", "2")).mean()
        p_ou = np.where(r["over"] == 1, p["O"], 1 - p["O"])
        print(f"   {w:4.2f}        {-np.log(p_res).mean():.4f}     {brier:.4f}      {-np.log(p_ou).mean():.4f}")

    # Calibración: cuando el bot dice X %, ¿ocurre X % de las veces?
    w = 0.35
    pares = []
    for k in ("1", "X", "2"):
        pares.append(pd.DataFrame({"p": w * r[f"mod_{k}"] + (1 - w) * r[f"mer_{k}"], "y": r["res"] == k}))
    po = w * r["mod_O"] + (1 - w) * r["mer_O"]
    pares += [pd.DataFrame({"p": po, "y": r["over"] == 1}), pd.DataFrame({"p": 1 - po, "y": r["over"] == 0})]
    cal = pd.concat(pares)
    cal["tramo"] = pd.cut(cal["p"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    print(f"\nCalibración (peso_modelo={w}): probabilidad que da el bot vs. frecuencia real")
    print(cal.groupby("tramo", observed=True).agg(n=("y", "size"), prob_media=("p", "mean"), acierto_real=("y", "mean"))
          .to_string(float_format=lambda x: f"{100 * x:.1f}%"))

    # Apostar 1 unidad cada vez que el bot ve valor.
    print("\nSimulación: 1 unidad por apuesta cuando el valor esperado supera el umbral")
    print("probabilidad        umbral  apuestas  aciertos  beneficio   ROI")
    for nombre, w in (("solo modelo", 1.0), ("modelo+mercado", 0.35)):
        for umbral in (0.0, 0.05, 0.10):
            n = gan = acc = 0
            for _, f in r.iterrows():
                for k, ok, c in (("1", f["res"] == "1", f["c_1"]), ("X", f["res"] == "X", f["c_X"]),
                                 ("2", f["res"] == "2", f["c_2"]), ("O", f["over"] == 1, f["c_O"])):
                    p = w * f[f"mod_{k}"] + (1 - w) * f[f"mer_{k}"]
                    if p * c - 1 > umbral:
                        n += 1; acc += ok; gan += (c - 1) if ok else -1
                pu = 1 - (w * f["mod_O"] + (1 - w) * f["mer_O"])
                if pu * f["c_U"] - 1 > umbral:
                    ok = f["over"] == 0
                    n += 1; acc += ok; gan += (f["c_U"] - 1) if ok else -1
            roi = f"{100 * gan / n:+.1f}%" if n else "-"
            print(f"{nombre:<18}  {umbral:>5.0%}  {n:>8}  {acc:>8}  {gan:>+8.1f}  {roi:>6}")


if __name__ == "__main__":
    main()
