# Bot de análisis de apuestas (La Liga + Champions, cuotas bet365)

Calcula la probabilidad de acierto de apuestas simples, combinadas y "Crear Apuesta" y la
compara con la cuota de bet365 para decirte si hay valor.

## Uso

```
pip install pandas numpy scipy requests
python bot.py                                  # próxima jornada de La Liga
python bot.py --equipos Barça Madrid           # solo esos equipos
python bot.py --ligas SP1 E0                   # La Liga + Premier
python bot.py --incluir-pasados                # prueba con la última jornada ya jugada
python backtest.py                             # ver qué tal acierta con datos pasados
```

Genera `informe.html` y lo muestra también en la consola.

## De dónde salen los datos

- **Resultados, tiros, xG y cuotas de bet365**: football-data.co.uk (gratis). La próxima
  jornada aparece hacia el jueves. No se hace scraping de bet365: sus condiciones lo
  prohíben y lo bloquea.
- **Champions y mercados que no vienen en el CSV** (ambos marcan, tu "Crear Apuesta"…):
  cópialos a mano en `cuotas_manual.csv`. El archivo explica el formato.

## Cómo calcula

1. **Modelo Dixon-Coles**: un ataque y una defensa por equipo, ventaja de local y
   corrección de marcadores bajos. Pesan más los partidos recientes y, cuando hay xG, se
   mezcla con los goles. Se entrena con 3 temporadas de 11 ligas europeas.
2. **Champions**: equipos de ligas distintas se comparan con un ajuste de fuerza por liga
   (`FUERZA_LIGA` en `modelo.py`). Es una estimación aproximada: estos partidos son
   menos fiables que los de liga.
3. **Mezcla con el mercado**: se quita el margen a las cuotas de bet365 y se combina con el
   modelo (`--peso-modelo`, 0.2 por defecto). De ahí sale una matriz de marcadores de la
   que se calculan todos los mercados. Así, las combinaciones del mismo partido tienen en
   cuenta la correlación (ganar y "más de 2.5" no son independientes).
4. **Valor esperado** = probabilidad × cuota − 1. Si es positivo, la cuota paga más de lo
   que debería.

## Lo que dice el backtest (482 partidos de La Liga, 2025-26 y 2026-27)

- Las probabilidades están **bien calibradas**: cuando dice 45 %, acierta ~45 %.
- El modelo solo **no gana al mercado**: las cuotas de bet365 sin margen predicen mejor y,
  cuanto más peso tiene el modelo, peor es la precisión.
- Apostar a todo lo que el modelo ve como "valor" **perdió dinero** (ROI de −12 % a −23 %).

### Corners, remates, faltas y jugadores (`backtest_estadisticas.py`)

Probado con unos 150 partidos de La Liga y Champions de 2026-27 que el modelo no había visto
(la calibración se ajustó con los de diciembre a junio):

- **Jugadores** (remates, tiros a puerta, entradas, faltas): bien calibrado; cuando dice 85 %
  sale ~85 %. Remates y tiros a puerta son lo que mejor predice (12–21 % mejor que la media);
  faltas cometidas casi no aporta (2–3 %).
- **Equipos**: corners, remates y tiros a puerta por equipo aportan bastante; tarjetas y
  faltas, poco (dependen mucho del árbitro, que el modelo no tiene en cuenta).
- **Totales del partido**: los más flojos. A principio de temporada los remates salieron más
  altos de lo previsto porque esta temporada se remata un 10 % más y el modelo tardó en
  ajustarse.

Úsalo para saber la probabilidad real de una apuesta y cuánto margen te cobra bet365
(sobre todo en "Crear Apuesta" y combinadas), no como máquina de ganar dinero.

## App del móvil

`docs/` es una web instalable: en el móvil se ve como una app (icono, pantalla completa,
funciona sin conexión con los últimos datos). Cada partido tiene tres pestañas:

- **Goles**: 1X2, doble oportunidad, más/menos, ambos marcan, hándicap, resultado exacto y
  combinaciones "Crear Apuesta" con probabilidad exacta. Muestra la cuota de bet365 y el valor
  cuando la hay.
- **Equipos**: corners, remates, tiros a puerta, faltas, tarjetas y entradas por equipo y en
  total, con la línea que quieras (botones − / +).
- **Jugadores**: remates, tiros a puerta, entradas, faltas recibidas y cometidas de cada
  jugador (1+, 2+, 3+), con la alineación probable o confirmada y las bajas.

Con el **+** añades selecciones al boleto: calcula la probabilidad de la combinada y su cuota
justa, y si escribes la cuota de bet365 te dice si tiene valor.

### Puesta en marcha (una sola vez)

1. Crea una cuenta en github.com y un repositorio **público** vacío (p. ej. `probabilidades`).
2. Sube esta carpeta al repositorio.
3. En el repositorio: **Settings → Pages → Source: GitHub Actions**.
4. **Actions → Actualizar datos y publicar app → Run workflow**. Tarda unos minutos.
5. Abre en Chrome del móvil `https://TU_USUARIO.github.io/probabilidades/` →
   menú ⋮ → **Añadir a pantalla de inicio** (o "Instalar app").

A partir de ahí GitHub actualiza los datos cada 3 horas (7:00–22:00) sin que tengas el PC
encendido. La alineación confirmada aparece si la actualización cae después de que se anuncie
(~1 h antes del partido); si quieres forzarla, pulsa "Run workflow".

### Fuentes

- FotMob: estadísticas por jugador y equipo, alineaciones y bajas (API no oficial: si
  cambia, habrá que adaptar `fotmob.py`).
- football-data.co.uk: resultados, xG y cuotas de bet365 de la jornada de Liga.

Las cuotas de bet365 de jugadores, corners, etc. no están en ninguna fuente gratuita: la app
te da la cuota justa y tú la comparas con la de bet365 (o la escribes en el boleto).

## Archivos

| Archivo | Qué hace |
|---|---|
| `bot.py` | Programa principal e informe |
| `datos.py` | Descarga y caché de datos |
| `modelo.py` | Modelo Dixon-Coles y cruces entre ligas |
| `mercados.py` | Mercados, cuotas y margen |
| `analisis.py` | Simples, combinadas y Crear Apuesta |
| `backtest.py` | Prueba histórica |
| `cuotas_manual.csv` | Cuotas que metes a mano |
| `fotmob.py` | Estadísticas por partido de jugadores y equipos |
| `estadisticas.py` | Modelos de corners, remates, faltas, tarjetas y jugadores |
| `backtest_estadisticas.py` | Prueba histórica de esos mercados |
| `exportar.py` | Genera `docs/data.json` para la app |
| `docs/` | La app del móvil |
| `.github/workflows/actualizar.yml` | Actualización automática en GitHub |
