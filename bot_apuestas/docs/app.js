"use strict";
// App de probabilidades: lee data.json (lo genera exportar.py) y calcula todo en el móvil.

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (p) => (p < 0.1 ? (100 * p).toFixed(1) : Math.round(100 * p)) + "%";
const cj = (p) => (p > 0 ? (1 / p).toFixed(2) : "–");
const guardar = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} };
const leer = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch (e) { return d; } };

const S = {
  data: null, filtro: leer("filtro", "todos"), tab: "goles", statJ: "tiros_puerta", lineaJ: 1,
  soloTitulares: true, lineas: {}, boleto: leer("boleto", []), cuotaBoleto: "",
};

// ---------- Mercados de goles (condiciones sobre el marcador h-a) ----------
const GOLES = {
  "1": ["Gana {L}", (h, a) => h > a], "X": ["Empate", (h, a) => h === a], "2": ["Gana {V}", (h, a) => h < a],
  "1X": ["{L} o empate", (h, a) => h >= a], "X2": ["Empate o {V}", (h, a) => h <= a], "12": ["No hay empate", (h, a) => h !== a],
  "O1.5": ["Más de 1.5 goles", (h, a) => h + a >= 2], "U1.5": ["Menos de 1.5 goles", (h, a) => h + a <= 1],
  "O2.5": ["Más de 2.5 goles", (h, a) => h + a >= 3], "U2.5": ["Menos de 2.5 goles", (h, a) => h + a <= 2],
  "O3.5": ["Más de 3.5 goles", (h, a) => h + a >= 4], "U3.5": ["Menos de 3.5 goles", (h, a) => h + a <= 3],
  "BTTS": ["Ambos marcan: Sí", (h, a) => h > 0 && a > 0], "NOBTTS": ["Ambos marcan: No", (h, a) => h === 0 || a === 0],
  "L_O0.5": ["{L} marca", (h) => h >= 1], "V_O0.5": ["{V} marca", (h, a) => a >= 1],
  "L_O1.5": ["{L} marca 2 o más", (h) => h >= 2], "V_O1.5": ["{V} marca 2 o más", (h, a) => a >= 2],
  "L-1.5": ["{L} gana por 2 o más", (h, a) => h - a >= 2], "V-1.5": ["{V} gana por 2 o más", (h, a) => a - h >= 2],
  "L_CS": ["{L} portería a cero", (h, a) => a === 0], "V_CS": ["{V} portería a cero", (h) => h === 0],
};
const GRUPOS_GOLES = [
  ["Resultado", ["1", "X", "2"]], ["Doble oportunidad", ["1X", "X2", "12"]],
  ["Goles del partido", ["O1.5", "U1.5", "O2.5", "U2.5", "O3.5", "U3.5"]], ["Ambos marcan", ["BTTS", "NOBTTS"]],
  ["Goles por equipo", ["L_O0.5", "V_O0.5", "L_O1.5", "V_O1.5"]],
  ["Hándicap y portería a cero", ["L-1.5", "V-1.5", "L_CS", "V_CS"]],
];
const PATAS_SUGERENCIAS = ["1", "X", "2", "1X", "X2", "O1.5", "O2.5", "U2.5", "U3.5", "BTTS", "NOBTTS", "L_O1.5", "V_O1.5", "L-1.5", "V-1.5", "L_CS", "V_CS"];

function condGol(k) {
  if (k.startsWith("CS:")) { const [x, y] = k.slice(3).split("-").map(Number); return (h, a) => h === x && a === y; }
  return GOLES[k][1];
}
function etqGol(k, m) {
  if (k.startsWith("CS:")) return "Resultado exacto " + k.slice(3);
  return GOLES[k][0].replace("{L}", m.local).replace("{V}", m.visitante);
}
function probGoles(m, claves) {
  const M = m.goles.matriz, fs = claves.map(condGol);
  let p = 0;
  for (let h = 0; h < M.length; h++) for (let a = 0; a < M[h].length; a++) if (fs.every((f) => f(h, a))) p += M[h][a];
  return p;
}

// ---------- Conteos (corners, remates...): Poisson o binomial negativa ----------
function pmf(mu, k, n = 90) {
  const p = new Array(n).fill(0);
  if (!k) { p[0] = Math.exp(-mu); for (let i = 1; i < n; i++) p[i] = p[i - 1] * mu / i; }
  else { const q = mu / (k + mu); p[0] = Math.pow(k / (k + mu), k); for (let i = 1; i < n; i++) p[i] = p[i - 1] * (i - 1 + k) / i * q; }
  return p;
}
function conv(a, b) {
  const r = new Array(a.length).fill(0);
  for (let i = 0; i < a.length; i++) for (let j = 0; i + j < a.length; j++) r[i + j] += a[i] * b[j];
  return r;
}
const alMenos = (p, n) => 1 - p.slice(0, n).reduce((s, x) => s + x, 0);

const STATS_EQ = {
  corners: { nombre: "Corners", eq: 4.5, total: 9.5 }, remates: { nombre: "Remates", eq: 11.5, total: 23.5 },
  tiros_puerta: { nombre: "Tiros a puerta", eq: 3.5, total: 8.5 }, faltas: { nombre: "Faltas", eq: 11.5, total: 23.5 },
  amarillas: { nombre: "Tarjetas amarillas", eq: 1.5, total: 4.5 }, entradas: { nombre: "Entradas", eq: 15.5, total: 31.5 },
};
const STATS_JUG = {
  tiros_puerta: "Tiros a puerta", remates: "Remates", entradas: "Entradas", faltas_recibidas: "Faltas recibidas", faltas: "Faltas cometidas",
};

function distEquipo(m, stat, lado) {
  const [ml, mv, k] = m.equipo[stat];
  if (lado === "local") return pmf(ml, k);
  if (lado === "visitante") return pmf(mv, k);
  return conv(pmf(ml, k), pmf(mv, k));
}
function probEquipo(m, stat, lado, dir, linea) {
  const p = alMenos(distEquipo(m, stat, lado), Math.ceil(linea));
  return dir === "mas" ? p : 1 - p;
}
function jugador(m, id) {
  return [...m.jugadores.local, ...m.jugadores.visitante].find((j) => j.id === id);
}
const probJugador = (j, stat, n) => alMenos(pmf(j[stat], S.data.k_jugador?.[stat], 30), n);

// ---------- Boleto ----------
function partidoPorId(id) { return S.data?.partidos.find((p) => p.id === id); }
function probPata(l) {
  const m = partidoPorId(l.m);
  if (!m) return null;
  if (l.t === "g") return m.goles ? probGoles(m, [l.k]) : null;
  if (l.t === "e") return m.equipo[l.stat] ? probEquipo(m, l.stat, l.lado, l.dir, l.linea) : null;
  const j = jugador(m, l.jid);
  return j ? probJugador(j, l.stat, l.n) : null;
}
function probBoleto() {
  // Patas de goles del mismo partido: probabilidad exacta con la matriz de marcadores.
  // El resto se trata como independiente (aproximación).
  let p = 1, aproximada = false;
  const porPartido = {};
  for (const l of S.boleto) (porPartido[l.m] ??= []).push(l);
  for (const [id, patas] of Object.entries(porPartido)) {
    const m = partidoPorId(Number(id));
    if (!m) return { p: null };
    const g = patas.filter((l) => l.t === "g"), otras = patas.filter((l) => l.t !== "g");
    if (g.length) p *= probGoles(m, g.map((l) => l.k));
    for (const l of otras) p *= probPata(l);
    if (otras.length && patas.length > 1) aproximada = true;
  }
  return { p, aproximada };
}
const idPata = (l) => JSON.stringify([l.t, l.m, l.k, l.stat, l.lado, l.dir, l.linea, l.jid, l.n]);
function alternarPata(l) {
  const id = idPata(l), i = S.boleto.findIndex((x) => idPata(x) === id);
  if (i >= 0) S.boleto.splice(i, 1); else S.boleto.push(l);
  guardar("boleto", S.boleto);
  pintarBotonBoleto();
}
const enBoleto = (l) => S.boleto.some((x) => idPata(x) === idPata(l));

function botonMas(l) {
  return `<button class="mas" aria-label="Añadir al boleto" aria-pressed="${enBoleto(l)}" data-pata='${esc(JSON.stringify(l))}'>${enBoleto(l) ? "✓" : "+"}</button>`;
}
function fila(nombre, sub, p, l, cuota) {
  let c = `<div class="c">justa <b>${cj(p)}</b></div>`;
  if (cuota) {
    const ev = p * cuota - 1;
    c = `<div class="c">bet365 <b>${cuota.toFixed(2)}</b><div class="ev ${ev > 0 ? "pos" : "neg"}">${ev > 0 ? "+" : ""}${(100 * ev).toFixed(1)}%</div></div>`;
  }
  return `<div class="fila"><div class="nombre">${esc(nombre)}${sub ? `<small>${esc(sub)}</small>` : ""}</div><div class="p">${pct(p)}</div>${c}${botonMas(l)}</div>`;
}

// ---------- Vistas ----------
const fmtFecha = (iso) => new Date(iso).toLocaleString("es-ES", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const FILTROS = [["todos", "Todos"], ["barca", "Barça"], ["madrid", "Real Madrid"], ["liga", "La Liga"], ["champions", "Champions"]];
function pasaFiltro(m) {
  const eqs = (m.local + " " + m.visitante).toLowerCase();
  return { todos: true, barca: eqs.includes("barcelona"), madrid: /real madrid/.test(eqs),
    liga: m.competicion === "La Liga", champions: m.competicion === "Champions" }[S.filtro];
}

function vistaLista() {
  $("#titulo").textContent = "Probabilidades";
  $("#subtitulo").textContent = S.data ? "Actualizado " + fmtFecha(S.data.generado) : "";
  $("#atras").hidden = true;
  const chips = FILTROS.map(([k, n]) => `<button class="chip" data-filtro="${k}" aria-pressed="${S.filtro === k}">${n}</button>`).join("");
  const ps = S.data.partidos.filter(pasaFiltro);
  const cards = ps.map((m) => {
    const g = m.goles;
    const p1 = g ? probGoles(m, ["1"]) : null, px = g ? probGoles(m, ["X"]) : null, p2 = g ? probGoles(m, ["2"]) : null;
    const tres = g ? `<div class="tres">
        <div><span>1</span><b>${pct(p1)}</b>${g.cuotas["1"] ? `<span>${g.cuotas["1"].toFixed(2)}</span>` : ""}</div>
        <div><span>X</span><b>${pct(px)}</b>${g.cuotas["X"] ? `<span>${g.cuotas["X"].toFixed(2)}</span>` : ""}</div>
        <div><span>2</span><b>${pct(p2)}</b>${g.cuotas["2"] ? `<span>${g.cuotas["2"].toFixed(2)}</span>` : ""}</div></div>
        <div class="barra-prob"><i style="width:${100 * p1}%"></i><i style="width:${100 * px}%"></i><i style="width:${100 * p2}%"></i></div>`
      : `<p class="nota">Sin modelo de goles para este partido</p>`;
    return `<button class="card partido" data-partido="${m.id}">
      <div class="meta"><span>${esc(m.competicion)}</span><span>${fmtFecha(m.fecha)}</span></div>
      <div class="equipos">${esc(m.local)} – ${esc(m.visitante)}</div>${tres}</button>`;
  }).join("");
  $("#vista").innerHTML = `<div class="chips">${chips}</div>` +
    (cards || `<p class="vacio">No hay partidos para este filtro en los próximos días.</p>`);
}

function vistaPartido(m) {
  $("#titulo").textContent = `${m.local} – ${m.visitante}`;
  $("#subtitulo").textContent = `${m.competicion} · ${fmtFecha(m.fecha)}`;
  $("#atras").hidden = false;
  const tabs = [["goles", "Goles"], ["equipos", "Equipos"], ["jugadores", "Jugadores"]]
    .map(([k, n]) => `<button role="tab" data-tab="${k}" aria-selected="${S.tab === k}">${n}</button>`).join("");
  const cuerpo = { goles: tabGoles, equipos: tabEquipos, jugadores: tabJugadores }[S.tab](m);
  $("#vista").innerHTML = `<div class="tabs" role="tablist">${tabs}</div>${cuerpo}`;
}

function tabGoles(m) {
  const g = m.goles;
  if (!g) return `<p class="vacio">No hay modelo de goles para este partido.</p>`;
  let h = `<div class="card"><div class="meta"><span>Goles esperados</span><span>${g.cuotas["1"] ? "Probabilidad modelo + bet365" : "Solo modelo (sin cuotas bet365)"}</span></div>
    <div class="equipos">${g.lh.toFixed(2)} – ${g.la.toFixed(2)}</div></div>`;
  for (const [titulo, claves] of GRUPOS_GOLES) {
    h += `<div class="card"><h3>${titulo}</h3>` +
      claves.map((k) => fila(etqGol(k, m), "", probGoles(m, [k]), { t: "g", m: m.id, k, label: etqGol(k, m) }, g.cuotas[k])).join("") + `</div>`;
  }
  // Marcadores más probables
  const ms = [];
  g.matriz.forEach((r, i) => r.forEach((p, j) => ms.push([p, i, j])));
  ms.sort((a, b) => b[0] - a[0]);
  h += `<div class="card"><h3>Resultado exacto</h3>` + ms.slice(0, 6).map(([p, i, j]) =>
    fila(`${m.local} ${i} – ${j} ${m.visitante}`, "", p, { t: "g", m: m.id, k: `CS:${i}-${j}`, label: `Resultado exacto ${i}-${j}` })).join("") + `</div>`;
  h += `<div class="card"><h3>Combinaciones más probables (Crear Apuesta)</h3>${sugerencias(m)}
    <p class="nota">Probabilidad exacta del combo, teniendo en cuenta que los mercados están relacionados. Compárala con la cuota de bet365: solo hay valor si su cuota supera la justa.</p></div>`;
  return h;
}

function sugerencias(m) {
  const res = [], vistas = new Set();
  for (let i = 0; i < PATAS_SUGERENCIAS.length; i++) for (let j = i + 1; j < PATAS_SUGERENCIAS.length; j++) {
    const a = PATAS_SUGERENCIAS[i], b = PATAS_SUGERENCIAS[j];
    const p = probGoles(m, [a, b]);
    if (p < 0.2 || 1 / p < 1.5) continue;
    if (probGoles(m, [a]) - p < 0.02 || probGoles(m, [b]) - p < 0.02) continue; // una pata sobra
    const huella = g2huella(m, [a, b]);
    if (vistas.has(huella)) continue;
    vistas.add(huella);
    res.push([p, a, b]);
  }
  res.sort((x, y) => y[0] - x[0]);
  if (!res.length) return `<p class="nota">No hay combinaciones con cuota justa ≥ 1.50.</p>`;
  return res.slice(0, 8).map(([p, a, b]) => {
    const etq = `${etqGol(a, m)} + ${etqGol(b, m)}`;
    const marcado = enBoleto({ t: "g", m: m.id, k: a }) && enBoleto({ t: "g", m: m.id, k: b });
    return `<div class="fila"><div class="nombre">${esc(etq)}</div><div class="p">${pct(p)}</div><div class="c">justa <b>${cj(p)}</b></div>
      <button class="mas" aria-label="Añadir al boleto" aria-pressed="${marcado}" data-combo='${esc(JSON.stringify([m.id, a, b]))}'>${marcado ? "✓" : "+"}</button></div>`;
  }).join("");
}
function g2huella(m, ks) {
  const fs = ks.map(condGol); let s = "";
  for (let h = 0; h < 11; h++) for (let a = 0; a < 11; a++) s += fs.every((f) => f(h, a)) ? "1" : "0";
  return s;
}

function tabEquipos(m) {
  if (!Object.keys(m.equipo).length) return `<p class="vacio">No hay estadísticas de estos equipos.</p>`;
  let h = "";
  for (const [stat, cfg] of Object.entries(STATS_EQ)) {
    if (!m.equipo[stat]) continue;
    const [ml, mv] = m.equipo[stat];
    h += `<div class="card"><h3>${cfg.nombre} <span class="nota">· esperado ${ml.toFixed(1)} – ${mv.toFixed(1)}</span></h3>`;
    for (const [lado, nombre, def] of [["local", m.local, cfg.eq], ["visitante", m.visitante, cfg.eq], ["total", "Total del partido", cfg.total]]) {
      const clave = `${stat}|${lado}`, linea = S.lineas[clave] ?? def;
      const pm = probEquipo(m, stat, lado, "mas", linea);
      const base = { t: "e", m: m.id, stat, lado, linea };
      const lMas = { ...base, dir: "mas", label: `${nombre}: más de ${linea} ${cfg.nombre.toLowerCase()}` };
      const lMenos = { ...base, dir: "menos", label: `${nombre}: menos de ${linea} ${cfg.nombre.toLowerCase()}` };
      h += `<div class="paso" style="margin-top:10px"><b style="flex:1">${esc(nombre)}</b>
        <button data-linea="${clave}" data-d="-1" aria-label="Bajar línea">−</button><output>${linea}</output>
        <button data-linea="${clave}" data-d="1" aria-label="Subir línea">+</button></div>` +
        fila(`Más de ${linea}`, "", pm, lMas) + fila(`Menos de ${linea}`, "", 1 - pm, lMenos);
    }
    h += `</div>`;
  }
  return h + `<p class="nota">Modelo por equipo con lo que cada uno genera y concede, ventaja de local y partidos recientes con más peso.</p>`;
}

function tabJugadores(m) {
  const chips = Object.entries(STATS_JUG).map(([k, n]) => `<button class="chip" data-statj="${k}" aria-pressed="${S.statJ === k}">${n}</button>`).join("");
  const lineas = [1, 2, 3].map((n) => `<button class="chip" data-lineaj="${n}" aria-pressed="${S.lineaJ === n}">${n}+</button>`).join("");
  const filtro = `<button class="chip" data-solotit aria-pressed="${S.soloTitulares}">Solo titulares probables</button>`;
  const estado = {
    confirmada: "Alineación oficial confirmada",
    prevista: "Alineación prevista por FotMob (la oficial sale ~1 h antes)",
    ultimo_once: "Alineación probable: el último once de cada equipo (la oficial sale ~1 h antes)",
  }[m.alineacion] || "Titulares probables: los habituales de los últimos 5 partidos";
  let h = `<div class="chips">${chips}</div><div class="chips">${lineas}${filtro}</div><div class="aviso">${estado}. Probabilidades si el jugador es titular.</div>`;
  for (const [lado, nombre] of [["local", m.local], ["visitante", m.visitante]]) {
    let js = m.jugadores[lado] || [];
    if (S.soloTitulares && js.some((j) => j.titular)) js = js.filter((j) => j.titular);
    js = js.map((j) => [probJugador(j, S.statJ, S.lineaJ), j]).sort((a, b) => b[0] - a[0]);
    h += `<div class="card"><h3>${esc(nombre)}</h3>` + (js.length ? js.map(([p, j]) => fila(
      j.nombre, `${j.pos} · ~${j.minutos}' · titular ${j.titular_ultimos5}/5 · esperado ${j[S.statJ].toFixed(2)}`, p,
      { t: "j", m: m.id, jid: j.id, stat: S.statJ, n: S.lineaJ, label: `${j.nombre}: ${S.lineaJ}+ ${STATS_JUG[S.statJ].toLowerCase()}` })).join("")
      : `<p class="nota">Sin datos de jugadores.</p>`);
    const bajas = m.bajas[lado] || [];
    if (bajas.length) h += `<p class="nota">Bajas: ${bajas.map((b) => esc(b.nombre) + (b.motivo ? ` (${esc(b.motivo === "injury" ? "lesión" : b.motivo === "suspension" ? "sanción" : b.motivo)})` : "")).join(", ")}</p>`;
    h += `</div>`;
  }
  return h;
}

// ---------- Boleto (hoja inferior) ----------
function pintarBotonBoleto() {
  const b = $("#boletoBtn");
  if (!S.boleto.length) { b.hidden = true; return; }
  const { p } = probBoleto();
  b.hidden = false;
  b.innerHTML = `<span>Boleto · ${S.boleto.length} ${S.boleto.length === 1 ? "selección" : "selecciones"}</span><span>${p != null ? pct(p) + " · justa " + cj(p) : ""}</span>`;
}
function abrirBoleto() {
  const hoja = $("#boleto");
  const { p, aproximada } = probBoleto();
  const filas = S.boleto.map((l, i) => {
    const m = partidoPorId(l.m), pp = probPata(l);
    return `<div class="fila" style="grid-template-columns:1fr auto auto"><div class="nombre">${esc(l.label)}<small>${m ? esc(m.local + " – " + m.visitante) : "Partido ya no disponible"}</small></div>
      <div class="p">${pp != null ? pct(pp) : "–"}</div><button class="quitar" data-quitar="${i}">Quitar</button></div>`;
  }).join("");
  const cuota = parseFloat(String(S.cuotaBoleto).replace(",", "."));
  let veredicto = "";
  if (p && cuota > 1) {
    const ev = p * cuota - 1;
    veredicto = `<div class="veredicto ${ev > 0 ? "pos" : "neg"}">${ev > 0 ? "Hay valor" : "Sin valor"}: valor esperado ${ev > 0 ? "+" : ""}${(100 * ev).toFixed(1)}% · ${ev > 0 ? "la cuota paga más de lo que debería" : `bet365 se queda un ${(100 * (1 - p * cuota)).toFixed(0)}% de media`}</div>`;
  }
  hoja.innerHTML = `<h2>Boleto <button class="quitar" data-vaciar>Vaciar</button></h2>${filas}
    <div class="resumen"><div><span>Probabilidad de acierto</span><b>${p != null ? pct(p) : "–"}</b></div><div><span>Cuota justa</span><b>${p ? cj(p) : "–"}</b></div></div>
    ${aproximada ? `<p class="nota">Hay estadísticas del mismo partido que se calculan como independientes: la probabilidad real puede ser algo distinta (p. ej. muchos remates y muchos corners suelen ir juntos).</p>` : ""}
    <label class="entrada"><input id="cuotaBoleto" inputmode="decimal" placeholder="Cuota que te da bet365" value="${esc(S.cuotaBoleto)}"></label>
    ${veredicto}
    <div style="display:flex;justify-content:flex-end;margin-top:12px"><button class="boton-sec" data-cerrar>Cerrar</button></div>`;
  hoja.hidden = false;
  if (!$(".hoja-fondo")) { const f = document.createElement("div"); f.className = "hoja-fondo"; f.dataset.cerrar = ""; document.body.appendChild(f); }
}
function cerrarBoleto() { $("#boleto").hidden = true; $(".hoja-fondo")?.remove(); }

// ---------- Navegación y eventos ----------
function pintar() {
  if (!S.data) return;
  const id = Number((location.hash.match(/^#\/p\/(\d+)/) || [])[1]);
  const m = id && partidoPorId(id);
  if (m) vistaPartido(m); else vistaLista();
  pintarBotonBoleto();
}

document.addEventListener("click", (e) => {
  const t = e.target.closest("button, .hoja-fondo");
  if (!t) return;
  const d = t.dataset;
  if (d.partido) { S.tab = "goles"; location.hash = `#/p/${d.partido}`; window.scrollTo(0, 0); return; }
  if (d.filtro) { S.filtro = d.filtro; guardar("filtro", S.filtro); return pintar(); }
  if (d.tab) { S.tab = d.tab; return pintar(); }
  if (d.statj) { S.statJ = d.statj; return pintar(); }
  if (d.lineaj) { S.lineaJ = Number(d.lineaj); return pintar(); }
  if ("solotit" in d) { S.soloTitulares = !S.soloTitulares; return pintar(); }
  if (d.linea) { const v = (S.lineas[d.linea] ?? null); const [stat, lado] = d.linea.split("|");
    const base = v ?? (lado === "total" ? STATS_EQ[stat].total : STATS_EQ[stat].eq);
    S.lineas[d.linea] = Math.max(0.5, base + Number(d.d)); return pintar(); }
  if (d.pata) { alternarPata(JSON.parse(d.pata)); return pintar(); }
  if (d.combo) {
    const [mid, a, b] = JSON.parse(d.combo), m = partidoPorId(mid);
    const pa = { t: "g", m: mid, k: a, label: etqGol(a, m) }, pb = { t: "g", m: mid, k: b, label: etqGol(b, m) };
    if (enBoleto(pa) && enBoleto(pb)) { alternarPata(pa); alternarPata(pb); }
    else { if (!enBoleto(pa)) alternarPata(pa); if (!enBoleto(pb)) alternarPata(pb); }
    return pintar();
  }
  if (t.id === "boletoBtn") return abrirBoleto();
  if (t.id === "atras") { location.hash = ""; return; }
  if (d.quitar != null) { S.boleto.splice(Number(d.quitar), 1); guardar("boleto", S.boleto); pintar(); return S.boleto.length ? abrirBoleto() : cerrarBoleto(); }
  if ("vaciar" in d) { S.boleto = []; guardar("boleto", S.boleto); cerrarBoleto(); return pintar(); }
  if ("cerrar" in d) return cerrarBoleto();
});
document.addEventListener("input", (e) => {
  if (e.target.id === "cuotaBoleto") {
    S.cuotaBoleto = e.target.value;
    const pos = e.target.selectionStart;
    abrirBoleto();
    const inp = $("#cuotaBoleto"); inp.focus(); inp.setSelectionRange(pos, pos);
  }
});
window.addEventListener("hashchange", () => { cerrarBoleto(); pintar(); });

async function cargar() {
  try {
    const r = await fetch("data.json", { cache: "no-cache" });
    S.data = await r.json();
    pintar();
  } catch (e) {
    $("#vista").innerHTML = `<p class="vacio">No se pudieron cargar los datos. Comprueba la conexión.</p>`;
  }
}
cargar();
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
