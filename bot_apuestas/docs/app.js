"use strict";
// App de probabilidades: lee data.json (lo genera exportar.py) y calcula todo en el móvil.

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (p) => (p < 0.1 ? (100 * p).toFixed(1) : Math.round(100 * p)) + "%";
const cj = (p) => (p > 0 ? (1 / p).toFixed(2) : "–");
const guardar = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} };
const leer = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch (e) { return d; } };

const S = {
  data: null, filtro: leer("filtro", "todos"), tab: "recomendadas", rango: "hasta4", statJ: "tiros_puerta", lineaJ: 1,
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

// ---------- Veces que se cumplió en los últimos 10 partidos ----------
// Goles y equipo: en los últimos partidos de cada equipo, visto desde su lado. Para "Gana
// Barcelona" cuenta las veces que ganó el Barça y las que perdió su rival; para "Barcelona: más
// de 5.5 corners", las que el Barça sacó 6+ y las que su rival los concedió. Jugadores: sus
// últimos 10 partidos como titular. Varias patas a la vez: partidos en los que se cumplen todas.
function cumpleEquipo(l, h, i, esLocal) {
  if (l.t === "g") {
    const [f, a] = h.goles[i] ?? [];
    if (f == null || a == null) return null;
    return esLocal ? condGol(l.k)(f, a) : condGol(l.k)(a, f);
  }
  const [f, a] = h[l.stat]?.[i] ?? [];
  if (f == null || a == null) return null;
  const v = l.lado === "total" ? f + a : (l.lado === "local") === esLocal ? f : a;
  return l.dir === "mas" ? v > l.linea : v < l.linea;
}
function veces(m, patas) {
  if (patas.every((l) => l.t !== "j")) {
    const res = [];
    for (const [lado, nombre, esLocal] of [["local", m.local, true], ["visitante", m.visitante, false]]) {
      const h = m.historial?.[lado];
      if (!h) continue;
      let si = 0, n = 0;
      for (let i = 0; i < h.goles.length; i++) {
        const r = patas.map((l) => cumpleEquipo(l, h, i, esLocal));
        if (r.some((x) => x == null)) continue;
        n++; if (r.every(Boolean)) si++;
      }
      if (n) res.push({ nombre, si, n });
    }
    return res;
  }
  if (patas.every((l) => l.t === "j" && l.jid === patas[0].jid)) {
    const h = jugador(m, patas[0].jid)?.ult10, n = h ? h[patas[0].stat].length : 0;
    let si = 0;
    for (let i = 0; i < n; i++) if (patas.every((l) => h[l.stat][i] >= l.n)) si++;
    return n ? [{ nombre: null, si, n }] : [];
  }
  return [];
}
function textoVeces(m, patas, corto) {
  const v = veces(m, patas);
  if (!v.length) return "";
  if (corto) return v.map((x) => `${x.si}/${x.n}`).join(" · ");
  if (v[0].nombre == null) return `${v[0].n === 10 ? "Últimos 10" : `Solo ${v[0].n} partidos`} como titular: ${v[0].si}/${v[0].n}`;
  return "Últimos 10: " + v.map((x) => `${x.nombre} ${x.si}/${x.n}`).join(" · ");
}
const lineaVeces = (m, patas) => { const t = m ? textoVeces(m, patas) : ""; return t ? `<small class="hist">${esc(t)}</small>` : ""; };
const NOTA_VECES = "«Últimos 10»: veces que se cumplió en los últimos 10 partidos de liga y Champions de cada equipo, visto desde su lado (para «Gana Barcelona», las que ganó el Barça y las que perdió su rival). En jugadores, sus últimos 10 partidos como titular.";

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
// ---------- Simulación conjunta del partido ----------
// Cada partido se simula N_SIM veces con una cópula gaussiana: cada mercado conserva su
// probabilidad y además la relación medida con partidos reales (correlaciones.py). Así las
// combinadas del mismo partido no se calculan como si sus patas fueran independientes.
const N_SIM = 20000, PALABRAS = N_SIM / 32;
const SIMS = new Map(), BITS = new Map();

function mulberry32(a) {
  return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function normales(rand) {
  let resto = null;
  return () => {
    if (resto !== null) { const r = resto; resto = null; return r; }
    let u = 0; while (u === 0) u = rand();
    const r = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * rand();
    resto = r * Math.sin(th); return r * Math.cos(th);
  };
}
function phiInv(p) { // cuantil de la normal (Acklam)
  if (p <= 0) return -8; if (p >= 1) return 8;
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
  const q = Math.min(p, 1 - p);
  let x;
  if (q > 0.02425) {
    const r = (p - 0.5), s = r * r;
    x = (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1);
  } else {
    const r = Math.sqrt(-2 * Math.log(q));
    x = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1);
    if (p > 0.5) x = -x;
  }
  return x;
}
function phi(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2);
  return x >= 0 ? (1 + y) / 2 : (1 - y) / 2;
}
function cholesky(R) {
  const n = R.length, L = R.map(() => new Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = R[i][j];
    for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-9)) : s / L[j][j];
  }
  return L;
}
const acumulada = (p) => { const c = new Float64Array(p.length); let s = 0; p.forEach((x, i) => { s += x; c[i] = s; }); return c; };
function cuantil(cdf, u) { let lo = 0, hi = cdf.length - 1; while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] >= u) hi = mid; else lo = mid + 1; } return lo; }

function simular(m) {
  if (SIMS.has(m.id)) return SIMS.get(m.id);
  const C = S.data.correlacion, orden = C.orden, L = cholesky(C.R), n = orden.length;
  const rand = mulberry32(m.id % 2147483647), nrm = normales(rand);
  const lat = {}, val = {};
  orden.forEach((v) => { lat[v] = new Float32Array(N_SIM); val[v] = new Int16Array(N_SIM).fill(-1); });
  // Goles: el marcador sale de la matriz (exacta) y se pasa a escala normal.
  const M = m.goles?.matriz;
  let celdas, pL, pV, FL, FV;
  if (M) {
    celdas = acumulada(M.flat());
    pL = M.map((r) => r.reduce((s, x) => s + x, 0));
    pV = M[0].map((_, a) => M.reduce((s, r) => s + r[a], 0));
    FL = acumulada(pL); FV = acumulada(pV);
  }
  const cdfs = {};
  for (const v of orden) {
    const [stat, lado] = [v.slice(0, v.lastIndexOf("_")), v.slice(-1)];
    if (stat !== "goles" && m.equipo[stat]) { const [ml, mv, k] = m.equipo[stat]; cdfs[v] = acumulada(pmf(lado === "L" ? ml : mv, k)); }
  }
  const E = new Float64Array(n);
  for (let i = 0; i < N_SIM; i++) {
    let zL, zV;
    if (M) {
      const c = Math.min(cuantil(celdas, rand() * celdas[celdas.length - 1]), celdas.length - 1);
      const h = Math.floor(c / M[0].length), a = c % M[0].length;
      val.goles_L[i] = h; val.goles_V[i] = a;
      zL = phiInv((h ? FL[h - 1] : 0) + rand() * pL[h]); zV = phiInv((a ? FV[a - 1] : 0) + rand() * pV[a]);
    } else { zL = nrm(); zV = nrm(); }
    E[0] = zL / L[0][0]; E[1] = (zV - L[1][0] * E[0]) / L[1][1];
    lat[orden[0]][i] = zL; lat[orden[1]][i] = zV;
    for (let j = 2; j < n; j++) {
      E[j] = nrm();
      let z = 0; for (let t = 0; t <= j; t++) z += L[j][t] * E[t];
      lat[orden[j]][i] = z;
      if (cdfs[orden[j]]) val[orden[j]][i] = cuantil(cdfs[orden[j]], phi(z));
    }
  }
  const sim = { lat, val };
  SIMS.set(m.id, sim);
  return sim;
}

const REF_JUGADOR = { remates: ["remates", false], tiros_puerta: ["tiros_puerta", false], entradas: ["entradas", false], faltas: ["faltas", false], faltas_recibidas: ["faltas", true] };
function bitsPata(l) {
  const id = idPata(l);
  if (BITS.has(id)) return BITS.get(id);
  const m = partidoPorId(l.m);
  if (!m) return null;
  const sim = simular(m), b = new Uint32Array(PALABRAS);
  const marca = (i) => { b[i >>> 5] |= 1 << (i & 31); };
  if (l.t === "g") {
    if (!m.goles) return null;
    const f = condGol(l.k), h = sim.val.goles_L, a = sim.val.goles_V;
    for (let i = 0; i < N_SIM; i++) if (f(h[i], a[i])) marca(i);
  } else if (l.t === "e") {
    if (!m.equipo[l.stat]) return null;
    const n = Math.ceil(l.linea), vl = sim.val[`${l.stat}_L`], vv = sim.val[`${l.stat}_V`];
    for (let i = 0; i < N_SIM; i++) {
      const x = l.lado === "local" ? vl[i] : l.lado === "visitante" ? vv[i] : vl[i] + vv[i];
      if ((x >= n) === (l.dir === "mas")) marca(i);
    }
  } else {
    const j = jugador(m, l.jid);
    if (!j) return null;
    const esLocal = m.jugadores.local.some((x) => x.id === j.id);
    const [ref, rival] = REF_JUGADOR[l.stat];
    const z = sim.lat[`${ref}_${esLocal !== rival ? "L" : "V"}`];
    const rho = S.data.correlacion.rho_jugador[l.stat], s = Math.sqrt(1 - rho * rho);
    // X ≥ n  ⇔  la normal latente del jugador supera el cuantil de P(X ≤ n-1)
    const umbral = phiInv(1 - probJugador(j, l.stat, l.n));
    const nrm = normales(mulberry32((j.id * 31 + l.stat.length * 7919) % 2147483647));
    for (let i = 0; i < N_SIM; i++) if (rho * z[i] + s * nrm() > umbral) marca(i);
  }
  BITS.set(id, b);
  return b;
}
function contar(b) { let c = 0; for (let i = 0; i < b.length; i++) { let v = b[i]; v -= (v >>> 1) & 0x55555555; v = (v & 0x33333333) + ((v >>> 2) & 0x33333333); c += (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24; } return c; }
function probConjunta(patas) {
  let acc = null;
  for (const l of patas) {
    const b = bitsPata(l);
    if (!b) return null;
    if (!acc) acc = b.slice(); else for (let i = 0; i < PALABRAS; i++) acc[i] &= b[i];
  }
  return contar(acc) / N_SIM;
}

function probBoleto() {
  // Patas del mismo partido: simulación conjunta. Partidos distintos: independientes.
  let p = 1, mismoJugador = false;
  const porPartido = {};
  for (const l of S.boleto) (porPartido[l.m] ??= []).push(l);
  for (const patas of Object.values(porPartido)) {
    const q = patas.length === 1 ? probPata(patas[0]) : probConjunta(patas);
    if (q == null) return { p: null };
    p *= q;
    const js = patas.filter((l) => l.t === "j").map((l) => l.jid);
    if (new Set(js).size < js.length) mismoJugador = true;
  }
  return { p, mismoJugador };
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
  return `<div class="fila"><div class="nombre">${esc(nombre)}${sub ? `<small>${esc(sub)}</small>` : ""}${lineaVeces(partidoPorId(l.m), [l])}</div><div class="p">${pct(p)}</div>${c}${botonMas(l)}</div>`;
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
  const tabs = [["recomendadas", "Combinadas"], ["chuleta", "Chuleta"], ["goles", "Goles"], ["equipos", "Equipos"], ["jugadores", "Jugadores"]]
    .map(([k, n]) => `<button role="tab" data-tab="${k}" aria-selected="${S.tab === k}">${n}</button>`).join("");
  const cuerpo = { recomendadas: tabRecomendadas, chuleta: tabChuleta, goles: tabGoles, equipos: tabEquipos, jugadores: tabJugadores }[S.tab](m);
  $("#vista").innerHTML = `<div class="tabs" role="tablist">${tabs}</div>${cuerpo}`;
}

// ---------- Combinadas recomendadas ----------
const RANGOS = [["hasta4", "Hasta 4", 1.8, 4], ["4a10", "De 4 a 10", 4, 10], ["10a15", "De 10 a 15", 10, 15], ["mas30", "Más de 30", 30, 120]];
// Cuánto se fía uno de cada mercado (de la prueba histórica: remates y tiros a puerta es lo que
// mejor predice el modelo; faltas y tarjetas, lo que menos).
const FIABILIDAD = {
  e: { remates: 0.9, tiros_puerta: 0.9, corners: 0.8, entradas: 0.5, faltas: 0.5 },
  j: { remates: 1, tiros_puerta: 1, entradas: 0.7, faltas_recibidas: 0.7 },
};
const NOMBRE_EQ = { corners: "corners", remates: "remates", tiros_puerta: "tiros a puerta", faltas: "faltas", entradas: "entradas" };
const GRUPO_GOL = (k) => /^(1|X|2|1X|X2|12|L-1.5|V-1.5)$/.test(k) ? "res" : /^[OU]/.test(k) ? "tot" : k.includes("BTTS") ? "btts" : k.slice(0, 1);

function candidatas(m) {
  const res = [];
  const add = (leg, tipo, grupo, w) => { const p = probPata(leg); if (p != null) res.push({ leg, tipo, grupo, w, p }); };
  if (m.goles) for (const k of ["1", "X", "2", "1X", "X2", "O1.5", "O2.5", "U2.5", "U3.5", "BTTS", "NOBTTS", "L_O0.5", "V_O0.5", "L_O1.5", "V_O1.5", "L-1.5", "V-1.5"]) {
    const p = probGoles(m, [k]);
    if (p >= 0.3 && p <= 0.95) add({ t: "g", m: m.id, k, label: etqGol(k, m) }, "g", "g:" + GRUPO_GOL(k), 1);
  }
  for (const [stat, w] of Object.entries(FIABILIDAD.e)) {
    if (!m.equipo[stat]) continue;
    for (const [lado, nombre] of [["local", m.local], ["visitante", m.visitante], ["total", "Total"]]) for (const dir of ["mas", "menos"]) {
      // Las dos líneas más interesantes: una bastante probable y otra más arriesgada.
      const lineas = [];
      for (let x = 0.5; x < 60; x++) { const p = probEquipo(m, stat, lado, dir, x); if (p >= 0.5 && p <= 0.92) lineas.push([x, p]); }
      for (const objetivo of [0.82, 0.62]) {
        const mejor = lineas.sort((a, b) => Math.abs(a[1] - objetivo) - Math.abs(b[1] - objetivo))[0];
        if (mejor && !res.some((c) => c.leg.t === "e" && c.leg.stat === stat && c.leg.lado === lado && c.leg.dir === dir && c.leg.linea === mejor[0]))
          add({ t: "e", m: m.id, stat, lado, dir, linea: mejor[0], label: `${nombre}: ${dir === "mas" ? "más" : "menos"} de ${mejor[0]} ${NOMBRE_EQ[stat]}` }, "e", `e:${stat}:${lado}`, w);
      }
    }
  }
  for (const lado of ["local", "visitante"]) for (const j of (m.jugadores[lado] || []).filter((x) => x.titular)) {
    for (const [stat, w] of Object.entries(FIABILIDAD.j)) for (const n of [1, 2]) {
      const p = probJugador(j, stat, n);
      if (p >= 0.35 && p <= 0.92) add({ t: "j", m: m.id, jid: j.id, stat, n, label: `${j.nombre}: ${n}+ ${STATS_JUG[stat].toLowerCase()}` }, "j", `j:${j.id}`, w);
    }
  }
  return res;
}
const choca = (a, b) => a.grupo === b.grupo ||
  (a.tipo === "e" && b.tipo === "e" && a.leg.stat === b.leg.stat && (a.leg.lado === "total" || b.leg.lado === "total"));
const MAX_TIPO = { g: 2, e: 3, j: 3 };

function recomendar(m, lo, hi) {
  const cands = candidatas(m);
  const porTipo = { g: cands.filter((c) => c.tipo === "g"), e: cands.filter((c) => c.tipo === "e"), j: cands.filter((c) => c.tipo === "j") };
  const obligatorios = ["g", "e", "j"].filter((t) => porTipo[t].length);
  const rand = mulberry32((m.id * 13 + lo * 1000) % 2147483647);
  const objetivo = Math.sqrt(lo * hi), hallados = new Map();
  const elegir = (lista, pesos) => { let s = pesos.reduce((a, b) => a + b, 0), r = rand() * s; for (let i = 0; i < lista.length; i++) { r -= pesos[i]; if (r <= 0) return lista[i]; } return lista[lista.length - 1]; };
  // Menos patas primero (cada pata suma margen de la casa), luego mercados más fiables y coherentes.
  const ordenar = () => [...hallados.values()].sort((a, b) => a.k - b.k || b.w - a.w || Math.abs(Math.log(a.p * objetivo)) - Math.abs(Math.log(b.p * objetivo)));
  for (let k = Math.max(3, obligatorios.length); k <= 8; k++) {
    // Cada pata, con una probabilidad parecida a la que haría falta para llegar a la cuota objetivo.
    // Las obligatorias (una de goles, una de equipo y una de jugador) se eligen con más margen
    // para que no salga siempre el único mercado de goles con esa probabilidad.
    const pPata = Math.pow(1 / objetivo, 1 / k);
    const peso = (ancho) => (c) => c.w * Math.exp(-((Math.log(c.p) - Math.log(pPata)) ** 2) / ancho);
    for (let intento = 0; intento < 1500; intento++) {
      const combo = [], cuenta = { g: 0, e: 0, j: 0 };
      let acc = null, p = 1, valido = true;
      const meter = (c) => {
        if (combo.some((x) => choca(x, c)) || cuenta[c.tipo] >= MAX_TIPO[c.tipo]) return false;
        const b = bitsPata(c.leg), nuevo = acc ? acc.map((x, i) => x & b[i]) : b.slice(), q = contar(nuevo) / N_SIM;
        if (acc && q > 0.97 * p) return false; // la pata no aporta nada
        combo.push(c); cuenta[c.tipo]++; acc = nuevo; p = q; return true;
      };
      for (const t of obligatorios) if (!meter(elegir(porTipo[t], porTipo[t].map(peso(0.8))))) { valido = false; break; }
      for (let extra = 0; valido && combo.length < k && extra < 20; extra++) meter(elegir(cands, cands.map(peso(0.18))));
      if (!valido || combo.length !== k || !p) continue;
      const cuota = 1 / p;
      if (cuota < lo || cuota > hi) continue;
      // Coherencia: probabilidad real frente a la de multiplicar. Por debajo de 0.9 las patas
      // se contradicen (p. ej. "no marcan ambos" con muchos tiros a puerta del visitante).
      const coherencia = p / combo.reduce((s, c) => s * c.p, 1);
      if (coherencia < 0.9) continue;
      const clave = combo.map((c) => idPata(c.leg)).sort().join("|");
      if (!hallados.has(clave)) hallados.set(clave, { combo, p, k, w: combo.reduce((s, c) => s + c.w, 0) / k + Math.log(coherencia) });
    }
    // Si ya hay 5 variadas con estas patas, no hace falta alargarlas.
    const elegidas = seleccionar(ordenar(), 2);
    if (elegidas.length === 5) return elegidas;
  }
  return seleccionar(ordenar(), 5);
}

// Variedad: dos combinadas comparten como mucho la mitad de sus patas y ninguna pata
// aparece en más de maxUsos de las 5.
function seleccionar(orden, maxUsos) {
  const elegidas = [], usos = new Map();
  for (const r of orden) {
    const ids = r.combo.map((c) => idPata(c.leg));
    if (ids.some((id) => (usos.get(id) || 0) >= maxUsos)) continue;
    if (!elegidas.every((e) => e.combo.filter((c) => ids.includes(idPata(c.leg))).length <= Math.floor(r.k / 2))) continue;
    elegidas.push(r);
    ids.forEach((id) => usos.set(id, (usos.get(id) || 0) + 1));
    if (elegidas.length === 5) break;
  }
  return elegidas;
}

const RECOS = new Map();
function tabRecomendadas(m) {
  if (!S.data.correlacion) return `<p class="vacio">Faltan datos para calcular combinadas.</p>`;
  const chips = RANGOS.map(([k, n]) => `<button class="chip" data-rango="${k}" aria-pressed="${S.rango === k}">${n}</button>`).join("");
  const [, nombre, lo, hi] = RANGOS.find((r) => r[0] === S.rango);
  const clave = `${m.id}|${S.rango}`;
  if (!RECOS.has(clave)) RECOS.set(clave, recomendar(m, lo, hi));
  const recos = RECOS.get(clave);
  let h = `<div class="chips">${chips}</div>`;
  if (!recos.length) return h + `<p class="vacio">No hay combinadas razonables con cuota justa ${nombre.toLowerCase()} en este partido.</p>`;
  recos.forEach((r, i) => {
    const indep = r.combo.reduce((s, c) => s * c.p, 1);
    h += `<div class="card"><div class="cabeza-reco"><h3>Combinada ${i + 1}</h3><div><b>${pct(r.p)}</b> · justa <b>${cj(r.p)}</b></div></div>` +
      r.combo.map((c) => `<div class="pata"><span class="tipo t-${c.tipo}">${{ g: "Goles", e: "Equipo", j: "Jugador" }[c.tipo]}</span><span class="pata-nombre">${esc(c.leg.label)}${lineaVeces(m, [c.leg])}</span><span class="p">${pct(c.p)}</span></div>`).join("") +
      `<p class="nota">Si las patas fueran independientes saldría ${pct(indep)}; la diferencia es la relación entre ellas.</p>
      <button class="boton-sec anadir" data-reco="${i}">Añadir al boleto</button></div>`;
  });
  return h + `<p class="nota">Probabilidad y cuota justa calculadas simulando el partido ${N_SIM.toLocaleString("es-ES")} veces. bet365 pagará menos que la cuota justa en "Crear Apuesta": solo tiene valor si su cuota es mayor. Se priorizan las combinadas con menos selecciones (cada una añade margen de la casa) y los mercados que mejor predice el modelo. ${NOTA_VECES}</p>`;
}

// ---------- Chuleta: qué buscar en bet365 y desde qué cuota ----------
// Sin las cuotas de bet365 no se sabe qué tiene valor, así que se da la vuelta: para cada apuesta
// interesante se da la cuota a partir de la cual compensa (la justa) y la que se queda a menos
// de un 5 % de ella (casi justa). El usuario solo tiene que compararlas al abrir bet365.
const CASI = 0.05, ALTA = 0.7, MAX_ALTA = 0.93;
const umbral = (p) => (Math.ceil(100 / p) / 100).toFixed(2);
const umbralCasi = (p) => (Math.ceil(100 * (1 - CASI) / p) / 100).toFixed(2);
const fiables = (t) => Object.entries(FIABILIDAD[t]).filter(([, w]) => w >= 0.7).map(([s]) => s);

function altaProbabilidad(m) {
  // Para cada mercado, la línea menos obvia que aún llega al 70 % (la que mejor paga).
  const mejores = new Map();
  const add = (grupo, leg, tipo, p) => {
    if (p < ALTA || p > MAX_ALTA) return;
    const actual = mejores.get(grupo);
    if (!actual || p < actual.p) mejores.set(grupo, { leg, tipo, p });
  };
  if (m.goles) for (const k of ["1", "2", "1X", "X2", "O1.5", "O2.5", "U2.5", "U3.5", "BTTS", "NOBTTS", "L_O0.5", "V_O0.5", "L_O1.5", "V_O1.5"])
    add("g:" + k, { t: "g", m: m.id, k, label: etqGol(k, m) }, "g", probGoles(m, [k]));
  for (const stat of fiables("e")) {
    if (!m.equipo[stat]) continue;
    for (const [lado, nombre] of [["local", m.local], ["visitante", m.visitante], ["total", "Total"]]) for (const dir of ["mas", "menos"])
      for (let x = 0.5; x < 60; x++)
        add(`e:${stat}:${lado}:${dir}`, { t: "e", m: m.id, stat, lado, dir, linea: x, label: `${nombre}: ${dir === "mas" ? "más" : "menos"} de ${x} ${NOMBRE_EQ[stat]}` }, "e", probEquipo(m, stat, lado, dir, x));
  }
  for (const lado of ["local", "visitante"]) for (const j of (m.jugadores[lado] || []).filter((x) => x.titular))
    for (const stat of fiables("j")) for (let n = 1; n <= 4; n++)
      add(`j:${j.id}:${stat}`, { t: "j", m: m.id, jid: j.id, stat, n, label: `${j.nombre}: ${n}+ ${STATS_JUG[stat].toLowerCase()}` }, "j", probJugador(j, stat, n));
  const todos = [...mejores.values()].sort((a, b) => b.p - a.p);
  // Variedad: no más de dos jugadores con la misma estadística (si no, salen seis "1+ entradas").
  const limite = { g: 3, e: 5, j: 6 }, porStat = {};
  return todos.filter((c) => {
    if (c.tipo === "j" && (porStat[c.leg.stat] = (porStat[c.leg.stat] || 0) + 1) > 2) return false;
    return limite[c.tipo]-- > 0;
  });
}

function filaChuleta(c, m) {
  let sub = { g: "Goles", e: "Equipo", j: "Jugador · si es titular" }[c.tipo];
  const cuota = c.tipo === "g" ? m.goles.cuotas[c.leg.k] : null;
  if (cuota) sub += ` · bet365 ahora ${cuota.toFixed(2)}: ${cuota * c.p >= 1 ? "con valor" : cuota * c.p >= 1 - CASI ? "casi justa" : "sin valor"}`;
  return `<div class="fila"><div class="nombre">${esc(c.leg.label)}<small>${esc(sub)}</small>${lineaVeces(m, [c.leg])}</div><div class="p">${pct(c.p)}</div>
    <div class="c umbral"><b>≥ ${umbral(c.p)}</b><span>casi ${umbralCasi(c.p)}</span></div>${botonMas(c.leg)}</div>`;
}

const celdaChuleta = (m, leg, texto, p) => `<button class="lado" aria-pressed="${enBoleto(leg)}" data-pata='${esc(JSON.stringify(leg))}'>
  <span>${texto} · ${pct(p)}</span><b>≥ ${umbral(p)}</b><small>casi ${umbralCasi(p)}</small><i class="hist">${textoVeces(m, [leg], true)}</i></button>`;

function tabChuleta(m) {
  let h = `<div class="aviso">Busca estas apuestas en bet365 y compara su cuota. Si es <b class="valor">igual o mayor que la verde</b>, tiene valor. Si llega a la <b class="casi">amarilla</b>, está casi justa (a menos de un 5 %). Si no llega, bet365 se queda demasiado margen.</div>`;
  const alta = altaProbabilidad(m);
  h += `<div class="card"><h3>Alta probabilidad <span class="nota">· 70 % o más</span></h3>` +
    (alta.length ? alta.map((c) => filaChuleta(c, m)).join("") : `<p class="nota">Nada llega al 70 % en este partido.</p>`) + `</div>`;

  // Líneas principales (cerca del 50 %), que es donde bet365 pone menos margen y más fácil es
  // encontrar una cuota casi justa. No se incluyen los goles: su probabilidad sale en su mayor
  // parte de las propias cuotas de bet365, así que nunca se alejaría de ellas.
  h += `<h2 class="seccion">Para comparar con bet365</h2>`;
  for (const stat of ["corners", "tiros_puerta", "remates"]) {
    if (!m.equipo[stat]) continue;
    h += `<div class="card"><h3>${STATS_EQ[stat].nombre}</h3>`;
    for (const [lado, nombre] of [["local", m.local], ["visitante", m.visitante], ["total", "Total del partido"]]) {
      let x = 0.5, mejor = 1;
      for (let y = 0.5; y < 60; y++) { const d = Math.abs(probEquipo(m, stat, lado, "mas", y) - 0.5); if (d < mejor) { mejor = d; x = y; } }
      const pm = probEquipo(m, stat, lado, "mas", x), base = { t: "e", m: m.id, stat, lado, linea: x };
      h += `<div class="dos"><div class="nombre">${esc(nombre)}<small>línea ${x}</small></div>` +
        celdaChuleta(m, { ...base, dir: "mas", label: `${nombre}: más de ${x} ${NOMBRE_EQ[stat]}` }, "Más", pm) +
        celdaChuleta(m, { ...base, dir: "menos", label: `${nombre}: menos de ${x} ${NOMBRE_EQ[stat]}` }, "Menos", 1 - pm) + `</div>`;
    }
    h += `</div>`;
  }

  const js = [];
  for (const lado of ["local", "visitante"]) for (const j of (m.jugadores[lado] || []).filter((x) => x.titular))
    for (const stat of ["tiros_puerta", "remates"]) {
      let mejor = null;
      for (let n = 1; n <= 5; n++) { const p = probJugador(j, stat, n); if (p >= 0.3 && p <= 0.7 && (!mejor || Math.abs(p - 0.5) < Math.abs(mejor.p - 0.5))) mejor = { n, p }; }
      if (mejor) js.push({ tipo: "j", p: mejor.p, esperado: j[stat], leg: { t: "j", m: m.id, jid: j.id, stat, n: mejor.n, label: `${j.nombre}: ${mejor.n}+ ${STATS_JUG[stat].toLowerCase()}` } });
    }
  js.sort((a, b) => Math.abs(a.p - 0.5) - Math.abs(b.p - 0.5));
  if (js.length) h += `<div class="card"><h3>Jugadores</h3>` + js.slice(0, 10).sort((a, b) => b.esperado - a.esperado).map((c) => filaChuleta(c, m)).join("") + `</div>`;

  return h + `<p class="nota">Si bet365 ofrece otra línea (por ejemplo 4.5 corners en vez de 5.5), búscala en las pestañas Equipos o Jugadores. Toca cualquier apuesta para añadirla al boleto. Que bet365 pague más que la cuota justa no garantiza nada: a veces sabe algo que el modelo no (una baja de última hora, por ejemplo). En las líneas principales, «7/10 · 4/10» son los últimos 10 del local y del visitante. ${NOTA_VECES}</p>`;
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
    <p class="nota">Probabilidad exacta del combo, teniendo en cuenta que los mercados están relacionados. Compárala con la cuota de bet365: solo hay valor si su cuota supera la justa.</p></div>
    <p class="nota">${NOTA_VECES}</p>`;
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
    return `<div class="fila"><div class="nombre">${esc(etq)}${lineaVeces(m, [{ t: "g", k: a }, { t: "g", k: b }])}</div><div class="p">${pct(p)}</div><div class="c">justa <b>${cj(p)}</b></div>
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
  return h + `<p class="nota">Modelo por equipo con lo que cada uno genera y concede, ventaja de local y partidos recientes con más peso. ${NOTA_VECES}</p>`;
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
  let h = `<div class="chips">${chips}</div><div class="chips">${lineas}${filtro}</div><div class="aviso">${estado}. Probabilidades si el jugador es titular. «Últimos 10»: veces que lo consiguió en sus últimos 10 partidos como titular.</div>`;
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
  const { p, mismoJugador } = probBoleto();
  const filas = S.boleto.map((l, i) => {
    const m = partidoPorId(l.m), pp = probPata(l);
    return `<div class="fila" style="grid-template-columns:1fr auto auto"><div class="nombre">${esc(l.label)}<small>${m ? esc(m.local + " – " + m.visitante) : "Partido ya no disponible"}</small>${lineaVeces(m, [l])}</div>
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
    <p class="nota">Las selecciones del mismo partido se calculan simulando el partido ${N_SIM.toLocaleString("es-ES")} veces con la relación real entre mercados (p. ej. muchos remates y muchos corners suelen ir juntos).${mismoJugador ? " Hay dos apuestas del mismo jugador: esa parte es aproximada." : ""}</p>
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
  if (d.partido) { S.tab = "recomendadas"; location.hash = `#/p/${d.partido}`; window.scrollTo(0, 0); return; }
  if (d.filtro) { S.filtro = d.filtro; guardar("filtro", S.filtro); return pintar(); }
  if (d.tab) { S.tab = d.tab; return pintar(); }
  if (d.rango) { S.rango = d.rango; return pintar(); }
  if (d.reco != null) {
    const m = partidoPorId(Number((location.hash.match(/^#\/p\/(\d+)/) || [])[1]));
    for (const c of RECOS.get(`${m.id}|${S.rango}`)[Number(d.reco)].combo) if (!enBoleto(c.leg)) alternarPata(c.leg);
    pintar();
    return abrirBoleto();
  }
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
