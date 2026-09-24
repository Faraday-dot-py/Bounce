import { loadWeights } from "./model.js";
import { Sim } from "./sim.js";
import { Physics } from "./physics.js";
import { Viz3D } from "./viz3d.js";

const $ = (id) => document.getElementById(id);
const S = 6;
const H = 32;
const DEFAULT_BALLS = 12;
const SPAWN_EVERY = 2;

const stops = [[0, [91, 141, 239]], [5, [127, 209, 185]], [12, [242, 227, 148]], [25, [255, 157, 92]]];
function speedColor(v) {
  let i = 0;
  while (i < stops.length - 2 && v > stops[i + 1][0]) i++;
  const [a, ca] = stops[i], [b, cb] = stops[i + 1];
  const t = Math.min(Math.max((v - a) / (b - a), 0), 1);
  return `rgb(${ca.map((x, k) => Math.round(x + (cb[k] - x) * t)).join(",")})`;
}

const weights = await loadWeights("weights");
const sim = new Sim(weights);
const cfg = weights.config;
const gt = new Physics(cfg.n, { dt: cfg.dt, gravity: cfg.gravity, radius: cfg.radius });
const gtIds = [];
const arena = $("arena"), gtCanvas = $("gt");
const viz = new Viz3D($("gl"), $("ov"), weights);

const state = {
  paused: false, tps: 30, held: false, cursor: [0, 0], selected: -1, edges: 0, gtOn: false,
  trace: null, auto: true, stepMs: 0, measured: 0, ticksSince: 0, sinceT: performance.now(),
};
const edgeModes = ["off", "selected", "all"];
const edgeLabels = ["Edges: off", "Edges: selected", "Edges: all"];
state.edges = 1;

function seedGT() {
  gt.clear();
  gtIds.length = 0;
  for (let i = 0; i < sim.count; i++) {
    gt.add(sim.pos[2 * i], sim.pos[2 * i + 1], sim.vel[2 * i], sim.vel[2 * i + 1]);
    gtIds.push(sim.ids[i]);
  }
}

function spawnBall(x, y, jitter) {
  const id = sim.spawn(x, y, jitter);
  if (id < 0) return;
  if (state.gtOn) {
    if (gtIds.length && gtIds[0] !== sim.ids[0]) { gt.remove(0); gtIds.shift(); }
    const i = sim.count - 1;
    gt.add(sim.pos[2 * i], sim.pos[2 * i + 1], sim.vel[2 * i], sim.vel[2 * i + 1]);
    gtIds.push(id);
  }
}

function clearAll() {
  sim.clear();
  gt.clear();
  gtIds.length = 0;
  state.selected = -1;
  refreshTrace();
}

function resetAll() {
  sim.reset();
  gt.clear();
  gtIds.length = 0;
  state.selected = -1;
  state.auto = true;
  sim.populate(DEFAULT_BALLS);
  if (state.gtOn) seedGT();
  refreshTrace();
}

function degree(i) {
  let c = 0;
  for (let j = 0; j < sim.count; j++) {
    if (j !== i && Math.hypot(sim.pos[2 * i] - sim.pos[2 * j], sim.pos[2 * i + 1] - sim.pos[2 * j + 1]) <= cfg.neighbor_radius) c++;
  }
  return c;
}

function busiest() {
  let best = -1, bc = -1;
  for (let i = 0; i < sim.count; i++) {
    const c = degree(i);
    if (c > bc) { bc = c; best = i; }
  }
  return best;
}

function ensureSelection() {
  if (sim.indexOf(state.selected) >= 0) return;
  const i = busiest();
  state.selected = i >= 0 ? sim.ids[i] : -1;
}

function previewTrace() {
  ensureSelection();
  if (state.selected < 0) return null;
  const c = sim.count;
  const p = sim.pos.slice(0, 2 * c), v = sim.vel.slice(0, 2 * c), h = sim.hidden.slice(0, H * c);
  const tr = sim.net.step(sim.pos, sim.vel, sim.hidden, c, sim.indexOf(state.selected));
  sim.pos.set(p); sim.vel.set(v); sim.hidden.set(h);
  return tr;
}

function labelTrace(tr) {
  if (tr) for (const e of tr.edges) e.srcId = e.self ? "self" : sim.ids[e.src];
  return tr;
}

function refreshTrace() {
  state.trace = labelTrace(previewTrace());
  viz.setTrace(state.trace, 400);
}

function gtStep() {
  if (!state.gtOn) return;
  gt.step();
}

function doTick() {
  if (state.held && sim.ticks % SPAWN_EVERY === 0) spawnBall(state.cursor[0], state.cursor[1], 0.4);
  ensureSelection();
  if (state.auto && sim.ticks % 20 === 0 && degree(sim.indexOf(state.selected)) === 0) state.selected = sim.ids[busiest()];
  const t0 = performance.now();
  const tr = sim.tick(state.selected);
  state.stepMs = state.stepMs * 0.9 + (performance.now() - t0) * 0.1;
  gtStep();
  state.ticksSince++;
  state.trace = labelTrace(tr);
  if (state.gtOn) syncGtIds();
}

function syncGtIds() {
  // model may drop non-finite tokens; keep the ground truth aligned by id
  for (let i = gtIds.length - 1; i >= 0; i--) {
    if (sim.indexOf(gtIds[i]) < 0) { gt.remove(i); gtIds.splice(i, 1); }
  }
}

function drawBall(ctx, x, y, r, color) {
  ctx.beginPath();
  ctx.arc((y + 0.5) * S, (x + 0.5) * S, r * S, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawArena() {
  const ctx = arena.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, arena.width, arena.height);
  const r = cfg.radius;
  const selIdx = sim.indexOf(state.selected);
  if (state.edges > 0 && sim.count > 1) {
    ctx.lineWidth = 1;
    const net = sim.net;
    if (state.edges === 2) {
      for (let i = 0; i < sim.count; i++) {
        for (let e = net.dstStart[i]; e < net.dstStart[i + 1]; e++) {
          const s = net.src[e];
          if (s === i || s >= sim.count) continue;
          ctx.strokeStyle = `rgba(160,190,255,${0.15 + 0.7 * net.weight[e]})`;
          ctx.beginPath();
          ctx.moveTo((sim.pos[2 * i + 1] + 0.5) * S, (sim.pos[2 * i] + 0.5) * S);
          ctx.lineTo((sim.pos[2 * s + 1] + 0.5) * S, (sim.pos[2 * s] + 0.5) * S);
          ctx.stroke();
        }
      }
    } else if (selIdx >= 0 && state.trace) {
      const cx = (sim.pos[2 * selIdx + 1] + 0.5) * S, cy = (sim.pos[2 * selIdx] + 0.5) * S;
      ctx.strokeStyle = "rgba(127,178,255,0.35)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(cx, cy, cfg.neighbor_radius * S, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      for (const e of state.trace.edges) {
        if (e.self) continue;
        const j = sim.indexOf(e.srcId);
        if (j < 0) continue;
        ctx.strokeStyle = `rgba(255,255,255,${0.2 + 0.8 * e.weight})`;
        ctx.lineWidth = 1 + 3 * e.weight;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo((sim.pos[2 * j + 1] + 0.5) * S, (sim.pos[2 * j] + 0.5) * S);
        ctx.stroke();
      }
    }
  }
  for (let i = 0; i < sim.count; i++) {
    drawBall(ctx, sim.pos[2 * i], sim.pos[2 * i + 1], r, speedColor(Math.hypot(sim.vel[2 * i], sim.vel[2 * i + 1])));
  }
  if (selIdx >= 0) {
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc((sim.pos[2 * selIdx + 1] + 0.5) * S, (sim.pos[2 * selIdx] + 0.5) * S, r * S + 4, 0, 2 * Math.PI);
    ctx.stroke();
  }
}

function drawGT() {
  const ctx = gtCanvas.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, gtCanvas.width, gtCanvas.height);
  for (let i = 0; i < gt.count; i++) {
    drawBall(ctx, gt.xs[i], gt.ys[i], cfg.radius, speedColor(Math.hypot(gt.vxs[i], gt.vys[i])));
  }
  const selIdx = gtIds.indexOf(state.selected);
  if (selIdx >= 0) {
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc((gt.ys[selIdx] + 0.5) * S, (gt.xs[selIdx] + 0.5) * S, cfg.radius * S + 4, 0, 2 * Math.PI);
    ctx.stroke();
  }
}

function energyModel() {
  let e = 0;
  for (let i = 0; i < sim.count; i++) e += 0.5 * (sim.vel[2 * i] ** 2 + sim.vel[2 * i + 1] ** 2) - cfg.gravity * sim.pos[2 * i];
  return e;
}

function energyGT() {
  let e = 0;
  for (let i = 0; i < gt.count; i++) e += 0.5 * (gt.vxs[i] ** 2 + gt.vys[i] ** 2) - cfg.gravity * gt.xs[i];
  return e;
}

function updateStats() {
  const rows = [["balls", sim.count], ["tick", sim.ticks], ["model step", `${state.stepMs.toFixed(2)} ms`], ["ticks/s", state.measured.toFixed(1)]];
  const i = sim.indexOf(state.selected);
  if (i >= 0) {
    const tr = state.trace;
    rows.push(["selected", `#${state.selected} at (${sim.pos[2 * i].toFixed(1)}, ${sim.pos[2 * i + 1].toFixed(1)}), speed ${Math.hypot(sim.vel[2 * i], sim.vel[2 * i + 1]).toFixed(2)}`]);
    if (tr) {
      const self = tr.edges.find((e) => e.self);
      rows.push(["neighbours", `${tr.edges.length - 1}, self-attention ${self ? self.weight.toFixed(3) : "-"}`]);
      rows.push(["step dp / dv", `(${tr.dp.map((x) => x.toFixed(3)).join(", ")}) / (${tr.dv.map((x) => x.toFixed(3)).join(", ")})`]);
    }
  }
  if (state.gtOn) {
    let sum = 0, m = 0;
    for (let k = 0; k < gtIds.length; k++) {
      const j = sim.indexOf(gtIds[k]);
      if (j < 0) continue;
      sum += Math.hypot(sim.pos[2 * j] - gt.xs[k], sim.pos[2 * j + 1] - gt.ys[k]);
      m++;
    }
    rows.push(["mean position gap", m ? `${(sum / m).toFixed(2)} cells` : "-"]);
    rows.push(["KE + gravity PE", `model ${energyModel().toFixed(0)}, truth ${energyGT().toFixed(0)}`]);
  }
  const sc = viz.scales;
  if (sc.input) rows.push(["colour full scale", `input ${sc.input.toFixed(1)}, qkv ${sc.qkv.toFixed(1)}, gru ${sc.gru.toFixed(1)}, heads ${sc.head.toFixed(1)}`]);
  $("stats").innerHTML = rows.map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
}

function cellFromEvent(e) {
  const r = arena.getBoundingClientRect();
  return [((e.clientY - r.top) / r.height) * cfg.n - 0.5, ((e.clientX - r.left) / r.width) * cfg.n - 0.5];
}

arena.addEventListener("pointerdown", (e) => {
  arena.setPointerCapture(e.pointerId);
  state.cursor = cellFromEvent(e);
  let best = -1, bd = 2.2;
  for (let i = 0; i < sim.count; i++) {
    const d = Math.hypot(sim.pos[2 * i] - state.cursor[0], sim.pos[2 * i + 1] - state.cursor[1]);
    if (d < bd) { bd = d; best = i; }
  }
  if (best >= 0) {
    state.auto = false;
    state.selected = sim.ids[best];
    refreshTrace();
  } else {
    state.held = true;
    spawnBall(state.cursor[0], state.cursor[1], 0.4);
    if (state.paused) refreshTrace();
  }
});
arena.addEventListener("pointermove", (e) => { if (state.held) state.cursor = cellFromEvent(e); });
const release = () => { state.held = false; };
arena.addEventListener("pointerup", release);
arena.addEventListener("pointercancel", release);

function setPaused(p) {
  state.paused = p;
  $("pause").textContent = p ? "Play" : "Pause";
  $("pause").setAttribute("aria-pressed", String(p));
}

function toggle(id, on) { $(id).setAttribute("aria-pressed", String(on)); }

function setGT(on) {
  state.gtOn = on;
  toggle("gtbtn", on);
  $("gtWrap").hidden = !on;
  $("arenas").classList.toggle("two", on);
  if (on) seedGT();
}

function setEdges(m) {
  state.edges = m;
  $("edges").textContent = edgeLabels[m];
}

function stepOnce() {
  if (!state.paused) setPaused(true);
  doTick();
  viz.setTrace(state.trace, 400);
}

$("pause").onclick = () => setPaused(!state.paused);
$("step").onclick = stepOnce;
$("add").onclick = () => { sim.populate(sim.count + 10); if (state.gtOn) seedGT(); };
$("clear").onclick = clearAll;
$("reset").onclick = resetAll;
$("speed").oninput = (e) => { state.tps = +e.target.value; $("speedv").textContent = e.target.value; };
$("edges").onclick = () => setEdges((state.edges + 1) % 3);
$("gtbtn").onclick = () => setGT(!state.gtOn);
$("wbtn").onclick = () => { viz.showWeights = !viz.showWeights; toggle("wbtn", viz.showWeights); };
$("lbtn").onclick = () => { viz.showLabels = !viz.showLabels; toggle("lbtn", viz.showLabels); };
$("cam").onclick = () => viz.resetCamera();
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.ctrlKey || e.metaKey) return;
  const k = e.key;
  if (k === " ") { e.preventDefault(); setPaused(!state.paused); }
  else if (k === ".") stepOnce();
  else if (k === "c") clearAll();
  else if (k === "r") resetAll();
  else if (k === "e") setEdges((state.edges + 1) % 3);
  else if (k === "g") setGT(!state.gtOn);
  else if (k === "w") $("wbtn").click();
  else if (k === "l") $("lbtn").click();
});

sim.populate(DEFAULT_BALLS);
refreshTrace();

let last = performance.now(), acc = 0;
function loop(now) {
  const dt = now - last;
  last = now;
  if (!state.paused) {
    acc += (dt * state.tps) / 1000;
    let k = 0;
    while (acc >= 1 && k < 4) { doTick(); acc -= 1; k++; }
    if (k) viz.setTrace(state.trace, 1000 / state.tps);
    if (acc > 4) acc = 0;
  }
  if (now - state.sinceT > 1000) {
    state.measured = (state.ticksSince * 1000) / (now - state.sinceT);
    state.ticksSince = 0;
    state.sinceT = now;
  }
  drawArena();
  if (state.gtOn) drawGT();
  viz.frame(now);
  updateStats();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

window.__bounce = { sim, viz, gt, state, setGT, setEdges, setPaused, benchTicks(count, ticks = 300) {
  sim.clear();
  sim.populate(count);
  const t0 = performance.now();
  for (let i = 0; i < ticks; i++) sim.tick(sim.ids[0]);
  return (ticks * 1000) / (performance.now() - t0);
} };
