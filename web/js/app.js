import { loadWeights } from "./model.js";
import { Sim } from "./sim.js";
import { Physics } from "./physics.js";
import { Scene } from "./scene.js";
import { fmt } from "./arch.js";
import { forceLut, drawPair, drawWall, drawEnergy } from "./plots.js";

const $ = (id) => document.getElementById(id);
const H = 32, SWEEP = 900, HIST = 300;

const weights = await loadWeights("weights");
const sim = new Sim(weights);
const cfg = weights.config;
const net = sim.net;
let lut = forceLut(net);
const gt = new Physics(cfg.n, { dt: cfg.dt, gravity: cfg.gravity, radius: cfg.radius, substeps: 64 });
const view = new Scene($("stage"), cfg);

const st = {
  paused: false, tps: 20, selected: -1, auto: true, edges: 1, gtOn: false, gravity: true, arch: true, panel: true,
  trace: null, stepMs: 0, gt, gtIds: [], hoverBall: -1, sweep: -1e9, eModel: [], eTruth: [], dirty: true,
};

function seedGT() {
  gt.clear();
  st.gtIds.length = 0;
  for (let i = 0; i < sim.count; i++) {
    gt.add(sim.pos[2 * i], sim.pos[2 * i + 1], sim.vel[2 * i], sim.vel[2 * i + 1]);
    st.gtIds.push(sim.ids[i]);
  }
  st.eModel.length = st.eTruth.length = 0;
}

function syncGT() {
  for (let i = st.gtIds.length - 1; i >= 0; i--) if (sim.indexOf(st.gtIds[i]) < 0) { gt.remove(i); st.gtIds.splice(i, 1); }
}

function degrees() {
  const deg = new Int32Array(sim.count), con = new Int32Array(sim.count), cd = 2 * cfg.radius;
  net.forPairs(sim.pos, sim.count, cfg.neighbor_radius, (i, j, d) => {
    deg[i]++; deg[j]++;
    if (d <= cd) { con[i]++; con[j]++; }
  });
  return { deg, con };
}

function busiest(random = false) {
  const { deg, con } = degrees();
  if (random) {
    const c = [];
    for (let i = 0; i < sim.count; i++) if (con[i]) c.push(i);
    if (c.length) return c[Math.floor(Math.random() * c.length)];
  }
  let best = -1, bc = -1;
  for (let i = 0; i < sim.count; i++) { const s = con[i] * 1000 + deg[i] * 10 - Math.min(Math.hypot(sim.vel[2 * i], sim.vel[2 * i + 1]), 9); if (s > bc) { bc = s; best = i; } }
  return best;
}

function contacts(i) {
  return i < 0 ? 0 : degrees().con[i];
}

function ensureSelection() {
  if (sim.indexOf(st.selected) >= 0) return;
  const i = busiest();
  st.selected = i >= 0 ? sim.ids[i] : -1;
}

function pickContact() {
  const i = busiest(true);
  if (i < 0) return;
  st.auto = false;
  st.selected = sim.ids[i];
  refreshTrace();
  st.sweep = performance.now();
}

function setTrace(tr) {
  if (tr) for (const e of tr.edges) e.srcId = sim.ids[e.src];
  st.trace = tr;
  st.dirty = true;
  view.arch.set(tr, (i) => sim.ids[i]);
  const now = performance.now();
  if (now - st.sweep > SWEEP) st.sweep = now;
}

function refreshTrace() {
  ensureSelection();
  if (st.selected < 0) return setTrace(null);
  const c = sim.count;
  const p = sim.pos.slice(0, 2 * c), v = sim.vel.slice(0, 2 * c), h = sim.hidden.slice(0, H * c);
  const tr = net.step(sim.pos, sim.vel, sim.hidden, c, sim.indexOf(st.selected));
  sim.pos.set(p); sim.vel.set(v); sim.hidden.set(h);
  setTrace(tr);
}

function energyModel() {
  let e = 0, ke = 0;
  for (let i = 0; i < sim.count; i++) {
    const k = 0.5 * (sim.vel[2 * i] ** 2 + sim.vel[2 * i + 1] ** 2);
    ke += k;
    e += k - net.gx * sim.pos[2 * i] - net.gy * sim.pos[2 * i + 1];
  }
  return [ke, e];
}

function energyTruth() {
  let e = 0;
  for (let i = 0; i < gt.count; i++) e += 0.5 * (gt.vxs[i] ** 2 + gt.vys[i] ** 2) - gt.gravity * gt.xs[i];
  return e;
}

function tick() {
  ensureSelection();
  const had = st.trace && st.trace.edges.some((e) => e.contact);
  st.idle = had ? 0 : (st.idle || 0) + 1;
  if (st.auto && sim.count && st.idle >= 1) { const i = busiest(); if (contacts(i)) { st.selected = sim.ids[i]; st.idle = 0; } }
  const t0 = performance.now();
  const tr = sim.tick(st.selected);
  st.stepMs = st.stepMs * 0.9 + (performance.now() - t0) * 0.1;
  if (st.gtOn) { gt.substeps = sim.count > 150 ? 32 : 64; gt.step(); syncGT(); }
  setTrace(tr);
  st.eModel.push(energyModel()[1]);
  if (st.gtOn) st.eTruth.push(energyTruth());
  if (st.eModel.length > HIST) st.eModel.shift();
  if (st.eTruth.length > HIST) st.eTruth.shift();
}

function countChanged() {
  $("r-balls").value = sim.count;
  $("o-balls").value = sim.count;
  if (st.gtOn) seedGT();
  if (st.paused) refreshTrace();
}

function setCount(n) {
  if (n > sim.count) sim.populate(n);
  while (sim.count > n) sim.drop(sim.count - 1);
  countChanged();
}

function burst(x, y, k = 16, jitter = 2.2) {
  for (let q = 0; q < k * 3 && k > 0; q++) if (sim.spawn(x, y, jitter) >= 0) k--;
  countChanged();
}

function reset() {
  sim.reset();
  st.selected = -1;
  st.auto = true;
  st.eModel.length = st.eTruth.length = 0;
  sim.populate(+$("r-balls").value);
  countChanged();
  refreshTrace();
}

const press = (id, on) => $(id).setAttribute("aria-pressed", String(on));
function setPaused(p) {
  st.paused = p;
  $("b-pause").firstChild.textContent = p ? "Play" : "Pause";
  press("b-pause", p);
}
function setGT(on) {
  st.gtOn = on;
  press("b-gt", on);
  if (on) seedGT();
}
function setGravity(on) {
  st.gravity = on;
  press("b-grav", on);
  net.gOn = on;
  gt.gravity = on ? cfg.gravity : 0;
  view.setGravity(net.gx, net.gy);
  st.eModel.length = st.eTruth.length = 0;
  refreshTrace();
}
const ws = { name: "gravity", i: 0 };
const wname = $("w-name"), widx = $("w-idx"), wval = $("w-val"), wslide = $("w-slide");
const wshape = (name) => name === "gravity" ? 2 : net.w[name].length;
for (const k of Object.keys(net.w)) wname.add(new Option(`${k} [${wshape(k)}]`, k));

function wrange() {
  if (ws.name === "gravity") return [-30, 30, 0.05];
  let m = 1;
  for (const v of net.trained[ws.name]) m = Math.max(m, 3 * Math.abs(v));
  return [-m, m, m / 500];
}

function syncWeight() {
  const n = net.edited();
  const v = net.get(ws.name, ws.i), t = ws.name === "gravity" ? net.trained.gravity[ws.i] * 10 : net.trained[ws.name][ws.i];
  const [lo, hi, step] = wrange();
  wname.value = ws.name;
  widx.max = wshape(ws.name) - 1;
  if (document.activeElement !== widx) widx.value = ws.i;
  if (document.activeElement !== wval) wval.value = +v.toPrecision(6);
  wslide.min = Math.min(lo, v); wslide.max = Math.max(hi, v); wslide.step = step; wslide.value = v;
  $("w-info").textContent = `trained ${+t.toPrecision(5)}${Math.abs(v - t) > 1e-9 ? " · edited" : ""}`;
  $("w-all").disabled = !n;
  $("w-note").textContent = n ? `${n} edited · truth ghost unedited` : "";
}

function selectWeight(name, i) {
  ws.name = name;
  ws.i = Math.max(0, Math.min(wshape(name) - 1, i | 0));
  syncWeight();
}

function editWeight(v) {
  net.set(ws.name, ws.i, v);
  lut = forceLut(net);
  st.eModel.length = st.eTruth.length = 0;
  view.setGravity(net.gx, net.gy);
  syncWeight();
  st.dirty = true;
  refreshTrace();
}

function resetWeights(all) {
  if (all) net.reset(); else net.reset(ws.name, ws.name === "gravity" ? -1 : ws.i);
  lut = forceLut(net);
  st.eModel.length = st.eTruth.length = 0;
  view.setGravity(net.gx, net.gy);
  syncWeight();
  st.dirty = true;
  refreshTrace();
}

wname.onchange = () => selectWeight(wname.value, 0);
widx.oninput = () => selectWeight(ws.name, +widx.value);
wval.oninput = () => { if (wval.value !== "") editWeight(+wval.value); };
wslide.oninput = () => editWeight(+wslide.value);
$("w-reset").onclick = () => resetWeights(false);
$("w-all").onclick = () => resetWeights(true);

function setPanel(on, save = true) {
  st.panel = on;
  press("b-panel", on);
  $("panel").classList.toggle("off", !on);
  $("tab").hidden = on;
  if (save) try { localStorage.setItem("bounce.panel", on ? "1" : "0"); } catch {}
  st.dirty = true;
  if (view.W) { insets(); camera(st.cam || "all"); }
}
function stepOnce() {
  setPaused(true);
  tick();
  st.sweep = performance.now();
}
function camera(name) {
  const i = sim.indexOf(st.selected);
  view.preset(name, i >= 0 ? [sim.pos[2 * i], sim.pos[2 * i + 1]] : null);
  st.follow = name === "follow" && i >= 0;
  st.cam = name;
  for (const n of ["all", "arena", "arch", "follow"]) press("cam-" + n, n === name);
}

$("b-pause").onclick = () => setPaused(!st.paused);
$("b-step").onclick = stepOnce;
$("b-reset").onclick = reset;
$("r-balls").oninput = (e) => { $("o-balls").value = e.target.value; setCount(+e.target.value); };
$("r-speed").oninput = (e) => { st.tps = +e.target.value; $("o-speed").value = e.target.value; };
$("b-burst").onclick = () => burst(10 + Math.random() * 45, 10 + Math.random() * 80);
$("b-grav").onclick = () => setGravity(!st.gravity);
$("b-gt").onclick = () => setGT(!st.gtOn);
$("b-edges").onclick = () => { st.edges = (st.edges + 1) % 3; press("b-edges", st.edges > 0); $("b-edges").firstChild.textContent = ["Edges", "Edges", "All edges"][st.edges]; };
$("b-panel").onclick = () => setPanel(!st.panel);
$("collapse").onclick = () => setPanel(false);
$("tab").onclick = () => setPanel(true);
$("b-pick").onclick = pickContact;
for (const n of ["all", "arena", "arch", "follow"]) $("cam-" + n).onclick = () => camera(n);

window.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || (e.target.tagName === "INPUT" && e.key !== " ") || e.target.tagName === "SELECT") return;
  const act = {
    " ": () => setPaused(!st.paused), ".": stepOnce, r: reset, b: () => $("b-burst").click(), v: () => setGravity(!st.gravity),
    g: () => setGT(!st.gtOn), p: pickContact, e: () => $("b-edges").click(), t: () => setPanel(!st.panel),
    1: () => camera("all"), 2: () => camera("arena"), 3: () => camera("arch"), 4: () => camera("follow"),
    "+": () => $("r-speed").stepUp(), "=": () => $("r-speed").stepUp(), "-": () => $("r-speed").stepDown(),
  }[e.key];
  if (!act) return;
  if (e.target.tagName === "BUTTON" && e.key === " ") return;
  e.preventDefault();
  act();
  if ("+-=".includes(e.key)) $("r-speed").dispatchEvent(new Event("input"));
});

const canvas = view.renderer.domElement;
const tip = $("tip");
let down = null, hoverAt = null;
canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
canvas.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6 || performance.now() - down.t > 500) return;
  down = null;
  const hit = view.pick(e.clientX, e.clientY, sim);
  if (hit && hit.kind === "ball") {
    st.auto = false;
    st.selected = sim.ids[hit.index];
    refreshTrace();
    st.sweep = performance.now();
  } else if (hit && hit.kind === "cell" && hit.cell.par) selectWeight(...hit.cell.par);
  else if (hit && hit.kind === "arena") burst(hit.x, hit.y, 8, 1.2);
  if (e.pointerType !== "mouse") { hoverAt = { x: e.clientX, y: e.clientY }; }
});
canvas.addEventListener("pointermove", (e) => { hoverAt = { x: e.clientX, y: e.clientY }; if (e.buttons) tip.classList.remove("on"); });
canvas.addEventListener("pointerleave", () => { hoverAt = null; tip.classList.remove("on"); view.arch.hover = -1; st.hoverBall = -1; });

const CH = ["position x", "position y", "velocity x", "velocity y"];
function hover() {
  view.arch.hover = -1;
  st.hoverBall = -1;
  const hit = hoverAt && view.pick(hoverAt.x, hoverAt.y, sim);
  if (!hit || hit.kind === "arena") { tip.classList.remove("on"); canvas.style.cursor = hit ? "copy" : ""; return; }
  canvas.style.cursor = "pointer";
  let html;
  if (hit.kind === "cell") {
    view.arch.hover = hit.id;
    html = `<b>${hit.cell.val}</b><br><span>${hit.cell.name}</span>`;
  } else {
    const i = hit.index;
    st.hoverBall = i;
    const v = [sim.pos[2 * i], sim.pos[2 * i + 1], sim.vel[2 * i], sim.vel[2 * i + 1]];
    html = `<b>token #${sim.ids[i]}</b>` + v.map((x, k) => `<br>${k === hit.channel ? "<b>" : "<span>"}${CH[k]} ${fmt(x)}${k === hit.channel ? "</b>" : "</span>"}`).join("");
  }
  tip.innerHTML = html;
  tip.classList.add("on");
  const w = tip.offsetWidth, h = tip.offsetHeight;
  const x = Math.min(hoverAt.x + 14, innerWidth - w - 8), y = hoverAt.y + 16 + h > innerHeight - 8 ? hoverAt.y - h - 10 : hoverAt.y + 16;
  tip.style.transform = `translate(${x}px, ${y}px)`;
}

const labelHost = $("labels"), pool = [];
const widths = new Map();
function measure(text, cls) {
  const key = cls + "|" + text;
  if (!widths.has(key)) {
    const el = labelHost.appendChild(document.createElement("div"));
    el.className = cls;
    el.textContent = text;
    widths.set(key, [el.offsetWidth, el.offsetHeight]);
    el.remove();
  }
  return widths.get(key);
}

const RANK = { arena: 0, sec: 1, op: 2, col: 3, "col off": 3 };
function labels() {
  const list = [{ text: "arena 100 × 100", u: 49.5, w: -4, y: 0, cls: "arena" }];
  if (st.gravity) list.push({ text: "gravity", u: 7, w: 50, y: 0, cls: "arena" });
  if (view.arch.group.visible) {
    const a = view.project(0, 0, 40, true), b = a && { x: a.x, y: a.y };
    const c = b && view.project(10, 0, 40, true);
    const ppu = c ? (Math.hypot((c.x - b.x) * innerWidth, (c.y - b.y) * innerHeight) / 20) : 0;
    for (const l of view.arch.labels) if (ppu >= (l.cls.startsWith("col") ? 6 : 3.5)) list.push({ ...l, y: 0.1, arch: true });
  }
  list.sort((a, b) => RANK[a.cls] - RANK[b.cls]);
  const placed = [];
  let n = 0;
  for (const l of list) {
    const p = view.project(l.u, l.y, l.w, l.arch);
    if (!p || Math.abs(p.x) > 1.1 || Math.abs(p.y) > 1.1) continue;
    const [w, h] = measure(l.text, l.cls);
    const x = ((p.x + 1) / 2) * innerWidth - (l.left ? 0 : w / 2), y = ((1 - p.y) / 2) * innerHeight - h / 2;
    if (placed.some((r) => x < r[2] + 4 && x + w + 4 > r[0] && y < r[3] + 1 && y + h + 1 > r[1])) continue;
    placed.push([x, y, x + w, y + h]);
    const el = pool[n] || labelHost.appendChild(document.createElement("div"));
    pool[n++] = el;
    if (el.textContent !== l.text) el.textContent = l.text;
    if (el.className !== l.cls) el.className = l.cls;
    el.style.display = "";
    el.style.transform = `translate(${x}px, ${y}px)`;
  }
  for (let k = n; k < pool.length; k++) pool[k].style.display = "none";
}

function panel() {
  if (!st.panel || !st.dirty) return;
  st.dirty = false;
  const tr = st.trace;
  const v2 = (a) => `(${fmt(a[0])}, ${fmt(a[1])})`;
  $("sel-id").textContent = tr ? `#${st.selected}` : "";
  $("sel").innerHTML = tr
    ? `pos <b>${v2(tr.pos)}</b><br>vel <b>${v2(tr.vel)}</b><br>neighbours <b>${tr.edges.length}</b> · contacts <b>${tr.edges.filter((e) => e.contact).length}</b> · walls <b>${tr.wall.filter((w) => w.pen > 0).length}</b><br>dp <b>${v2(tr.dp)}</b><br>dv <b>${v2(tr.dv)}</b>`
    : "no balls";
  drawPair($("c-pair"), lut, tr);
  drawWall($("c-wall"), lut, tr);
  drawEnergy($("c-energy"), st.eModel, st.gtOn ? st.eTruth : []);
  const em = st.eModel.at(-1), et = st.eTruth.at(-1);
  $("e-read").textContent = em === undefined ? "" : `model ${em.toFixed(0)}` + (st.gtOn && et !== undefined ? ` · truth ${et.toFixed(0)}` : "");
}

function insets() {
  const small = innerWidth <= 760, pr = $("panel").getBoundingClientRect();
  const top = document.querySelector("header").getBoundingClientRect().bottom + 4;
  const right = st.panel && !small ? pr.width + 24 : 0;
  const bottom = innerHeight - document.querySelector("nav").getBoundingClientRect().top + 8 + (st.panel && small ? pr.height + 8 : 0);
  view.insets(innerWidth, innerHeight, [0, top, right, bottom]);
}

let wasSmall = null;
function resize() {
  view.resize(innerWidth, innerHeight);
  const small = innerWidth <= 760;
  if (small !== wasSmall) { if (wasSmall !== null) setPanel(!small, false); wasSmall = small; }
  insets();
  camera(st.cam || "all");
  st.dirty = true;
}
addEventListener("resize", resize);

const small = matchMedia("(max-width: 760px)").matches;
if (small) { setPanel(false, false); $("r-balls").value = 30; }
try { const p = localStorage.getItem("bounce.panel"); if (p !== null) setPanel(p === "1", false); } catch {}
sim.populate(+$("r-balls").value);
countChanged();
refreshTrace();
selectWeight("gravity", 0);
resize();
view.camera.position.copy(view.goal.pos).multiplyScalar(1.6);

let last = performance.now(), acc = 0, frames = 0, fpsT = last;
function loop(now) {
  const dt = now - last;
  last = now;
  if (!st.paused) {
    acc += (dt * st.tps) / 1000;
    let k = 0;
    while (acc >= 1 && k < 3) { tick(); acc -= 1; k++; }
    if (acc > 3) acc = 0;
  }
  const u = (now - st.sweep) / SWEEP;
  const fi = st.follow ? sim.indexOf(st.selected) : -1;
  if (fi >= 0 && !view.goal) {
    const c = view.controls.target, dx = sim.pos[2 * fi + 1] - c.x, dz = sim.pos[2 * fi] - c.z;
    c.x += dx * 0.1; c.z += dz * 0.1;
    view.camera.position.x += dx * 0.1; view.camera.position.z += dz * 0.1;
  }
  view.update(sim, st, st.trace, lut, now);
  view.frame(u < 1 ? u * 96 - 4 : -1);
  hover();
  labels();
  panel();
  if (++frames && now - fpsT > 500) {
    $("s-fps").textContent = Math.round((frames * 1000) / (now - fpsT));
    frames = 0; fpsT = now;
    const [ke, e] = energyModel();
    $("s-tick").textContent = sim.ticks;
    $("s-balls").textContent = sim.count;
    $("s-ms").textContent = st.stepMs.toFixed(2);
    $("s-ke").textContent = ke.toFixed(0);
    $("s-e").textContent = e.toFixed(0);
    st.dirty = true;
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

window.__bounce = { net, selectWeight, editWeight, sim, view, st, gt, setGT, setPaused, setGravity, camera, benchTicks(count, ticks = 300) {
  sim.clear();
  sim.populate(count);
  const t0 = performance.now();
  for (let i = 0; i < ticks; i++) sim.tick(sim.ids[0]);
  return (ticks * 1000) / (performance.now() - t0);
} };
