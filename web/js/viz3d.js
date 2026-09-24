// Instanced-cube WebGL2 view of one ball's activations through a token-model
// step, in the style of bbycroft.net/llm: every tensor is a grid of cells,
// colour is the value (diverging, symmetric), height is |value|.

const MAXE = 12;
const H = 32;

const VS = `#version 300 es
in vec3 aPos; in vec3 aNorm; in vec2 aOff; in float aKind; in float aVal;
uniform mat4 uMVP; uniform float uCell;
out vec3 vCol;
vec3 cmap(float v) {
  float t = abs(v);
  vec3 c0 = vec3(0.13, 0.15, 0.20);
  vec3 e1 = v < 0.0 ? vec3(0.22, 0.50, 0.92) : vec3(0.93, 0.50, 0.08);
  vec3 e2 = v < 0.0 ? vec3(0.72, 0.88, 1.00) : vec3(1.00, 0.88, 0.50);
  vec3 c = mix(c0, e1, smoothstep(0.0, 0.6, t));
  return mix(c, e2, smoothstep(0.6, 1.0, t));
}
void main() {
  if (aVal > 1.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float v = clamp(aVal, -1.0, 1.0);
  float a = abs(v);
  float h = aKind > 0.5 ? 0.06 + 0.2 * a : 0.1 + 1.0 * a;
  vec3 p = vec3(aOff.x + aPos.x * uCell, aPos.y * h, aOff.y + aPos.z * uCell);
  gl_Position = uMVP * vec4(p, 1.0);
  float l = 0.55 + 0.45 * max(dot(aNorm, normalize(vec3(0.4, 1.0, 0.6))), 0.0);
  vCol = cmap(v) * l * (aKind > 0.5 ? 0.8 : 1.0);
}`;
const FS = `#version 300 es
precision mediump float;
in vec3 vCol; out vec4 o;
void main() { o = vec4(vCol, 1.0); }`;

function cubeGeometry() {
  const faces = [
    [[0, 0, 1], [[-1, 0, 1], [1, 0, 1], [1, 1, 1], [-1, 1, 1]]],
    [[0, 0, -1], [[1, 0, -1], [-1, 0, -1], [-1, 1, -1], [1, 1, -1]]],
    [[1, 0, 0], [[1, 0, 1], [1, 0, -1], [1, 1, -1], [1, 1, 1]]],
    [[-1, 0, 0], [[-1, 0, -1], [-1, 0, 1], [-1, 1, 1], [-1, 1, -1]]],
    [[0, 1, 0], [[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]]],
  ];
  const data = [];
  for (const [n, q] of faces) {
    for (const k of [0, 1, 2, 0, 2, 3]) data.push(q[k][0] * 0.5, q[k][1], q[k][2] * 0.5, ...n);
  }
  return new Float32Array(data);
}

function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f; m[10] = (far + near) / (near - far); m[11] = -1; m[14] = (2 * far * near) / (near - far);
  return m;
}

function lookAt(eye, c, up) {
  const z = norm(sub(eye, c)), x = norm(cross(up, z)), y = cross(z, x);
  const m = new Float32Array(16);
  m[0] = x[0]; m[4] = x[1]; m[8] = x[2]; m[12] = -dot(x, eye);
  m[1] = y[0]; m[5] = y[1]; m[9] = y[2]; m[13] = -dot(y, eye);
  m[2] = z[0]; m[6] = z[1]; m[10] = z[2]; m[14] = -dot(z, eye);
  m[15] = 1;
  return m;
}

function mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const ease = (t) => { t = Math.min(Math.max(t, 0), 1); return t * t * (3 - 2 * t); };

const STAGE = { input: 0, qkv: 0.12, attn: 0.25, agg: 0.42, gru: 0.55, head: 0.72, final: 0.9 };

export class Viz3D {
  constructor(canvas, overlay, weights) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.weights = weights;
    this.gl = canvas.getContext("webgl2", { antialias: true });
    this.ok = !!this.gl;
    this.showWeights = true;
    this.showLabels = true;
    this.canvasReady = false;
    this.tickMs = 300;
    this.tickStart = -1e9;
    this.scales = {};
    this.hover = null;
    this.buildLayout();
    if (this.ok) this.initGL();
    this.bindInput();
  }

  buildLayout() {
    const T = this.tensors = [];
    const inst = this.inst = { off: [], kind: [], tensor: [], r: [], c: [], delay: [] };
    const add = (name, label, x, z, rows, cols, o) => {
      const t = { name, label, x, z, rows, cols, start: inst.off.length / 2, key: o.key || name, stage: o.stage ?? 0,
        center: o.center ?? 0, get: o.get, kind: o.kind || 0, groups: o.groups || [], ldz: o.ldz || 0, rowLabels: o.rowLabels, segs: o.segs, id: T.length };
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          let gap = 0;
          for (const g of t.groups) if (c >= g) gap += 0.7;
          inst.off.push(x + c + gap, z + r);
          inst.kind.push(t.kind);
          inst.tensor.push(t.id);
          inst.r.push(r);
          inst.c.push(c);
          inst.delay.push(t.stage);
        }
      }
      t.end = inst.off.length / 2;
      T.push(t);
      return t;
    };
    const flatEdges = (field, cols) => (tr) => {
      const out = new Float64Array(MAXE * cols).fill(NaN);
      tr.edgesShown.forEach((ed, i) => { const v = field(ed); for (let c = 0; c < cols; c++) out[i * cols + c] = v[c]; });
      return out;
    };
    add("input", "input node state", 0, 0, 1, 46,
      { key: "input", stage: STAGE.input, groups: [2, 6, 10, 14], get: (t) => t.ns,
        segs: [[0, "vel"], [2, "wall"], [6, "pen"], [10, "pen+dt"], [14, "h(t-1)"]] });
    add("q", "q = Wq . input", 0, 5, 1, H, { key: "qkv", stage: STAGE.qkv, get: (t) => t.q });
    add("score", "score", -6, 9, MAXE, 1, { ldz: -1.4, key: "score", stage: STAGE.attn, get: flatEdges((x) => [x.score], 1),
      rowLabels: true });
    add("weight", "softmax", -3, 9, MAXE, 1, { ldz: -2.6, key: "weight", stage: STAGE.attn + 0.05, get: flatEdges((x) => [x.weight], 1) });
    add("K", "K per edge (rows: neighbours + self)", 0, 9, MAXE, H, { key: "qkv", stage: STAGE.qkv + 0.04, get: flatEdges((x) => x.k, H) });
    add("V", "V per edge", 38, 9, MAXE, H, { key: "qkv", stage: STAGE.attn + 0.05, get: flatEdges((x) => x.v, H) });
    add("attn", "attention output = sum(weight * V)", 0, 24, 1, H, { key: "agg", stage: STAGE.agg, get: (t) => t.attnOut });
    add("hprev", "h(t-1)", 38, 24, 1, H, { key: "input", stage: STAGE.input, get: (t) => t.ns.subarray(14) });
    add("r", "GRU reset gate r", 0, 28, 1, H, { key: "gate", stage: STAGE.gru, center: 0.5, get: (t) => t.r });
    add("zg", "GRU update gate z", 0, 31, 1, H, { key: "gate", stage: STAGE.gru + 0.03, center: 0.5, get: (t) => t.z });
    add("ng", "GRU candidate n", 0, 34, 1, H, { key: "gru", stage: STAGE.gru + 0.06, get: (t) => t.n });
    add("hnew", "h(t) = (1-z)*n + z*h(t-1)", 0, 38, 1, H, { key: "gru", stage: STAGE.gru + 0.1, get: (t) => t.hNew });
    add("delta", "delta head", 0, 43, 1, 4, { key: "head", stage: STAGE.head, groups: [2], get: (t) => t.deltaHead });
    add("wallh", "wall head hidden", 0, 47, 1, H, { key: "head", stage: STAGE.head, get: (t) => t.wallHidden });
    add("wallo", "wall head out", 24, 43, 1, 4, { key: "head", stage: STAGE.head + 0.04, groups: [2], get: (t) => t.wallOut });
    add("pcoef", "pair MLP coef", 0, 51, MAXE, 4,
      { key: "head", stage: STAGE.head + 0.02, get: flatEdges((x) => (x.pair ? x.pair.coef : [NaN, NaN, NaN, NaN]), 4) });
    add("pforce", "pair impulse", 12, 51, MAXE, 4, { key: "head", stage: STAGE.head + 0.06, groups: [2],
      get: flatEdges((x) => (x.pair ? [...x.pair.dp, ...x.pair.dv] : [NaN, NaN, NaN, NaN]), 4) });
    add("psum", "pair sum", 48, 43, 1, 4, { key: "head", stage: STAGE.head + 0.1, groups: [2], get: (t) => t.pairSum });
    add("final", "final = delta + wall + pair", 0, 66, 1, 4, { key: "final", stage: STAGE.final, groups: [2], get: (t) => t.delta });

    // static weight tiles
    const W = this.weights.w;
    const mats = [["query.weight", "Wq", 32, 46], ["key.weight", "Wk", 32, 48], ["value.weight", "Wv", 32, 48],
      ["gru.weight_ih", "GRU W_ih (r|z|n)", 96, 32], ["gru.weight_hh", "GRU W_hh (r|z|n)", 96, 32],
      ["delta_head.weight", "delta head", 4, 32], ["wall_head.0.weight", "wall head 1", 32, 14], ["wall_head.2.weight", "wall head 2", 4, 32],
      ["pair_head.0.weight", "pair MLP 1", 32, 5], ["pair_head.2.weight", "pair MLP 2", 32, 32], ["pair_head.4.weight", "pair MLP 3", 4, 32]];
    let cx = 0, cz = 75, rowH = 0;
    this.weightTiles = [];
    for (const [key, label, rows, cols] of mats) {
      if (cx + cols > 110) { cx = 0; cz += rowH + 3; rowH = 0; }
      const arr = W[key];
      let mx = 1e-9;
      for (let i = 0; i < arr.length; i++) mx = Math.max(mx, Math.abs(arr[i]));
      const t = add(key, label, cx, cz, rows, cols, { key: "w", kind: 1, get: () => arr, stage: 0 });
      t.scale = mx;
      this.weightTiles.push(t);
      cx += cols + 3;
      rowH = Math.max(rowH, rows);
    }
    this.nInst = inst.off.length / 2;
    this.cur = new Float32Array(this.nInst).fill(2);
    this.prev = new Float32Array(this.nInst).fill(2);
    this.next = new Float32Array(this.nInst).fill(2);
    this.raw = new Float32Array(this.nInst);
    for (const t of this.weightTiles) {
      const arr = t.get();
      for (let i = 0; i < arr.length; i++) { this.cur[t.start + i] = arr[i] / t.scale; this.prev[t.start + i] = this.cur[t.start + i]; this.next[t.start + i] = this.cur[t.start + i]; this.raw[t.start + i] = arr[i]; }
    }
  }

  initGL() {
    const gl = this.gl;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const prog = this.prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    const buf = (data, usage = gl.STATIC_DRAW) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, usage); return b; };
    const attr = (name, size, stride, offset, div) => {
      const l = gl.getAttribLocation(prog, name);
      gl.enableVertexAttribArray(l);
      gl.vertexAttribPointer(l, size, gl.FLOAT, false, stride, offset);
      if (div) gl.vertexAttribDivisor(l, 1);
    };
    const cube = cubeGeometry();
    const offBuf = buf(new Float32Array(this.inst.off));
    const kindBuf = buf(new Float32Array(this.inst.kind));
    this.valBuf = buf(this.cur, gl.DYNAMIC_DRAW);
    const ws = this.weightTiles[0].start;
    // activations draw full cubes; weight tiles draw only the top face, sharing the instance buffers at an offset
    const setup = (geom, first) => {
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      buf(geom);
      attr("aPos", 3, 24, 0, 0);
      attr("aNorm", 3, 24, 12, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, offBuf);
      attr("aOff", 2, 8, first * 8, 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, kindBuf);
      attr("aKind", 1, 4, first * 4, 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.valBuf);
      attr("aVal", 1, 4, first * 4, 1);
      return vao;
    };
    this.vaoAct = setup(cube, 0);
    this.actVerts = cube.length / 6;
    this.vaoW = setup(cube.slice(cube.length - 36), ws);
    this.actCount = ws;
    this.wCount = this.nInst - ws;
    this.uMVP = gl.getUniformLocation(prog, "uMVP");
    this.uCell = gl.getUniformLocation(prog, "uCell");
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
  }

  // Sets the target values from a trace; `tickMs` is the animation period.
  setTrace(trace, tickMs) {
    this.trace = trace;
    this.tickMs = tickMs;
    this.tickStart = performance.now();
    this.prev.set(this.cur);
    if (!trace) { this.hideActivations(); return; }
    const edges = trace.edges.slice().sort((a, b) => (b.self - a.self) || (b.weight - a.weight));
    trace.edgesShown = edges.slice(0, MAXE);
    trace.edgesHidden = edges.length - trace.edgesShown.length;
    const groups = {};
    for (const t of this.tensors) {
      if (t.kind) continue;
      const vals = t.get(trace);
      t.vals = vals;
      const g = groups[t.key] || (groups[t.key] = 0);
      let m = 0;
      for (let i = 0; i < vals.length; i++) if (!Number.isNaN(vals[i])) m = Math.max(m, Math.abs(vals[i] - t.center));
      groups[t.key] = Math.max(g, m);
    }
    for (const k in groups) {
      const old = this.scales[k] ?? 0;
      this.scales[k] = Math.max(groups[k], old * 0.97, 1e-3);
    }
    for (const t of this.tensors) {
      if (t.kind) continue;
      const s = this.scales[t.key];
      for (let i = 0; i < t.vals.length; i++) {
        const v = t.vals[i];
        this.next[t.start + i] = Number.isNaN(v) ? 2 : Math.min(Math.max((v - t.center) / s, -1), 1);
        this.raw[t.start + i] = v;
      }
    }
  }

  hideActivations() {
    for (const t of this.tensors) if (!t.kind) this.next.fill(2, t.start, t.end);
  }

  setTraceIdle() { this.setTrace(null, this.tickMs); }

  bindInput() {
    const c = this.canvas;
    let drag = null;
    const pts = new Map();
    c.addEventListener("pointerdown", (e) => { c.setPointerCapture(e.pointerId); pts.set(e.pointerId, e); drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey, d: this.pinchDist(pts) }; });
    c.addEventListener("pointerup", (e) => { pts.delete(e.pointerId); drag = null; });
    c.addEventListener("pointercancel", (e) => { pts.delete(e.pointerId); drag = null; });
    c.addEventListener("pointermove", (e) => {
      const rect = c.getBoundingClientRect();
      this.mouse = [e.clientX - rect.left, e.clientY - rect.top];
      if (!pts.has(e.pointerId)) return;
      pts.set(e.pointerId, e);
      if (pts.size === 2) {
        const d = this.pinchDist(pts);
        if (drag.d) this.dist = Math.min(Math.max(this.dist * drag.d / d, 8), 400);
        drag.d = d;
        return;
      }
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        const s = this.dist * 0.0016;
        const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
        this.target[0] -= (dx * cy) * s; this.target[2] -= (-dx * sy) * s;
        this.target[0] -= (dy * sy) * s; this.target[2] -= (dy * cy) * s;
      } else {
        this.yaw -= dx * 0.006;
        this.pitch = Math.min(Math.max(this.pitch + dy * 0.006, 0.15), 1.5);
      }
    });
    c.addEventListener("pointerleave", () => { this.mouse = null; });
    c.addEventListener("wheel", (e) => { e.preventDefault(); this.dist = Math.min(Math.max(this.dist * Math.exp(e.deltaY * 0.001), 8), 400); }, { passive: false });
    c.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  pinchDist(pts) {
    if (pts.size < 2) return 0;
    const [a, b] = [...pts.values()];
    return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
  }

  resetCamera() {
    this.yaw = 0; this.pitch = 1.0;
    const aspect = this.canvas.clientWidth / Math.max(this.canvas.clientHeight, 1);
    this.dist = Math.max(52 / (Math.tan(0.35) * aspect), 38 / Math.tan(0.35)) * 0.95;
    this.target = [33, 0, 34];
  }

  matrix() {
    const { yaw, pitch, dist, target } = this;
    const eye = [target[0] + dist * Math.cos(pitch) * Math.sin(yaw), target[1] + dist * Math.sin(pitch), target[2] + dist * Math.cos(pitch) * Math.cos(yaw)];
    const aspect = this.canvas.width / this.canvas.height;
    return mul(perspective(0.7, aspect, 1, 1000), lookAt(eye, target, [0, 1, 0]));
  }

  project(m, x, y, z) {
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (w <= 0) return null;
    const nx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w, ny = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
    return [(nx * 0.5 + 0.5) * this.canvas.clientWidth, (0.5 - ny * 0.5) * this.canvas.clientHeight];
  }

  frame(now) {
    const canvas = this.canvas;
    if (!this.canvasReady && canvas.clientWidth > 0) { this.resetCamera(); this.canvasReady = true; }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; this.overlay.width = w; this.overlay.height = h; }
    const D = Math.max(this.tickMs, 120);
    const el = now - this.tickStart;
    for (let i = 0; i < this.nInst; i++) {
      if (this.inst.kind[i]) continue;
      const a = this.prev[i], b = this.next[i];
      if (a > 1.5 || b > 1.5) { this.cur[i] = b; continue; }
      this.cur[i] = a + (b - a) * ease((el - this.inst.delay[i] * D * 0.8) / (D * 0.3));
    }
    const m = this.matrix();
    this.mvp = m;
    if (this.ok) {
      const gl = this.gl;
      gl.viewport(0, 0, w, h);
      gl.clearColor(0.05, 0.06, 0.08, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.uniformMatrix4fv(this.uMVP, false, m);
      gl.uniform1f(this.uCell, 0.88);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.valBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.cur.subarray(0, this.actCount));
      gl.bindVertexArray(this.vaoAct);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, this.actVerts, this.actCount);
      if (this.showWeights) {
        gl.bindVertexArray(this.vaoW);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.wCount);
      }
    }
    this.drawOverlay(m, dpr);
  }

  drawOverlay(m, dpr) {
    const ctx = this.overlay.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);
    if (!this.ok) {
      ctx.fillStyle = "#ccd";
      ctx.font = "14px sans-serif";
      ctx.fillText("WebGL2 unavailable: 3D view disabled", 12, 24);
      return;
    }
    if (this.showLabels) {
      ctx.font = "11px ui-monospace, Menlo, monospace";
      ctx.textBaseline = "bottom";
      ctx.textAlign = "left";
      for (const t of this.tensors) {
        if (t.kind && !this.showWeights) continue;
        const p = this.project(m, t.x, 0, t.z - 0.2 + t.ldz);
        if (!p) continue;
        ctx.fillStyle = t.kind ? "rgba(150,160,180,0.75)" : "rgba(225,230,240,0.95)";
        ctx.fillText(t.label, p[0], p[1] - 2);
        if (t.segs) {
          ctx.fillStyle = "rgba(170,180,200,0.85)";
          ctx.textBaseline = "top";
          for (const [c, txt] of t.segs) {
            const q = this.project(m, t.x + c + 0.7 * t.groups.filter((g) => c >= g).length, 0, t.z + 1.1);
            if (q) ctx.fillText(txt, q[0], q[1]);
          }
          ctx.textBaseline = "bottom";
        }
      }
      if (this.trace && this.trace.edgesShown) {
        ctx.fillStyle = "rgba(190,200,215,0.9)";
        const p0 = this.project(m, -15, 0, 9.8), p1 = this.project(m, -15, 0, 10.8);
        if (p0 && p1 && p1[1] - p0[1] >= 11) {
          this.trace.edgesShown.forEach((ed, i) => {
            const p = this.project(m, -15, 0, 9 + i + 0.8);
            if (p) ctx.fillText(ed.self ? "self" : "ball " + (ed.srcId ?? ed.src), p[0], p[1]);
          });
        }
        if (this.trace.edgesHidden > 0) {
          const p = this.project(m, 0, 0, 9 + MAXE + 0.6);
          if (p) ctx.fillText(`+${this.trace.edgesHidden} more neighbours not shown`, p[0], p[1]);
        }
      }
    }
    if (this.mouse && this.trace) {
      const hit = this.pick();
      if (hit) {
        const txt = `${hit.label} [${hit.r},${hit.c}] = ${hit.value.toFixed(4)}`;
        ctx.font = "12px ui-monospace, Menlo, monospace";
        const w = ctx.measureText(txt).width + 10;
        const x = Math.min(this.mouse[0] + 12, this.canvas.clientWidth - w - 4), y = Math.max(this.mouse[1] - 22, 4);
        ctx.fillStyle = "rgba(10,12,16,0.92)";
        ctx.fillRect(x, y, w, 20);
        ctx.fillStyle = "#e6ebf5";
        ctx.textBaseline = "middle";
        ctx.fillText(txt, x + 5, y + 10);
      }
    }
  }

  pick() {
    const m = this.mvp;
    let best = null, bd = 100;
    const limit = this.showWeights ? this.nInst : this.weightTiles[0].start;
    for (let i = 0; i < limit; i++) {
      const v = this.cur[i];
      if (v > 1.5) continue;
      const a = Math.abs(v);
      const h = this.inst.kind[i] ? 0.06 + 0.2 * a : 0.1 + 1.0 * a;
      const p = this.project(m, this.inst.off[2 * i], h, this.inst.off[2 * i + 1]);
      if (!p) continue;
      const d = (p[0] - this.mouse[0]) ** 2 + (p[1] - this.mouse[1]) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    if (best === null) return null;
    const t = this.tensors[this.inst.tensor[best]];
    return { label: t.name, r: this.inst.r[best], c: this.inst.c[best], value: this.raw[best] };
  }

  legendScales() { return this.scales; }
}
