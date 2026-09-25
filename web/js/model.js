// From-scratch port of TokenFreeDynamics(conservative_contact=True) +
// TokenModel.step_free (model/token_free.py, model/token_model.py): learned
// distance-only pair force and wall force, learned uniform acceleration,
// velocity-Verlet substeps.

const H = 32;
const W = 64;

export async function loadWeights(base = "weights") {
  const [manifest, buf] = await Promise.all([
    fetch(base + ".json").then((r) => r.json()),
    fetch(base + ".bin").then((r) => r.arrayBuffer()),
  ]);
  return parseWeights(manifest, buf);
}

export function parseWeights(manifest, buf) {
  const all = new Float32Array(buf);
  const w = {};
  for (const [name, t] of Object.entries(manifest.tensors)) {
    const size = t.shape.reduce((a, b) => a * b, 1);
    w[name] = all.subarray(t.offset, t.offset + size);
  }
  return { w, config: manifest.config, checkpoint: manifest.checkpoint };
}

class RadialMlp {
  constructor(w, prefix) {
    this.W0 = w[prefix + ".0.weight"]; this.b0 = w[prefix + ".0.bias"];
    this.W1 = w[prefix + ".2.weight"]; this.b1 = w[prefix + ".2.bias"];
    this.W2 = w[prefix + ".4.weight"]; this.b2 = w[prefix + ".4.bias"][0];
    this.h1 = new Float64Array(W);
    this.h2 = new Float64Array(W);
  }

  // scalar in, scalar out; hidden activations are left in this.h1 / this.h2
  eval(x) {
    const { W0, b0, W1, b1, W2, h1, h2 } = this;
    for (let a = 0; a < W; a++) h1[a] = Math.tanh(W0[a] * x + b0[a]);
    let out = this.b2;
    for (let a = 0; a < W; a++) {
      let s = b1[a];
      const o = a * W;
      for (let b = 0; b < W; b++) s += W1[o + b] * h1[b];
      h2[a] = Math.tanh(s);
      out += W2[a] * h2[a];
    }
    return out;
  }
}

export class TokenNet {
  constructor({ w, config }) {
    this.cfg = config;
    this.pairMlp = new RadialMlp(w, "pair_force");
    this.wallMlp = new RadialMlp(w, "wall_force");
    this.gx = w.gravity[0] * 10;
    this.gy = w.gravity[1] * 10;
    this.cap = 0;
    this.grow(64);
  }

  grow(n) {
    if (n <= this.cap) return;
    this.cap = n;
    this.px = new Float64Array(2 * n);
    this.vx = new Float64Array(2 * n);
    this.a = new Float64Array(2 * n);
    this.aWall = new Float64Array(2 * n);
    this.aPair = new Float64Array(2 * n);
    this.next = new Int32Array(n);
    this.cellKey = new Float64Array(n);
  }

  // pairs within `radius` (dist <= radius, as build_radius_graph), each
  // unordered pair once, calling visit(i, j, dist, dx, dy) with d = pos[i] - pos[j]
  forPairs(pos, count, radius, visit) {
    const next = this.next, keys = this.cellKey;
    const heads = new Map();
    for (let i = 0; i < count; i++) {
      const x = pos[2 * i], y = pos[2 * i + 1];
      if (!(Number.isFinite(x) && Number.isFinite(y))) { keys[i] = NaN; continue; }
      const cx = Math.min(Math.max(Math.floor(x / radius), -1000), 1000);
      const cy = Math.min(Math.max(Math.floor(y / radius), -1000), 1000);
      const k = (cx + 1024) * 4096 + cy + 1024;
      keys[i] = k;
      next[i] = heads.has(k) ? heads.get(k) : -1;
      heads.set(k, i);
    }
    for (let i = 0; i < count; i++) {
      const k = keys[i];
      if (k !== k) continue;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const h = heads.get(k + dx * 4096 + dy);
          if (h === undefined) continue;
          for (let j = h; j >= 0; j = next[j]) {
            if (j <= i) continue;
            const rx = pos[2 * i] - pos[2 * j], ry = pos[2 * i + 1] - pos[2 * j + 1];
            const d = Math.sqrt(rx * rx + ry * ry + 1e-12);
            if (d <= radius) visit(i, j, d, rx, ry);
          }
        }
      }
    }
  }

  // TokenFreeDynamics.contact_accel; rec = {wall, pair} optionally receives the components
  accel(p, count, a, rec = null) {
    const { n, radius, force_scale: fs } = this.cfg;
    const wall = rec ? rec.wall : null, pair = rec ? rec.pair : null;
    for (let i = 0; i < count; i++) {
      const x = p[2 * i], y = p[2 * i + 1];
      const d = [x, n - 1 - x, y, n - 1 - y];
      const f = [0, 0, 0, 0];
      for (let c = 0; c < 4; c++) {
        const pen = Math.max(radius - d[c], 0) / radius;
        if (pen > 0) f[c] = pen * this.wallMlp.eval(pen) * fs;
      }
      const wx = f[0] - f[1], wy = f[2] - f[3];
      a[2 * i] = wx + this.gx;
      a[2 * i + 1] = wy + this.gy;
      if (rec) { wall[2 * i] = wx; wall[2 * i + 1] = wy; pair[2 * i] = 0; pair[2 * i + 1] = 0; }
    }
    const cd = 2 * radius;
    this.forPairs(p, count, cd, (i, j, d, rx, ry) => {
      const pen = Math.max(cd - d, 0) / cd;
      const fp = pen * this.pairMlp.eval(pen) * fs / d;
      const fx = fp * rx, fy = fp * ry;
      a[2 * i] += fx; a[2 * i + 1] += fy;
      a[2 * j] -= fx; a[2 * j + 1] -= fy;
      if (rec) { pair[2 * i] += fx; pair[2 * i + 1] += fy; pair[2 * j] -= fx; pair[2 * j + 1] -= fy; }
    });
  }

  // One step; writes results into pos/vel in place (hidden passes through
  // unchanged, the conservative model has no recurrent state). `traceIdx` >= 0
  // returns the activation record for that ball.
  step(pos, vel, hidden, count, traceIdx = -1) {
    const { dt, contact_substeps: S } = this.cfg;
    this.grow(count);
    const { px, vx, a } = this;
    const h = dt / S;
    for (let i = 0; i < 2 * count; i++) { px[i] = pos[i]; vx[i] = vel[i]; }
    const tracing = traceIdx >= 0 && traceIdx < count;
    let trace = null, rec = null;
    if (tracing) {
      rec = { wall: this.aWall, pair: this.aPair };
      trace = this.startTrace(pos, vel, hidden, count, traceIdx);
    }
    this.accel(px, count, a, rec);
    if (tracing) this.recordSub(trace, 0, px, vx, a, traceIdx);
    for (let s = 0; s < S; s++) {
      for (let i = 0; i < 2 * count; i++) {
        vx[i] += 0.5 * h * a[i];
        px[i] += h * vx[i];
      }
      this.accel(px, count, a, rec);
      for (let i = 0; i < 2 * count; i++) vx[i] += 0.5 * h * a[i];
      if (tracing) this.recordSub(trace, s + 1, px, vx, a, traceIdx);
    }
    if (tracing) {
      const i = traceIdx;
      trace.newPos = [px[2 * i], px[2 * i + 1]];
      trace.newVel = [vx[2 * i], vx[2 * i + 1]];
      trace.dp = [px[2 * i] - (pos[2 * i] + vel[2 * i] * dt), px[2 * i + 1] - (pos[2 * i + 1] + vel[2 * i + 1] * dt)];
      trace.dv = [vx[2 * i] - vel[2 * i], vx[2 * i + 1] - vel[2 * i + 1]];
    }
    for (let i = 0; i < 2 * count; i++) { pos[i] = px[i]; vel[i] = vx[i]; }
    return trace;
  }

  recordSub(trace, s, p, v, a, i) {
    trace.subPos[2 * s] = p[2 * i]; trace.subPos[2 * s + 1] = p[2 * i + 1];
    trace.subVel[2 * s] = v[2 * i]; trace.subVel[2 * s + 1] = v[2 * i + 1];
    trace.subAcc[2 * s] = a[2 * i]; trace.subAcc[2 * s + 1] = a[2 * i + 1];
    trace.subAccWall[2 * s] = this.aWall[2 * i]; trace.subAccWall[2 * s + 1] = this.aWall[2 * i + 1];
    trace.subAccPair[2 * s] = this.aPair[2 * i]; trace.subAccPair[2 * s + 1] = this.aPair[2 * i + 1];
  }

  // Start-of-step record for ball i (see web/README.md, "Trace"). Pair and
  // wall activations are evaluated at the pre-step state, i.e. the first force
  // evaluation of the substep loop.
  startTrace(pos, vel, hidden, count, i) {
    const { n, radius, force_scale: fs, neighbor_radius: R, contact_substeps: S } = this.cfg;
    const x = pos[2 * i], y = pos[2 * i + 1];
    const dist = [x, n - 1 - x, y, n - 1 - y];
    const wall = dist.map((d) => {
      const pen = Math.max(radius - d, 0) / radius;
      const out = this.wallMlp.eval(pen);
      return { pen, out, force: pen * out * fs, h1: Float64Array.from(this.wallMlp.h1), h2: Float64Array.from(this.wallMlp.h2) };
    });
    const cd = 2 * radius;
    const edges = [];
    for (let j = 0; j < count; j++) {
      if (j === i) continue;
      const rx = x - pos[2 * j], ry = y - pos[2 * j + 1];
      const d = Math.sqrt(rx * rx + ry * ry + 1e-12);
      if (d > R) continue;
      const pen = Math.max(cd - d, 0) / cd;
      const e = { src: j, dist: d, pen, unit: [rx / d, ry / d], contact: d <= cd, out: 0, mag: 0, force: [0, 0], h1: null, h2: null };
      if (e.contact) {
        e.out = this.pairMlp.eval(pen);
        e.mag = pen * e.out * fs;
        e.force = [e.mag * rx / d, e.mag * ry / d];
        e.h1 = Float64Array.from(this.pairMlp.h1);
        e.h2 = Float64Array.from(this.pairMlp.h2);
      }
      edges.push(e);
    }
    const sub = S + 1;
    return {
      index: i, pos: [x, y], vel: [vel[2 * i], vel[2 * i + 1]],
      hidden: Float32Array.from(hidden.subarray(i * H, (i + 1) * H)),
      wallDist: dist, wall, wallForce: [wall[0].force - wall[1].force, wall[2].force - wall[3].force],
      gravity: [this.gx, this.gy], edges,
      subPos: new Float64Array(2 * sub), subVel: new Float64Array(2 * sub), subAcc: new Float64Array(2 * sub),
      subAccWall: new Float64Array(2 * sub), subAccPair: new Float64Array(2 * sub),
      newPos: null, newVel: null, dp: null, dv: null,
    };
  }

  // flat [i, j, i, j, ...] of pairs within `radius`, for drawing
  edgeList(pos, count, radius) {
    const out = [];
    this.forPairs(pos, count, radius, (i, j) => { out.push(i, j); });
    return out;
  }
}

// Guard from scripts/realtime_sim.py: clamp to the box, zero outward
// velocity, cap speed, drop non-finite tokens. Returns the new count.
export function containState(pos, vel, hidden, count, n, maxSpeed, ids = null) {
  let m = 0;
  const top = n - 1;
  for (let i = 0; i < count; i++) {
    const p0 = pos[2 * i], p1 = pos[2 * i + 1], v0 = vel[2 * i], v1 = vel[2 * i + 1];
    if (!(Number.isFinite(p0) && Number.isFinite(p1) && Number.isFinite(v0) && Number.isFinite(v1))) continue;
    if (m !== i) {
      pos[2 * m] = p0; pos[2 * m + 1] = p1; vel[2 * m] = v0; vel[2 * m + 1] = v1;
      for (let a = 0; a < H; a++) hidden[m * H + a] = hidden[i * H + a];
      if (ids) ids[m] = ids[i];
    }
    m++;
  }
  for (let i = 0; i < m; i++) {
    for (let c = 0; c < 2; c++) {
      const p = pos[2 * i + c], v = vel[2 * i + c];
      if ((p < 0 && v < 0) || (p > top && v > 0)) vel[2 * i + c] = 0;
      pos[2 * i + c] = Math.min(Math.max(p, 0), top);
    }
    const sp = Math.max(Math.hypot(vel[2 * i], vel[2 * i + 1]), 1e-6);
    const k = Math.min(sp, maxSpeed) / sp;
    vel[2 * i] *= k;
    vel[2 * i + 1] *= k;
  }
  return m;
}
