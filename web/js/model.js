// From-scratch port of TokenFreeDynamics._core + TokenModel.step_free
// (model/token_free.py, model/token_model.py): cell graph, global-max softmax.

const H = 32;
const NODE = 46;
const WALL = 12;

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

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class TokenNet {
  constructor({ w, config }) {
    this.w = w;
    this.cfg = config;
    this.cap = 0;
    this.grow(64);
    this.edgeCap = 0;
    this.growEdges(512);
    this.cellSide = config.neighbor_radius;
  }

  grow(n) {
    if (n <= this.cap) return;
    this.cap = n;
    this.ns = new Float64Array(n * NODE);
    this.q = new Float64Array(n * H);
    this.kNode = new Float64Array(n * H);
    this.vNode = new Float64Array(n * H);
    this.attn = new Float64Array(n * H);
    this.delta = new Float64Array(n * 4);
    this.dstStart = new Int32Array(n + 1);
    this.walls = new Float64Array(n * WALL);
    this.cellKey = new Int32Array(n);
    this.order = new Int32Array(n);
  }

  growEdges(e) {
    if (e <= this.edgeCap) return;
    this.edgeCap = e;
    this.src = new Int32Array(e);
    this.dst = new Int32Array(e);
    this.score = new Float64Array(e);
    this.weight = new Float64Array(e);
  }

  wallFeatures(pos, vel, i, out) {
    const { n, wall_range: wr, radius, dt } = this.cfg;
    const x = pos[2 * i], y = pos[2 * i + 1];
    const d = [x, n - 1 - x, y, n - 1 - y];
    for (let a = 0; a < 4; a++) out[a] = (wr - Math.min(Math.max(d[a], 0), wr)) / wr;
    const x2 = x + vel[2 * i] * dt, y2 = y + vel[2 * i + 1] * dt;
    const d2 = [x2, n - 1 - x2, y2, n - 1 - y2];
    for (let a = 0; a < 4; a++) {
      out[4 + a] = Math.max(radius - d[a], 0) / radius;
      out[8 + a] = Math.max(radius - d2[a], 0) / radius;
    }
  }

  buildGraph(pos, count) {
    const R = this.cfg.neighbor_radius;
    const side = this.cellSide;
    let minX = Infinity, minY = Infinity, maxY = -Infinity;
    const cx = new Int32Array(count), cy = new Int32Array(count);
    for (let i = 0; i < count; i++) {
      cx[i] = Math.floor(pos[2 * i] / side);
      cy[i] = Math.floor(pos[2 * i + 1] / side);
      if (cx[i] < minX) minX = cx[i];
      if (cy[i] < minY) minY = cy[i];
      if (cy[i] > maxY) maxY = cy[i];
    }
    const stride = maxY - minY + 3;
    const keys = this.cellKey;
    for (let i = 0; i < count; i++) keys[i] = (cx[i] - minX + 1) * stride + (cy[i] - minY + 1);
    const order = Array.from({ length: count }, (_, i) => i).sort((a, b) => keys[a] - keys[b]);
    const buckets = new Map();
    for (const i of order) {
      const b = buckets.get(keys[i]);
      if (b) b.push(i); else buckets.set(keys[i], [i]);
    }
    let e = 0;
    for (let i = 0; i < count; i++) {
      this.dstStart[i] = e;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const b = buckets.get(keys[i] + dx * stride + dy);
          if (!b) continue;
          for (const j of b) {
            if (j === i) continue;
            const ddx = pos[2 * j] - pos[2 * i], ddy = pos[2 * j + 1] - pos[2 * i + 1];
            if (Math.sqrt(ddx * ddx + ddy * ddy + 1e-12) > R) continue;
            if (e + 2 > this.edgeCap) this.growEdges(this.edgeCap * 2);
            this.src[e] = j;
            this.dst[e] = i;
            e++;
          }
        }
      }
      if (e + 1 > this.edgeCap) this.growEdges(this.edgeCap * 2);
      this.src[e] = i;
      this.dst[e] = i;
      e++;
    }
    this.dstStart[count] = e;
    return e;
  }

  // One step; writes results into pos/vel/hidden in place. `traceIdx` >= 0
  // returns the full activation record for that ball.
  step(pos, vel, hidden, count, traceIdx = -1) {
    const { w, cfg } = this;
    const { dt, radius } = cfg;
    this.grow(count);
    const ns = this.ns, walls = this.walls;
    for (let i = 0; i < count; i++) {
      const o = i * NODE;
      ns[o] = vel[2 * i];
      ns[o + 1] = vel[2 * i + 1];
      const wo = i * WALL;
      this.wallFeatures(pos, vel, i, walls.subarray(wo, wo + WALL));
      for (let a = 0; a < WALL; a++) ns[o + 2 + a] = walls[wo + a];
      for (let a = 0; a < H; a++) ns[o + 14 + a] = hidden[i * H + a];
    }
    const Wq = w["query.weight"], bq = w["query.bias"];
    const Wk = w["key.weight"], bk = w["key.bias"];
    const Wv = w["value.weight"], bv = w["value.bias"];
    const { q, kNode, vNode } = this;
    for (let i = 0; i < count; i++) {
      const o = i * NODE;
      for (let a = 0; a < H; a++) {
        let sq = bq[a], sk = bk[a], sv = bv[a];
        for (let b = 0; b < NODE; b++) {
          const x = ns[o + b];
          sq += Wq[a * NODE + b] * x;
          sk += Wk[a * (NODE + 2) + b] * x;
          sv += Wv[a * (NODE + 2) + b] * x;
        }
        q[i * H + a] = sq;
        kNode[i * H + a] = sk;
        vNode[i * H + a] = sv;
      }
    }

    const E = this.buildGraph(pos, count);
    const { src, dst, score, weight, attn } = this;
    const inv = 1 / Math.sqrt(H);
    let gmax = -Infinity;
    for (let e = 0; e < E; e++) {
      const s = src[e], d = dst[e];
      const rx = pos[2 * s] - pos[2 * d], ry = pos[2 * s + 1] - pos[2 * d + 1];
      let sc = 0;
      for (let a = 0; a < H; a++) {
        const k = kNode[s * H + a] + Wk[a * (NODE + 2) + NODE] * rx + Wk[a * (NODE + 2) + NODE + 1] * ry;
        sc += q[d * H + a] * k;
      }
      score[e] = sc * inv;
      if (score[e] > gmax) gmax = score[e];
    }
    attn.fill(0, 0, count * H);
    for (let i = 0; i < count; i++) {
      let denom = 0;
      const e0 = this.dstStart[i], e1 = this.dstStart[i + 1];
      for (let e = e0; e < e1; e++) {
        weight[e] = Math.exp(score[e] - gmax);
        denom += weight[e];
      }
      denom = Math.max(denom, 1e-6);
      for (let e = e0; e < e1; e++) {
        weight[e] /= denom;
        const s = src[e];
        const rx = pos[2 * s] - pos[2 * i], ry = pos[2 * s + 1] - pos[2 * i + 1];
        for (let a = 0; a < H; a++) {
          const v = vNode[s * H + a] + Wv[a * (NODE + 2) + NODE] * rx + Wv[a * (NODE + 2) + NODE + 1] * ry;
          attn[i * H + a] += weight[e] * v;
        }
      }
    }

    // GRU + heads
    const Wih = w["gru.weight_ih"], Whh = w["gru.weight_hh"], bih = w["gru.bias_ih"], bhh = w["gru.bias_hh"];
    const Wd = w["delta_head.weight"], bd = w["delta_head.bias"];
    const W1 = w["wall_head.0.weight"], b1 = w["wall_head.0.bias"], W2 = w["wall_head.2.weight"], b2 = w["wall_head.2.bias"];
    const delta = this.delta;
    const newHidden = new Float32Array(count * H);
    const rr = new Float64Array(H), zz = new Float64Array(H), nn = new Float64Array(H), hn = new Float64Array(H);
    const wh = new Float64Array(H);
    let trace = null;
    for (let i = 0; i < count; i++) {
      for (let a = 0; a < H; a++) {
        let ir = bih[a], iz = bih[H + a], inn = bih[2 * H + a];
        let hr = bhh[a], hz = bhh[H + a], hnn = bhh[2 * H + a];
        for (let b = 0; b < H; b++) {
          const x = attn[i * H + b], h = hidden[i * H + b];
          ir += Wih[a * H + b] * x;
          iz += Wih[(H + a) * H + b] * x;
          inn += Wih[(2 * H + a) * H + b] * x;
          hr += Whh[a * H + b] * h;
          hz += Whh[(H + a) * H + b] * h;
          hnn += Whh[(2 * H + a) * H + b] * h;
        }
        const r = sigmoid(ir + hr), z = sigmoid(iz + hz);
        const n = Math.tanh(inn + r * hnn);
        rr[a] = r; zz[a] = z; nn[a] = n;
        hn[a] = (1 - z) * n + z * hidden[i * H + a];
        newHidden[i * H + a] = hn[a];
      }
      for (let c = 0; c < 4; c++) {
        let s = bd[c];
        for (let a = 0; a < H; a++) s += Wd[c * H + a] * hn[a];
        delta[i * 4 + c] = s;
      }
      // wall head on [velocity, wall features]
      for (let a = 0; a < H; a++) {
        let s = b1[a];
        s += W1[a * 14] * vel[2 * i] + W1[a * 14 + 1] * vel[2 * i + 1];
        for (let b = 0; b < WALL; b++) s += W1[a * 14 + 2 + b] * walls[i * WALL + b];
        wh[a] = s > 0 ? s : 0;
      }
      const wallOut = [0, 0, 0, 0];
      for (let c = 0; c < 4; c++) {
        let s = b2[c];
        for (let a = 0; a < H; a++) s += W2[c * H + a] * wh[a];
        wallOut[c] = s;
        delta[i * 4 + c] += s;
      }
      if (i === traceIdx) {
        trace = {
          i, ns: Float64Array.from(ns.subarray(i * NODE, (i + 1) * NODE)),
          q: Float64Array.from(q.subarray(i * H, (i + 1) * H)),
          attnOut: Float64Array.from(attn.subarray(i * H, (i + 1) * H)),
          r: Float64Array.from(rr), z: Float64Array.from(zz), n: Float64Array.from(nn),
          hNew: Float64Array.from(hn),
          deltaHead: Float64Array.from(delta.subarray(i * 4, i * 4 + 4)).map((v, c) => v - wallOut[c]),
          wallHidden: Float64Array.from(wh), wallOut: Float64Array.from(wallOut),
          edges: [], pairSum: [0, 0, 0, 0], gmax,
        };
        const e0 = this.dstStart[i], e1 = this.dstStart[i + 1];
        for (let e = e0; e < e1; e++) {
          const s = src[e];
          const rx = pos[2 * s] - pos[2 * i], ry = pos[2 * s + 1] - pos[2 * i + 1];
          const k = new Float64Array(H), v = new Float64Array(H);
          for (let a = 0; a < H; a++) {
            k[a] = kNode[s * H + a] + Wk[a * (NODE + 2) + NODE] * rx + Wk[a * (NODE + 2) + NODE + 1] * ry;
            v[a] = vNode[s * H + a] + Wv[a * (NODE + 2) + NODE] * rx + Wv[a * (NODE + 2) + NODE + 1] * ry;
          }
          trace.edges.push({ src: s, self: s === i, score: score[e], weight: weight[e], k, v, rel: [rx, ry] });
        }
      }
    }

    // pair head: antisymmetric impulse summed over neighbours (pair edges exclude self loops)
    const P0 = w["pair_head.0.weight"], pb0 = w["pair_head.0.bias"];
    const P2 = w["pair_head.2.weight"], pb2 = w["pair_head.2.bias"];
    const P4 = w["pair_head.4.weight"], pb4 = w["pair_head.4.bias"];
    const cd = 2 * radius;
    const f = new Float64Array(5), a1 = new Float64Array(H), a2 = new Float64Array(H);
    for (let i = 0; i < count; i++) {
      for (let e = this.dstStart[i]; e < this.dstStart[i + 1]; e++) {
        const s = src[e];
        if (s === i) continue;
        const rpx = pos[2 * i] - pos[2 * s], rpy = pos[2 * i + 1] - pos[2 * s + 1];
        const rvx = vel[2 * i] - vel[2 * s], rvy = vel[2 * i + 1] - vel[2 * s + 1];
        const d = Math.sqrt(rpx * rpx + rpy * rpy + 1e-12);
        const ux = rpx / d, uy = rpy / d;
        const ns_ = rvx * ux + rvy * uy;
        const tx = rvx - ns_ * ux, ty = rvy - ns_ * uy;
        const nx = rpx + rvx * dt, ny = rpy + rvy * dt;
        const dNext = Math.sqrt(nx * nx + ny * ny + 1e-12);
        f[0] = d;
        f[1] = ns_;
        f[2] = Math.sqrt(tx * tx + ty * ty + 1e-12);
        f[3] = Math.max(cd - d, 0) / cd;
        f[4] = Math.max(cd - dNext, 0) / cd;
        for (let a = 0; a < H; a++) {
          let s1 = pb0[a];
          for (let b = 0; b < 5; b++) s1 += P0[a * 5 + b] * f[b];
          a1[a] = s1 > 0 ? s1 : 0;
        }
        for (let a = 0; a < H; a++) {
          let s2 = pb2[a];
          for (let b = 0; b < H; b++) s2 += P2[a * H + b] * a1[b];
          a2[a] = s2 > 0 ? s2 : 0;
        }
        const c0 = [0, 0, 0, 0];
        for (let c = 0; c < 4; c++) {
          let s3 = pb4[c];
          for (let b = 0; b < H; b++) s3 += P4[c * H + b] * a2[b];
          c0[c] = s3;
        }
        const dpx = c0[0] * ux + c0[1] * tx, dpy = c0[0] * uy + c0[1] * ty;
        const dvx = c0[2] * ux + c0[3] * tx, dvy = c0[2] * uy + c0[3] * ty;
        delta[i * 4] += dpx;
        delta[i * 4 + 1] += dpy;
        delta[i * 4 + 2] += dvx;
        delta[i * 4 + 3] += dvy;
        if (i === traceIdx) {
          const edge = trace.edges.find((x) => x.src === s);
          edge.pair = { feats: Float64Array.from(f), coef: c0, dp: [dpx, dpy], dv: [dvx, dvy] };
          trace.pairSum[0] += dpx; trace.pairSum[1] += dpy; trace.pairSum[2] += dvx; trace.pairSum[3] += dvy;
        }
      }
    }

    for (let i = 0; i < count; i++) {
      const vx = vel[2 * i], vy = vel[2 * i + 1];
      pos[2 * i] += vx * dt + delta[i * 4];
      pos[2 * i + 1] += vy * dt + delta[i * 4 + 1];
      vel[2 * i] = vx + delta[i * 4 + 2];
      vel[2 * i + 1] = vy + delta[i * 4 + 3];
      for (let a = 0; a < H; a++) hidden[i * H + a] = newHidden[i * H + a];
    }
    if (trace) {
      trace.dp = [delta[traceIdx * 4], delta[traceIdx * 4 + 1]];
      trace.dv = [delta[traceIdx * 4 + 2], delta[traceIdx * 4 + 3]];
      trace.delta = Float64Array.from(delta.subarray(traceIdx * 4, traceIdx * 4 + 4));
    }
    return trace;
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
