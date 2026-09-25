import * as THREE from "three";

export const MAXE = 6;
const CAP = 1800, MAXW = 400;
const WALLS = ["x = 0", "x = 99", "y = 0", "y = 99"];

export function divColor(t, c, k = 1) {
  t = Math.max(-1, Math.min(1, t || 0));
  const a = Math.pow(Math.abs(t), 0.6);
  const e = t >= 0 ? [1.0, 0.56, 0.16] : [0.26, 0.58, 1.0];
  const base = [0.085, 0.1, 0.14];
  const w = Math.max(0, a - 0.8) * 2;
  c.setRGB(...[0, 1, 2].map((q) => (base[q] + (e[q] - base[q]) * a + w * (1 - e[q])) * k));
  return c;
}

// signed log scale onto [-1, 1]: s is the value that maps to ~0.15, 100 s to 1
export const sl = (x, s = 1) => Math.sign(x) * Math.min(1, Math.log1p(Math.abs(x) / s) / Math.log1p(100));

export const fmt = (x) => {
  const a = Math.abs(x);
  return a !== 0 && (a >= 1e4 || a < 1e-3) ? x.toExponential(3) : x.toFixed(4);
};

export class Arch {
  constructor(scene) {
    this.group = new THREE.Group();
    scene.add(this.group);
    const geo = new THREE.BoxGeometry(1, 1, 1);
    geo.translate(0, 0.5, 0);
    this.mesh = new THREE.InstancedMesh(geo, new THREE.MeshLambertMaterial(), CAP);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.setColorAt(0, new THREE.Color());
    this.mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
    this.mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(50, 0, 50), 400);
    this.group.add(this.mesh);
    const wg = new THREE.BufferGeometry();
    this.wirePos = new Float32Array(6 * MAXW);
    wg.setAttribute("position", new THREE.BufferAttribute(this.wirePos, 3).setUsage(THREE.DynamicDrawUsage));
    this.wires = new THREE.LineSegments(wg, new THREE.LineBasicMaterial({ color: 0x2b5f86, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.wires.frustumCulled = false;
    this.group.add(this.wires);
    const plate = (u0, w0, u1, w1) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(u1 - u0, w1 - w0), new THREE.MeshBasicMaterial({ color: 0x0c1119 }));
      m.rotation.x = -Math.PI / 2;
      m.position.set((u0 + u1) / 2, -0.02, (w0 + w1) / 2);
      const ed = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: 0x1b2636 }));
      m.add(ed);
      this.group.add(m);
    };
    plate(-1, -3.5, 9, 3.5);
    plate(-1, 3.5, 80, 34);
    plate(-1, 35, 72, 62);
    plate(-1, 63, 38, 67);
    plate(-1, 68.5, 96, 81);
    plate(-1, 82.5, 22, 87);
    this.cells = [];
    this.labels = [];
    this.anchors = { input: [4, 1], output: [4, 85], cols: [] };
    this.hover = -1;
    this.m = new THREE.Matrix4();
    this.c = new THREE.Color();
  }

  set(tr, ids) {
    const C = this.cells, L = this.labels, A = this.anchors;
    C.length = 0; L.length = 0; A.cols.length = 0;
    let nw = 0;
    const wire = (u0, w0, u1, w1) => {
      if (nw >= MAXW) return;
      this.wirePos.set([u0, 0.05, w0, u1, 0.05, w1], 6 * nw++);
    };
    const cell = (u, w, s, v, name, val, dim = false) => C.push({ u, w, s, v, name, val, dim, h: s * (0.3 + 1.4 * Math.abs(v)) });
    const lab = (text, u, w, cls = "", left = false) => L.push({ text, u, w, cls, left });
    const vec = (u, w, s, gap, a, sc, name, unit = 1) => {
      cell(u, w, s, sc(a[0] / unit), name + " x", fmt(a[0]));
      cell(u + gap, w, s, sc(a[1] / unit), name + " y", fmt(a[1]));
    };
    if (!tr) { this.mesh.count = 0; this.wires.geometry.setDrawRange(0, 0); return; }
    const id = ids(tr.index);
    lab(`input token #${id}`, -1, -2.4, "sec", true);
    vec(0, 1, 1.6, 2, tr.pos, (x) => (x - 49.5) / 49.5, "position");
    vec(4, 1, 1.6, 2, tr.vel, (x) => x / 15, "velocity");

    const edges = tr.edges.slice().sort((a, b) => (b.contact - a.contact) || (Math.abs(b.mag) - Math.abs(a.mag)) || (a.dist - b.dist));
    const nc = tr.edges.filter((e) => e.contact).length;
    lab(`pair force MLP 1→64→64→1 · ${tr.edges.length} neighbours ≤ 4.0, ${nc ? nc + " in contact (d ≤ 1.5)" : "none in contact → MLP idle"}`, 20, -2.4, "sec", true);
    edges.slice(0, MAXE).forEach((e, k) => {
      const u0 = k * 13 + 1;
      const tag = `edge #${e.srcId} → #${id}`;
      const off = !e.contact;
      lab(`#${e.srcId} d ${e.dist.toFixed(2)}`, u0 + 4, 5.2, off ? "col off" : "col");
      A.cols.push({ srcId: e.srcId, u: u0 + 3.5, w: 8 });
      cell(u0 + 1.5, 8.5, 1.2, e.unit[0] * e.dist / 4, `${tag} · offset x`, fmt(e.unit[0] * e.dist));
      cell(u0 + 3.5, 8.5, 1.2, e.unit[1] * e.dist / 4, `${tag} · offset y`, fmt(e.unit[1] * e.dist));
      cell(u0 + 5.5, 8.5, 1.2, 1 - e.dist / 4, `${tag} · distance d`, fmt(e.dist));
      cell(u0 + 7.5, 8.5, 1.2, e.pen, `${tag} · penetration (2r - d)/2r`, fmt(e.pen), off);
      for (let r = 0; r < 8; r++) {
        for (let q = 0; q < 8; q++) {
          const a = r * 8 + q;
          cell(u0 + q, 11 + r, 0.8, off ? 0 : e.h1[a], `${tag} · h1[${a}] = tanh(W0 pen + b0)`, off ? "inactive (no contact)" : fmt(e.h1[a]), off);
          cell(u0 + q, 20 + r, 0.8, off ? 0 : e.h2[a], `${tag} · h2[${a}] = tanh(W1 h1 + b1)`, off ? "inactive (no contact)" : fmt(e.h2[a]), off);
        }
      }
      cell(u0 + 1.5, 29.5, 1.2, sl(e.out, 0.05), `${tag} · MLP out`, fmt(e.out), off);
      cell(u0 + 3.5, 29.5, 1.2, sl(e.mag), `${tag} · |F| = pen · out · 100`, fmt(e.mag), off);
      cell(u0 + 5.5, 29.5, 1.2, sl(e.force[0]), `${tag} · force x`, fmt(e.force[0]), off);
      cell(u0 + 7.5, 29.5, 1.2, sl(e.force[1]), `${tag} · force y`, fmt(e.force[1]), off);
      wire(4, 2.5, u0 + 3.5, 8);
      wire(u0 + 3.5, 9.5, u0 + 3.5, 11);
      wire(u0 + 3.5, 18.8, u0 + 3.5, 20);
      wire(u0 + 3.5, 27.8, u0 + 3.5, 29.5);
      if (!off) wire(u0 + 3.5, 31, 11, 64.2);
    });
    if (tr.edges.length > MAXE) lab(`+${tr.edges.length - MAXE} more`, 76, 32.6, "col off");
    if (!tr.edges.length) lab("no neighbours within 4.0 · pair force MLP idle", 40, 19, "sec");

    lab("wall force MLP 1→64→64→1 · per wall, active when d < r", -1, 34.4, "sec", true);
    tr.wall.forEach((wl, k) => {
      const u0 = k * 13 + 1;
      const tag = `wall ${WALLS[k]}`;
      const off = wl.pen <= 0;
      const d = tr.wallDist[k];
      lab(`${WALLS[k]}  d ${d.toFixed(1)}`, u0 + 4, 37.3, off ? "col off" : "col");
      cell(u0 + 2.5, 39.5, 1.2, 1 - Math.min(d, 10) / 10, `${tag} · distance`, fmt(d));
      cell(u0 + 4.5, 39.5, 1.2, wl.pen, `${tag} · penetration (r - d)/r`, fmt(wl.pen), off);
      for (let r = 0; r < 8; r++) {
        for (let q = 0; q < 8; q++) {
          const a = r * 8 + q;
          cell(u0 + q, 42 + r, 0.8, wl.h1[a], `${tag} · h1[${a}]`, fmt(wl.h1[a]), off);
          cell(u0 + q, 51 + r, 0.8, wl.h2[a], `${tag} · h2[${a}]`, fmt(wl.h2[a]), off);
        }
      }
      cell(u0 + 2.5, 60.3, 1.2, sl(wl.out, 0.05), `${tag} · MLP out`, fmt(wl.out), off);
      cell(u0 + 4.5, 60.3, 1.2, sl(wl.force), `${tag} · force = pen · out · 100`, fmt(wl.force), off);
      wire(u0 + 3.5, 40.5, u0 + 3.5, 42);
      wire(u0 + 3.5, 49.8, u0 + 3.5, 51);
      wire(u0 + 3.5, 58.8, u0 + 3.5, 60.3);
      if (!off) wire(u0 + 3.5, 61.5, 3, 64.2);
    });
    lab("learned gravity", 60, 37.3, "col");
    vec(57, 42, 2.2, 3, tr.gravity, (x) => sl(x), "learned gravity");
    wire(59.5, 44.5, 19, 64.2);

    lab("acceleration = wall + pair + gravity", -1, 62.3, "sec", true);
    vec(2, 64.5, 1.3, 2, tr.subAccWall, sl, "wall accel");
    vec(10, 64.5, 1.3, 2, tr.subAccPair, sl, "pair accel");
    vec(18, 64.5, 1.3, 2, tr.gravity, sl, "gravity");
    vec(29, 64.5, 1.3, 2, tr.subAcc, sl, "total accel");
    lab("+", 7, 65.4, "op"); lab("+", 15, 65.4, "op"); lab("=", 24.5, 65.4, "op");

    lab("velocity Verlet · 8 substeps of dt/8", 42, 65.5, "sec", true);
    const S = tr.subPos.length / 2;
    const rows = [["wall accel", tr.subAccWall, sl], ["pair accel", tr.subAccPair, sl], ["accel", tr.subAcc, sl], ["Δpos", null, (x) => sl(x, 0.01)], ["velocity", tr.subVel, (x) => x / 15]];
    for (let s = 0; s < S; s++) {
      const u0 = s * 10.5 + 1;
      lab(s ? `sub ${s}` : "start", u0 + 1.5, 69.1, "col");
      rows.forEach(([nm, arr, sc], r) => {
        const v = arr ? [arr[2 * s], arr[2 * s + 1]] : [tr.subPos[2 * s] - tr.pos[0], tr.subPos[2 * s + 1] - tr.pos[1]];
        vec(u0 + 0.5, 71 + r * 1.9, 1.2, 2, v, sc, `substep ${s} · ${nm}`);
      });
      if (s) wire(u0 - 6.2, 75, u0, 75);
      else wire(31, 66, u0 + 1.5, 71);
    }
    wire((S - 1) * 10.5 + 3, 80, 8, 84);

    lab(`output token #${id}`, -1, 82.3, "sec", true);
    vec(0, 84.5, 1.6, 2, tr.newPos, (x) => (x - 49.5) / 49.5, "new position");
    vec(4, 84.5, 1.6, 2, tr.newVel, (x) => x / 15, "new velocity");
    vec(11, 84.5, 1.2, 2, tr.dp, (x) => sl(x, 0.01), "dp (vs pos + vel dt)");
    vec(16, 84.5, 1.2, 2, tr.dv, (x) => x / 5, "dv");
    this.wires.geometry.attributes.position.needsUpdate = true;
    this.wires.geometry.setDrawRange(0, 2 * nw);
    this.mesh.count = C.length;
  }

  frame(front) {
    const { m, c, cells: C, mesh } = this;
    for (let i = 0; i < C.length; i++) {
      const e = C[i];
      const b = front < 0 ? 0 : Math.exp(-(((e.w - front) / 3) ** 2));
      const k = 1 + 1.2 * b + (i === this.hover ? 1.5 : 0);
      m.makeScale(e.s, e.h * (1 + 0.8 * b) + 0.02, e.s);
      m.setPosition(e.u, 0, e.w);
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, e.dim ? c.setRGB(0.1 * k, 0.13 * k, 0.19 * k) : divColor(e.v, c, k));
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
  }
}
