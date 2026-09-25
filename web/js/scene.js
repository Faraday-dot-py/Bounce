import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Arch, divColor, sl } from "./arch.js";

THREE.ColorManagement.enabled = false;

const MAXB = 300, MAXL = 3000, BG = 0x06080c;

const stops = [[0, [0.36, 0.55, 0.94]], [5, [0.5, 0.82, 0.73]], [12, [0.95, 0.89, 0.58]], [25, [1, 0.62, 0.36]]];
export function speedColor(v, out) {
  let i = 0;
  while (i < stops.length - 2 && v > stops[i + 1][0]) i++;
  const [a, ca] = stops[i], [b, cb] = stops[i + 1];
  const t = Math.min(Math.max((v - a) / (b - a), 0), 1);
  for (let k = 0; k < 3; k++) out[k] = ca[k] + (cb[k] - ca[k]) * t;
  return out;
}

function tex(ring) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  if (ring) {
    r.addColorStop(0.55, "rgba(255,255,255,0)"); r.addColorStop(0.72, "rgba(255,255,255,1)"); r.addColorStop(0.9, "rgba(255,255,255,0)");
  } else {
    r.addColorStop(0, "rgba(255,255,255,1)"); r.addColorStop(0.3, "rgba(255,255,255,0.6)"); r.addColorStop(0.6, "rgba(255,255,255,0.15)"); r.addColorStop(1, "rgba(255,255,255,0)");
  }
  g.fillStyle = r;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

function buffers(obj, n) {
  const g = obj.geometry;
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * n), 3).setUsage(THREE.DynamicDrawUsage));
  g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(3 * n), 3).setUsage(THREE.DynamicDrawUsage));
  g.setDrawRange(0, 0);
  obj.frustumCulled = false;
  return obj;
}
const additive = { vertexColors: true, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false };
const points = (n, size, map) => buffers(new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size, map, ...additive })), n);
const lines = (n) => buffers(new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial(additive)), 2 * n);

class Writer {
  constructor(obj) { this.obj = obj; this.p = obj.geometry.attributes.position; this.c = obj.geometry.attributes.color; this.n = 0; }
  put(x, y, z, r, g, b) {
    if (this.n >= this.p.count) return false;
    this.p.array.set([x, y, z], 3 * this.n);
    this.c.array.set([r, g, b], 3 * this.n++);
    return true;
  }
  done() { this.p.needsUpdate = this.c.needsUpdate = true; this.obj.geometry.setDrawRange(0, this.n); this.n = 0; }
}

export class Scene {
  constructor(host, cfg) {
    this.cfg = cfg;
    const R = this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    R.setPixelRatio(Math.min(devicePixelRatio, 2));
    R.outputColorSpace = THREE.LinearSRGBColorSpace;
    R.setClearColor(BG);
    host.appendChild(R.domElement);
    const S = this.scene = new THREE.Scene();
    S.fog = new THREE.Fog(BG, 260, 560);
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.5, 2000);
    this.camera.position.set(106, 190, 200);
    const C = this.controls = new OrbitControls(this.camera, R.domElement);
    C.enableDamping = true; C.dampingFactor = 0.08; C.maxPolarAngle = 1.48; C.minDistance = 6; C.maxDistance = 700;
    C.addEventListener("start", () => { this.goal = null; });
    S.add(new THREE.HemisphereLight(0xe4ecff, 0x1a1f2a, 2.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.3);
    sun.position.set(60, 140, 30);
    S.add(sun);

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(100, 100), new THREE.MeshBasicMaterial({ color: 0x0a0e15 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(49.5, -0.03, 49.5);
    S.add(floor);
    const grid = new THREE.GridHelper(100, 20, 0x18202e, 0x121824);
    grid.position.set(49.5, -0.01, 49.5);
    S.add(grid);
    const e = [-0.5, 99.5];
    const border = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([[e[0], e[0]], [e[1], e[0]], [e[1], e[1]], [e[0], e[1]]].map(([a, b]) => new THREE.Vector3(a, 0, b))), new THREE.LineBasicMaterial({ color: 0x4f9ad6 }));
    S.add(border);
    this.gravArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(-5, 0, 38), 24, 0x8f7cff, 5, 3);
    S.add(this.gravArrow);

    const geo = new THREE.BoxGeometry(1, 1, 1);
    geo.translate(0, 0.5, 0);
    this.tokens = new THREE.InstancedMesh(geo, new THREE.MeshLambertMaterial(), 4 * MAXB);
    this.tokens.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.tokens.setColorAt(0, new THREE.Color());
    this.tokens.frustumCulled = false;
    this.tokens.count = 0;
    this.tokens.boundingSphere = new THREE.Sphere(new THREE.Vector3(50, 0, 50), 400);
    S.add(this.tokens);
    const glow = tex(false);
    this.glow = new Writer(points(MAXB, 7, glow));
    this.spark = new Writer(points(1200, 3.2, glow));
    this.links = new Writer(lines(MAXL));
    this.ghost = new Writer(points(MAXB, 2.6, tex(true)));
    this.gap = new Writer(lines(MAXB));
    this.wires = new Writer(lines(MAXB));
    this.flow = new Writer(points(600, 1.3, glow));
    for (const w of [this.glow, this.spark, this.links, this.ghost, this.gap, this.wires, this.flow]) S.add(w.obj);

    this.ring = new THREE.Mesh(new THREE.RingGeometry(1.25, 1.55, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false }));
    this.ring.rotation.x = -Math.PI / 2;
    this.reach = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 64 }, (_, k) => new THREE.Vector3(4 * Math.cos(k * Math.PI / 32), 0.05, 4 * Math.sin(k * Math.PI / 32)))), new THREE.LineBasicMaterial({ color: 0x3f78a8, transparent: true, opacity: 0.8 }));
    this.arrows = [0xff9d5c, 0x6fd6b4, 0x9d86ff, 0xffffff].map((c) => new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, c));
    for (const o of [this.ring, this.reach, ...this.arrows]) S.add(o);

    this.arch = new Arch(S);
    this.ray = new THREE.Raycaster();
    this.plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    this.m = new THREE.Matrix4();
    this.col = new THREE.Color();
    this.rgb = [0, 0, 0];
    this.v = new THREE.Vector3();
    this.goal = null;
    this.portrait = null;
  }

  setGravity(gx, gy) {
    const m = Math.hypot(gx, gy);
    this.gravArrow.visible = m > 1e-6;
    if (m > 1e-6) this.gravArrow.setDirection(this.v.set(gy / m, 0, gx / m));
  }

  resize(w, h) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const p = w / h < 0.9;
    if (p !== this.portrait) {
      this.portrait = p;
      this.arch.group.position.set(p ? 1 : 116, 0, p ? 114 : 5);
      return true;
    }
    return false;
  }

  bounds(name) {
    const a = this.arch.group.position;
    if (name === "arena") return [49.5, 49.5, 104, 104];
    if (name === "arch") return [a.x + 47, a.z + 42, 100, 94];
    return this.portrait ? [49.5, 100, 104, 206] : [106, 48, 216, 104];
  }

  preset(name, at = null) {
    let cx, cz, w, h, el = name === "arena" ? 1.2 : 0.95;
    if (name === "follow" && at) { cx = at[1]; cz = at[0]; w = h = 24; el = 0.75; }
    else [cx, cz, w, h] = this.bounds(name);
    const tgt = new THREE.Vector3(cx, 0, cz), dir = new THREE.Vector3(0, Math.sin(el), Math.cos(el));
    const cam = this.camera.clone();
    cam.clearViewOffset();
    const [l, t, r, b] = this.ins, fw = this.W - l - r, fh = this.H - t - b;
    const pts = [];
    for (const y of [0, 3]) for (const sx of [-1, 1]) for (const sz of [-1, 1]) pts.push(new THREE.Vector3(cx + (sx * w) / 2, y, cz + (sz * h) / 2));
    let d = Math.max(w, h) * 2;
    const right = new THREE.Vector3(1, 0, 0), up = new THREE.Vector3().crossVectors(dir, right);
    for (let k = 0; k < 10; k++) {
      cam.position.copy(dir).multiplyScalar(d).add(tgt);
      cam.lookAt(tgt);
      cam.updateMatrixWorld();
      let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
      for (const p of pts) {
        const v = this.v.copy(p).project(cam);
        x0 = Math.min(x0, v.x); x1 = Math.max(x1, v.x); y0 = Math.min(y0, v.y); y1 = Math.max(y1, v.y);
      }
      const hh = d * Math.tan((cam.fov * Math.PI) / 360), hw = hh * cam.aspect;
      tgt.addScaledVector(right, ((x0 + x1) / 2) * hw * 0.8).addScaledVector(up, ((y0 + y1) / 2) * hh * 0.8);
      d *= Math.max(((x1 - x0) / 2) * this.W / fw, ((y1 - y0) / 2) * this.H / fh) / 0.94;
    }
    this.goal = { pos: cam.position.clone(), target: tgt };
  }

  insets(W, H, ins) {
    this.W = W; this.H = H; this.ins = ins;
    const cx = (ins[0] + W - ins[2]) / 2, cy = (ins[1] + H - ins[3]) / 2;
    this.camera.setViewOffset(W, H, W / 2 - cx, H / 2 - cy, W, H);
  }

  world(u, w, out = this.v) { const a = this.arch.group.position; return out.set(a.x + u, 0, a.z + w); }

  update(sim, st, tr, lut, t) {
    const { pos, vel, count, ids } = sim;
    const net = sim.net, r2 = 2 * this.cfg.radius, rgb = this.rgb, m = this.m, c = this.col;
    const sel = st.selected >= 0 ? sim.indexOf(st.selected) : -1;
    this.tokens.count = 4 * count;
    for (let i = 0; i < count; i++) {
      const x = pos[2 * i], y = pos[2 * i + 1];
      const ch = [(x - 49.5) / 49.5, (y - 49.5) / 49.5, vel[2 * i] / 15, vel[2 * i + 1] / 15];
      const s = i === sel ? 0.8 : 0.62;
      for (let k = 0; k < 4; k++) {
        m.makeScale(s, s * 0.95, s);
        m.setPosition(y, 0.06 + k * s * 1.12, x);
        this.tokens.setMatrixAt(4 * i + k, m);
        this.tokens.setColorAt(4 * i + k, divColor(ch[k], c, i === st.hoverBall ? 1.8 : 1.1));
      }
      speedColor(Math.hypot(vel[2 * i], vel[2 * i + 1]), rgb);
      const k = i === sel ? 1.2 : 0.7;
      this.glow.put(y, 0.1, x, rgb[0] * k, rgb[1] * k, rgb[2] * k);
    }
    this.tokens.instanceMatrix.needsUpdate = true;
    if (this.tokens.instanceColor) this.tokens.instanceColor.needsUpdate = true;
    this.glow.done();

    const pulse = 0.75 + 0.25 * Math.sin(t * 0.012);
    const link = (i, j, d) => {
      const yi = 0.5, a = [pos[2 * i + 1], yi, pos[2 * i]], b = [pos[2 * j + 1], yi, pos[2 * j]];
      if (d <= r2) {
        const f = Math.abs(lut.pairAt(d));
        const k = (0.35 + 0.65 * sl(f)) * pulse;
        this.links.put(...a, k, 0.62 * k, 0.28 * k);
        this.links.put(...b, k, 0.62 * k, 0.28 * k);
        const g = 0.3 + 0.7 * sl(f);
        this.spark.put((a[0] + b[0]) / 2, 0.6, (a[2] + b[2]) / 2, g * pulse, 0.7 * g * pulse, 0.35 * g * pulse);
      } else {
        const k = 0.28 * (1 - d / 4) + 0.04;
        this.links.put(...a, 0.35 * k, 0.6 * k, k);
        this.links.put(...b, 0.35 * k, 0.6 * k, k);
      }
    };
    if (st.edges === 2) net.forPairs(pos, count, this.cfg.neighbor_radius, link);
    else if (st.edges === 1 && tr && sel >= 0) for (const e of tr.edges) { const j = sim.indexOf(e.srcId); if (j >= 0) link(sel, j, e.dist); }
    this.links.done();
    this.spark.done();

    for (let k = 0; k < st.gt.count && st.gtOn; k++) {
      const gx = st.gt.xs[k], gy = st.gt.ys[k];
      this.ghost.put(gy, 0.2, gx, 0.55, 0.6, 0.7);
      const j = sim.indexOf(st.gtIds[k]);
      if (j >= 0) { this.gap.put(gy, 0.2, gx, 0.5, 0.18, 0.25); this.gap.put(pos[2 * j + 1], 0.2, pos[2 * j], 0.5, 0.18, 0.25); }
    }
    this.ghost.done();
    this.gap.done();

    const show = sel >= 0;
    this.ring.visible = this.reach.visible = show;
    for (const a of this.arrows) a.visible = false;
    this.arch.group.visible = show && !!tr && st.arch;
    if (show) {
      const x = pos[2 * sel], y = pos[2 * sel + 1];
      this.ring.position.set(y, 0.08, x);
      this.ring.scale.setScalar(1 + 0.12 * Math.sin(t * 0.006));
      this.reach.position.set(y, 0, x);
      if (tr) {
        const acc = [[tr.subAccPair[0], tr.subAccPair[1]], [tr.subAccWall[0], tr.subAccWall[1]], tr.gravity, [tr.subAcc[0], tr.subAcc[1]]];
        acc.forEach(([ax, ay], k) => {
          const mag = Math.hypot(ax, ay);
          if (mag < 1e-6) return;
          const A = this.arrows[k], L = 2.5 + 9 * sl(mag, 0.5);
          A.visible = true;
          A.position.set(y, 3.4 + 0.25 * k, x);
          A.setDirection(this.v.set(ay / mag, 0, ax / mag));
          A.setLength(L, Math.min(1.6, L * 0.35), 0.9);
        });
      }
    }
    if (this.arch.group.visible) {
      const A = this.arch.anchors, p = this.v;
      const wire = (ax, az, bx, bz, k) => { this.wires.put(ax, 2.9, az, 0.1 * k, 0.45 * k, 0.7 * k); this.wires.put(bx, 0.1, bz, 0.1 * k, 0.45 * k, 0.7 * k); };
      this.world(...A.input, p);
      wire(pos[2 * sel + 1], pos[2 * sel], p.x, p.z, 1);
      for (const col of A.cols) {
        const j = sim.indexOf(col.srcId);
        if (j < 0) continue;
        this.world(col.u, col.w, p);
        wire(pos[2 * j + 1], pos[2 * j], p.x, p.z, 0.45);
      }
      this.world(...A.output, p);
      wire(pos[2 * sel + 1], pos[2 * sel], p.x, p.z, 0.6);
      const ph = (t * 0.00025) % 1;
      const wp = this.arch.wirePos, nw = this.arch.wires.geometry.drawRange.count / 2, o = this.arch.group.position;
      for (let q = 0; q < nw; q++) {
        for (let h = 0; h < 2; h++) {
          const u = (ph * (1 + (q % 3) * 0.3) + h * 0.5 + q * 0.137) % 1;
          const b = 6 * q;
          const bright = 0.5 + 0.5 * Math.sin(u * Math.PI);
          this.flow.put(o.x + wp[b] + (wp[b + 3] - wp[b]) * u, 0.3, o.z + wp[b + 2] + (wp[b + 5] - wp[b + 2]) * u, 0.3 * bright, 0.75 * bright, bright);
        }
      }
    }
    this.wires.done();
    this.flow.done();
  }

  frame(front) {
    if (this.goal) {
      this.camera.position.lerp(this.goal.pos, 0.09);
      this.controls.target.lerp(this.goal.target, 0.09);
      if (this.camera.position.distanceTo(this.goal.pos) < 0.2) this.goal = null;
    }
    this.controls.update();
    const fd = this.camera.position.distanceTo(this.controls.target);
    this.scene.fog.near = fd * 1.2;
    this.scene.fog.far = fd * 3 + 150;
    if (this.arch.group.visible) this.arch.frame(front);
    this.renderer.render(this.scene, this.camera);
  }

  pick(cx, cy, sim) {
    const el = this.renderer.domElement.getBoundingClientRect();
    this.ray.setFromCamera({ x: ((cx - el.left) / el.width) * 2 - 1, y: -((cy - el.top) / el.height) * 2 + 1 }, this.camera);
    if (this.arch.group.visible) {
      const h = this.ray.intersectObject(this.arch.mesh)[0];
      if (h) return { kind: "cell", id: h.instanceId, cell: this.arch.cells[h.instanceId] };
    }
    const tk = this.ray.intersectObject(this.tokens)[0];
    if (tk) return { kind: "ball", index: Math.floor(tk.instanceId / 4), channel: tk.instanceId % 4 };
    const p = this.ray.ray.intersectPlane(this.plane, this.v);
    if (!p || p.x < -0.5 || p.x > 99.5 || p.z < -0.5 || p.z > 99.5) return null;
    let best = -1, bd = 2;
    for (let i = 0; i < sim.count; i++) {
      const d = Math.hypot(sim.pos[2 * i] - p.z, sim.pos[2 * i + 1] - p.x);
      if (d < bd) { bd = d; best = i; }
    }
    return best >= 0 ? { kind: "ball", index: best, channel: -1 } : { kind: "arena", x: p.z, y: p.x };
  }

  project(u, y, w, arch) {
    const v = arch ? this.world(u, w) : this.v.set(u, 0, w);
    v.y = y;
    v.project(this.camera);
    return v.z < 1 ? v : null;
  }
}
