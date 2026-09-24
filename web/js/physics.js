export class Physics {
  constructor(n = 100, { dt = 0.15, gravity = 9.0, radius = 0.75, stiffness = 400, substeps = 8 } = {}) {
    this.n = n;
    this.dt = dt;
    this.gravity = gravity;
    this.radius = radius;
    this.stiffness = stiffness;
    this.substeps = substeps;
    this.count = 0;
    this._alloc(64);
  }

  _alloc(cap) {
    const grow = (a) => {
      const b = new Float64Array(cap);
      if (a) b.set(a.subarray(0, this.count));
      return b;
    };
    this.cap = cap;
    this.xs = grow(this.xs);
    this.ys = grow(this.ys);
    this.vxs = grow(this.vxs);
    this.vys = grow(this.vys);
    this.fx = new Float64Array(cap);
    this.fy = new Float64Array(cap);
  }

  add(x, y, vx, vy) {
    if (this.count === this.cap) this._alloc(this.cap * 2);
    const i = this.count++;
    this.xs[i] = x;
    this.ys[i] = y;
    this.vxs[i] = vx;
    this.vys[i] = vy;
    return i;
  }

  clear() {
    this.count = 0;
  }

  remove(i) {
    for (const a of [this.xs, this.ys, this.vxs, this.vys]) a.copyWithin(i, i + 1, this.count);
    this.count--;
  }

  _forces() {
    const { xs, ys, fx, fy, radius, stiffness, gravity, count: m } = this;
    const hi = this.n - 1;
    const diam = 2 * radius;
    for (let i = 0; i < m; i++) {
      const x = xs[i], y = ys[i];
      let ax = 0, ay = 0;
      if (x < radius) { const p = radius - x; ax = stiffness * p * p; }
      else if (hi - x < radius) { const p = radius - (hi - x); ax = -(stiffness * p * p); }
      if (y < radius) { const p = radius - y; ay = stiffness * p * p; }
      else if (hi - y < radius) { const p = radius - (hi - y); ay = -(stiffness * p * p); }
      let sx = gravity + ax, sy = ay;
      for (let j = 0; j < m; j++) {
        if (j === i) continue;
        const dx = xs[j] - x, dy = ys[j] - y;
        const d = Math.hypot(dx, dy);
        const pen = diam - d;
        if (!(pen > 0)) continue;
        let nx, ny, dd = d;
        if (d < 1e-9) { nx = 1; ny = 0; dd = 0; } else { nx = dx / d; ny = dy / d; }
        const f = stiffness * (diam - dd) * (diam - dd);
        sx += -(f * nx);
        sy += -(f * ny);
      }
      fx[i] = sx;
      fy[i] = sy;
    }
  }

  step() {
    const sub = this.dt / this.substeps;
    const { xs, ys, vxs, vys, fx, fy } = this;
    for (let s = 0; s < this.substeps; s++) {
      this._forces();
      for (let i = 0; i < this.count; i++) {
        vxs[i] += fx[i] * sub;
        vys[i] += fy[i] * sub;
        xs[i] += vxs[i] * sub;
        ys[i] += vys[i] * sub;
      }
    }
  }

  energy() {
    const { xs, ys, vxs, vys, radius, stiffness, gravity, count: m } = this;
    const hi = this.n - 1;
    const diam = 2 * radius;
    const cube = (p) => (p > 0 ? stiffness * p * p * p / 3 : 0);
    let e = 0;
    for (let i = 0; i < m; i++) {
      e += 0.5 * (vxs[i] * vxs[i] + vys[i] * vys[i]) - gravity * xs[i];
      e += cube(radius - xs[i]) + cube(radius - (hi - xs[i]));
      e += cube(radius - ys[i]) + cube(radius - (hi - ys[i]));
      for (let j = i + 1; j < m; j++) e += cube(diam - Math.hypot(xs[j] - xs[i], ys[j] - ys[i]));
    }
    return e;
  }
}
