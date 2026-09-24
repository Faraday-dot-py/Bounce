import { TokenNet, containState } from "./model.js";

const H = 32;

function rng(seed) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  return { uniform, normal };
}

export class Sim {
  constructor(weights, { maxBalls = 300, spawnSpeed = 2.3, seed = 4738 } = {}) {
    this.net = new TokenNet(weights);
    this.n = weights.config.n;
    this.maxSpeed = weights.config.max_speed;
    this.maxBalls = maxBalls;
    this.spawnSpeed = spawnSpeed;
    this.minGap = 1.5;
    this.seed = seed;
    this.pos = new Float32Array(2 * maxBalls);
    this.vel = new Float32Array(2 * maxBalls);
    this.hidden = new Float32Array(H * maxBalls);
    this.ids = new Int32Array(maxBalls);
    this.reset();
  }

  reset() {
    this.rand = rng(this.seed);
    this.nextId = 0;
    this.clear();
  }

  clear() {
    this.count = 0;
    this.ticks = 0;
  }

  spawn(x, y, jitter = 0.4) {
    const { n, rand } = this;
    const px = Math.min(Math.max(x + rand.normal() * jitter, 1), n - 2);
    const py = Math.min(Math.max(y + rand.normal() * jitter, 1), n - 2);
    for (let i = 0; i < this.count; i++) {
      if (Math.hypot(this.pos[2 * i] - px, this.pos[2 * i + 1] - py) < this.minGap) return -1;
    }
    if (this.count >= this.maxBalls) this.drop(0);
    const i = this.count++;
    const s = this.spawnSpeed;
    this.pos[2 * i] = px;
    this.pos[2 * i + 1] = py;
    this.vel[2 * i] = (rand.uniform() * 2 - 1) * s;
    this.vel[2 * i + 1] = (rand.uniform() * 2 - 1) * s;
    this.hidden.fill(0, i * H, (i + 1) * H);
    this.ids[i] = this.nextId++;
    return this.ids[i];
  }

  drop(i) {
    const c = this.count;
    this.pos.copyWithin(2 * i, 2 * (i + 1), 2 * c);
    this.vel.copyWithin(2 * i, 2 * (i + 1), 2 * c);
    this.hidden.copyWithin(i * H, (i + 1) * H, c * H);
    this.ids.copyWithin(i, i + 1, c);
    this.count--;
  }

  populate(count) {
    let tries = 0;
    while (this.count < count && tries++ < 10000) {
      this.spawn(3 + this.rand.uniform() * (this.n * 0.6 - 6), 3 + this.rand.uniform() * (this.n - 6), 0);
    }
  }

  indexOf(id) {
    for (let i = 0; i < this.count; i++) if (this.ids[i] === id) return i;
    return -1;
  }

  tick(traceId = -1) {
    this.ticks++;
    if (this.count === 0) return null;
    const trace = this.net.step(this.pos, this.vel, this.hidden, this.count, traceId >= 0 ? this.indexOf(traceId) : -1);
    this.count = containState(this.pos, this.vel, this.hidden, this.count, this.n, this.maxSpeed, this.ids);
    return trace;
  }
}
