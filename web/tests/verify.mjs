// Compares the JS port against PyTorch rollouts dumped by scripts/dump_web_testvectors.py.
import { readFileSync } from "node:fs";
import { TokenNet, containState, parseWeights } from "../js/model.js";

const dir = new URL("..", import.meta.url).pathname;
const manifest = JSON.parse(readFileSync(dir + "weights.json"));
const bin = readFileSync(dir + "weights.bin");
const net = new TokenNet(parseWeights(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)));
const vec = JSON.parse(readFileSync(dir + "tests/vectors.json"));
const H = 32;

const maxDiff = (a, flat) => {
  let m = 0;
  for (let i = 0; i < flat.length; i++) m = Math.max(m, Math.abs(a[i] - flat[i]));
  return m;
};
const flat = (rows) => Float32Array.from(rows.flat());
const load = (s, t) => ({ pos: flat(s.pos[t]), vel: flat(s.vel[t]), hidden: flat(s.hidden[t]) });

let worst1 = 0, worstN = 0;
for (const [name, s] of Object.entries(vec.scenes)) {
  let one = 0;
  for (let t = 0; t < s.steps; t++) {
    const a = load(s, t), b = load(s, t + 1);
    const c = s.count[t];
    net.step(a.pos, a.vel, a.hidden, c);
    const m = containState(a.pos, a.vel, a.hidden, c, vec.n, vec.max_speed);
    one = Math.max(one, maxDiff(a.pos, b.pos), maxDiff(a.vel, b.vel), maxDiff(a.hidden, b.hidden));
    if (m !== s.count[t + 1]) throw new Error("count mismatch");
  }
  const st = load(s, 0);
  let count = s.count[0];
  const curve = [];
  for (let t = 0; t < s.steps; t++) {
    net.step(st.pos, st.vel, st.hidden, count);
    count = containState(st.pos, st.vel, st.hidden, count, vec.n, vec.max_speed);
    const b = load(s, t + 1);
    curve.push(Math.max(maxDiff(st.pos, b.pos), maxDiff(st.vel, b.vel)));
  }
  const nstep = curve[Math.min(19, curve.length - 1)];
  const full = curve[curve.length - 1];
  console.log(`${name}: balls ${s.count[0]}  1-step max|d| ${one.toExponential(2)}  ${s.steps}-step: max|d| at step ${Math.min(20, s.steps)} ${nstep.toExponential(2)}, at end ${full.toExponential(2)}`);
  worst1 = Math.max(worst1, one);
  worstN = Math.max(worstN, nstep);
}

{
  const s = vec.scenes.spread12;
  const a = load(s, 1);
  const tr = net.step(a.pos, a.vel, a.hidden, s.count[1], 0);
  void tr;
  const h = s.hooks_step2;
  const d = {
    q: maxDiff(net.q.subarray(0, 12 * H), flat(h.q)),
    gru: maxDiff(a.hidden, flat(h.gru)),
  };
  console.log("hooks (step 2) q/gru max|d|:", d.q.toExponential(2), d.gru.toExponential(2));
  const tr0 = (() => { const b = load(s, 1); return net.step(b.pos, b.vel, b.hidden, s.count[1], 3); })();
  console.log("trace ball 3: hNew vs gru", maxDiff(tr0.hNew, flat([h.gru[3]])).toExponential(2),
    "delta vs wall+delta", maxDiff(Float64Array.from(tr0.wallOut), flat([h.wall[3]])).toExponential(2));
}
console.log(`worst 1-step ${worst1.toExponential(2)}, worst multi-step ${worstN.toExponential(2)}`);
process.exit(worst1 < 1e-4 ? 0 : 1);
