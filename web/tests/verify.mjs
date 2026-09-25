// Compares the JS port against PyTorch rollouts dumped by scripts/export_web_testvectors.py.
import { readFileSync } from "node:fs";
import { TokenNet, containState, parseWeights } from "../js/model.js";

const dir = new URL("..", import.meta.url).pathname;
const manifest = JSON.parse(readFileSync(dir + "weights.json"));
const bin = readFileSync(dir + "weights.bin");
const net = new TokenNet(parseWeights(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)));
const vec = JSON.parse(readFileSync(dir + "tests/vectors.json"));

const maxDiff = (a, flat) => {
  let m = 0;
  for (let i = 0; i < flat.length; i++) m = Math.max(m, Math.abs(a[i] - flat[i]));
  return m;
};
const flat = (rows) => Float32Array.from(rows.flat());
const load = (s, t) => ({ pos: flat(s.pos[t]), vel: flat(s.vel[t]), hidden: flat(s.hidden[t]) });

let worst1 = 0, worstN = 0;
for (const [name, s] of Object.entries(vec.scenes)) {
  let one = 0, one32 = 0;
  for (let t = 0; t < s.steps; t++) {
    const a = load(s, t), b = load(s, t + 1);
    const c = s.count[t];
    net.step(a.pos, a.vel, a.hidden, c);
    const m = containState(a.pos, a.vel, a.hidden, c, vec.n, vec.max_speed);
    one32 = Math.max(one32, maxDiff(a.pos, b.pos), maxDiff(a.vel, b.vel), maxDiff(a.hidden, b.hidden));
    one = Math.max(one, maxDiff(a.pos, flat(s.pos64[t])), maxDiff(a.vel, flat(s.vel64[t])));
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
  console.log(`${name}: balls ${s.count[0]}  1-step max|d| vs float64 torch ${one.toExponential(2)} (vs float32 torch ${one32.toExponential(2)})  free-run max|d| at step ${Math.min(20, s.steps)} ${nstep.toExponential(2)}, at end (${s.steps}) ${full.toExponential(2)}`);
  worst1 = Math.max(worst1, one);
  worstN = Math.max(worstN, nstep);
}

let traceDiff = 0;
{
  const s = vec.scenes.contact_pack, r = s.trace;
  const a = load(s, r.t);
  const tr = net.step(a.pos, a.vel, a.hidden, s.count[r.t], r.idx);
  traceDiff = Math.max(maxDiff(tr.subPos, flat(r.pos)), maxDiff(tr.subVel, flat(r.vel)), maxDiff(tr.subAcc, flat(r.acc)));
  const b = load(s, r.t + 1);
  const fin = Math.max(Math.abs(tr.newPos[0] - b.pos[2 * r.idx]), Math.abs(tr.newPos[1] - b.pos[2 * r.idx + 1]));
  console.log(`trace ball ${r.idx} at t=${r.t}: ${tr.subAcc.length / 2} substep records, max|d| pos/vel/acc ${traceDiff.toExponential(2)}, newPos ${fin.toExponential(2)}, ${tr.edges.length} neighbours, ${tr.edges.filter((e) => e.contact).length} in contact`);
  traceDiff = Math.max(traceDiff, fin);
}
console.log(`worst 1-step ${worst1.toExponential(2)}, worst step-20 ${worstN.toExponential(2)}`);
process.exit(worst1 < 1e-4 && traceDiff < 1e-4 ? 0 : 1);
