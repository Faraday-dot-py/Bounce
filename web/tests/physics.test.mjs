import { readFileSync } from 'fs';
import { Physics } from '../js/physics.js';

const cases = JSON.parse(readFileSync(process.argv[2], 'utf8'));
for (const c of cases) {
  const p = new Physics(c.n);
  for (const b of c.init) p.add(b.x, b.y, b.vx, b.vy);
  const out = [];
  const t0 = performance.now();
  for (let f = 1; f <= 200; f++) {
    p.step();
    if (f === 1 || f === 20 || f === 200) {
      let d = 0;
      c.frames[f - 1].forEach((b, i) => {
        d = Math.max(d, Math.abs(b[0] - p.xs[i]), Math.abs(b[1] - p.ys[i]));
      });
      out.push(`f${f}=${d.toExponential(2)}`);
    }
  }
  console.log(c.name, out.join(' '), `${((performance.now() - t0) / 200).toFixed(4)} ms/step`);
}
for (const m of [12, 100]) {
  const p = new Physics(100);
  for (let i = 0; i < m; i++) p.add(Math.random() * 99, Math.random() * 99, Math.random() - 0.5, Math.random() - 0.5);
  for (let i = 0; i < 50; i++) p.step();
  const t0 = performance.now();
  for (let i = 0; i < 500; i++) p.step();
  console.log(`bench ${m} balls: ${((performance.now() - t0) / 500).toFixed(4)} ms/step`);
}
