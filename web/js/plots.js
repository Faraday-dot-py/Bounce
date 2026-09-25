const N = 240;

// force curves sampled from the model's own MLPs (magnitude along the contact normal, positive = repulsive)
export function forceLut(net) {
  const { radius: r, force_scale: fs } = net.cfg;
  const pair = new Float64Array(N + 1), wall = new Float64Array(N + 1);
  for (let k = 0; k <= N; k++) {
    const pp = 1 - k / N;
    pair[k] = pp > 0 ? pp * net.pairMlp.eval(pp) * fs : 0;
    wall[k] = pp > 0 ? pp * net.wallMlp.eval(pp) * fs : 0;
  }
  const at = (a, d, span) => {
    const x = Math.min(Math.max(d / span, 0), 1) * N, i = Math.min(Math.floor(x), N - 1), f = x - i;
    return a[i] * (1 - f) + a[i + 1] * f;
  };
  return { pair, wall, r, pairAt: (d) => (d > 2 * r ? 0 : at(pair, d, 2 * r)), wallAt: (d) => (d > r ? 0 : at(wall, d, r)) };
}

function setup(cv) {
  const dpr = Math.min(devicePixelRatio, 2), w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  g.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  return { g, w, h };
}

function frame(g, w, h, x0, x1, y0, y1, xl) {
  const L = 44, B = 16, T = 7, Rr = 4;
  const X = (x) => L + ((x - x0) / (x1 - x0)) * (w - L - Rr);
  const Y = (y) => T + (1 - (y - y0) / (y1 - y0)) * (h - T - B);
  g.strokeStyle = "#1f2838"; g.lineWidth = 1;
  if (y0 < 0 && y1 > 0) { g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(w - Rr, Y(0)); g.stroke(); }
  g.fillStyle = "#6b7892"; g.textAlign = "right"; g.textBaseline = "middle";
  const tk = (v) => (Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(v && Math.abs(v) < 10 ? 1 : 0));
  for (const v of new Set([y0, y0 < 0 && y1 > 0 ? 0 : y0, y1])) g.fillText(tk(v), L - 5, Math.min(Math.max(Y(v), T), h - B - 4));
  g.textBaseline = "top";
  if (x1 !== 1) { g.textAlign = "left"; g.fillText(x0.toFixed(1), L, h - B + 3); g.textAlign = "right"; g.fillText(x1.toFixed(1), w - Rr, h - B + 3); }
  g.textAlign = "center";
  g.fillText(xl, (L + w) / 2, h - B + 3);
  return { X, Y };
}

function curve(g, X, Y, a, span, color) {
  g.strokeStyle = color; g.lineWidth = 1.6; g.beginPath();
  for (let k = 0; k <= N; k++) g[k ? "lineTo" : "moveTo"](X((k / N) * span), Y(a[k]));
  g.stroke();
}

function dot(g, x, y, color) {
  g.fillStyle = color; g.beginPath(); g.arc(x, y, 3.5, 0, 2 * Math.PI); g.fill();
  g.strokeStyle = "#06080c"; g.lineWidth = 1.2; g.stroke();
}

function range(a) {
  let lo = 0, hi = 0;
  for (const v of a) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  return [lo * 1.08, hi * 1.08 || 1];
}

export function drawPair(cv, lut, tr) {
  const { g, w, h } = setup(cv);
  const span = 2 * lut.r, [y0, y1] = range(lut.pair);
  const { X, Y } = frame(g, w, h, 0, span + 0.3, y0, y1, "distance d");
  g.fillStyle = "rgba(79,154,214,0.08)"; g.fillRect(X(0), 6, X(span) - X(0), h - 22);
  curve(g, X, Y, lut.pair, span, "#6fc3ff");
  g.strokeStyle = "#6fc3ff"; g.beginPath(); g.moveTo(X(span), Y(0)); g.lineTo(X(span + 0.3), Y(0)); g.stroke();
  if (tr) for (const e of tr.edges) if (e.contact) dot(g, X(e.dist), Y(Math.min(Math.max(e.mag, y0), y1)), "#ffa24c");
}

export function drawWall(cv, lut, tr) {
  const { g, w, h } = setup(cv);
  const [y0, y1] = range(lut.wall);
  const { X, Y } = frame(g, w, h, 0, lut.r + 0.25, y0, y1, "distance to wall");
  g.fillStyle = "rgba(111,214,180,0.08)"; g.fillRect(X(0), 6, X(lut.r) - X(0), h - 22);
  curve(g, X, Y, lut.wall, lut.r, "#6fd6b4");
  if (tr) tr.wall.forEach((wl, k) => { if (wl.pen > 0) dot(g, X(tr.wallDist[k]), Y(Math.min(wl.force, y1)), "#ffa24c"); });
}

export function drawEnergy(cv, model, truth) {
  const { g, w, h } = setup(cv);
  let lo = Infinity, hi = -Infinity;
  for (const a of [model, truth]) for (const v of a) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!model.length) return;
  if (hi - lo < 1) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.1;
  const { X, Y } = frame(g, w, h, 0, 1, lo - pad, hi + pad, "last 300 ticks");
  const line = (a, color, dash) => {
    if (a.length < 2) return;
    g.strokeStyle = color; g.lineWidth = 1.5; g.setLineDash(dash); g.beginPath();
    a.forEach((v, k) => g[k ? "lineTo" : "moveTo"](X(1 - (a.length - 1 - k) / 299), Y(v)));
    g.stroke(); g.setLineDash([]);
  };
  line(truth, "#ffa24c", [4, 3]);
  line(model, "#6fc3ff", []);
}
