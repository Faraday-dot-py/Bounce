import sys
import time
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/index.html"
EXE = sys.argv[2] if len(sys.argv) > 2 else None
ARGS = ["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]

JS_CELLS = """() => {
  const a = window.__bounce.view.arch, out = [];
  const m = a.mesh.instanceMatrix.array;
  a.cells.forEach((c, i) => out.push({ v: c.v, s: c.s, par: c.par, sy: m[16 * i + 5], y: m[16 * i + 13] }));
  return out;
}"""


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name, extra)
    if not ok:
        check.failed += 1


check.failed = 0

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, args=ARGS)
    pg = b.new_context(viewport={"width": 1500, "height": 900}).new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.goto(URL)
    pg.wait_for_function("window.__bounce", timeout=20000)
    time.sleep(3)
    pg.evaluate("window.__bounce.setPaused(true)")

    n = pg.evaluate("document.getElementById('w-name').options.length")
    check("weight dropdown populated", n > 0, n)

    pg.click("#collapse"); time.sleep(0.6)
    check("collapse hides panel", pg.evaluate("getComputedStyle(document.getElementById('panel')).visibility") == "hidden" and not pg.evaluate("document.getElementById('tab').hidden"))
    pg.click("#tab"); time.sleep(0.6)
    check("tab restores panel", pg.evaluate("getComputedStyle(document.getElementById('panel')).visibility") == "visible" and pg.evaluate("document.getElementById('tab').hidden"))

    pg.evaluate("window.__bounce.camera('arch')"); time.sleep(1.5)
    cells = pg.evaluate(JS_CELLS)
    neg = [c for c in cells if c["v"] < -0.05]
    pos = [c for c in cells if c["v"] > 0.05]
    base = lambda c: c["s"] * 1.4 * 1.2 + 0.01 * c["s"]
    check("negative cubes exist", len(neg) > 0, len(neg))
    check("negative cubes lie below baseline", all(c["y"] < base(c) - 1e-6 and c["y"] + c["sy"] <= base(c) + 0.08 * c["s"] + 1e-6 for c in neg))
    check("positive cubes start at baseline and rise", all(abs(c["y"] - base(c)) < 1e-6 and c["y"] + c["sy"] > base(c) for c in pos))

    grav = [i for i, c in enumerate(cells) if c["par"] and c["par"][0] == "gravity"]
    pg.evaluate("window.__bounce.view.goal = null")
    pos_xy = pg.evaluate("""(i) => {
      const v = window.__bounce.view, c = v.arch.cells[i], o = v.arch.group.position;
      const p = v.v.set(o.x + c.u, c.s * 1.4 * 1.2 + 0.4, o.z + c.w).project(v.camera);
      return [(p.x + 1) / 2 * innerWidth, (1 - p.y) / 2 * innerHeight];
    }""", grav[0])
    pg.mouse.move(*pos_xy); pg.mouse.down(); pg.mouse.up(); time.sleep(0.5)
    sel = pg.evaluate("[window.__bounce.ws.name, window.__bounce.ws.i]")
    check("clicking a gravity cube selects its weight", sel[0] == "gravity", sel)

    pg.select_option("#w-name", "gravity")
    pg.fill("#w-idx", "0"); pg.fill("#w-val", "0")
    pg.fill("#w-idx", "1"); pg.fill("#w-val", "9")
    g = pg.evaluate("[window.__bounce.net.gx, window.__bounce.net.gy]")
    check("UI sets gravity to (0, 9)", abs(g[0]) < 1e-6 and abs(g[1] - 9) < 1e-4, g)
    moved = pg.evaluate("""() => {
      const s = window.__bounce.sim; s.count = 1; s.pos[0] = 50; s.pos[1] = 50; s.vel.fill(0);
      for (let t = 0; t < 15; t++) s.tick();
      return [s.pos[0], s.pos[1]];
    }""")
    check("dropped ball moves in +y", moved[1] > 60 and abs(moved[0] - 50) < 1, moved)

    pg.evaluate("window.__bounce.selectWeight('pair_force.4.bias', 0); window.__bounce.editWeight(3)")
    same = pg.evaluate("""() => { const n = window.__bounce.net; const t = n.get('pair_force.4.bias', 0); return [t, n.pairMlp.B2[0], n.trained['pair_force.4.bias'][0]]; }""")
    check("pair_force edit reaches the shared MLP", same[0] == same[1] == 3 or (same[0] == 3 and same[1] == 3), same)
    check("no page errors", not errs, errs[:3])
    b.close()
sys.exit(1 if check.failed else 0)
