"""Browser check for the room demo (ADR-008), in a real Chrome.

The gateway's Rust tests prove the demo is *served* correctly: routes,
headers, CSP, no people in the served scenes. They cannot prove it
*works*: that WebGL draws, that the stricter CSP does not silently block
a module, that a click selects the thing under the cursor, that nothing
overflows at phone width. Those failed in ways no unit test caught while
this page was being built — the `hidden` attribute losing to a CSS
`display: grid`, a hash change not redrawing the scene, dimension labels
colliding off-screen at 375 px — so this script drives the page.

Usage (from the repo root, with a gateway built with `--features roomsim`
running; see RUN.md §4b):

    .venv/bin/python -I rust/crates/saans-gateway/tests/browser/roomsim_check.py \
        --base http://127.0.0.1:18080 --out /tmp/roomsim-shots

Needs Playwright (`pip install playwright` into .venv) and Google Chrome;
no Playwright browser download is used. Exits non-zero on any failure and
leaves screenshots in --out for a human to look at, because a page can
pass every check here and still look wrong.

All numbers on the page are SCENE (declared geometry). This checks they
are drawn and labelled, not that any of them is true of a real room.
"""
import argparse
import json
import os
import statistics
import sys
import tempfile

from playwright.sync_api import sync_playwright
from PIL import Image

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--base", default="http://127.0.0.1:18080", help="gateway origin")
ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "roomsim-shots"),
                help="directory for screenshots and results.json (default: outside the repo)")
ap.add_argument("--chrome", default="/usr/bin/google-chrome", help="Chrome executable")
args = ap.parse_args()
BASE = args.base.rstrip("/") + "/roomsim"
OUT = args.out
os.makedirs(OUT, exist_ok=True)
VIEWPORTS = {"desktop": (1366, 860), "phone": (375, 812)}
results = {"checks": [], "console": [], "pageerrors": []}

def check(name, ok, detail=""):
    results["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

def not_blank(path, box=None):
    im = Image.open(path).convert("L")
    if box: im = im.crop(box)
    px = list(im.get_flattened_data())[::7]
    return statistics.pstdev(px)

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=args.chrome, headless=True,
        args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    for vname, (w, h) in VIEWPORTS.items():
        ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=1,
                                  is_mobile=(vname == "phone"), has_touch=(vname == "phone"))
        page = ctx.new_page()
        page.on("console", lambda m, v=vname: results["console"].append(f"[{v}] {m.type}: {m.text}"))
        page.on("pageerror", lambda e, v=vname: results["pageerrors"].append(f"[{v}] {e}"))
        for scene in ["bedroom", "office", "hall-l-shaped"]:
            page.goto(f"{BASE}#scene={scene}")
            page.wait_for_selector(f"body[data-scene='{scene}']", timeout=20000)
            page.wait_for_timeout(400)
            shot = f"{OUT}/{scene}-{vname}.png"
            page.screenshot(path=shot)
            canvas = page.locator("canvas")
            bb = canvas.bounding_box()
            check(f"{vname}/{scene}: canvas present and sized", bb and bb["width"] > 200 and bb["height"] > 200, bb)
            sd = not_blank(shot, (int(bb["x"]), int(bb["y"]), int(bb["x"] + bb["width"]), int(bb["y"] + bb["height"])))
            check(f"{vname}/{scene}: the 3-D view drew something (pixel stdev)", sd > 8, f"{sd:.1f}")
            sw = page.evaluate("[document.documentElement.scrollWidth, document.documentElement.clientWidth]")
            check(f"{vname}/{scene}: no horizontal overflow", sw[0] <= sw[1], sw)
            info = page.locator("#scene-info").inner_text()
            check(f"{vname}/{scene}: panel shows provenance", "hand-written" in info, info.splitlines()[:2])
            want_name = {"bedroom": "bedroom-4.2x3.6", "office": "office-6x5", "hall-l-shaped": "hall-l-shaped-9x7"}[scene]
            check(f"{vname}/{scene}: the requested scene is the one drawn", want_name in info, want_name)
            nlab = page.locator(".label:not(.hidden)").count()
            check(f"{vname}/{scene}: labels placed", nlab >= 5, nlab)
            geo = page.evaluate("""() => {
              const vp = document.getElementById('viewport').getBoundingClientRect();
              const rs = [...document.querySelectorAll('.label:not(.hidden)')].map(e => e.getBoundingClientRect());
              let overlaps = 0, outside = 0;
              for (let i = 0; i < rs.length; i++) {
                const a = rs[i];
                if (a.left < vp.left - 0.5 || a.right > vp.right + 0.5 || a.top < vp.top - 0.5 || a.bottom > vp.bottom + 0.5) outside++;
                for (let j = i + 1; j < rs.length; j++) {
                  const b = rs[j];
                  if (a.left < b.right - 0.5 && a.right > b.left + 0.5 && a.top < b.bottom - 0.5 && a.bottom > b.top + 0.5) overlaps++;
                }
              }
              return {n: rs.length, overlaps, outside};
            }""")
            check(f"{vname}/{scene}: no two visible labels overlap", geo["overlaps"] == 0, geo)
            dims = page.locator(".label.dim:not(.hidden)").all_inner_texts()
            if scene != "hall-l-shaped":
                for tag in ("L ", "B ", "H "):
                    check(f"{vname}/{scene}: room dimension {tag.strip()} is shown", any(d.startswith(tag) for d in dims), dims)
            else:
                check(f"{vname}/{scene}: H and every edge of the L are dimensioned", len(dims) == 7, dims)
            check(f"{vname}/{scene}: every visible label is inside the view", geo["outside"] == 0, geo)
            if vname == "phone":
                fold = page.evaluate("""() => {
                  const f = document.querySelector('footer').getBoundingClientRect();
                  const p = document.querySelector('.panel').getBoundingClientRect();
                  return {footerTop: f.top + scrollY, panelBottom: p.bottom + scrollY};
                }""")
                check(f"phone/{scene}: footer comes after the panel, not on top of it",
                      fold["footerTop"] >= fold["panelBottom"] - 0.5, fold)
        ctx.close()

    # ---- interaction, desktop, bedroom ----
    ctx = browser.new_context(viewport={"width": 1366, "height": 860})
    page = ctx.new_page()
    page.on("console", lambda m: results["console"].append(f"[interact] {m.type}: {m.text}"))
    page.on("pageerror", lambda e: results["pageerrors"].append(f"[interact] {e}"))
    page.goto(f"{BASE}#scene=bedroom")
    page.wait_for_selector("body[data-ready=true]")
    page.wait_for_timeout(300)

    check("nothing looks selected at load", page.locator("#sel-info").is_hidden())
    # 1. Select from the list (keyboard/screen-reader path).
    page.get_by_role("button", name="almirah").click()
    sel = page.locator("#sel-info").inner_text()
    check("list click selects the almirah", "almirah" in sel and "1.20 × 0.60 × 2.10 m" in sel, sel.replace("\n", " | "))

    # 2. Real raycast. A selected object's label always wins the declutter,
    #    so select the bed from the list to learn where it is on screen,
    #    deselect, then click the 3-D view just under that label - on the
    #    bed's top surface - and expect the raycast to select it again.
    page.get_by_role("button", name="bed").click()
    # The highlight moves on the next animation frame; wait for the bed's
    # own label rather than whichever label was selected a frame ago.
    lab = page.locator(".label.selected", has_text="1.90×1.40×0.50 m")
    lab.wait_for(state="visible", timeout=5000)
    check("a selected object's label is shown even when decluttered", page.locator(".label.selected").count() == 1)
    lb = lab.bounding_box()
    page.keyboard.press("Escape")
    page.wait_for_timeout(100)
    page.mouse.click(lb["x"] + lb["width"] / 2, lb["y"] + lb["height"] + 14)
    sel = page.locator("#sel-info").inner_text()
    check("clicking the bed in the 3-D view selects it (raycast)", "bed" in sel and "fabric" in sel, sel.replace("\n", " | "))
    page.screenshot(path=f"{OUT}/bedroom-selected.png")

    page.keyboard.press("Escape")
    check("Escape clears the selection", page.locator("#sel-info").is_hidden())
    # 3. A drag must not select.
    page.mouse.move(600, 500); page.mouse.down(); page.mouse.move(700, 520, steps=6); page.mouse.up()
    check("a drag orbits without selecting", page.locator("#sel-info").is_hidden())

    # 4. Dimensions off hides every dim label.
    page.click("#dims-toggle")
    dims_vis = page.locator(".label.dim:not(.hidden)").count()
    check("dimensions toggle hides dimension labels", dims_vis == 0, dims_vis)
    check("dimensions toggle reports its state", page.get_attribute("#dims-toggle", "aria-pressed") == "false")
    page.click("#dims-toggle")
    check("dimensions back on shows them", page.locator(".label.dim:not(.hidden)").count() >= 3)

    # 5. Top-down 2D.
    page.click("#view-toggle")
    page.wait_for_timeout(250)
    page.screenshot(path=f"{OUT}/bedroom-topdown.png")
    check("top-down toggle reports its state", page.get_attribute("#view-toggle", "aria-pressed") == "true")
    check("top-down is in the URL so it can be shared", "view=top" in page.url, page.url)

    # 6. The L-shape in plan, with every edge dimensioned.
    page.goto(f"{BASE}#scene=hall-l-shaped&view=top")
    page.wait_for_selector("body[data-scene='hall-l-shaped']"); page.wait_for_timeout(300)
    page.screenshot(path=f"{OUT}/hall-l-shaped-topdown.png")
    dims = page.locator(".label.dim:not(.hidden)").all_inner_texts()
    check("L-shaped hall: six edge dimensions plus H", len(dims) == 7, dims)
    for want in ["9.00 m", "3.00 m", "5.00 m", "4.00 m", "H 3.20 m"]:
        check(f"L-shaped hall shows {want}", any(want in d for d in dims))

    page.goto(f"{BASE}#scene=office")
    page.wait_for_selector("body[data-scene='office']")
    page.go_back()
    page.wait_for_selector("body[data-scene='hall-l-shaped']", timeout=10000)
    check("browser Back returns to the previous scene", page.get_attribute("body", "data-scene") == "hall-l-shaped")
    # 7. No person is drawn anywhere; the served scene has none.
    served = page.evaluate("fetch('/roomsim/scenes/bedroom').then(r => r.json())")
    check("served scene carries no people or phone", "people" not in served and "phone" not in served)
    ctx.close()

    # ---- regressions found by review, one check each ----
    ctx = browser.new_context(viewport={"width": 1366, "height": 860})
    page = ctx.new_page()
    page.on("console", lambda m: results["console"].append(f"[review] {m.type}: {m.text}"))
    page.on("pageerror", lambda e: results["pageerrors"].append(f"[review] {e}"))
    page.goto(f"{BASE}#scene=bedroom")
    page.wait_for_selector("body[data-scene='bedroom']"); page.wait_for_timeout(300)

    # A label names one thing: clicking it selects that thing (#31).
    shown = page.locator(".label[data-why='shown']:not(.dim)")
    tried = 0
    for i in range(shown.count()):
        lab = shown.nth(i)
        first = lab.inner_text().split("\n")[0]
        bb = lab.bounding_box()
        page.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        sel = page.locator("#sel-info").inner_text()
        check(f"clicking the label '{first}' selects '{first}'", first in sel, sel.replace("\n", " | ")[:80])
        page.keyboard.press("Escape")
        tried += 1
    check("label clicks were actually tried", tried >= 4, tried)

    # The URL records the selection and Back undoes a scene pick (#42).
    page.get_by_role("button", name="almirah").click()
    check("the selection is in the URL", "select=almirah" in page.url, page.url)
    page.select_option("#scene-picker", "office")
    page.wait_for_selector("body[data-scene='office']")
    page.go_back()
    page.wait_for_selector("body[data-scene='bedroom']", timeout=10000)
    check("Back after a picker change returns to the previous scene", True)

    # Keyboard: zoom moves the view, 0 puts it back; Reset view does too (#41).
    # The page redraws on the next animation frame, so every comparison
    # waits for two frames rather than a fixed time: a fixed sleep made
    # these checks flaky under load (1 run in 5 failed).
    def settle():
        page.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
    def anchor_pos():
        settle()
        return page.evaluate("[...document.querySelectorAll('.label.dim[data-why=shown]')].map(e => e.style.transform).join('|')")
    page.keyboard.press("Escape")
    home = anchor_pos()
    page.focus("canvas")
    page.keyboard.press("+")
    check("+ zooms the view from the keyboard", anchor_pos() != home)
    page.keyboard.press("Control+0")
    check("Ctrl+0 is left to the browser (the view stays zoomed)", anchor_pos() != home)
    page.keyboard.press("0")
    check("0 resets the view from the keyboard", anchor_pos() == home)
    page.keyboard.press("ArrowLeft")
    check("arrow keys pan the view", anchor_pos() != home)
    page.click("#reset-view")
    check("Reset view restores the view", anchor_pos() == home)

    # Plan view: the primary drag pans (#36).
    page.click("#view-toggle")
    before = anchor_pos()
    page.mouse.move(500, 400); page.mouse.down(); page.mouse.move(600, 450, steps=6); page.mouse.up()
    check("in plan view a left-drag pans", anchor_pos() != before)
    page.click("#view-toggle")

    # A failed load says so, keeps the picker honest, and clears once a
    # later load succeeds (#1, #30).
    page.goto(f"{BASE}#scene=bedroom"); page.wait_for_selector("body[data-scene='bedroom']")
    page.route("**/roomsim/scenes/office", lambda r: r.fulfill(status=503, body="down"))
    page.select_option("#scene-picker", "office"); page.wait_for_timeout(500)
    check("a failed load shows a notice", page.locator("#notice").is_visible() and "Could not load" in page.locator("#notice").inner_text())
    check("after a failed load the picker still names what is drawn",
          page.eval_on_selector("#scene-picker", "e => e.value") == page.get_attribute("body", "data-scene") == "bedroom")
    check("after a failed load the URL still names what is drawn", "scene=bedroom" in page.url and "office" not in page.url, page.url)
    page.unroute("**/roomsim/scenes/office")
    page.select_option("#scene-picker", "hall-l-shaped")
    page.wait_for_selector("body[data-scene='hall-l-shaped']")
    check("a later successful load clears the error", page.locator("#notice").is_hidden())

    # A scene the gateway does not serve is reported, not swapped (#2).
    page.goto(f"{BASE}#scene=garage"); page.wait_for_timeout(500)
    txt = page.locator("#notice").inner_text() if page.locator("#notice").is_visible() else ""
    check("an unknown scene id is reported", 'No scene named "garage"' in txt, txt)
    check("an unknown scene id is not silently replaced", "scene=garage" in page.url and page.get_attribute("body", "data-scene") == "", page.url)
    ctx.close()

    # ---- crafted scenes, served in place of the bedroom ----
    def crafted(mutate):
        c = browser.new_context(viewport={"width": 1366, "height": 860})
        pg = c.new_page()
        pg.on("pageerror", lambda e: results["pageerrors"].append(f"[crafted] {e}"))
        raw = json.loads(pg.request.get(f"{args.base}/roomsim/scenes/bedroom").text())
        mutate(raw)
        pg.route("**/roomsim/scenes/bedroom", lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps(raw)))
        pg.goto(f"{BASE}#scene=bedroom")
        pg.wait_for_selector("body[data-scene='bedroom']"); pg.wait_for_timeout(300)
        return c, pg

    # No walls listed: implied walls must not claim a thickness (#3).
    c, pg = crafted(lambda sc: sc.pop("walls"))
    pg.get_by_role("button", name="Wall 1").click()
    sel = pg.locator("#sel-info").inner_text()
    check("an implied wall's thickness is shown as not declared", "not declared" in sel and "Thickness\n0.10 m" not in sel, sel.replace("\n", " | ")[:120])
    c.close()

    # An empty room: clicking bare floor clears the selection (#34).
    def empty(sc):
        sc["objects"] = []
    c, pg = crafted(empty)
    pg.get_by_role("button", name="N0").click()
    vp = pg.locator("#viewport").bounding_box()
    pg.mouse.click(vp["x"] + vp["width"] / 2, vp["y"] + vp["height"] / 2)
    check("clicking bare floor clears the selection", pg.locator("#sel-info").is_hidden())
    c.close()

    # A very small room: every dimension still fits in both views (#35).
    def small(sc):
        sc["room"] = {"w_m": 1.5, "d_m": 1.2, "h_m": 2.4}
        sc["walls"] = []; sc["objects"] = []
        sc["nodes"] = [{"id": 0, "pos": [0.2, 0.2, 1.0]}, {"id": 1, "pos": [1.3, 1.0, 1.0]}]
    c, pg = crafted(small)
    for view in ("orbit", "plan"):
        if view == "plan":
            pg.click("#view-toggle"); pg.wait_for_timeout(250)
        dims = pg.locator(".label.dim[data-why='shown']").all_inner_texts()
        check(f"a 1.5 x 1.2 m room shows L, B and H in the {view} view", all(any(d.startswith(t) for d in dims) for t in ("L ", "B ", "H ")), dims)
    c.close()

    # A short, wide window (a landscape phone): the footer stays below (#38).
    c = browser.new_context(viewport={"width": 812, "height": 375}, is_mobile=True, has_touch=True)
    pg = c.new_page()
    pg.goto(f"{BASE}#scene=office"); pg.wait_for_selector("body[data-scene='office']")
    fold = pg.evaluate("""() => {
      const f = document.querySelector('footer').getBoundingClientRect();
      const m = document.querySelector('main').getBoundingClientRect();
      return {footerTop: f.top + scrollY, mainBottom: m.bottom + scrollY};
    }""")
    check("landscape phone: the footer comes after main, not over it", fold["footerTop"] >= fold["mainBottom"] - 0.5, fold)
    c.close()
    browser.close()

    # ---- without WebGL the page still describes the room (#25, #39) ----
    nogl = p.chromium.launch(executable_path=args.chrome, headless=True,
                             args=["--disable-webgl", "--disable-webgl2", "--disable-3d-apis"])
    pg = nogl.new_page(viewport={"width": 1366, "height": 860})
    pg.on("pageerror", lambda e: results["pageerrors"].append(f"[no-webgl] {e}"))
    pg.on("console", lambda m: results["console"].append(f"[no-webgl] {m.type}: {m.text}"))
    pg.goto(f"{BASE}#scene=bedroom")
    pg.wait_for_selector("body[data-ready=true]", timeout=20000)
    check("without WebGL the page says the 3-D view is unavailable", "WebGL is not available" in pg.locator("#notice").inner_text())
    items = pg.locator("#object-list li").all_inner_texts()
    check("without WebGL every wall, object and node is listed as text",
          sum(t.startswith("Wall ") for t in items) == 4 and sum(t.startswith("N") and "ESP32" in t for t in items) == 4
          and any(t.startswith("bed:") for t in items), items[:3])
    check("without WebGL the view controls are disabled",
          all(pg.locator(f"#{i}").is_disabled() for i in ("view-toggle", "dims-toggle", "reset-view")))
    check("without WebGL the room panel still names the scene and its provenance",
          "bedroom-4.2x3.6" in pg.locator("#scene-info").inner_text() and "hand-written" in pg.locator("#scene-info").inner_text())
    # A failed load and then a good one must not clear the WebGL notice.
    pg.route("**/roomsim/scenes/office", lambda r: r.fulfill(status=503, body="down"))
    pg.select_option("#scene-picker", "office"); pg.wait_for_timeout(400)
    pg.unroute("**/roomsim/scenes/office")
    pg.select_option("#scene-picker", "hall-l-shaped")
    pg.wait_for_selector("body[data-scene='hall-l-shaped']")
    check("without WebGL, the notice survives a failed load followed by a good one",
          pg.locator("#notice").is_visible() and "WebGL is not available" in pg.locator("#notice").inner_text())
    pg.goto(f"{BASE}#scene=garage"); pg.wait_for_timeout(300)
    pg.select_option("#scene-picker", "office")
    pg.wait_for_selector("body[data-scene='office']")
    check("without WebGL, the notice survives an unknown scene followed by a good one",
          pg.locator("#notice").is_visible() and "WebGL is not available" in pg.locator("#notice").inner_text())
    nogl.close()

csp = [c for c in results["console"] if "Content Security Policy" in c or "Refused to" in c]
# Two kinds of error are caused on purpose and nothing else may error: the
# failed-load checks answer one scene with a 503 (the browser and the page
# both log it), and the no-WebGL run disables WebGL, so three.js logs that
# it could not create a context.
def expected(c):
    if c.startswith(("[review]", "[no-webgl]")) and "503" in c:
        return True
    return c.startswith("[no-webgl]") and ("WebGL context could not be created" in c or "Error creating WebGL context" in c)
errs = [c for c in results["console"] if " error:" in c and not expected(c)]
check("no CSP violations in the console", not csp, csp[:3])
check("no console errors", not errs, errs[:5])
check("no uncaught page errors", not results["pageerrors"], results["pageerrors"][:3])
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
failed = [c for c in results["checks"] if not c["ok"]]
print(f"\n{len(results['checks']) - len(failed)}/{len(results['checks'])} checks passed")
sys.exit(1 if failed else 0)
