"""Browser check for the room demo (ADR-008), in a real Chrome.

The gateway's Rust tests prove the demo is *served* correctly: routes,
headers, CSP, no people in the served scenes, and the room stream's
frames. They cannot prove it *works*: that WebGL draws, that the stricter
CSP does not silently block a module, that a click selects the thing
under the cursor, that nothing overflows at phone width, that the
simulation view draws what spec/livedemo.md §9 says and nothing else.
Those failed in ways no unit test caught while this page was being
built — the `hidden` attribute losing to a CSS `display: grid`, a hash
change not redrawing the scene, dimension labels colliding off-screen at
375 px — so this script drives the page.

Setup (spec/livedemo.md §11 "Browser"; run from the repo root). The
script starts nothing itself; it needs two gateways from a release build
with the `roomsim` feature, and one recorded run:

    export PATH="$HOME/.cargo/bin:$PATH"
    (cd rust && cargo build --release -p saans-gateway --features roomsim \
             && cargo build --release -p saans-cli)
    # 1. A recorded SIM run of the hall, for REPLAY (§6):
    rust/target/release/saans roomsim-eval --scene spec/scenes/hall-l-shaped.json \
        --variant with_walker --seed 0 --records-out $SCRATCH/hall-with_walker-0.jsonl
    # 2. The page's gateway (--base): a scratch copy of
    #    rust/crates/saans-gateway/roomsim-demo.toml with data_dir and both
    #    ports moved into scratch (here http 127.0.0.1:18080), started with
    #    the recorded run:
    rust/target/release/saans-gateway --config $SCRATCH/sim.toml \
        --room-replay $SCRATCH/hall-with_walker-0.jsonl
    # 3. A signed-out view (--base-auth): another scratch copy, its own
    #    data_dir and ports (here 127.0.0.1:18081), plus
    #        [auth]
    #        required = true
    #        [[token]]
    #        name = "page-check"
    #        role = "admin"
    #        sha256 = "<sha256 of any token>"
    rust/target/release/saans-gateway --config $SCRATCH/auth.toml

Then:

    .venv/bin/python -I rust/crates/saans-gateway/tests/browser/roomsim_check.py \
        --base http://127.0.0.1:18080 --base-auth http://127.0.0.1:18081 --out /tmp/roomsim-shots

The simulation checks use the hall, `with_walker`, seed 0, where WLM, RTI
and RTT all publish early. Every timing they assert comes from the frames
themselves (`t` and `seq`, read from the page's `data-sim-*` test hooks),
never from wall time; wall time is only ever waited on. "The page does not
reconnect" is checked on Playwright's clock: ten minutes of the page's own
time pass, and no socket may open in them. The page opens at most a few
sessions at once, below the gateway's cap of 4, except where a check fills
the cap on purpose.

The page's read-only test hooks, on document.body.dataset: simState,
simMode, simSeq, simT, simRun, simLog (the last 120 frames drawn, as
seq:t:speed:paused:run), simRings, simTruth, simErr, simErrLen (each
error segment's drawn length, metres), simTrails, simLinks, simLayer (0
when cleared), simCleared (why) and simPlanFit (in the plan view, whether
every drawn ring is in it). None holds a position.

Needs Playwright (`pip install playwright` into .venv) and Pillow, and
Google Chrome; no Playwright browser download is used. Exits non-zero on
any failure and leaves screenshots in --out for a human to look at,
because a page can pass every check here and still look wrong.

Numbers on the page are SCENE (declared geometry) or SIM (simulated
people, estimated from simulated WiFi). This checks they are drawn and
labelled, not that any of them is true of a real room.
"""
import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from urllib.parse import urlsplit, parse_qsl

from playwright.sync_api import sync_playwright
from PIL import Image

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--base", default="http://127.0.0.1:18080", help="gateway origin")
ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "roomsim-shots"),
                help="directory for screenshots and results.json (default: outside the repo)")
ap.add_argument("--chrome", default="/usr/bin/google-chrome", help="Chrome executable")
ap.add_argument("--base-auth", default="http://127.0.0.1:18081",
                help="origin of a second gateway with [auth] required = true (the signed-out checks)")
args = ap.parse_args()
BASE = args.base.rstrip("/") + "/roomsim"
AUTH = args.base_auth.rstrip("/") + "/roomsim"
AUTH_HOST = urlsplit(args.base_auth).netloc
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

    # =================================================================
    # The simulation view (spec/livedemo.md §9; the checks of §11
    # "Browser"). Every timing asserted below is read from the frames the
    # page drew (`data-sim-log`: seq, t, speed, paused, run), never from
    # wall time; wall time is only ever waited on.
    # =================================================================
    HALL = "hall-l-shaped"
    TEXT_SIGN_IN = "Sign in at / with an admin or safety account to run the simulation (SIM)"
    TEXT_UNOPENED = ("The room stream could not be opened (signed out, not permitted, "
                     "or not built into this gateway)")
    TEXT_LEGEND = ("link disturbance: y ÷ WLM gate (SIM) — 0 at or below the empty room, "
                   "1 at the gate, 2 or more twice the gate")
    EST_LABEL = re.compile(r"(WLM \(baseline\)|RTI|phone RTT|phone RSSI) ± \d+\.\d\d m · model radius, "
                           r"uncalibrated \(SIM\)( · (covers the whole room|centre outside the room|"
                           r"false alarm — the room is empty \(SIM\)))*")
    ERR_LABEL = re.compile(r"(WLM|RTI) error to P\d+: \d+\.\d\d m \(SIM\)|"
                           r"phone (RTT|RSSI) error to the phone: \d+\.\d\d m \(SIM\)")
    HASH_KEYS = {"scene", "view", "dims", "select", "sim", "variant", "seed", "replay"}
    # Opens n extra sessions from the page's origin, one at a time, and
    # says what each was first told: the server's own session count.
    EXTRA = """async (n) => {
      window.checkSockets = window.checkSockets || [];
      const one = () => new Promise((res) => {
        const ws = new WebSocket('ws://' + location.host + '/api/roomsim?scene=bedroom&variant=empty');
        window.checkSockets.push(ws);
        ws.onmessage = (e) => { const m = JSON.parse(e.data); res(m.type === 'error' ? 'error:' + m.param : m.type + ':' + m.state); };
        ws.onclose = (e) => res('closed:' + e.code);
      });
      const out = [];
      for (let i = 0; i < n; i++) out.push(await one());
      return out;
    }"""
    template = {}  # a SIM frame of the hall, from the SIM session, for the protocol checks
    # Test doubles for the page's WebSocket, installed before the page's
    # own scripts. (Playwright's route_web_socket needs URL.parse, which
    # Chrome 123 lacks.) KEEP_SOCKETS keeps a handle on each real socket,
    # so a check can drop the connection under the page and count every
    # socket the page opened; SCRIPTED replaces the socket with one that
    # plays the given messages and records how the page closed it. LINKS
    # counts the shader programs WebGL links.
    KEEP_SOCKETS = """(() => {
      const Native = window.WebSocket;
      window.WebSocket = class extends Native {
        constructor(...a) { super(...a); (window.checkKept = window.checkKept || []).push(this); }
      };
    })();"""
    SCRIPTED = """((script) => {
      class Scripted extends EventTarget {
        constructor(url) {
          super();
          this.url = url;
          this.readyState = 0;
          window.checkClosed = null;
          setTimeout(() => {
            this.readyState = 1;
            this.dispatchEvent(new Event('open'));
            for (const m of script) if (this.readyState === 1) this.dispatchEvent(new MessageEvent('message', { data: m }));
          }, 0);
        }
        send() {}
        close(code) {
          if (this.readyState > 1) return;
          this.readyState = 3;
          window.checkClosed = code ?? 1005;
          this.dispatchEvent(new CloseEvent('close', { code: code ?? 1005 }));
        }
      }
      Object.assign(Scripted, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
      window.WebSocket = Scripted;
    })(SCRIPT);"""
    # Every animation frame, looks at the simulation labels as placed: an
    # estimate's under a corner box of the view, or any of them hidden by
    # the declutter, is counted (spec/livedemo.md §9.3; ADR-008 d5).
    WATCH_LABELS = """(() => {
      const u = window.checkLabels = { frames: 0, under: 0, hidden: 0, seen: [] };
      const tick = () => {
        const boxes = ['sim-overlay', 'overlay-key'].map((id) => document.getElementById(id))
          .filter((e) => e && e.getClientRects().length).map((e) => e.getBoundingClientRect());
        let under = false;
        for (const l of document.querySelectorAll('#labels .label.est[data-why=shown]')) {
          const a = l.getBoundingClientRect();
          for (const b of boxes) {
            if (a.left < b.right - 0.5 && a.right > b.left + 0.5 && a.top < b.bottom - 0.5 && a.bottom > b.top + 0.5) {
              under = true;
              if (u.seen.length < 3) u.seen.push(l.textContent.slice(0, 40));
            }
          }
        }
        const hidden = document.querySelectorAll('#labels .label.est[data-why=overlap], #labels .label.truth[data-why=overlap], #labels .label.err[data-why=overlap]').length;
        u.frames++;
        if (under) u.under++;
        if (hidden) u.hidden++;
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    })();"""
    LINKS = """(() => {
      window.checkLinks = 0;
      for (const C of [WebGLRenderingContext, WebGL2RenderingContext]) {
        const link = C.prototype.linkProgram;
        C.prototype.linkProgram = function (...a) { window.checkLinks++; return link.apply(this, a); };
      }
    })();"""

    def ds(pg, key):
        return pg.evaluate("(k) => document.body.dataset[k] ?? null", key)

    def sim_log(pg):
        """The (seq, t, speed, paused, run) of every frame the page drew lately."""
        out = []
        for e in (ds(pg, "simLog") or "").split():
            seq, t, speed, paused, run = e.split(":")
            out.append({"seq": int(seq), "t": float(t), "speed": float(speed), "paused": paused == "1", "run": int(run)})
        return out

    def wait_state(pg, states, timeout=30000):
        pg.wait_for_function("(s) => s.includes(document.body.dataset.simState)", arg=list(states), timeout=timeout)

    def wait_frames(pg, n, timeout=15000):
        """Wait until the page has drawn n frames more than it has now."""
        now = int(ds(pg, "simSeq") or -1)
        pg.wait_for_function("(s) => Number(document.body.dataset.simSeq ?? -1) >= s", arg=now + n, timeout=timeout)

    def settle(pg):
        pg.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")

    def sim_section(name, fn):
        """A section that times out or crashes fails its own check, not the run."""
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - report it and go on
            check(f"{name}: section completed", False, repr(e)[:300])

    def sim_ctx(**kw):
        """A context whose sockets are counted, on Playwright's clock (for no_reconnect)."""
        c = browser.new_context(**({"viewport": {"width": 1366, "height": 860}} | kw))
        c.add_init_script(script=KEEP_SOCKETS)
        c.clock.install()
        return c

    def sim_page(ctx, tag, frag, base=BASE):
        pg = ctx.new_page()
        pg.on("console", lambda m: results["console"].append(f"[{tag}] {m.type}: {m.text}"))
        pg.on("pageerror", lambda e: results["pageerrors"].append(f"[{tag}] {e}"))
        socks = []
        pg.on("websocket", lambda ws: socks.append(ws))
        pg.goto(f"{base}#{frag}")
        pg.wait_for_selector("body[data-ready=true]", timeout=20000)
        return pg, socks

    def shot(pg, name, **kw):
        settle(pg)
        pg.screenshot(path=f"{OUT}/{name}", **kw)

    def own(socks):
        """The page's own sockets, not the extra sessions a check opened."""
        return [w for w in socks if "variant=empty" not in w.url]

    def wait_closed(socks, timeout_s=5.0):
        end = time.monotonic() + timeout_s
        while time.monotonic() < end and not all(w.is_closed() for w in socks):
            time.sleep(0.05)
        return all(w.is_closed() for w in socks)

    def no_reconnect(pg, socks, n):
        """Ten minutes of the page's own time pass (Playwright's clock runs
        every timer due in them): no socket may open, so a retry on any
        back-off is caught, not only one within a wall-time wait."""
        opened = pg.evaluate("(window.checkKept || []).length")
        pg.clock.run_for(600_000)
        pg.wait_for_timeout(300)  # a socket a timer opened is seen by now
        return len(own(socks)) == n and pg.evaluate("(window.checkKept || []).length") == opened

    def overlaps(pg, sel_a, sel_b):
        """Pairs (a, b) of visible elements whose boxes overlap, as text."""
        return pg.evaluate("""([a, b]) => {
          const vis = (e) => e.getClientRects().length > 0;
          const out = [];
          for (const x of document.querySelectorAll(a)) for (const y of document.querySelectorAll(b)) {
            if (x === y || !vis(x) || !vis(y)) continue;
            const p = x.getBoundingClientRect(), q = y.getBoundingClientRect();
            if (p.left < q.right - 0.5 && p.right > q.left + 0.5 && p.top < q.bottom - 0.5 && p.bottom > q.top + 0.5)
              out.push([x.textContent.slice(0, 40), y.id || y.textContent.slice(0, 40)]);
          }
          return out;
        }""", [sel_a, sel_b])

    def stats_rules_hold(text):
        """§9.4: a median only from k >= 5, a p90 only from k >= 10."""
        bad = []
        for block in text.split("\n\n"):
            m = re.search(r"published (\d+) of", block)
            if not m or "error to" not in block:
                continue
            k = int(m.group(1))
            line = next(l for l in block.splitlines() if l.startswith("error to"))
            if k < 5 and ("median" in line or f"too few to summarise ({k})" not in line):
                bad.append(line)
            elif 5 <= k < 10 and ("median" not in line or f"p90: too few to summarise ({k})" not in line):
                bad.append(line)
            elif k >= 10 and ("median" not in line or "p90 " not in line):
                bad.append(line)
        return bad

    # ---- SIM, desktop: the hall, with_walker, seed 0 ----
    def sim_hall():
        ctx = sim_ctx()
        ctx.add_init_script(script=LINKS)
        try:
            pg, socks = sim_page(ctx, "sim", f"scene={HALL}")
            received = []
            pg.on("websocket", lambda ws: ws.on("framereceived", lambda m: received.append(m)))
            check("sim: the static view opens no socket until the simulation is asked for",
                  not socks and pg.locator("#sim-card").is_hidden()
                  and pg.get_attribute("#sim-toggle", "aria-pressed") == "false", len(socks))
            check("sim: the static view says people appear only in the SIM view",
                  "appear only in the SIM view" in pg.locator("footer").inner_text()
                  and "appear only in the SIM view" in pg.locator("#scene-info").inner_text())
            pg.click("#sim-toggle")
            check("sim: 'Run simulation (SIM)' puts sim=1 in the URL", "sim=1" in pg.url, pg.url)
            check("sim: the header gains the SIM badge",
                  pg.locator("#sim-badge").inner_text() == "SIM · simulated people, estimates from simulated WiFi")
            wait_state(pg, ["running"], 30000)
            check("sim: the hall is running within 30 s", ds(pg, "simState") == "running")
            check("sim: the mode badge is the stream's 'SIM'",
                  pg.locator("#mode-badge").inner_text() == "SIM" and ds(pg, "simMode") == "sim")
            aria = pg.evaluate("""() => {
                const live = '[aria-live], [role=status], [role=log], [role=alert]';
                return [document.getElementById('sim-announce').getAttribute('aria-live'),
                  ['sim-card', 'sim-stats-card'].some((id) => document.getElementById(id).closest(live)
                    || document.getElementById(id).querySelector(live)),
                  document.getElementById('sim-announce').textContent]; }""")
            check("sim: state changes go to a polite region; the figures are in no live region, above or within",
                  aria == ["polite", False, "Simulation running (SIM)"], aria)
            check("sim: the canvas description says estimates are drawn in the SIM view",
                  "Simulation view on" in (pg.get_attribute("canvas", "aria-label") or ""))
            pg.wait_for_function("() => Number(document.body.dataset.simRings) >= 2", timeout=20000)
            pg.wait_for_function("() => document.getElementById('sim-now').innerText.includes('abstained — ')", timeout=40000)
            seq, now = pg.evaluate("() => [Number(document.body.dataset.simSeq), document.getElementById('sim-now').innerText]")
            check("sim: an abstention is said as text in the panel", "abstained — " in now, now[:300])
            check("sim: an abstention is said in the view too",
                  "abstained — " in pg.locator("#sim-overlay").inner_text())
            m = re.search(r"links over the WLM gate: (\d+) of (\d+) pairs", now)
            def drawn_frame(seq):
                """The frame the page drew as `seq`, as it came off the wire."""
                for _ in range(40):  # its framereceived event may trail the page by a moment
                    f = next((x for x in map(json.loads, (r for r in received if '"type":"frame"' in r)) if x["seq"] == seq), None)
                    if f:
                        return f
                    pg.wait_for_timeout(50)
                return None
            f = drawn_frame(seq)
            pairs = {}
            for it in (f or {}).get("links", {}).get("items", []):
                key = (min(it["tx"], it["rx"]), max(it["tx"], it["rx"]))
                pairs[key] = pairs.get(key, False) or it["disturbed"] is True
            check("sim: 'links over the WLM gate: k of K pairs' counts the drawn frame's disturbed pairs",
                  m and f and (int(m.group(1)), int(m.group(2))) == (sum(pairs.values()), len(pairs)) == (sum(pairs.values()), 28),
                  (m and m.groups(), sum(pairs.values()), len(pairs)))
            check("sim: each estimate is listed as (x, y) ± r m or its abstention",
                  all(re.search(rf"{re.escape(n)}: (\(-?\d+\.\d\d, -?\d+\.\d\d\) ± \d+\.\d\d m|abstained — |estimate not drawn|not drawn)", now)
                      for n in ("WLM (baseline)", "RTI", "phone RTT", "phone RSSI")), now[:400])
            key = pg.locator("#sim-key").inner_text()
            check("sim: the key shows each line style and the link legend",
                  all(s in key for s in ("WLM (baseline): solid", "RTI: dashed", "phone RTT: dotted", "phone RSSI: dash-dot", "truth: a cross"))
                  and TEXT_LEGEND in key, key[:200])
            stats = pg.locator("#sim-stats").inner_text()
            heads = pg.locator("#sim-stats h4").all_inner_texts()
            check("sim: one block per estimator, WLM beside RTI",
                  heads == ["WLM (baseline) · SIM", "RTI · SIM", "phone RTT · SIM", "phone RSSI · SIM"], heads)
            items = pg.locator("#sim-stats li").all_inner_texts()
            check("sim: every figure in the statistics is labelled SIM", items and all("SIM" in t for t in items),
                  [t for t in items if "SIM" not in t][:3])
            check("sim: coverage, the empty room and the reference are beside each person estimator",
                  stats.count("empty room: ") == 2 and "published ticks/h" in stats and "events/h (SIM)" in stats
                  and stats.count("reference: the floor's centroid with the same radius — an estimator that knows nothing") == 4)
            bad = stats_rules_hold(pg.evaluate("[...document.querySelectorAll('#sim-stats .est-block')].map(b => b.innerText).join('\\n\\n')"))
            check("sim: a median only from 5 published, a p90 only from 10", not bad, bad[:2])
            check("sim: truth is off by default", ds(pg, "simTruth") == "0" and pg.locator(".label.truth").count() == 0
                  and pg.get_attribute("#sim-truth", "aria-pressed") == "false")
            shot(pg, "sim-hall-running.png")

            # Pause: t stops, the heartbeat repeats it, the state is announced.
            pg.click("#sim-play")
            pg.wait_for_function("""() => (document.body.dataset.simLog || '').trim().split(' ')
                .slice(-2).filter(e => e.split(':')[3] === '1').length === 2""", timeout=10000)
            log = sim_log(pg)
            i0 = next(i for i, e in enumerate(log) if e["paused"])
            held = log[i0:]
            check("sim: t is unchanged while paused (from the frames)",
                  i0 > 0 and len(held) >= 2 and all(e["paused"] and e["t"] == log[i0 - 1]["t"] for e in held)
                  and all(b["seq"] == a["seq"] + 1 for a, b in zip(log[i0 - 1:], held)),
                  [(e["seq"], e["t"], e["paused"]) for e in log[i0 - 1:]])
            check("sim: pausing is announced", pg.locator("#sim-announce").text_content() == "Simulation paused (SIM)")
            # A paused session still sends a frame every 100 ms, each drawn
            # anew: the materials outlive frames, so no shader is relinked.
            links0 = pg.evaluate("window.checkLinks")
            wait_frames(pg, 4)
            links1 = pg.evaluate("window.checkLinks")
            check("sim: redrawing a frame links no shader program (materials are reused)", links1 == links0, (links0, links1))
            # While paused the drawing is steady: check the labels now.
            pg.wait_for_function("""() => { const ls = [...document.querySelectorAll('.label.est')];
                return ls.length === Number(document.body.dataset.simRings) && ls.every(l => l.dataset.why === 'shown'); }""",
                                 timeout=5000)
            labels = pg.locator(".label.est").all_inner_texts()
            check("sim: every ring has its label, never hidden by the declutter",
                  len(labels) == int(ds(pg, "simRings")) >= 2, labels)
            check("sim: each label reads '<name> ± r m · model radius, uncalibrated (SIM)'",
                  all(EST_LABEL.fullmatch(t.replace("\n", " ")) for t in labels), labels)
            # The Dimensions toggle hides dimensions, never the estimates or the key.
            pg.click("#dims-toggle")
            # A paused session still redraws each heartbeat, and a frame's
            # labels stay hidden until placed: read a placed frame.
            shown = pg.wait_for_function("""() => { const ls = [...document.querySelectorAll('.label.est')];
                return ls.length && ls.every((l) => l.dataset.why) ? [ls.filter((l) => !l.classList.contains('hidden')).length,
                  document.querySelectorAll('.label.dim:not(.hidden)').length] : null; }""", timeout=5000).json_value()
            check("sim: the Dimensions toggle leaves estimate labels and the key alone",
                  shown == [len(labels), 0] and pg.locator("#sim-key").is_visible(), shown)
            pg.click("#dims-toggle")
            pg.click("#sim-play")
            wait_frames(pg, 2)
            check("sim: playing again moves t", sim_log(pg)[-1]["t"] > log[-1]["t"])

            # Restart: a new run from t = 0, and a cleared layer first.
            run0 = int(ds(pg, "simRun"))
            pg.click("#sim-restart")
            pg.wait_for_function("(r) => Number(document.body.dataset.simRun) === r", arg=run0 + 1, timeout=10000)
            log = sim_log(pg)
            first = min((e for e in log if e["run"] == run0 + 1), key=lambda e: e["seq"])
            check("sim: restart raises run.index and returns t to 0", first["t"] == 0.0, first)
            check("sim: a new run clears trails, circles, truth and figures first", ds(pg, "simCleared") == "run")

            # Each speed: t[seq+1] - t[seq] = speed x 0.1 s, from the frames.
            for v in ("0.25", "0.5", "1", "2", "4", "8"):
                s = float(v)
                pg.select_option("#sim-speed", v)
                pg.wait_for_function("""(s) => (document.body.dataset.simLog || '').trim().split(' ')
                    .filter(e => Number(e.split(':')[2]) === s && e.split(':')[3] === '0').length >= 5""", arg=s, timeout=10000)
                log = sim_log(pg)
                steps = [(a, b) for a, b in zip(log, log[1:])
                         if b["seq"] == a["seq"] + 1 and b["run"] == a["run"] and b["speed"] == s and not b["paused"]]
                bad = [(b["seq"], b["t"] - a["t"]) for a, b in steps if abs((b["t"] - a["t"]) - s * 0.1) > 1e-9]
                check(f"sim: at {v}x every frame advances t by {s * 0.1:g} s (from seq and t)",
                      len(steps) >= 3 and not bad, f"{len(steps)} steps; off: {bad[:3]}")
            pg.select_option("#sim-speed", "1")
            wait_frames(pg, 3)

            # Truth (SIM only): crosses, error segments, the people count.
            pg.click("#sim-truth")
            pg.wait_for_function("() => Number(document.body.dataset.simTruth) >= 2", timeout=10000)
            pg.wait_for_function("() => document.getElementById('sim-now').innerText.includes('Simulated people: ')", timeout=10000)
            truth = pg.locator(".label.truth").all_inner_texts()
            check("sim: truth on draws each simulated person as a cross labelled 'P<i+1> · truth (SIM)'",
                  sum(bool(re.fullmatch(r"P\d+ · truth \(SIM\)", t)) for t in truth) == 3, truth)
            check("sim: truth on draws the phone as 'phone · truth (SIM)'", "phone · truth (SIM)" in truth, truth)
            check("sim: truth on says 'Simulated people: N'", "Simulated people: 3" in pg.locator("#sim-now").inner_text())
            pg.wait_for_function("() => Number(document.body.dataset.simErr) >= 1", timeout=20000)
            # One frame as placed, with a person's error segment and the
            # phone's: a frame drawn since the last placement has labels
            # with no data-why yet, so wait for one that has.
            drawn = pg.wait_for_function("""() => {
              const ls = [...document.querySelectorAll('.label.truth, .label.err')];
              if (!ls.length || ls.some((e) => !e.dataset.why)) return null;
              const errs = [...document.querySelectorAll('.label.err')].map((e) => e.textContent);
              if (!errs.some((t) => / error to P\\d+: /.test(t)) || !errs.some((t) => t.includes('to the phone'))) return null;
              return { truth: Number(document.body.dataset.simTruth), err: Number(document.body.dataset.simErr),
                lens: (document.body.dataset.simErrLen || '').split(' ').filter(Boolean).map(Number),
                errs: [...document.querySelectorAll('.label.err')].map((e) => [e.textContent, e.dataset.why]),
                truths: [...document.querySelectorAll('.label.truth')].map((e) => [e.textContent, e.dataset.why]) }; }""",
                timeout=40000).json_value()
            errs = [t for t, _ in drawn["errs"]]
            check("sim: each scored estimate gets an error segment labelled like 'WLM error to P2: 1.23 m (SIM)'",
                  errs and all(ERR_LABEL.fullmatch(t) for t in errs), errs)
            check("sim: no truth cross or error segment is drawn without its label shown (ADR-008 d5)",
                  len(drawn["truths"]) == drawn["truth"] and len(drawn["errs"]) == drawn["err"]
                  and all(w == "shown" for _, w in drawn["truths"] + drawn["errs"]), drawn)
            printed = [float(re.search(r": (\d+\.\d\d) m \(SIM\)$", t).group(1)) for t in errs]
            check("sim: each error segment is drawn as long as the error printed on it",
                  printed and len(drawn["lens"]) == len(printed)
                  and all(abs(a - b) <= 0.0051 for a, b in zip(drawn["lens"], printed)), list(zip(drawn["lens"], printed)))
            # Each drawn segment joins an estimate to the record's own truth,
            # so its length is the printed error: checked on the frame drawn.
            f = drawn_frame(int(ds(pg, "simSeq")))
            frames = [json.loads(x) for x in received if '"type":"frame"' in x]
            gaps = []
            if f and f.get("persons") and f["truth"].get("persons"):
                for k in ("wlm", "rti"):
                    est, sc = f["persons"][k].get("estimate"), f["truth"]["persons"][k]
                    if est and "error_m" in sc:
                        p = next(q for q in f["truth"]["persons"]["people"] if q["i"] == sc["person"])
                        gaps.append(abs(((est["x"] - p["x"]) ** 2 + (est["y"] - p["y"]) ** 2) ** 0.5 - sc["error_m"]))
            if f and f.get("phone") and f["truth"].get("phone"):
                tp = f["truth"]["phone"]
                for k in ("rtt", "rssi"):
                    est, sc = f["phone"][k].get("estimate"), tp[k]
                    if est and "error_m" in sc:
                        gaps.append(abs(((est["x"] - tp["x"]) ** 2 + (est["y"] - tp["y"]) ** 2) ** 0.5 - sc["error_m"]))
            check("sim: the frame's own truth is as far from each estimate as its printed error",
                  f and gaps and all(g < 1e-9 for g in gaps), gaps)
            shot(pg, "sim-hall-truth.png")
            for x in reversed(frames):
                if (x.get("persons") and x["persons"]["wlm"].get("estimate") and x.get("phone")
                        and (x["phone"]["rtt"].get("estimate") or x["phone"]["rssi"].get("estimate"))):
                    template["frame"] = x
                    break
            pg.click("#sim-truth")
            wait_frames(pg, 2)
            check("sim: truth off removes every cross and segment",
                  ds(pg, "simTruth") == "0" and ds(pg, "simErr") == "0" and pg.locator(".label.truth, .label.err").count() == 0)

            # Nothing position-like leaves the page: the URL, storage, the console.
            keys = {k for k, _ in parse_qsl(urlsplit(pg.url).fragment)}
            check("sim: the URL holds only view keys, never a position or a time", keys <= HASH_KEYS, keys)
            check("sim: nothing is kept in browser storage",
                  pg.evaluate("localStorage.length + sessionStorage.length") == 0)
            check("sim: no text on the page calls the stream live",
                  re.search(r"\blive\b", pg.locator("body").inner_text(), re.I) is None)

            # A scene change: one session, for the new scene, and a cleared layer.
            before = [w for w in own(socks) if not w.is_closed()]
            check("sim: one session is open before the scene change", len(before) == 1, [w.url for w in before])
            pg.select_option("#scene-picker", "office")
            pg.wait_for_selector("body[data-scene='office']")
            cleared = [ds(pg, "simLayer"), ds(pg, "simRings"), ds(pg, "simCleared"), pg.locator(".label.est").count()]
            check("sim: a scene change clears the layer", cleared == ["0", "0", "scene", 0], cleared)
            wait_state(pg, ["running"], 30000)
            check("sim: the old session is closed", wait_closed(before))
            opened = [w for w in own(socks) if not w.is_closed()]
            check("sim: after a scene change the page holds exactly one open session, for the new scene",
                  len(opened) == 1 and "scene=office" in opened[0].url, [w.url for w in opened])
            # And so does the gateway: 3 more sessions fit, a 4th does not.
            pg.wait_for_timeout(1000)  # a closed session's thread exits within 0.5 s (§2.4)
            three = pg.evaluate(EXTRA, 3)
            fourth = pg.evaluate(EXTRA, 1)
            check("sim: the gateway counts one session for this page (3 more open, a 4th is refused)",
                  three == ["state:calibrating"] * 3 and fourth == ["error:session"], (three, fourth))
            # The page's own refusal: error, then close 1013, shows the message.
            pg.click("#sim-toggle")
            check("sim: turning the view off closes the session and clears the layer",
                  wait_closed(own(socks)) and ds(pg, "simLayer") == "0" and pg.locator("#sim-card").is_hidden()
                  and "sim=1" not in pg.url)
            # The office was framed as it loaded, with none of the hall's
            # rings (they were no longer drawn): the very view a fresh load
            # of the office gets. Turning the view off moves no camera.
            def dims_at(q):
                settle(q)
                return q.evaluate("""() => [...document.querySelectorAll('.label.dim[data-why=shown]')]
                    .map((e) => e.textContent + '@' + e.style.transform).sort().join(' | ')""")
            fresh = ctx.new_page()
            fresh.goto(f"{BASE}#scene=office")
            fresh.wait_for_selector("body[data-ready=true]")
            switched, loaded = dims_at(pg), dims_at(fresh)
            fresh.close()
            check("sim: a scene change frames the new room as a fresh load does, not around the old rings",
                  switched and switched == loaded, (switched[:160], loaded[:160]))
            pg.wait_for_timeout(1000)
            check("sim: the slot was returned", pg.evaluate(EXTRA, 1) == ["state:calibrating"])
            n = len(own(socks))
            pg.click("#sim-toggle")
            wait_state(pg, ["refused"], 10000)
            msg = pg.locator("#sim-msg").inner_text()
            check("sim: a fifth session shows the gateway's message (error, then close 1013)",
                  msg == "4 room simulations are running; try again later" and ds(pg, "simLayer") == "0", msg)
            said = pg.locator("#sim-announce").text_content()
            check("sim: the refusal is announced, so no one believes the run goes on",
                  said == "Simulation stopped — nothing is shown: 4 room simulations are running; try again later", said)
            check("sim: the page does not reconnect on its own", no_reconnect(pg, socks, n + 1), len(own(socks)))
            pg.evaluate("() => window.checkSockets.forEach((w) => w.close())")
            pg.wait_for_timeout(1000)
        finally:
            ctx.close()

    # ---- SIM, desktop: the bedroom as declared; the hash keys; a refused parameter ----
    def sim_bedroom():
        ctx = sim_ctx()
        try:
            pg, socks = sim_page(ctx, "sim-bedroom", "scene=bedroom&sim=1&variant=declared")
            wait_state(pg, ["running"], 30000)
            pg.wait_for_function("() => document.getElementById('sim-now').innerText.includes('WLM (baseline): ')", timeout=20000)
            check("sim/bedroom: #variant=declared is the control's value and the panel's",
                  pg.eval_on_selector("#sim-variant", "e => e.value") == "declared"
                  and "declared · seed 0" in pg.locator("#sim-facts").inner_text())
            check("sim/bedroom: the session asked for what the URL says",
                  "variant=declared" in own(socks)[-1].url and "seed=0" in own(socks)[-1].url, own(socks)[-1].url)
            shot(pg, "sim-bedroom-declared.png")
            for b in ("#dims-toggle", "#view-toggle", "#dims-toggle", "#view-toggle"):
                pg.click(b)
            check("sim/bedroom: the simulation keys survive the page's own hash rewrites",
                  "sim=1" in pg.url and "variant=declared" in pg.url, pg.url)
            pg.fill("#sim-seed", "5")
            pg.press("#sim-seed", "Enter")
            check("sim/bedroom: a seed typed in the panel goes into the URL", "seed=5" in pg.url, pg.url)
            wait_state(pg, ["running"], 30000)
            check("sim/bedroom: a new seed is a new session (§3.3)",
                  "seed=5" in own(socks)[-1].url and "declared · seed 5" in pg.locator("#sim-facts").inner_text()
                  and wait_closed(own(socks)[:-1]))
            n = len(own(socks))
            pg.goto(f"{BASE}#scene=bedroom&sim=1&variant=walker")
            wait_state(pg, ["refused"], 15000)
            msg = pg.locator("#sim-msg").inner_text()
            check("sim/bedroom: a bad parameter shows the gateway's message (error, then close 1008)",
                  msg == "variant must be declared, with_walker or empty" and ds(pg, "simLayer") == "0", msg)
            check("sim/bedroom: the refusal is announced",
                  pg.locator("#sim-announce").text_content()
                  == "Simulation stopped — nothing is shown: variant must be declared, with_walker or empty")
            check("sim/bedroom: after a refusal the page does not reconnect", no_reconnect(pg, socks, n + 1), len(own(socks)))
        finally:
            ctx.close()

    # ---- SIM in the plan view, opened from the URL: every ring is in it ----
    def sim_plan():
        ctx = browser.new_context(viewport={"width": 1366, "height": 860})
        try:
            pg, socks = sim_page(ctx, "sim-plan", "scene=bedroom&sim=1&view=top")
            wait_state(pg, ["running"], 30000)
            pg.wait_for_function("() => Number(document.body.dataset.simRings) >= 2", timeout=20000)
            fits = []
            for _ in range(8):
                wait_frames(pg, 3)
                fits.append(ds(pg, "simPlanFit"))
            check("sim/plan: opened from the URL, the plan view keeps every drawn ring in it (§9.3)",
                  set(fits) == {"all"}, fits)
            shot(pg, "sim-plan.png")
            # A viewer's pan is theirs: the page does not undo it; Reset view fits again.
            vp = pg.locator("#viewport").bounding_box()
            pg.mouse.move(vp["x"] + vp["width"] * 0.2, vp["y"] + vp["height"] / 2)
            pg.mouse.down()
            pg.mouse.move(vp["x"] + vp["width"] * 0.85, vp["y"] + vp["height"] / 2, steps=8)
            pg.mouse.up()
            wait_frames(pg, 3)
            panned = ds(pg, "simPlanFit")
            pg.click("#reset-view")
            wait_frames(pg, 3)
            check("sim/plan: a viewer's pan is left alone, and Reset view fits every ring again",
                  panned == "clipped" and ds(pg, "simPlanFit") == "all", (panned, ds(pg, "simPlanFit")))
        finally:
            ctx.close()

    # ---- REPLAY, desktop; then the server goes away ----
    def sim_replay():
        ctx = sim_ctx()
        from_server = []
        ctx.on("page", lambda pg: pg.on("websocket", lambda ws: ws.on("framereceived", lambda m: from_server.append(m))))
        try:
            # A variant and seed asked for in REPLAY are not what runs.
            pg, socks = sim_page(ctx, "sim-replay", f"scene={HALL}&sim=1&replay=1&variant=empty&seed=7")
            wait_state(pg, ["running"], 30000)
            pg.wait_for_function("() => Number(document.body.dataset.simRings) >= 1", timeout=20000)
            check("replay: the mode badge is 'REPLAY · recorded SIM run'",
                  pg.locator("#mode-badge").inner_text() == "REPLAY · recorded SIM run" and ds(pg, "simMode") == "replay")
            check("replay: estimates are drawn with their rings", int(ds(pg, "simRings")) >= 1)
            check("replay: truth is not offered, and the page says so",
                  pg.locator("#sim-truth").is_hidden() and pg.locator("#sim-notruth").inner_text() == "no truth in REPLAY")
            check("replay: no truth and no error is drawn",
                  ds(pg, "simTruth") == "0" and ds(pg, "simErr") == "0" and pg.locator(".label.truth, .label.err").count() == 0)
            stats = pg.locator("#sim-stats").inner_text()
            check("replay: the statistics show coverage, the radius median and the empty room, and no error figures",
                  all(s in stats for s in ("published", "radius median", "empty room: ", "no truth in REPLAY"))
                  and not any(s in stats for s in ("error to", "p90", "reference:", "inside")),
                  stats[:300])
            frames = [json.loads(m) for m in from_server if isinstance(m, str) and '"type":"frame"' in m]
            check("replay: no frame from the gateway carries a truth key", frames and not any("truth" in f for f in frames), len(frames))
            check("replay: the variant and seed are the recorded run's",
                  "with_walker · seed 0 (fixed by the recorded run)" in pg.locator("#sim-facts").inner_text()
                  and pg.locator("#sim-variant").is_disabled() and pg.locator("#sim-seed").is_disabled())
            ctl = [pg.eval_on_selector("#sim-variant", "e => e.value"), pg.eval_on_selector("#sim-seed", "e => e.value")]
            keys = dict(parse_qsl(urlsplit(pg.url).fragment))
            check("replay: the controls show the recorded run, never the variant and seed asked for",
                  ctl == ["with_walker", "0"], ctl)
            check("replay: the URL keeps no variant or seed, which a recording fixes",
                  keys.get("replay") == "1" and "variant" not in keys and "seed" not in keys, keys)
            shot(pg, "sim-replay.png")
            # The connection drops under the page: the layer is cleared and
            # nothing is held.
            pg.evaluate("window.checkKept[window.checkKept.length - 1].close(4001)")
            wait_state(pg, ["disconnected"], 10000)
            check("replay: a closed socket clears the layer and says 'Disconnected — nothing is shown'",
                  pg.locator("#sim-msg").inner_text() == "Disconnected — nothing is shown"
                  and ds(pg, "simLayer") == "0" and ds(pg, "simRings") == "0"
                  and pg.locator("#sim-now li").count() == 0 and pg.locator("#sim-stats .est-block").count() == 0)
            check("replay: disconnecting is announced",
                  pg.locator("#sim-announce").text_content() == "Simulation disconnected — nothing is shown")
            corner = pg.locator("#sim-overlay").inner_text().splitlines()
            check("replay: the view's corner keeps no line of the last frame",
                  corner == ["Disconnected — nothing is shown", "no truth in REPLAY"], corner)
            shot(pg, "sim-disconnected.png")
            check("replay: the page does not reconnect on its own", no_reconnect(pg, socks, 1), len(socks))
        finally:
            ctx.close()

    # ---- SIM at phone width ----
    def sim_phone():
        ctx = browser.new_context(viewport={"width": 375, "height": 812}, device_scale_factor=1, is_mobile=True, has_touch=True)
        ctx.add_init_script(script=WATCH_LABELS)
        try:
            pg, socks = sim_page(ctx, "sim-phone", f"scene={HALL}&sim=1")
            wait_state(pg, ["running"], 30000)
            pg.wait_for_function("() => Number(document.body.dataset.simRings) >= 1", timeout=20000)
            settle(pg)
            sw = pg.evaluate("[document.documentElement.scrollWidth, document.documentElement.clientWidth]")
            check("sim/phone: nothing overflows at 375 px", sw[0] <= sw[1], sw)
            geo = pg.evaluate("""() => {
              const W = document.documentElement.clientWidth;
              const wide = [...document.querySelectorAll('#sim-card *, #sim-stats-card *, header *')]
                .filter(e => e.getClientRects().length && e.getBoundingClientRect().right > W + 0.5).map(e => e.id || e.tagName);
              const lefts = [...document.querySelectorAll('#sim-stats .est-block')].map(b => Math.round(b.getBoundingClientRect().left));
              const vp = document.getElementById('viewport').getBoundingClientRect();
              const out = [...document.querySelectorAll('.label.est:not(.hidden)')].filter(l => {
                const r = l.getBoundingClientRect();
                return r.left < vp.left - 0.5 || r.right > vp.right + 0.5 || r.top < vp.top - 0.5 || r.bottom > vp.bottom + 0.5; }).length;
              const f = document.querySelector('footer').getBoundingClientRect();
              const p = document.querySelector('.panel').getBoundingClientRect();
              return {wide, lefts, out, footerAfter: f.top + scrollY >= p.bottom + scrollY - 0.5};
            }""")
            check("sim/phone: every control and figure fits the width", not geo["wide"], geo["wide"][:5])
            check("sim/phone: the estimator blocks stack", len(geo["lefts"]) == 4 and len(set(geo["lefts"])) == 1, geo["lefts"])
            check("sim/phone: estimate labels stay inside the view", geo["out"] == 0, geo)
            check("sim/phone: the footer comes after the panel", geo["footerAfter"], geo)
            shot(pg, "sim-phone.png", full_page=True)
            # Most of a run at 8x with truth on, every frame looked at as
            # placed: the crowded phone view is where labels collide.
            pg.evaluate("window.scrollTo(0, 0)")
            pg.select_option("#sim-speed", "8")
            pg.click("#sim-truth")
            pg.evaluate("() => Object.assign(window.checkLabels, { frames: 0, under: 0, hidden: 0, seen: [] })")
            pg.wait_for_function("() => Number(document.body.dataset.simT) >= 100 || document.body.dataset.simState === 'ended'",
                                 timeout=30000)
            u = pg.evaluate("window.checkLabels")
            check("sim/phone: over a run at 8x with truth on, no estimate label ever lies under a corner box",
                  u["frames"] > 100 and u["under"] == 0, u)
            check("sim/phone: over the same run, no estimate, truth or error label is ever hidden by the declutter",
                  u["frames"] > 100 and u["hidden"] == 0, u)
        finally:
            ctx.close()

    # ---- The page's side of the protocol, against a scripted stream ----
    def sim_protocol():
        f0 = template.get("frame")
        check("protocol: a SIM frame of the hall was captured to script from", f0 is not None)
        if f0 is None:
            return
        def frame(seq, **change):
            f = json.loads(json.dumps(f0))
            f["seq"] = seq
            f["new"] = {"persons": [], "phone": []}
            for k, v in change.items():
                f[k] = v
            return f
        def state(mode="sim"):
            return {"type": "state", "state": "calibrating", "mode": mode, "label": "SIM", "scene": f0["scene"], "run": f0["run"]}
        def scripted(name, msgs, frag=f"scene={HALL}&sim=1", **kw):
            ctx = browser.new_context(**({"viewport": {"width": 1366, "height": 860}} | kw))
            ctx.add_init_script(script=SCRIPTED.replace("SCRIPT", json.dumps([json.dumps(m) for m in msgs])))
            pg, _ = sim_page(ctx, f"sim-protocol-{name}", frag)
            return ctx, pg, lambda: pg.evaluate("window.checkClosed")
        def estimate(radius):
            e = dict(f0["persons"]["wlm"]["estimate"])
            e["radius_m"] = radius
            return {"estimate": e}

        ctx, pg, closed = scripted("gap", [state(), frame(0), frame(2)])
        try:
            wait_state(pg, ["stopped"], 10000)
            check("protocol: a gap in seq stops the stream: 'frames lost — stopped', cleared, closed",
                  pg.locator("#sim-msg").inner_text() == "frames lost — stopped" and ds(pg, "simLayer") == "0"
                  and closed() == 1000, closed())
            said = pg.locator("#sim-announce").text_content()
            check("protocol: the stop is announced, not left at 'running'",
                  said == "Simulation stopped — nothing is shown: frames lost — stopped", said)
            corner = pg.locator("#sim-overlay").inner_text().splitlines()
            check("protocol: the view's corner keeps no line of the cleared frame", corner == ["frames lost — stopped"], corner)
        finally:
            ctx.close()
        ctx, pg, closed = scripted("truth", [state("replay"), frame(0, mode="replay")])
        try:
            wait_state(pg, ["stopped"], 10000)
            check("protocol: truth outside SIM stops the stream, closes it and draws nothing",
                  "ground truth outside SIM" in pg.locator("#sim-msg").inner_text() and ds(pg, "simLayer") == "0"
                  and ds(pg, "simRings") == "0" and closed() == 1000, (pg.locator("#sim-msg").inner_text(), closed()))
        finally:
            ctx.close()
        ctx, pg, closed = scripted("mode", [frame(0, mode="demo")])
        try:
            wait_state(pg, ["stopped"], 10000)
            badge = pg.locator("#mode-badge")
            check("protocol: an unknown mode is the 'unknown mode' error, never a mode",
                  badge.inner_text() == "unknown mode" and "mode-unknown" in badge.get_attribute("class")
                  and ds(pg, "simLayer") == "0" and closed() == 1000, badge.inner_text())
        finally:
            ctx.close()
        ctx, pg, closed = scripted("live", [frame(0, mode="live")])
        try:
            wait_state(pg, ["stopped"], 10000)
            check("protocol: a LIVE frame is badged LIVE and not drawn",
                  pg.locator("#mode-badge").inner_text() == "LIVE" and ds(pg, "simRings") == "0" and closed() == 1000)
        finally:
            ctx.close()
        ctx, pg, closed = scripted("live-state", [state("live")])
        try:
            wait_state(pg, ["stopped"], 10000)
            # The badge reads as the error it is, in the --bad tokens.
            bad = pg.evaluate("""() => { const p = document.createElement('span');
                p.style.color = 'var(--bad)'; document.body.append(p);
                const want = getComputedStyle(p).color; p.remove();
                return getComputedStyle(document.getElementById('mode-badge')).color === want; }""")
            check("protocol: a LIVE message of any type stops the stream, its badge an error",
                  pg.locator("#mode-badge").inner_text() == "LIVE" and bad and closed() == 1000
                  and "LIVE" in pg.locator("#sim-msg").inner_text(), pg.locator("#sim-msg").inner_text())
        finally:
            ctx.close()
        calibrated = dict(f0["persons"], wlm={"estimate": dict(f0["persons"]["wlm"]["estimate"], radius_calibrated=True)})
        ctx, pg, closed = scripted("calibrated", [state(), frame(0, persons=calibrated)])
        try:
            wait_state(pg, ["stopped"], 10000)
            check("protocol: a radius said to be calibrated stops the stream (Phase 5 calibrates none)",
                  "calibrated" in pg.locator("#sim-msg").inner_text() and ds(pg, "simRings") == "0"
                  and pg.locator(".label.est").count() == 0 and closed() == 1000, pg.locator("#sim-msg").inner_text())
        finally:
            ctx.close()
        other = dict(f0["scene"], sha256="0" * 64)
        ctx, pg, closed = scripted("scene", [frame(0, scene=other)])
        try:
            wait_state(pg, ["other-scene"], 10000)
            check("protocol: a frame of another scene is not drawn: 'Stream is for another scene'",
                  pg.locator("#sim-msg").inner_text() == "Stream is for another scene" and ds(pg, "simRings") == "0")
        finally:
            ctx.close()
        ctx, pg, closed = scripted("ended", [state(), frame(0, persons=dict(f0["persons"], wlm=estimate(2.0)), t=120.0, ended=True)])
        try:
            wait_state(pg, ["ended"], 10000)
            check("protocol: after `ended` the last frame stays drawn: 'Ended at t = 120.0 s (SIM) — Restart to run again'",
                  pg.locator("#sim-msg").inner_text() == "Ended at t = 120.0 s (SIM) — Restart to run again"
                  and int(ds(pg, "simRings")) >= 1 and pg.locator("#sim-announce").text_content() == "Simulation ended (SIM)")
        finally:
            ctx.close()
        persons = dict(f0["persons"], wlm=estimate(0.2), rti=estimate(1.0))
        ctx, pg, closed = scripted("radius", [frame(0, persons=persons)])
        try:
            wait_state(pg, ["running", "paused", "ended"], 10000)
            now = pg.locator("#sim-now").inner_text()
            check("protocol: a radius below 0.3 m is not drawn: 'estimate not drawn: radius'",
                  "WLM (baseline): estimate not drawn: radius (SIM)" in now and pg.locator(".label.est-wlm").count() == 0, now[:300])
            check("protocol: RTI is not drawn without its WLM baseline",
                  "RTI: not drawn — the WLM baseline of this tick is not drawn (SIM)" in now
                  and pg.locator(".label.est-rti").count() == 0)
        finally:
            ctx.close()
        persons = dict(f0["persons"], wlm=estimate(4.0))
        ctx, pg, closed = scripted("empty", [frame(0, persons=persons, run=dict(f0["run"], variant="empty"))])
        try:
            wait_state(pg, ["running", "paused", "ended"], 10000)
            settle(pg)
            wlm = pg.locator(".label.est-wlm").all_inner_texts()
            heads = pg.locator("#sim-stats h4").all_inner_texts()
            check("protocol: in variant empty a person estimate is labelled a false alarm",
                  wlm and "false alarm — the room is empty (SIM)" in wlm[0].replace("\n", " "), wlm)
            check("protocol: in variant empty the person blocks are the empty room's false alarms",
                  heads[:2] == ["WLM (baseline) · False alarms — empty room (SIM)", "RTI · False alarms — empty room (SIM)"], heads)
        finally:
            ctx.close()
        est = f0["phone"]["rtt"] if f0["phone"]["rtt"].get("estimate") else f0["phone"]["rssi"]
        def epoch(t, ok):
            o = est if ok else {"abstain": {"reason": "gated"}}
            return {"t": t, "rtt": o, "rssi": o}
        trail = [frame(0, new={"persons": [], "phone": [epoch(0.0, True), epoch(0.2, True)]}),
                 frame(1, new={"persons": [], "phone": [epoch(0.4, False)]}),
                 frame(2, new={"persons": [], "phone": [epoch(0.6, True), epoch(0.8, True)]})]
        ctx, pg, closed = scripted("trail", [state()] + trail)
        try:
            pg.wait_for_function("() => document.body.dataset.simSeq === '2'", timeout=10000)
            check("protocol: phone trails come from new.phone and break at an abstention",
                  ds(pg, "simTrails") == "4", ds(pg, "simTrails"))
        finally:
            ctx.close()
        trail.append(frame(3, run=dict(f0["run"], index=f0["run"]["index"] + 1)))
        ctx, pg, closed = scripted("trail-run", [state()] + trail)
        try:
            pg.wait_for_function("() => document.body.dataset.simSeq === '3'", timeout=10000)
            check("protocol: a new run clears the trails", ds(pg, "simTrails") == "0" and ds(pg, "simCleared") == "run")
        finally:
            ctx.close()
        # The crowded case of §9.3 "Labels are never hidden": at phone width,
        # in the plan view, all four estimates on one centre with one radius,
        # truth on and an object selected. Every simulation label is shown,
        # none under the view's corner boxes, and no estimate's under another.
        same = {"estimate": dict(f0["persons"]["wlm"]["estimate"], radius_m=1.5)}
        crowd = frame(0, persons=dict(f0["persons"], wlm=same, rti=same), phone=dict(f0["phone"], rtt=same, rssi=same))
        ctx, pg, closed = scripted("crowd", [state(), crowd], frag=f"scene={HALL}&sim=1&view=top",
                                   viewport={"width": 375, "height": 812}, is_mobile=True, has_touch=True)
        try:
            wait_state(pg, ["running", "paused", "ended"], 10000)
            pg.locator("#object-list button:has(.swatch:not(.sw-wall):not(.sw-node))").first.click()
            pg.click("#sim-truth")
            pg.wait_for_function("() => Number(document.body.dataset.simTruth) >= 1", timeout=5000)
            settle(pg)
            why = pg.evaluate("""() => [...document.querySelectorAll('#labels .label.est, #labels .label.truth, #labels .label.err, #labels .label.selected')]
                .map((e) => [e.className.replace('label ', ''), e.dataset.why])""")
            est = [w for c, w in why if c.startswith("est")]
            check("protocol: crowded at 375 px, every estimate label is shown (never hidden by the declutter)",
                  len(est) == 4 == int(ds(pg, "simRings")) and set(est) == {"shown"}, why)
            check("protocol: crowded at 375 px, every truth and error label and the selection are shown too",
                  all(w == "shown" for _, w in why) and any("selected" in c for c, _ in why), why)
            under = overlaps(pg, "#labels .label.est", "#sim-overlay, #overlay-key")
            check("protocol: crowded at 375 px, no estimate label lies under the view's corner boxes", not under, under)
            stacked = overlaps(pg, "#labels .label.est", "#labels .label.est, #labels .label.selected")
            check("protocol: crowded at 375 px, no estimate label lies under another estimate's or the selection's",
                  not stacked, stacked)
            pg.evaluate("scrollTo(0, 0)")
            shot(pg, "sim-crowded-phone.png")
        finally:
            ctx.close()

    # ---- Signed out, and refused at the handshake (the --base-auth gateway) ----
    def sim_auth():
        ctx = sim_ctx()
        try:
            pg, socks = sim_page(ctx, "sim-auth", "scene=bedroom&sim=1", base=AUTH)
            wait_state(pg, ["signed-out"], 15000)
            check("auth: signed out, the page asks for a sign-in and opens no socket",
                  pg.locator("#sim-msg").inner_text() == TEXT_SIGN_IN and no_reconnect(pg, socks, 0), len(socks))
            check("auth: being signed out is announced",
                  pg.locator("#sim-announce").text_content() == f"Simulation not started: {TEXT_SIGN_IN}")
            shot(pg, "sim-signed-out.png")
        finally:
            ctx.close()
        # A role without position scope (ADR-008 d10), as /api/me reports it.
        ctx = sim_ctx()
        try:
            ctx.route("**/api/me", lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps(
                {"auth_required": True, "authenticated": True, "role": "ops", "mode": "live",
                 "can": {"system": True, "audit": False, "position": False}})))
            pg, socks = sim_page(ctx, "sim-auth-role", "scene=bedroom&sim=1", base=AUTH)
            wait_state(pg, ["not-permitted"], 15000)
            check("auth: a role without position scope is told so, and no socket opens",
                  pg.locator("#sim-msg").inner_text() == "Your role (ops) cannot see simulated positions; an admin or safety account can"
                  and no_reconnect(pg, socks, 0))
        finally:
            ctx.close()
        # /api/me says yes, the handshake says 401: closed before any message.
        ctx = sim_ctx()
        try:
            ctx.route("**/api/me", lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps(
                {"auth_required": True, "authenticated": True, "role": "admin", "mode": "live",
                 "can": {"system": True, "audit": True, "position": True}})))
            pg, socks = sim_page(ctx, "sim-refused", "scene=bedroom&sim=1", base=AUTH)
            wait_state(pg, ["unopened"], 15000)
            check("auth: a socket refused at the handshake shows 'could not be opened'",
                  pg.locator("#sim-msg").inner_text() == TEXT_UNOPENED and ds(pg, "simLayer") == "0")
            check("auth: and the page does not try again", no_reconnect(pg, socks, 1), len(socks))
        finally:
            ctx.close()

    t_sim = time.monotonic()
    sim_section("sim", sim_hall)
    sim_section("sim/bedroom", sim_bedroom)
    sim_section("sim/plan", sim_plan)
    sim_section("replay", sim_replay)
    sim_section("sim/phone", sim_phone)
    sim_section("protocol", sim_protocol)
    sim_section("auth", sim_auth)
    print(f"(simulation checks took {time.monotonic() - t_sim:.1f} s)")
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
    # The simulation view without WebGL: the stream still runs, and the
    # panel lists every outcome as text (spec/livedemo.md §9.5).
    def sim_no_webgl():
        sp = nogl.new_page(viewport={"width": 1366, "height": 860})
        sp.on("pageerror", lambda e: results["pageerrors"].append(f"[no-webgl] {e}"))
        sp.on("console", lambda m: results["console"].append(f"[no-webgl] {m.type}: {m.text}"))
        try:
            sp.goto(f"{BASE}#scene={HALL}&sim=1")
            wait_state(sp, ["running"], 30000)
            sp.wait_for_function("""() => { const t = document.getElementById('sim-now').innerText;
                return ['WLM (baseline): ', 'RTI: ', 'phone RTT: ', 'phone RSSI: '].every(n => t.includes(n)); }""", timeout=20000)
            now = sp.locator("#sim-now").inner_text()
            listed = [re.search(rf"^{re.escape(n)}: (\(-?\d+\.\d\d, -?\d+\.\d\d\) ± \d+\.\d\d m|abstained — |estimate not drawn|not drawn)", now, re.M)
                      for n in ("WLM (baseline)", "RTI", "phone RTT", "phone RSSI")]
            check("without WebGL the simulation still runs and lists every outcome as text", all(listed), now[:300])
            check("without WebGL no ring is drawn, and the notice still says why",
                  ds(sp, "simRings") == "0" and "WebGL is not available" in sp.locator("#notice").inner_text())
            sp.locator("#sim-now").evaluate("(e) => e.scrollIntoView({ block: 'center' })")
            sp.screenshot(path=f"{OUT}/sim-no-webgl.png")
        finally:
            sp.close()
    sim_section("sim/no-webgl", sim_no_webgl)
    nogl.close()

csp = [c for c in results["console"] if "Content Security Policy" in c or "Refused to" in c]
# Two kinds of error are caused on purpose and nothing else may error: the
# failed-load checks answer one scene with a 503 (the browser and the page
# both log it), and the no-WebGL run disables WebGL, so three.js logs that
# it could not create a context.
#
# The simulation checks add one: a socket the auth gateway refuses at the
# handshake (401) is logged by the browser, matched here by its URL.
def expected(c):
    if c.startswith(("[review]", "[no-webgl]")) and "503" in c:
        return True
    if c.startswith("[sim-refused]") and f"ws://{AUTH_HOST}/api/roomsim?" in c:
        return True
    return c.startswith("[no-webgl]") and ("WebGL context could not be created" in c or "Error creating WebGL context" in c)
errs = [c for c in results["console"] if " error:" in c and not expected(c)]
check("no CSP violations in the console", not csp, csp[:3])
check("no console errors", not errs, errs[:5])
# ADR-008 d11: nothing position-like in the console. The simulation view
# logs nothing at all, so nothing it holds can leak there.
said = [c for c in results["console"] if c.startswith("[sim") and not expected(c)]
check("the simulation view logs nothing to the console", not said, said[:3])
check("no uncaught page errors", not results["pageerrors"], results["pageerrors"][:3])
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
failed = [c for c in results["checks"] if not c["ok"]]
print(f"\n{len(results['checks']) - len(failed)}/{len(results['checks'])} checks passed")
sys.exit(1 if failed else 0)
