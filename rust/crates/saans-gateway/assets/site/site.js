// Saans site: progressive enhancement only. Every section reads correctly
// without this file; it adds reveal motion, marks the current page in the
// header, and runs the interactive pieces (pipeline, links, detectors,
// frame, tabs, waveform).
//
// One file serves all seven pages. Every block below is gated on an
// element that exists on exactly one of them, so the pages that do not
// have it simply skip it — which is cheaper and far less brittle than
// seven bundles that can drift apart.
(() => {
  "use strict";

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];

  // Run `fn(true/false)` as `el` enters / leaves the viewport.
  function whileVisible(el, fn, margin = "0px") {
    if (!el) return;
    if (!("IntersectionObserver" in window)) return fn(true);
    new IntersectionObserver((es) => es.forEach((e) => fn(e.isIntersecting)), { rootMargin: margin }).observe(el);
  }

  /* ---------- reveal on scroll ---------- */
  if (!reduce && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        e.target.classList.add("is-in");
        e.target.classList.remove("is-pending");
        io.unobserve(e.target);
      }
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.01 });
    for (const el of $$("[data-reveal]")) {
      if (el.getBoundingClientRect().top < innerHeight * 0.92) continue; // already on screen
      el.classList.add("is-pending");
      io.observe(el);
    }
  }

  const bars = $("#bars");
  if (bars) {
    if (reduce || !("IntersectionObserver" in window)) bars.classList.add("is-in");
    else {
      const io = new IntersectionObserver(([e]) => {
        if (e.isIntersecting) { bars.classList.add("is-in"); io.disconnect(); }
      }, { rootMargin: "0px 0px -15% 0px" });
      io.observe(bars);
    }
  }

  /* ---------- page navigation ----------
     The site used to be one document with a scroll-spy rail. It is seven
     pages now, so "where am I" is answered by the URL, not by an
     observer: one source of truth, and it is correct before any script
     has run on a page that never scrolls. */
  const sitenav = $(".sitenav");
  if (sitenav) {
    const page = document.body.dataset.page;
    for (const a of $$("a[data-page]", sitenav)) {
      if (a.dataset.page === page) a.setAttribute("aria-current", "page");
    }
    // A hairline under the header, but only once something has scrolled
    // beneath it. Sentinel over scroll listener: no work per frame.
    const probe = document.createElement("div");
    probe.setAttribute("aria-hidden", "true");
    probe.style.cssText = "position:absolute;top:0;height:1px;width:1px";
    document.body.prepend(probe);
    whileVisible(probe, (vis) => sitenav.classList.toggle("is-stuck", !vis));
  }

  /* ---------- links kept alive after the move ----------
     The dashboard used to live at `/`, so bookmarks and anything that was
     pasted into a chat look like `/#system`. Those hashes now name a tab
     inside `/app`. Forward them rather than landing someone on a home
     page with a fragment that means nothing here. The home page is found
     by its marker, not by `location.pathname === "/"`, because the static
     copy of the site lives under a path prefix, and `app#…` is relative
     for the same reason. */
  if (document.body.dataset.page === "home" && location.hash.length > 1) {
    const TABS = ["overview", "explorer", "signals", "topology", "events", "system"];
    const want = location.hash.slice(1);
    if (TABS.includes(want)) location.replace("app#" + want);
  }

  /* ---------- pipeline: follow the phase at the centre ---------- */
  const phases = $$(".phases li");
  const steps = $$(".flow-step");
  const setPhase = (n) => {
    phases.forEach((li) => li.classList.toggle("is-active", +li.dataset.phase === n));
    steps.forEach((s) => s.classList.toggle("is-active", +s.dataset.step === n));
  };
  if (phases.length) {
    setPhase(0);
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => {
        for (const e of entries) if (e.isIntersecting) setPhase(+e.target.dataset.phase);
      }, { rootMargin: "-45% 0px -50% 0px" });
      phases.forEach((li) => io.observe(li));
    }
  }

  /* ---------- what just changed, for the two filter controls ----------
     Both controls change what is on the page without moving focus, so
     nothing is spoken. The polite region is built here rather than written
     into the markup because it only has a job while this file is running:
     with script off neither control filters anything. */
  const srLive = (anchor, id) => {
    if (!anchor) return null;
    let el = document.getElementById(id);
    if (!el) {
      el = document.createElement("p");
      el.id = id;
      el.className = "sr-only";
      el.setAttribute("aria-live", "polite");
      anchor.after(el);
    }
    return el;
  };
  const liveAfter = (btns, id) => (btns.length ? srLive(btns[0].closest(".seg") || btns[0].parentElement, id) : null);

  /* ---------- link diversity toggle ---------- */
  const linkButtons = $$("[data-links]");
  const linksLive = liveAfter(linkButtons, "links-live");
  const setLinks = (n) => {
    linkButtons.forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.links === n)));
    $$("#links [data-k]").forEach((g) => g.classList.toggle("is-off", +g.dataset.k > n));
    $$("#links [data-node]").forEach((g) => (g.style.opacity = +g.dataset.node > n ? "0.25" : "1"));
    $$("#links [data-col]").forEach((c) => c.classList.toggle("is-sel", +c.dataset.col === n));
  };
  linkButtons.forEach((b) => b.addEventListener("click", () => {
    const n = +b.dataset.links;
    setLinks(n);
    // only on a press: the initial state is already spoken by aria-pressed
    if (linksLive) linksLive.textContent = `${n} link${n === 1 ? "" : "s"} selected`;
  }));
  if (linkButtons.length) setLinks(2);

  /* ---------- detector filter ---------- */
  const filterButtons = $$("[data-filter]");
  const detLive = liveAfter(filterButtons, "det-live");
  filterButtons.forEach((b) => b.addEventListener("click", () => {
    const f = b.dataset.filter;
    filterButtons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    const dets = $$("#det-grid .det");
    dets.forEach((d) => (d.hidden = f !== "all" && d.dataset.cat !== f));
    if (detLive) detLive.textContent = `showing ${dets.filter((d) => !d.hidden).length} of ${dets.length} detectors`;
  }));

  /* ---------- FeatureFrame v1, encoded per spec/wire.md §2 ---------- */
  const FIELDS = [
    ["magic", 0, "u32", "head"], ["version", 4, "u8", "head"], ["node_id", 5, "u8", "head"],
    ["tx_id", 6, "u8", "head"], ["flags", 7, "u8", "head"], ["seq", 8, "u32", "head"],
    ["uptime_ms", 12, "u32", "head"], ["csi_rate_chz", 16, "u16", "link"],
    ["loss_permille", 18, "u16", "link"], ["motion_energy", 20, "f32", "presence"],
    ["presence_threshold", 24, "f32", "presence"], ["gross_motion", 28, "f32", "presence"],
    ["breath_rms", 32, "f32", "vitals"], ["breathing_bpm", 36, "f32", "vitals"],
    ["breathing_conf", 40, "f32", "vitals"], ["heart_bpm", 44, "f32", "heart"],
    ["heart_conf", 48, "f32", "heart"], ["best_bnr", 52, "f32", "quality"],
    ["calib_age_s", 56, "u32", "quality"], ["rssi_dbm", 60, "i8", "quality"],
    ["noise_floor_dbm", 61, "i8", "quality"], ["n_subcarriers", 62, "u8", "quality"],
    ["reserved", 63, "u8", "quality"], ["crc32", 64, "u32", "crc"],
  ].map(([name, off, type, g]) => ({ name, off, type, g, size: { u8: 1, i8: 1, u16: 2, u32: 4, f32: 4 }[type] }));

  const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();
  const crc32 = (bytes) => {
    let c = 0xffffffff;
    for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };

  const bytesEl = $("#bytes");
  const fieldsEl = $("#fields");
  const readout = $("#readout");
  const frameLive = $("#frame-live");
  const consent = $("#consent");
  if (bytesEl && fieldsEl && readout && consent) {
    const buf = new ArrayBuffer(68);
    const dv = new DataView(buf);
    const u8 = new Uint8Array(buf);
    let prev = new Uint8Array(68);
    const cells = [];
    const fieldOf = [];
    const jitter = (v, a) => v + (Math.random() * 2 - 1) * a;
    const st = { seq: 4211, uptime: 2105500, calib: 7262 };
    const values = {};

    for (const f of FIELDS) for (let i = 0; i < f.size; i++) fieldOf[f.off + i] = f;
    for (let i = 0; i < 68; i++) {
      const c = document.createElement("span");
      c.className = `byte g-${fieldOf[i].g}`;
      c.dataset.i = i;
      bytesEl.append(c);
      cells.push(c);
    }
    // The 24 field rows are static markup: they are the wire format, they
    // never change, and they have to be readable with this file switched
    // off. Script adds the tab stops, because script is what gives a row
    // something to say when you reach it.
    const rows = new Map();
    for (const li of $$("li[data-field]", fieldsEl)) {
      li.tabIndex = 0;
      rows.set(li.dataset.field, li);
    }
    const byName = (name) => FIELDS.find((f) => f.name === name);

    const gatedNote = (f) =>
      f.g === "heart" ? "zeroed on the node: heart rate needs its own switch, off by default"
      : f.g === "vitals" && !consent.checked ? "zeroed on the node: this space has not opted in"
      : "";
    const fmt = (f) => {
      const v = values[f.name];
      if (f.name === "magic") return `0x${v.toString(16).toUpperCase()} "SANS"`;
      if (f.name === "crc32") return `0x${v.toString(16).padStart(8, "0")} ✓`;
      if (f.name === "flags") return `0b${v.toString(2).padStart(8, "0")}`;
      if (f.type === "f32") return Math.fround(v).toFixed(4);
      return String(v);
    };

    let hovered = null;
    const showIdle = () => {
      readout.innerHTML = `seq <b>${values.seq}</b> · 68 bytes · crc32 <b>${fmt(FIELDS[FIELDS.length - 1])}</b> · flags <b>${fmt(FIELDS[4])}</b>`;
    };
    const showField = (f) => {
      const note = gatedNote(f);
      readout.innerHTML = `offset ${f.off} · ${f.type} · <b>${f.name}</b> = <b>${fmt(f)}</b>${note ? ` · ${note}` : ""}`;
    };
    // #readout is rewritten twice a second, so it must never be a live
    // region. The inspected field is announced here instead, and only when
    // the field — or what the gate does to it — actually changes.
    let said = "";
    const announce = (f) => {
      if (!frameLive) return;
      const note = f ? gatedNote(f) : "";
      const key = f ? `${f.name}|${note}` : "";
      if (key === said) return;
      said = key;
      frameLive.textContent = f ? `${f.name}, offset ${f.off}, ${f.type}, ${fmt(f)}${note ? `, ${note}` : ""}` : "";
    };
    const highlight = (f) => {
      hovered = f;
      bytesEl.classList.toggle("is-hovering", !!f);
      cells.forEach((c, i) => c.classList.toggle("is-hl", !!f && fieldOf[i] === f));
      rows.forEach((li, name) => li.classList.toggle("is-hl", !!f && name === f.name));
      f ? showField(f) : showIdle();
      announce(f);
    };
    // A pointer leaving must not steal the row the keyboard is sitting on.
    const focusedField = () => {
      const li = fieldsEl.contains(document.activeElement) && document.activeElement.closest("li[data-field]");
      return li ? byName(li.dataset.field) : null;
    };
    bytesEl.addEventListener("pointerover", (e) => { const i = e.target.dataset?.i; if (i != null) highlight(fieldOf[+i]); });
    bytesEl.addEventListener("pointerleave", () => highlight(focusedField()));
    fieldsEl.addEventListener("pointerover", (e) => {
      const li = e.target.closest("li[data-field]");
      if (li) highlight(byName(li.dataset.field));
    });
    fieldsEl.addEventListener("pointerleave", () => highlight(focusedField()));
    fieldsEl.addEventListener("focusin", (e) => {
      const li = e.target.closest("li[data-field]");
      if (li) highlight(byName(li.dataset.field));
    });
    fieldsEl.addEventListener("focusout", (e) => {
      if (!fieldsEl.contains(e.relatedTarget)) highlight(null);
    });

    const encode = () => {
      const vit = consent.checked;
      // The gated marker is state, not markup. The rows ship with it set
      // because that is what an unconsented node sends, which is the
      // honest default for a reader with no JavaScript; from here on the
      // switch owns it, or the list would contradict the bytes beside it.
      for (const li of fieldsEl.children) {
        if (li.classList.contains("g-vitals")) li.classList.toggle("is-gated", !vit);
      }
      Object.assign(values, {
        magic: 0x534e4153, version: 1, node_id: 2, tx_id: 1,
        // calibrated | occupied | vitals_enabled | breathing_valid
        flags: 0b1 | 0b10 | (vit ? 0b100 | 0b10000 : 0),
        seq: st.seq, uptime_ms: st.uptime,
        csi_rate_chz: Math.round(jitter(1998, 3)), loss_permille: Math.round(jitter(3, 2)),
        motion_energy: jitter(0.151, 0.004), presence_threshold: 0.0527, gross_motion: jitter(0.079, 0.003),
        breath_rms: vit ? jitter(0.0123, 0.0004) : 0, breathing_bpm: vit ? jitter(15.0, 0.06) : 0,
        breathing_conf: vit ? jitter(0.92, 0.01) : 0, heart_bpm: 0, heart_conf: 0,
        best_bnr: jitter(6.3, 0.15), calib_age_s: st.calib,
        rssi_dbm: Math.round(jitter(-52, 1)), noise_floor_dbm: -92, n_subcarriers: 52, reserved: 0,
      });
      for (const f of FIELDS) {
        if (f.name === "crc32") continue;
        const v = values[f.name];
        ({ u8: () => dv.setUint8(f.off, v), i8: () => dv.setInt8(f.off, v), u16: () => dv.setUint16(f.off, v, true),
           u32: () => dv.setUint32(f.off, v >>> 0, true), f32: () => dv.setFloat32(f.off, v, true) })[f.type]();
      }
      values.crc32 = crc32(u8.subarray(0, 64));
      dv.setUint32(64, values.crc32, true);

      cells.forEach((c, i) => {
        c.textContent = u8[i].toString(16).padStart(2, "0").toUpperCase();
        const f = fieldOf[i];
        c.classList.toggle("is-zero", (f.g === "vitals" || f.g === "heart") && u8[i] === 0);
        if (!reduce && u8[i] !== prev[i]) {
          c.classList.remove("is-flash");
          void c.offsetWidth; // restart the animation
          c.classList.add("is-flash");
        }
      });
      prev = u8.slice();
      if (hovered) { showField(hovered); announce(hovered); } else showIdle();
    };
    const tick = () => {
      st.seq = (st.seq + 1) >>> 0;
      st.uptime += 500;
      if (st.seq % 2 === 0) st.calib += 1;
      encode();
    };
    consent.addEventListener("change", encode);
    encode();
    let timer = null;
    whileVisible(bytesEl, (on) => {
      if (on && !timer && !reduce) timer = setInterval(() => !document.hidden && tick(), 500);
      if (!on && timer) { clearInterval(timer); timer = null; }
    });
  }

  /* ---------- code tabs + copy ---------- */
  const tabs = $$('[role="tab"]');
  const selectTab = (t, focus) => {
    tabs.forEach((x) => {
      const on = x === t;
      x.setAttribute("aria-selected", String(on));
      x.tabIndex = on ? 0 : -1;
      document.getElementById(x.getAttribute("aria-controls")).hidden = !on;
    });
    if (focus) t.focus();
  };
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => selectTab(t));
    t.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (d) { e.preventDefault(); selectTab(tabs[(i + d + tabs.length) % tabs.length], true); }
    });
  });
  // Collapsing the panels is this script's job, not the markup's. The
  // page ships all four visible so that someone with no JavaScript can
  // read all four; the tab strip is the enhancement, so the enhancement
  // is what hides three of them.
  if (tabs.length) {
    // The bar does nothing without script, and the headings are what
    // stand in for it, so the two swap over exactly here.
    $('[role="tablist"]')?.removeAttribute("hidden");
    $$(".code-h").forEach((h) => (h.hidden = true));
    selectTab(tabs.find((t) => t.getAttribute("aria-selected") === "true") || tabs[0]);
  }
  const copy = $("#copy");
  copy?.addEventListener("click", async () => {
    const pane = $('.code pre:not([hidden])');
    if (!pane) return;
    const text = pane.textContent;
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; } catch {
      const r = document.createRange();
      r.selectNodeContents(pane);
      const s = getSelection();
      s.removeAllRanges(); s.addRange(r);
      try { ok = document.execCommand("copy"); } catch { /* leave it selected */ }
      s.removeAllRanges();
    }
    const label = $("span", copy);
    label.textContent = ok ? "copied" : "select and copy";
    setTimeout(() => (label.textContent = "copy"), 1600);
  });

  /* ---------- hero: simulated breathing-band waveform ---------- */
  const canvas = $("#wave");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    const FS = 20, WINDOW_S = 30, BPM = 15;
    const hash = (k) => { const x = Math.sin(k * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
    // breathing fundamental + a harmonic + a faint cardiac ripple + noise,
    // sampled on a fixed 20 Hz grid so scrolling never re-rolls the noise
    const sample = (k) => {
      const t = k / FS, f = BPM / 60;
      return Math.sin(2 * Math.PI * f * t) + 0.18 * Math.sin(4 * Math.PI * f * t + 1)
        + 0.04 * Math.sin(2 * Math.PI * 1.13 * t) + 0.06 * (hash(k) - 0.5);
    };
    let w = 0, h = 0, dpr = 1;
    const resize = () => {
      const r = canvas.parentElement.getBoundingClientRect();
      dpr = Math.min(devicePixelRatio || 1, 2);
      w = r.width; h = r.height;
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const t0 = performance.now();
    const draw = (now) => {
      if (!w || !h) return;
      const tNow = (now - t0) / 1000 + 40;
      const kNow = tNow * FS;
      const n = WINDOW_S * FS;
      const padT = 34, padB = 16, mid = padT + (h - padT - padB) / 2, amp = (h - padT - padB) * 0.36;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "#2b2a28"; ctx.lineWidth = 1;
      for (let s = 0; s <= WINDOW_S; s += 5) {
        const x = Math.round(w - ((s * FS) / n) * w) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, padT - 6); ctx.lineTo(x, h - padB + 6); ctx.stroke();
      }
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, "rgba(255,181,71,0)");
      grad.addColorStop(0.25, "rgba(255,181,71,0.85)");
      grad.addColorStop(0.65, "#ff6b3d");
      grad.addColorStop(1, "#ff4f79");
      ctx.strokeStyle = grad; ctx.lineWidth = 2; ctx.lineJoin = "round";
      ctx.beginPath();
      const k0 = Math.floor(kNow - n), frac = kNow - Math.floor(kNow);
      for (let i = 0; i <= n + 1; i++) {
        const k = k0 + i;
        // shift by the fractional sample so the scroll is smooth between frames
        const y = mid - amp * sample(k);
        const x = ((i - frac) / n) * w;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      const yEnd = mid - amp * (sample(Math.floor(kNow)) * (1 - frac) + sample(Math.floor(kNow) + 1) * frac);
      ctx.fillStyle = "#ff4f79";
      ctx.beginPath(); ctx.arc(w - 1, yEnd, 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "rgba(255,79,121,0.18)";
      ctx.beginPath(); ctx.arc(w - 1, yEnd, 9, 0, Math.PI * 2); ctx.fill();
    };
    resize();
    if ("ResizeObserver" in window) new ResizeObserver(() => { resize(); draw(performance.now()); }).observe(canvas.parentElement);
    draw(performance.now());

    if (!reduce) {
      let raf = 0, readings = 0;
      const loop = (now) => { draw(now); raf = requestAnimationFrame(loop); };
      const bpm = $("#bpm"), conf = $("#conf"), motion = $("#motion"), still = $("#still");
      let stillS = 372;
      const update = () => {
        if (document.hidden) return;
        stillS += 1;
        still.textContent = `${Math.floor(stillS / 60)}:${String(stillS % 60).padStart(2, "0")}`;
        if (++readings % 2) return; // vitals change at the 2 Hz frame rate, shown every 2 s
        bpm.textContent = (15 + (Math.random() - 0.5) * 0.2).toFixed(1);
        conf.textContent = (0.92 + (Math.random() - 0.5) * 0.03).toFixed(2);
        motion.textContent = (0.81 + (Math.random() - 0.5) * 0.05).toFixed(2);
      };
      let timer = null;
      whileVisible(canvas.closest(".console"), (on) => {
        if (on) {
          if (!raf) raf = requestAnimationFrame(loop);
          if (!timer) timer = setInterval(update, 1000);
        } else {
          cancelAnimationFrame(raf); raf = 0;
          clearInterval(timer); timer = null;
        }
      });
    }
  }
})();
