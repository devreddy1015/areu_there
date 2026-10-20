# three.js 0.186.1, vendored

Served by `saans-gateway` (with the `roomsim` feature) at
`/roomsim/vendor/0.186.1-saans1/` for the static room demo (ADR-008). Vendored rather than fetched so the gateway works on a box with
no route out, and so the demo page's CSP can be `script-src 'self'`.

| | |
|---|---|
| Package | `three@0.186.1`, MIT (see `LICENSE`, and `NOTICE` at the repo root) |
| Tarball | `https://registry.npmjs.org/three/-/three-0.186.1.tgz` |
| Tarball integrity | `sha512-blFeqb49wRCSGUGj7gtpfnSGHy2lwDk94RhUmS1c/hTby70kvChbWpkJ4Pm1390LqzzvTmzgXKHPEafJwCb8jA==` (checked against the registry before extraction) |

| File | From the package | sha256 as vendored |
|---|---|---|
| `three.core.js` | `build/three.core.js`, unmodified | `9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6` |
| `three.module.js` | `build/three.module.js`, unmodified | `9052042d676cb0fdc1ddfefe193053f34b7ac0513a616fdac4535d49987812ea` |
| `OrbitControls.js` | `examples/jsm/controls/OrbitControls.js`, **one line changed** | `30386f7141d5ce5d5e0a3ab643e4eb16265f4564466108105ce449857f62fdda` |
| `LICENSE` | `LICENSE`, unmodified | `8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc` |

## The one modification

`OrbitControls.js` line 12 upstream is `} from 'three';`. A bare specifier
like `'three'` only resolves through an import map, and an import map is
an inline `<script>`, which the demo's CSP deliberately forbids. The line
is changed to `} from './three.module.js';`. Nothing else in the file is
touched. Upstream sha256 of the unmodified file:
`3d79d07ecb686b4e5d93232eedab255331c1beef711e13164eaa1f68655a5f2b`.

## Audited before vendoring

None of the three JavaScript files contains `eval(`, `new Function` or
`new Worker`, so the page needs no `'unsafe-eval'`. `three.core.js` has
three `fetch(` calls, all inside loaders the demo does not use; the page's
`connect-src 'self'` would refuse a cross-origin fetch from them anyway.
`three.module.js` imports only `./three.core.js`; `three.core.js` imports
nothing.

## The served revision

The files are served under `/roomsim/vendor/0.186.1-saans1/` and cached as
immutable for a year. `0.186.1` is the upstream version; `-saans1` is the
local revision, because `OrbitControls.js` carries a local change. **Any
change to any vendored byte bumps the suffix** (`-saans2`, …), or browsers
that cached the old bytes keep running them. `VENDOR_REV` in
`src/roomsim.rs` is the single source; a test fails if this file does not
name it.

## Upgrading

Every place the version lives, in order:

1. Download the new tarball and check its `sha512` integrity against
   `https://registry.npmjs.org/three/<version>` **before** extracting.
2. Copy the same four files into a new `assets/vendor/three-<version>/`,
   re-apply the one-line change to `OrbitControls.js`, and re-audit (no
   `eval(`, `new Function`, `new Worker`; every import relative).
3. In `src/roomsim.rs`: the three `include_str!` paths, the
   `include_str!` of this file in the pinned-hash test, `VENDOR_REV`
   (reset the suffix to `-saans1`), and the pinned SHA-256 values.
4. In `assets/roomsim.js`: the two `import` paths. A test fails if they
   disagree with `VENDOR_REV`.
5. In `tests/e2e.rs`: nothing — it reads the import paths from the page.
6. This file, `NOTICE` at the repo root (version and copyright years),
   and the mentions in `RUN.md` and `ROOMSIM.md`.

Then run `cargo test -p saans-gateway --features roomsim` and the browser
check in `tests/browser/`. The pinned hashes fail until step 3 is done,
which is the point.
