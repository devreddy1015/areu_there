//! Golden-vector parity with the Python reference `wifisense/roomsim/`
//! (ADR-002, spec/roomsim.md §14). Everything compared here is SIM.
//!
//! Each `roomsim_*` set in `spec/golden/` (written by
//! `spec/golden/generate.py` on CPython 3.14; the array layouts are the
//! docstrings of its `gen_roomsim_*` functions) is loaded by name — the
//! `saans-dsp/tests/golden.rs` pattern, `NAME.kv` scalars plus
//! `NAME__ARRAY.f64` arrays — its inputs are rebuilt in Rust, and the
//! outputs are compared by the §14 rule:
//!
//! - a float `a` matches the reference `b` iff
//!   `|a - b| <= 1e-9*|b| + 1e-12*M`, `M` the largest `|b_j|` of the array
//!   `b` belongs to (one packet's 64 bins or 128 buffer entries, one link's
//!   coefficients, lengths or points, one trajectory's coordinates);
//! - RNG output is bit-identical; lost flags, sequence numbers, AGC
//!   states, kinds and path counts are identical;
//! - RSSI, int8 and datagram bytes are identical unless the pre-rounding
//!   value is within 1e-6 of a .5 tie in either implementation.
//!
//! A set built from a scene must carry the `scene_sha256` of that file's
//! bytes; `Vectors::load` checks it, so a scene edit cannot leave a stale
//! set passing. `roomsim_export` is compared with `saans roomsim-export` in
//! `saans-cli/tests/roomsim_export.rs`.
//!
//! Every test prints its worst relative error (`cargo test -p
//! saans-roomsim --test golden -- --nocapture`): `rel` is the largest
//! `|a - b| / |b|` over values with `|b| >= 1e-6*M`, `budget` the largest
//! `|a - b| / (1e-9*|b| + 1e-12*M)` (a match needs `budget <= 1`).

use saans_roomsim::impair::{buffer_active_all, buffer_k_all, PacketDraws};
use saans_roomsim::materials::{eta, interface_g, object_rho, slab, ItuMaterial, Pol};
use saans_roomsim::motion::SegmentKind;
use saans_roomsim::numerics::{carrier_hz, cexpj, cmul, pow10, round_half_even, Cx, C};
use saans_roomsim::paths::{draw_tail, Face};
use saans_roomsim::people::{j1, leg_shadow_db, three_edge_j, Cylinder, Pose};
use saans_roomsim::{
    v3, Buffer, Config, Mode, Packet, PathKind, PeopleAt, Posture, Simulator, Transmission, LABEL,
};
use saans_scene::Scene;
use saans_sim::rng::Rng;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;

const REL: f64 = 1e-9;
const ABS: f64 = 1e-12;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn golden_dir() -> PathBuf {
    repo().join("spec/golden")
}

/// One golden set: `NAME.kv` scalars and `NAME__ARRAY.f64` arrays.
struct Vectors {
    name: String,
    kv: BTreeMap<String, String>,
}

impl Vectors {
    fn load(name: &str) -> Self {
        let path = golden_dir().join(format!("{name}.kv"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run spec/golden/generate.py)", path.display()));
        let kv = text
            .lines()
            .filter_map(|l| l.split_once(' '))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let v = Self {
            name: name.to_string(),
            kv,
        };
        assert_eq!(v.s("label"), LABEL, "{name}: every golden set is SIM");
        // §14: a set built from a scene records the digest of the scene
        // file's bytes, so a golden set never silently outlives a scene
        // edit; the formula-only sets (materials, shadow) have no scene.
        match v.kv.get("scene") {
            Some(rel) => {
                let path = repo().join(rel);
                let bytes =
                    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                let sha: String = Sha256::digest(&bytes)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                assert_eq!(
                    v.s("scene_sha256"),
                    sha,
                    "{name}: {rel} changed since the set was generated (rerun spec/golden/generate.py)"
                );
            }
            None => assert!(
                !v.kv.contains_key("scene_sha256"),
                "{name}: scene_sha256 without a scene"
            ),
        }
        v
    }

    fn s(&self, k: &str) -> &str {
        self.kv
            .get(k)
            .unwrap_or_else(|| panic!("{}: no scalar {k}", self.name))
    }

    fn f(&self, k: &str) -> f64 {
        let v = self.s(k);
        v.parse()
            .unwrap_or_else(|_| panic!("{}: {k}={v} not a number", self.name))
    }

    fn u(&self, k: &str) -> usize {
        let x = self.f(k);
        assert!(x >= 0.0 && x.fract() == 0.0, "{}: {k}={x}", self.name);
        x as usize
    }

    fn a(&self, k: &str) -> Vec<f64> {
        let n: usize = self.s(&format!("array.{k}")).parse().unwrap();
        let path = golden_dir().join(format!("{}__{k}.f64", self.name));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(bytes.len(), n * 8, "{}: array {k} length", self.name);
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    }

    fn scene(&self) -> Scene {
        Scene::load(repo().join(self.s("scene"))).unwrap()
    }

    /// The run's configuration: the §3.2 defaults with every `cfg.*` scalar
    /// applied (maps as `cfg.name.key`, pairs as `cfg.name.0`/`.1`, bools
    /// as 0/1), read through `Config::from_json_str`.
    fn config(&self) -> Config {
        self.config_of("")
    }

    /// The configuration of the run whose scalars carry `prefix` (`a_`
    /// reads the `a_cfg.*` scalars), by the `cfg.*` convention of
    /// [`Vectors::config`].
    fn config_of(&self, prefix: &str) -> Config {
        let head = format!("{prefix}cfg.");
        let mut j = serde_json::to_value(Config::default()).unwrap();
        for (k, text) in &self.kv {
            let Some(rest) = k.strip_prefix(head.as_str()) else {
                continue;
            };
            let parts: Vec<&str> = rest.split('.').collect();
            let slot = &mut j[parts[0]];
            assert!(
                !slot.is_null() || parts[0] == "n_people",
                "{}: unknown config {k}",
                self.name
            );
            let value = |default: &serde_json::Value| -> serde_json::Value {
                if default.is_boolean() {
                    serde_json::Value::Bool(text != "0")
                } else if let Ok(i) = text.parse::<i64>() {
                    i.into()
                } else if let Ok(x) = text.parse::<f64>() {
                    x.into()
                } else {
                    text.as_str().into()
                }
            };
            match parts.len() {
                1 => *slot = value(slot),
                2 => {
                    if let Some(arr) = slot.as_array_mut() {
                        let i: usize = parts[1].parse().unwrap();
                        arr[i] = value(&arr[i]);
                    } else {
                        let obj = slot.as_object_mut().expect("a map");
                        obj.insert(parts[1].to_string(), value(&serde_json::Value::Null));
                    }
                }
                _ => panic!("{}: config key {k}", self.name),
            }
        }
        Config::from_json_str(&j.to_string()).unwrap()
    }
}

/// The largest `|b_j|`.
fn max_abs(b: &[f64]) -> f64 {
    b.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// The largest complex magnitude of `(re[j], im[j])`.
fn max_mag(re: &[f64], im: &[f64]) -> f64 {
    re.iter()
        .zip(im)
        .fold(0.0, |m, (a, b)| m.max((a * a + b * b).sqrt()))
}

/// Whether `x` is within 1e-6 of a .5 tie (§14).
fn near_tie(x: f64) -> bool {
    ((x - x.floor()) - 0.5).abs() <= 1e-6
}

/// Comparisons for one set: collects every failure and the worst errors.
struct Check {
    set: String,
    rel: f64,
    rel_at: String,
    budget: f64,
    budget_at: String,
    floats: usize,
    exact: usize,
    ties: usize,
    failures: Vec<String>,
}

impl Check {
    fn new(set: &str) -> Self {
        Self {
            set: set.into(),
            rel: 0.0,
            rel_at: String::new(),
            budget: 0.0,
            budget_at: String::new(),
            floats: 0,
            exact: 0,
            ties: 0,
            failures: Vec::new(),
        }
    }

    fn fail(&mut self, msg: String) {
        if self.failures.len() < 40 {
            self.failures.push(msg);
        } else if self.failures.len() == 40 {
            self.failures.push("(more failures not shown)".into());
        }
    }

    /// §14 float rule with `M = m`.
    fn close_m(&mut self, what: &str, got: &[f64], want: &[f64], m: f64) {
        if got.len() != want.len() {
            self.fail(format!("{what}: length {} != {}", got.len(), want.len()));
            return;
        }
        for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
            self.floats += 1;
            let err = (a - b).abs();
            let tol = REL * b.abs() + ABS * m;
            let ok = err <= tol || a == b;
            if !ok {
                self.fail(format!(
                    "{what}[{i}]: got {a:e}, want {b:e} (err {err:e}, tol {tol:e})"
                ));
            }
            if err > 0.0 {
                if tol > 0.0 && err / tol > self.budget {
                    self.budget = err / tol;
                    self.budget_at = format!("{what}[{i}]");
                }
                if b.abs() >= 1e-6 * m && b != 0.0 && err / b.abs() > self.rel {
                    self.rel = err / b.abs();
                    self.rel_at = format!("{what}[{i}]");
                }
            }
        }
    }

    /// §14 float rule, `M` the largest `|b_j|` of `want`.
    fn close(&mut self, what: &str, got: &[f64], want: &[f64]) {
        self.close_m(what, got, want, max_abs(want));
    }

    /// A complex array: `M` the largest magnitude.
    fn close_cx(&mut self, what: &str, got: &[Cx], re: &[f64], im: &[f64]) {
        let m = max_mag(re, im);
        let gr: Vec<f64> = got.iter().map(|c| c.re).collect();
        let gi: Vec<f64> = got.iter().map(|c| c.im).collect();
        self.close_m(&format!("{what}.re"), &gr, re, m);
        self.close_m(&format!("{what}.im"), &gi, im, m);
    }

    /// Identical values (integers, flags, kinds, RNG output).
    fn same(&mut self, what: &str, got: &[f64], want: &[f64]) {
        if got.len() != want.len() {
            self.fail(format!("{what}: length {} != {}", got.len(), want.len()));
            return;
        }
        for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
            self.exact += 1;
            if a.to_bits() != b.to_bits() && a != b {
                self.fail(format!("{what}[{i}]: got {a:?}, want {b:?}"));
            }
        }
    }

    /// A rounded integer: identical unless either pre-rounding value is
    /// within 1e-6 of a .5 tie.
    fn rounded(&mut self, what: &str, got: f64, want: f64, pre_got: f64, pre_want: f64) {
        self.exact += 1;
        if got == want {
            return;
        }
        if near_tie(pre_got) || near_tie(pre_want) {
            self.ties += 1;
            return;
        }
        self.fail(format!(
            "{what}: got {got}, want {want} (pre-rounding {pre_got:?} vs {pre_want:?})"
        ));
    }

    fn finish(self) {
        println!(
            "{:20} worst rel {:.3e} ({}), worst tolerance budget {:.3e} ({}); {} floats, {} exact, {} tie exemptions",
            self.set, self.rel, self.rel_at, self.budget, self.budget_at, self.floats, self.exact, self.ties
        );
        if !self.failures.is_empty() {
            panic!(
                "{}: {} mismatches:\n  {}",
                self.set,
                self.failures.len(),
                self.failures.join("\n  ")
            );
        }
    }
}

/// §5.1 material codes of the golden sets: 0 concrete, 1 brick,
/// 2 drywall, 3 glass, 4 wood, 5 metal.
fn itu(code: f64) -> ItuMaterial {
    match code as u32 {
        0 => ItuMaterial::Concrete,
        1 => ItuMaterial::Brick,
        2 => ItuMaterial::Plasterboard,
        3 => ItuMaterial::Glass,
        4 => ItuMaterial::Wood,
        5 => ItuMaterial::Metal,
        c => panic!("material code {c}"),
    }
}

fn pol(code: f64) -> Pol {
    if code == 0.0 {
        Pol::Te
    } else {
        Pol::Tm
    }
}

fn posture_code(p: Posture) -> f64 {
    match p {
        Posture::Stand => 0.0,
        Posture::Sit => 1.0,
        Posture::Lie => 2.0,
    }
}

fn b(x: bool) -> f64 {
    if x {
        1.0
    } else {
        0.0
    }
}

#[test]
fn roomsim_materials() {
    let v = Vectors::load("roomsim_materials");
    let mut c = Check::new(&v.name);
    let fc = carrier_hz(v.u("channel") as u32);
    let lam = C / fc;
    c.same("carrier_hz", &[fc], &[v.f("carrier_hz")]);
    c.same("lambda", &[lam], &[v.f("lambda")]);
    let cos_t = v.a("cos_t");
    // The inputs as the generator documents them: cos((deg * PI) / 180.0).
    let want_cos: Vec<f64> = (0..v.u("n_angles"))
        .map(|i| ((v.f(&format!("angle_deg_{i}")) * std::f64::consts::PI) / 180.0).cos())
        .collect();
    c.same("cos_t", &want_cos, &cos_t);
    let etas: Vec<Cx> = (0..6).map(|m| eta(itu(m as f64), fc)).collect();
    c.close(
        "eta_re",
        &etas.iter().map(|e| e.re).collect::<Vec<_>>(),
        &v.a("eta_re"),
    );
    c.close(
        "eta_im",
        &etas.iter().map(|e| e.im).collect::<Vec<_>>(),
        &v.a("eta_im"),
    );

    let (mat, thick, cos, pl) = (
        v.a("row_material"),
        v.a("row_thickness"),
        v.a("row_cos_t"),
        v.a("row_pol"),
    );
    let mut out: BTreeMap<&str, Vec<Cx>> = BTreeMap::new();
    for r in 0..pl.len() {
        let s = slab(etas[mat[r] as usize], cos[r], thick[r], lam, pol(pl[r]));
        out.entry("G").or_default().push(s.g);
        out.entry("R").or_default().push(s.r);
        out.entry("T").or_default().push(s.t);
    }
    for (k, got) in &out {
        c.close_cx(k, got, &v.a(&format!("{k}_re")), &v.a(&format!("{k}_im")));
    }

    let (om, oc, op) = (v.a("obj_material"), v.a("obj_cos_t"), v.a("obj_pol"));
    let got: Vec<Cx> = (0..om.len())
        .map(|r| interface_g(eta(itu(om[r]), fc), oc[r], pol(op[r])))
        .collect();
    c.close_cx("obj_G", &got, &v.a("obj_G_re"), &v.a("obj_G_im"));
    c.finish();
}

#[test]
fn roomsim_shadow() {
    let v = Vectors::load("roomsim_shadow");
    let mut c = Check::new(&v.name);
    let lam = v.f("lambda");
    let cap = v.f("shadow_cap_db");
    let radius = v.f("body_radius_m");
    let nu = v.a("nu");
    let want_nu: Vec<f64> = (0..17).map(|n| -1.0 + 0.25 * f64::from(n)).collect();
    c.same("nu", &want_nu, &nu);
    c.close(
        "J1",
        &nu.iter().map(|&x| j1(x)).collect::<Vec<_>>(),
        &v.a("J1"),
    );
    let (g1, g2, gt) = (v.a("grid_nu_1"), v.a("grid_nu_2"), v.a("grid_nu_t"));
    let got: Vec<f64> = (0..gt.len())
        .map(|r| three_edge_j(g1[r], g2[r], gt[r], cap))
        .collect();
    c.close("grid_J", &got, &v.a("grid_J"));

    let col = |k: &str| v.a(&format!("leg_{k}"));
    let (p1x, p1y, p1z, p2x, p2y, p2z, cx_, cy_, top) = (
        col("p1x"),
        col("p1y"),
        col("p1z"),
        col("p2x"),
        col("p2y"),
        col("p2z"),
        col("cx"),
        col("cy"),
        col("top"),
    );
    let mut js = Vec::new();
    let mut fs = Vec::new();
    for i in 0..v.u("n_legs") {
        let cyl = Cylinder {
            c: (cx_[i], cy_[i]),
            radius,
            top: top[i],
        };
        let j = leg_shadow_db(
            v3(p1x[i], p1y[i], p1z[i]),
            v3(p2x[i], p2y[i], p2z[i]),
            &cyl,
            lam,
            cap,
        );
        js.push(j.unwrap_or(0.0));
        fs.push(j.map_or(1.0, |j| pow10(-j / 20.0)));
    }
    c.close("leg_J", &js, &col("J"));
    c.close("leg_factor", &fs, &col("factor"));
    c.finish();
}

/// The static paths of `links` against the `roomsim_paths` layout (with
/// `prefix` for `roomsim_multi`).
fn check_paths(c: &mut Check, v: &Vectors, sim: &Simulator, links: &[(u8, u8)], prefix: &str) {
    let a = |k: &str| v.a(&format!("{prefix}{k}"));
    let (path_link, kind, surf_i, surf_j, obj, face, clutter, tail, npts) = (
        a("path_link"),
        a("kind"),
        a("surf_i"),
        a("surf_j"),
        a("obj"),
        a("face"),
        a("clutter"),
        a("tail"),
        a("npts"),
    );
    let (l_all, sigma_all) = (a("L"), a("sigma"));
    let (cre, cim, c0re, c0im) = (a("coef_re"), a("coef_im"), a("coef0_re"), a("coef0_im"));
    let (px, py, pz) = (a("pts_x"), a("pts_y"), a("pts_z"));
    let link_npaths = a("link_npaths");
    let (h0re, h0im) = (a("H0_re"), a("H0_im"));
    let (tail_x, tail_phi) = (a("tail_x"), a("tail_phi"));
    let n_tail = sim.config().tail_count as usize;
    let no_people = PeopleAt {
        t: 0.0,
        poses: Vec::new(),
        cylinders: Vec::new(),
        points: Vec::new(),
    };
    let room = sim.room();

    let mut row = 0usize;
    let mut pt = 0usize;
    for (li, &(tx, rx)) in links.iter().enumerate() {
        let what = format!("{prefix}link {tx}->{rx}");
        let index = sim
            .link_index(tx, rx)
            .unwrap_or_else(|| panic!("{what}: not a link"));
        let paths = sim.paths(index);
        c.same(
            &format!("{what} path count"),
            &[paths.len() as f64],
            &[link_npaths[li]],
        );
        if paths.len() as f64 != link_npaths[li] {
            // Show both lists of kinds/surfaces so the difference is visible.
            let got: Vec<String> = paths
                .iter()
                .map(|p| format!("{}{:?}{:?}", p.kind.code(), p.surfaces, p.object))
                .collect();
            let n = link_npaths[li] as usize;
            let want: Vec<String> = (row..row + n)
                .map(|r| {
                    format!(
                        "{}[{},{}]({},{})",
                        kind[r], surf_i[r], surf_j[r], obj[r], face[r]
                    )
                })
                .collect();
            c.fail(format!(
                "{what}: rust paths {got:?}\n    python paths {want:?}"
            ));
            return;
        }
        let lo = row;
        let mut got: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        let mut gpts: Vec<[f64; 3]> = Vec::new();
        for p in paths {
            let si = p.surfaces.first().map_or(-1.0, |&s| s as f64);
            let sj = p.surfaces.get(1).map_or(-1.0, |&s| s as f64);
            let (o, f) = p.object.map_or((-1.0, -1.0), |(o, f)| (o as f64, f as f64));
            let sigma = match p.kind {
                PathKind::ObjectPoint => {
                    let (oi, fi) = p.object.unwrap();
                    let og = &room.objects[oi];
                    let fc = Face::of(og, fi);
                    sim.config().object_scatter_scale * object_rho(og.material) * (fc.e1 * fc.e2)
                }
                PathKind::Clutter => sim.clutter()[p.clutter.unwrap()].sigma,
                _ => 0.0,
            };
            for (k, x) in [
                ("path_link", li as f64),
                ("kind", f64::from(p.kind.code())),
                ("surf_i", si),
                ("surf_j", sj),
                ("obj", o),
                ("face", f),
                ("clutter", p.clutter.map_or(-1.0, |x| x as f64)),
                ("tail", p.tail.map_or(-1.0, |x| x as f64)),
                ("npts", p.points.len() as f64),
                ("L", p.length),
                ("sigma", sigma),
            ] {
                got.entry(k).or_default().push(x);
            }
            got.entry("coef_re").or_default().push(p.coeff.re);
            got.entry("coef_im").or_default().push(p.coeff.im);
            got.entry("coef0_re").or_default().push(p.coeff_free.re);
            got.entry("coef0_im").or_default().push(p.coeff_free.im);
            for q in &p.points {
                gpts.push([q.x, q.y, q.z]);
            }
        }
        row += paths.len();
        let hi = row;
        for (k, want) in [
            ("path_link", &path_link),
            ("kind", &kind),
            ("surf_i", &surf_i),
            ("surf_j", &surf_j),
            ("obj", &obj),
            ("face", &face),
            ("clutter", &clutter),
            ("tail", &tail),
            ("npts", &npts),
        ] {
            c.same(&format!("{what} {k}"), &got[k], &want[lo..hi]);
        }
        c.close(&format!("{what} L"), &got["L"], &l_all[lo..hi]);
        c.close(&format!("{what} sigma"), &got["sigma"], &sigma_all[lo..hi]);
        let m = max_mag(&cre[lo..hi], &cim[lo..hi]);
        c.close_m(&format!("{what} coef_re"), &got["coef_re"], &cre[lo..hi], m);
        c.close_m(&format!("{what} coef_im"), &got["coef_im"], &cim[lo..hi], m);
        let m0 = max_mag(&c0re[lo..hi], &c0im[lo..hi]);
        c.close_m(
            &format!("{what} coef0_re"),
            &got["coef0_re"],
            &c0re[lo..hi],
            m0,
        );
        c.close_m(
            &format!("{what} coef0_im"),
            &got["coef0_im"],
            &c0im[lo..hi],
            m0,
        );
        let n = gpts.len();
        let want_pts: Vec<f64> = (pt..pt + n).flat_map(|i| [px[i], py[i], pz[i]]).collect();
        let got_pts: Vec<f64> = gpts.iter().flatten().copied().collect();
        c.close(&format!("{what} points"), &got_pts, &want_pts);
        pt += n;

        let h = sim.channel(index, &no_people);
        c.close_cx(
            &format!("{what} H0"),
            &h,
            &h0re[li * 64..(li + 1) * 64],
            &h0im[li * 64..(li + 1) * 64],
        );
        let pair = sim.links()[index].pair_index;
        let draws = draw_tail(sim.config(), pair);
        let (gx, gphi): (Vec<f64>, Vec<f64>) = draws.into_iter().unzip();
        c.same(
            &format!("{what} tail_x"),
            &gx,
            &tail_x[li * n_tail..(li + 1) * n_tail],
        );
        c.same(
            &format!("{what} tail_phi"),
            &gphi,
            &tail_phi[li * n_tail..(li + 1) * n_tail],
        );
    }
    c.same(
        &format!("{prefix}path rows"),
        &[row as f64],
        &[path_link.len() as f64],
    );
    let cl = sim.clutter();
    c.same(
        &format!("{prefix}clutter_x"),
        &cl.iter().map(|p| p.p.x).collect::<Vec<_>>(),
        &a("clutter_x"),
    );
    c.same(
        &format!("{prefix}clutter_y"),
        &cl.iter().map(|p| p.p.y).collect::<Vec<_>>(),
        &a("clutter_y"),
    );
    c.same(
        &format!("{prefix}clutter_z"),
        &cl.iter().map(|p| p.p.z).collect::<Vec<_>>(),
        &a("clutter_z"),
    );
    c.close(
        &format!("{prefix}clutter_sigma"),
        &cl.iter().map(|p| p.sigma).collect::<Vec<_>>(),
        &a("clutter_sigma"),
    );
}

/// The stored links, `prefix` naming the run (`""`, or `a_` in a set of
/// several runs).
fn links_of(v: &Vectors, prefix: &str) -> Vec<(u8, u8)> {
    (0..v.u(&format!("{prefix}n_links")))
        .map(|i| {
            (
                v.u(&format!("{prefix}link{i}_tx")) as u8,
                v.u(&format!("{prefix}link{i}_rx")) as u8,
            )
        })
        .collect()
}

fn paths_set(name: &str) {
    let v = Vectors::load(name);
    let mut c = Check::new(&v.name);
    let sim = Simulator::new(&v.scene(), v.config()).unwrap();
    c.same(
        "carrier_hz",
        &[sim.world().carrier_hz],
        &[v.f("carrier_hz")],
    );
    c.same("lambda", &[sim.world().lambda], &[v.f("lambda")]);
    c.same(
        "n_surfaces",
        &[sim.world().surfaces.len() as f64],
        &[v.f("n_surfaces")],
    );
    c.same(
        "n_walls",
        &[sim.room().walls.len() as f64],
        &[v.f("n_walls")],
    );
    c.same(
        "implied_walls",
        &[b(sim.room().walls_implied)],
        &[v.f("implied_walls")],
    );
    c.same(
        "n_clutter",
        &[sim.clutter().len() as f64],
        &[v.f("n_clutter")],
    );
    check_paths(&mut c, &v, &sim, &links_of(&v, ""), "");
    c.finish();
}

#[test]
fn roomsim_paths() {
    paths_set("roomsim_paths");
}

#[test]
fn roomsim_paths_hall() {
    paths_set("roomsim_paths_hall");
}

/// Object-specular faces (§6.5 step 1, kind 3), yawed, TE and TM.
#[test]
fn roomsim_paths_specular() {
    let v = Vectors::load("roomsim_paths_specular");
    let n3 = v.a("kind").iter().filter(|&&k| k == 3.0).count();
    assert_eq!(n3, 10, "the set exists for its kind-3 paths");
    paths_set("roomsim_paths_specular");
}

/// Every person's pose at `t`, built independently of the run (stationary
/// people from `stationary_pose`, walkers from their own tracks).
struct Poses<'a> {
    sim: &'a Simulator,
    tracks: Vec<Option<saans_roomsim::PersonTrack<'a>>>,
}

impl<'a> Poses<'a> {
    fn new(sim: &'a Simulator) -> Self {
        let tracks = (0..sim.people().len()).map(|i| sim.track(i)).collect();
        Self { sim, tracks }
    }

    fn at(&mut self, t: f64) -> Vec<Pose> {
        let sim = self.sim;
        sim.people()
            .iter()
            .zip(self.tracks.iter_mut())
            .map(|(p, tr)| match tr {
                None => sim.stationary_pose(p),
                Some(tr) => {
                    let s = tr.at(t);
                    Pose {
                        x: s.x,
                        y: s.y,
                        heading: s.heading,
                        walking: s.walking,
                        speed: s.speed,
                        posture: Posture::Stand,
                    }
                }
            })
            .collect()
    }
}

#[test]
fn roomsim_motion() {
    let v = Vectors::load("roomsim_motion");
    let mut c = Check::new(&v.name);
    let sim = Simulator::new(&v.scene(), v.config()).unwrap();
    let i = v.u("person");
    let p = &sim.people()[i];
    c.same("walking", &[b(p.mode == Mode::Walk)], &[v.f("walking")]);
    c.same(
        "start",
        &[p.start.0, p.start.1],
        &[v.f("start_x"), v.f("start_y")],
    );
    c.same(
        "body draws",
        &[
            p.body.height,
            p.body.f_b,
            p.body.p2p,
            p.body.phi_b,
            p.body.phi_g,
        ],
        &[
            v.f("height"),
            v.f("f_b"),
            v.f("p2p"),
            v.f("phi_b"),
            v.f("phi_g"),
        ],
    );

    let ts = v.a("t");
    let want_t: Vec<f64> = (0..v.u("n_samples")).map(|n| n as f64 / 10.0).collect();
    c.same("t", &want_t, &ts);
    let mut track = sim.track(i).expect("people[i] walks");
    let mut got: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut body: Vec<f64> = Vec::new();
    for &t in &ts {
        let s = track.at(t);
        for (k, x) in [
            ("x", s.x),
            ("y", s.y),
            ("heading", s.heading),
            ("s", b(s.walking)),
            ("v", s.speed),
        ] {
            got.entry(k).or_default().push(x);
        }
        let pose = Pose {
            x: s.x,
            y: s.y,
            heading: s.heading,
            walking: s.walking,
            speed: s.speed,
            posture: Posture::Stand,
        };
        let mut poses: Vec<Pose> = sim
            .people()
            .iter()
            .map(|q| sim.stationary_pose(q))
            .collect();
        poses[i] = pose;
        let at = sim.people_at(t, &poses);
        for q in &at.points[i] {
            body.extend([q.x, q.y, q.z]);
        }
    }
    // One trajectory's coordinates share M (§14).
    let mut xy_want = v.a("x");
    xy_want.extend(v.a("y"));
    let mut xy_got = got["x"].clone();
    xy_got.extend(&got["y"]);
    c.close("xy", &xy_got, &xy_want);
    c.close("heading", &got["heading"], &v.a("heading"));
    c.same("s", &got["s"], &v.a("s"));
    c.close("v", &got["v"], &v.a("v"));
    let (bx, by, bz) = (v.a("body_x"), v.a("body_y"), v.a("body_z"));
    let want_body: Vec<f64> = (0..bx.len()).flat_map(|j| [bx[j], by[j], bz[j]]).collect();
    c.close("body", &body, &want_body);

    // The segments, enumerated by a second, identical track.
    let n_seg = v.u("n_segments");
    let mut track = sim.track(i).unwrap();
    let mut seg: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut t = 0.0;
    for _ in 0..n_seg {
        let s = track.segment(t);
        let (walk, from, to, speed) = match s.kind {
            SegmentKind::Walk { from, to, speed } => (1.0, from, to, speed),
            SegmentKind::Pause { at } => (0.0, at, at, 0.0),
        };
        for (k, x) in [
            ("walk", walk),
            ("t0", s.t0),
            ("t1", s.t1),
            ("x0", from.0),
            ("y0", from.1),
            ("x1", to.0),
            ("y1", to.1),
            ("v", speed),
            ("heading", s.heading),
        ] {
            seg.entry(k).or_default().push(x);
        }
        t = s.t1;
    }
    c.same("seg_walk", &seg["walk"], &v.a("seg_walk"));
    let mut t_want = v.a("seg_t0");
    t_want.extend(v.a("seg_t1"));
    let mut t_got = seg["t0"].clone();
    t_got.extend(&seg["t1"]);
    c.close("seg_t", &t_got, &t_want);
    let mut p_want = Vec::new();
    let mut p_got = Vec::new();
    for k in ["x0", "y0", "x1", "y1"] {
        p_want.extend(v.a(&format!("seg_{k}")));
        p_got.extend(&seg[k]);
    }
    c.close("seg_xy", &p_got, &p_want);
    c.close("seg_v", &seg["v"], &v.a("seg_v"));
    c.close("seg_heading", &seg["heading"], &v.a("seg_heading"));
    c.finish();
}

/// Truth of every transmission against the `tr_*`, `p_*` and `phone_*`
/// arrays, `prefix` naming the run (`""`, or `a_` in a set of several).
fn check_truth(c: &mut Check, v: &Vectors, trs: &[Transmission], prefix: &str) {
    let a = |k: &str| v.a(&format!("{prefix}{k}"));
    let mut got: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for tr in trs {
        for (k, x) in [
            ("tr_t", tr.t),
            ("tr_m", tr.m as f64),
            ("tr_slot", tr.slot as f64),
            ("tr_tx", f64::from(tr.tx)),
        ] {
            got.entry(k).or_default().push(x);
        }
        assert_eq!(tr.truth.t, tr.t);
        for p in &tr.truth.people {
            for (k, x) in [
                ("p_x", p.x),
                ("p_y", p.y),
                ("p_heading", p.heading),
                ("p_walking", b(p.walking)),
                ("p_posture", posture_code(p.posture)),
            ] {
                got.entry(k).or_default().push(x);
            }
        }
        if let Some(ph) = tr.truth.phone {
            got.entry("phone_x").or_default().push(ph.x);
            got.entry("phone_y").or_default().push(ph.y);
            got.entry("phone_z").or_default().push(ph.z);
        }
    }
    for k in ["tr_m", "tr_slot", "tr_tx", "p_walking", "p_posture"] {
        c.same(
            &format!("{prefix}{k}"),
            got.get(k).map_or(&[][..], Vec::as_slice),
            &a(k),
        );
    }
    c.close(&format!("{prefix}tr_t"), &got["tr_t"], &a("tr_t"));
    let mut xy_got = got.get("p_x").cloned().unwrap_or_default();
    xy_got.extend(got.get("p_y").cloned().unwrap_or_default());
    let mut xy_want = a("p_x");
    xy_want.extend(a("p_y"));
    c.close(&format!("{prefix}p_xy"), &xy_got, &xy_want);
    c.close(
        &format!("{prefix}p_heading"),
        got.get("p_heading").map_or(&[][..], Vec::as_slice),
        &a("p_heading"),
    );
    let mut ph_got = Vec::new();
    let mut ph_want = Vec::new();
    for k in ["phone_x", "phone_y", "phone_z"] {
        ph_got.extend(got.get(k).cloned().unwrap_or_default());
        ph_want.extend(a(k));
    }
    c.close(&format!("{prefix}phone"), &ph_got, &ph_want);
}

/// Body points of every transmission (transmission-major, people in order,
/// table order) from poses built independently of the run, against the
/// `body_x/_y/_z` arrays; one run's points share `M` (§14).
fn check_body(c: &mut Check, v: &Vectors, sim: &Simulator, trs: &[Transmission], prefix: &str) {
    let mut poses = Poses::new(sim);
    let mut body = Vec::new();
    for tr in trs {
        let at = sim.people_at(tr.t, &poses.at(tr.t));
        for pts in &at.points {
            for q in pts {
                body.extend([q.x, q.y, q.z]);
            }
        }
    }
    let a = |k: &str| v.a(&format!("{prefix}{k}"));
    let (bx, by, bz) = (a("body_x"), a("body_y"), a("body_z"));
    let want_body: Vec<f64> = (0..bx.len()).flat_map(|j| [bx[j], by[j], bz[j]]).collect();
    c.close(&format!("{prefix}body"), &body, &want_body);
}

#[test]
fn roomsim_channel() {
    let v = Vectors::load("roomsim_channel");
    let mut c = Check::new(&v.name);
    let cfg = v.config();
    assert!(
        !cfg.impair,
        "roomsim_channel is generated with impair false"
    );
    let sim = Simulator::new(&v.scene(), cfg).unwrap();
    let trs: Vec<Transmission> = sim.run().collect();
    c.same(
        "n_transmissions",
        &[trs.len() as f64],
        &[v.f("n_transmissions")],
    );
    let pkts: Vec<_> = trs.iter().flat_map(|t| t.packets.iter()).collect();
    c.same("n_packets", &[pkts.len() as f64], &[v.f("n_packets")]);
    let mut got: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for p in &pkts {
        assert!(!p.lost);
        let r = p.received.as_ref().unwrap();
        for (k, x) in [
            ("t", p.t),
            ("m", p.m as f64),
            ("slot", p.slot as f64),
            ("tx", f64::from(p.tx)),
            ("rx", f64::from(p.rx)),
            ("link_index", p.link_index as f64),
            ("seq", f64::from(r.seq)),
            ("rssi", f64::from(r.rssi)),
            ("rssi_true", r.rssi_true),
        ] {
            got.entry(k).or_default().push(x);
        }
    }
    for k in ["m", "slot", "tx", "rx", "link_index", "seq"] {
        c.same(k, &got[k], &v.a(k));
    }
    c.close("t", &got["t"], &v.a("t"));
    c.close("rssi_true", &got["rssi_true"], &v.a("rssi_true"));
    let (want_rssi, want_rt) = (v.a("rssi"), v.a("rssi_true"));
    for i in 0..pkts.len().min(want_rssi.len()) {
        c.rounded(
            &format!("rssi[{i}]"),
            got["rssi"][i],
            want_rssi[i],
            got["rssi_true"][i],
            want_rt[i],
        );
    }
    let (hre, him) = (v.a("H_re"), v.a("H_im"));
    for (i, p) in pkts.iter().enumerate().take(hre.len() / 64) {
        c.close_cx(
            &format!("H[{i}]"),
            &p.received.as_ref().unwrap().h,
            &hre[i * 64..(i + 1) * 64],
            &him[i * 64..(i + 1) * 64],
        );
    }
    check_truth(&mut c, &v, &trs, "");
    check_body(&mut c, &v, &sim, &trs, "");
    c.finish();
}

/// The received packets of `pkts` (§10 order, lost included) against a
/// set's per-received-packet arrays, `prefix` naming them (`nq_` for the
/// quantize-false run of `roomsim_switches`): `rssi` (identical unless the
/// pre-rounding `rssi_true + rj` is near a tie, `rj` the packet's §12.2
/// draw), `rssi_true` (0.0 for a lost packet), `buffer_pre`, `buffer`
/// (int8 under the tie rule, or the floats when `quantize` is off, which
/// equal `buffer_pre`) and every datagram byte (§14). The lost flags must
/// already agree; returns the number of received packets.
fn check_received(c: &mut Check, v: &Vectors, pkts: &[&Packet], rj: &[f64], prefix: &str) -> usize {
    let a = |k: &str| v.a(&format!("{prefix}{k}"));
    let w_lost = a("lost");
    if pkts.iter().map(|p| b(p.lost)).ne(w_lost.iter().copied()) {
        c.fail(format!(
            "{prefix}lost flags differ, so the received packets cannot be paired"
        ));
        return 0;
    }
    let (w_rssi, w_rt) = (a("rssi"), a("rssi_true"));
    let (bpre_re, bpre_im, buf_re, buf_im) = (
        a("buffer_pre_re"),
        a("buffer_pre_im"),
        a("buffer_re"),
        a("buffer_im"),
    );
    let (dgram, dgram_len) = (a("dgram"), a("dgram_len"));
    let rt_got: Vec<f64> = pkts
        .iter()
        .map(|p| p.received.as_ref().map_or(0.0, |r| r.rssi_true))
        .collect();
    c.close(&format!("{prefix}rssi_true"), &rt_got, &w_rt);
    let mut n = 0usize; // received packet index
    let mut nd = 0usize; // datagram index
    let mut off = 0usize; // datagram byte offset
    for (i, p) in pkts.iter().enumerate() {
        let Some(r) = &p.received else {
            c.same(&format!("{prefix}rssi[{i}] (lost)"), &[0.0], &[w_rssi[i]]);
            continue;
        };
        let (rssi_pre_got, rssi_pre_want) = (r.rssi_true + rj[i], w_rt[i] + rj[i]);
        c.rounded(
            &format!("{prefix}rssi[{i}]"),
            f64::from(r.rssi),
            w_rssi[i],
            rssi_pre_got,
            rssi_pre_want,
        );
        let g = n * 128..(n + 1) * 128;
        if g.end > bpre_re.len() {
            c.fail(format!(
                "{prefix}buffer_pre: no entries for received packet {n}"
            ));
            return n + 1;
        }
        let (pr, pi) = (&bpre_re[g.clone()], &bpre_im[g.clone()]);
        c.close_cx(&format!("{prefix}buffer_pre[{n}]"), &r.buffer_pre, pr, pi);
        match &r.buffer {
            Buffer::Int8(q) => {
                for e in 0..128 {
                    for (got_q, want_q, pre_got, pre_want, part) in [
                        (q[e].0, buf_re[n * 128 + e], r.buffer_pre[e].re, pr[e], "re"),
                        (q[e].1, buf_im[n * 128 + e], r.buffer_pre[e].im, pi[e], "im"),
                    ] {
                        c.rounded(
                            &format!("{prefix}buffer[{n}][{e}].{part}"),
                            f64::from(got_q),
                            want_q,
                            pre_got,
                            pre_want,
                        );
                    }
                }
            }
            Buffer::Float(f) => {
                c.close_cx(
                    &format!("{prefix}buffer[{n}]"),
                    f,
                    &buf_re[g.clone()],
                    &buf_im[g.clone()],
                );
                if *f != r.buffer_pre {
                    c.fail(format!(
                        "{prefix}buffer[{n}]: quantize off but not buffer_pre"
                    ));
                }
            }
        }
        if let Some(d) = &r.datagram {
            // Bytes identical, except the rssi byte and payload bytes whose
            // value took a tie exemption above.
            if nd >= dgram_len.len() {
                c.fail(format!(
                    "{prefix}dgram: a datagram the reference does not have (packet {i})"
                ));
                return n + 1;
            }
            let len = dgram_len[nd] as usize;
            c.same(
                &format!("{prefix}dgram_len[{nd}]"),
                &[d.len() as f64],
                &[dgram_len[nd]],
            );
            let want = &dgram[off..off + len];
            for (j, (&gb, &wb)) in d.iter().zip(want).enumerate() {
                c.exact += 1;
                if f64::from(gb) == wb {
                    continue;
                }
                let exempt = match j {
                    16 => near_tie(rssi_pre_got) || near_tie(rssi_pre_want),
                    j if j >= 20 => {
                        let e = (j - 20) / 2;
                        let (pg, pw) = if (j - 20) % 2 == 0 {
                            (r.buffer_pre[e].im, pi[e])
                        } else {
                            (r.buffer_pre[e].re, pr[e])
                        };
                        near_tie(pg) || near_tie(pw)
                    }
                    _ => false,
                };
                if exempt {
                    c.ties += 1;
                } else {
                    c.fail(format!("{prefix}dgram[{nd}][{j}]: got {gb}, want {wb}"));
                }
            }
            off += len;
            nd += 1;
        }
        n += 1;
    }
    c.same(
        &format!("{prefix}received buffers"),
        &[(n * 128) as f64],
        &[bpre_re.len() as f64],
    );
    c.same(
        &format!("{prefix}datagrams"),
        &[nd as f64, off as f64],
        &[dgram_len.len() as f64, dgram.len() as f64],
    );
    n
}

#[test]
fn roomsim_impair() {
    let v = Vectors::load("roomsim_impair");
    let mut c = Check::new(&v.name);
    let cfg = v.config();
    assert!(cfg.impair && !cfg.smoothing && cfg.quantize);
    let sim = Simulator::new(&v.scene(), cfg.clone()).unwrap();
    let pkts: Vec<_> = sim.packets().collect();
    c.same("n_packets", &[pkts.len() as f64], &[v.f("n_packets")]);

    // The §12.2 draws, re-drawn independently from stream 1000 + link_index.
    let mut rngs: BTreeMap<usize, Rng> = BTreeMap::new();
    let mut got: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for p in &pkts {
        let rng = rngs
            .entry(p.link_index)
            .or_insert_with(|| Rng::stream(cfg.seed, 1000 + p.link_index as u64));
        let d = PacketDraws::draw(rng, &cfg);
        let r = p.received.as_ref();
        for (k, x) in [
            ("t", p.t),
            ("tx", f64::from(p.tx)),
            ("rx", f64::from(p.rx)),
            ("link_index", p.link_index as f64),
            ("lost", b(p.lost)),
            ("agc_db", p.agc_db),
            ("seq", r.map_or(-1.0, |r| f64::from(r.seq))),
            ("noise_floor", r.map_or(0.0, |r| f64::from(r.noise_floor))),
            ("u_loss", d.u_loss),
            ("u_jump", d.u_jump),
            ("u_dir", d.u_dir),
            ("phi0", d.phi0),
            ("tau", d.tau),
            ("rj", d.rj),
        ] {
            got.entry(k).or_default().push(x);
        }
        got.entry("noise").or_default().extend(&d.noise);
    }
    for k in [
        "tx",
        "rx",
        "link_index",
        "lost",
        "agc_db",
        "seq",
        "noise_floor",
        "u_loss",
        "u_jump",
        "u_dir",
        "phi0",
        "tau",
        "rj",
        "noise",
    ] {
        c.same(k, &got[k], &v.a(k));
    }
    c.close("t", &got["t"], &v.a("t"));

    // Received packets: rssi, buffers and datagrams.
    let refs: Vec<&Packet> = pkts.iter().collect();
    let n = check_received(&mut c, &v, &refs, &got["rj"], "");
    c.same("n_received", &[n as f64], &[v.f("n_received")]);

    c.same(
        "buffer_k",
        &buffer_k_all()
            .iter()
            .map(|&k| f64::from(k))
            .collect::<Vec<_>>(),
        &v.a("buffer_k"),
    );
    c.same(
        "buffer_active",
        &buffer_active_all()
            .iter()
            .map(|&x| b(x))
            .collect::<Vec<_>>(),
        &v.a("buffer_active"),
    );
    c.same(
        "sim.buffer_k",
        &sim.buffer_k()
            .iter()
            .map(|&k| f64::from(k))
            .collect::<Vec<_>>(),
        &v.a("buffer_k"),
    );

    // Devices (§12.1), node id ascending.
    let nodes = v.a("dev_node");
    let mut dev: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut rdev: Vec<Cx> = Vec::new();
    for &id in &nodes {
        let d = sim.device(id as u8).expect("a node");
        dev.entry("alpha").or_default().extend(&d.alpha);
        dev.entry("beta").or_default().extend(&d.beta);
        dev.entry("gamma").or_default().extend(&d.gamma);
        dev.entry("delta").or_default().extend(&d.delta);
        dev.entry("g_db").or_default().push(d.g_dev_db);
        dev.entry("nf").or_default().push(d.nf_dbm);
        dev.entry("tau").or_default().push(d.tau_dev);
        rdev.extend(&d.rdev);
    }
    c.same(
        "dev_node",
        &sim.room()
            .nodes
            .iter()
            .map(|n| f64::from(n.id))
            .collect::<Vec<_>>(),
        &nodes,
    );
    for k in ["alpha", "beta", "gamma", "delta", "g_db", "nf", "tau"] {
        c.same(&format!("dev_{k}"), &dev[k], &v.a(&format!("dev_{k}")));
    }
    let (rre, rim) = (v.a("dev_rdev_re"), v.a("dev_rdev_im"));
    for i in 0..nodes.len() {
        c.close_cx(
            &format!("dev_rdev[{i}]"),
            &rdev[i * 64..(i + 1) * 64],
            &rre[i * 64..(i + 1) * 64],
            &rim[i * 64..(i + 1) * 64],
        );
    }
    // The noise floor is an integer drawn with round-half-even.
    for &nf in &dev["nf"] {
        assert_eq!(nf, round_half_even(nf));
    }
    c.finish();
}

#[test]
fn roomsim_multi() {
    let v = Vectors::load("roomsim_multi");
    let mut c = Check::new(&v.name);
    let cfg = v.config();
    assert!(!cfg.impair, "roomsim_multi is generated with impair false");
    let sim = Simulator::new(&v.scene(), cfg).unwrap();
    c.same(
        "implied_walls",
        &[b(sim.room().walls_implied)],
        &[v.f("implied_walls")],
    );
    c.same(
        "n_walls",
        &[sim.room().walls.len() as f64],
        &[v.f("n_walls")],
    );
    c.same("n_people", &[sim.people().len() as f64], &[v.f("n_people")]);
    for p in sim.people() {
        let i = p.index;
        let k = |s: &str| v.f(&format!("person{i}_{s}"));
        c.same(
            &format!("person{i}"),
            &[
                p.start.0,
                p.start.1,
                b(p.mode == Mode::Walk),
                p.body.height,
                p.body.f_b,
                p.body.p2p,
                p.body.phi_b,
                p.body.phi_g,
            ],
            &[
                k("start_x"),
                k("start_y"),
                k("walking"),
                k("height"),
                k("f_b"),
                k("p2p"),
                k("phi_b"),
                k("phi_g"),
            ],
        );
    }
    let links = links_of(&v, "");
    let trs: Vec<Transmission> = sim.run().collect();
    c.same(
        "n_transmissions",
        &[trs.len() as f64],
        &[v.f("n_transmissions")],
    );
    let pkts: Vec<_> = trs
        .iter()
        .flat_map(|t| t.packets.iter())
        .filter(|p| links.contains(&(p.tx, p.rx)))
        .collect();
    c.same("n_packets", &[pkts.len() as f64], &[v.f("n_packets")]);
    let mut got: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for p in &pkts {
        for (k, x) in [
            ("t", p.t),
            ("m", p.m as f64),
            ("slot", p.slot as f64),
            ("tx", f64::from(p.tx)),
            ("rx", f64::from(p.rx)),
            ("link_index", p.link_index as f64),
        ] {
            got.entry(k).or_default().push(x);
        }
    }
    for k in ["m", "slot", "tx", "rx", "link_index"] {
        c.same(k, &got[k], &v.a(k));
    }
    c.close("t", &got["t"], &v.a("t"));
    let (hre, him) = (v.a("H_re"), v.a("H_im"));
    for (i, p) in pkts.iter().enumerate().take(hre.len() / 64) {
        let r = p.received.as_ref().expect("impair is off: nothing is lost");
        c.close_cx(
            &format!("H[{i}] t={} {}->{}", p.t, p.tx, p.rx),
            &r.h,
            &hre[i * 64..(i + 1) * 64],
            &him[i * 64..(i + 1) * 64],
        );
    }
    check_truth(&mut c, &v, &trs, "");
    check_paths(&mut c, &v, &sim, &links, "paths_");
    c.finish();
}

/// The packets of one `roomsim_switches` run against its per-packet arrays
/// (`prefix` "" or "nq_"), lost included, and then its received packets.
fn check_switch_run(c: &mut Check, v: &Vectors, sim: &Simulator, pkts: &[&Packet], prefix: &str) {
    let cfg = sim.config();
    // The §12.2 draws, re-drawn independently from stream 1000 + link_index:
    // the rssi tie rule needs each packet's jitter `rj`.
    let mut rngs: BTreeMap<usize, Rng> = BTreeMap::new();
    let mut got: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for p in pkts {
        let rng = rngs
            .entry(p.link_index)
            .or_insert_with(|| Rng::stream(cfg.seed, 1000 + p.link_index as u64));
        let d = PacketDraws::draw(rng, cfg);
        let r = p.received.as_ref();
        for (k, x) in [
            ("t", p.t),
            ("m", p.m as f64),
            ("slot", p.slot as f64),
            ("tx", f64::from(p.tx)),
            ("rx", f64::from(p.rx)),
            ("link_index", p.link_index as f64),
            ("lost", b(p.lost)),
            ("agc_db", p.agc_db),
            ("seq", r.map_or(-1.0, |r| f64::from(r.seq))),
            ("noise_floor", r.map_or(0.0, |r| f64::from(r.noise_floor))),
            ("rj", d.rj),
        ] {
            got.entry(k).or_default().push(x);
        }
    }
    let a = |k: &str| v.a(&format!("{prefix}{k}"));
    for k in [
        "m",
        "slot",
        "tx",
        "rx",
        "link_index",
        "lost",
        "agc_db",
        "seq",
        "noise_floor",
    ] {
        c.same(
            &format!("{prefix}{k}"),
            got.get(k).map_or(&[][..], Vec::as_slice),
            &a(k),
        );
    }
    c.close(
        &format!("{prefix}t"),
        got.get("t").map_or(&[][..], Vec::as_slice),
        &a("t"),
    );
    let rj = got.remove("rj").unwrap_or_default();
    let n = check_received(c, v, pkts, &rj, prefix);
    c.same(
        &format!("{prefix}n_received"),
        &[n as f64],
        &[v.f(&format!("{prefix}n_received"))],
    );
}

/// Every non-default branch the other sets leave alone (§14
/// `roomsim_switches`): the hall, impaired, `roles` with a `tx`-only node 0
/// and an `rx`-only node 7, `tx_period`, `loss_prob_link` (link (0, 7)
/// always lost), `smoothing` on, `cpe`/`ripple`/`ltf_merge`/`snr_noise`
/// off, `timing_offset_dev_s`, channel 11, 64 firmware subcarriers, 16
/// body points, and the phone carried by the seated `people[1]` on the
/// sofa with `start_heading_rad` 0.7; plus the same configuration with
/// `quantize` false for its first 0.05 s.
#[test]
fn roomsim_switches() {
    let v = Vectors::load("roomsim_switches");
    let mut c = Check::new(&v.name);
    let cfg = v.config();
    assert!(cfg.impair && cfg.quantize && cfg.smoothing);
    assert!(!cfg.cpe && !cfg.ripple && !cfg.ltf_merge && !cfg.snr_noise);
    let sim = Simulator::new(&v.scene(), cfg.clone()).unwrap();
    c.same(
        "carrier_hz",
        &[sim.world().carrier_hz],
        &[v.f("carrier_hz")],
    );
    c.same("lambda", &[sim.world().lambda], &[v.f("lambda")]);
    c.same(
        "counts",
        &[
            sim.room().nodes.len() as f64,
            sim.links().len() as f64,
            sim.transmitters().len() as f64,
            sim.people().len() as f64,
        ],
        &[
            v.f("n_nodes"),
            v.f("n_links"),
            v.f("n_transmitters"),
            v.f("n_people"),
        ],
    );
    assert_eq!(
        sim.transmitters(),
        &[0, 1, 2, 3, 4, 5, 6],
        "node 7 is rx only"
    );
    assert_eq!(sim.receivers(), &[1, 2, 3, 4, 5, 6, 7], "node 0 is tx only");
    for p in sim.people() {
        let i = p.index;
        c.same(
            &format!("person{i}"),
            &[b(p.mode == Mode::Walk), p.body.height],
            &[
                v.f(&format!("person{i}_walking")),
                v.f(&format!("person{i}_height")),
            ],
        );
        assert_eq!(p.posture.name(), v.s(&format!("person{i}_posture")));
    }
    let carrier = &sim.people()[1];
    assert!(
        carrier.mode != Mode::Walk && carrier.posture == Posture::Sit,
        "the phone's carrier sits on the sofa"
    );

    let trs: Vec<Transmission> = sim.run().collect();
    c.same(
        "n_transmissions",
        &[trs.len() as f64],
        &[v.f("n_transmissions")],
    );
    let pkts: Vec<&Packet> = trs.iter().flat_map(|t| t.packets.iter()).collect();
    c.same(
        "n_packets",
        &[
            pkts.len() as f64,
            pkts.iter().filter(|p| p.lost).count() as f64,
        ],
        &[v.f("n_packets"), v.f("n_lost")],
    );
    assert!(
        pkts.iter()
            .filter(|p| (p.tx, p.rx) == (0, 7))
            .all(|p| p.lost),
        "loss_prob_link 1.0 loses every packet on (0, 7)"
    );
    check_switch_run(&mut c, &v, &sim, &pkts, "");
    check_truth(&mut c, &v, &trs, "");

    // The seated carrier keeps its start_heading_rad and carries the phone
    // at phone_offset_m in its body frame, z scaled by 0.72 (§9): rebuilt
    // here from the reference's own truth.
    let (px, py, ph) = (v.a("p_x"), v.a("p_y"), v.a("p_heading"));
    let (fx, fy, fz) = (v.a("phone_x"), v.a("phone_y"), v.a("phone_z"));
    let o = cfg.phone_offset_m;
    for t in 0..trs.len() {
        let j = 2 * t + 1;
        assert_eq!(ph[j], 0.7, "people[1]'s heading in transmission {t}");
        let (cs, sn) = (ph[j].cos(), ph[j].sin());
        let want = [
            px[j] + (o.0 * cs - o.1 * sn),
            py[j] + (o.0 * sn + o.1 * cs),
            o.2 * 0.72,
        ];
        for (g, w) in [fx[t], fy[t], fz[t]].iter().zip(want) {
            assert!(
                (g - w).abs() < 1e-12,
                "phone in transmission {t}: {g} vs {w}"
            );
        }
    }

    // Body points, 16 per person, from poses built independently of the run.
    let mut poses = Poses::new(&sim);
    let mut body = Vec::new();
    for tr in &trs {
        let at = sim.people_at(tr.t, &poses.at(tr.t));
        for pts in &at.points {
            assert_eq!(pts.len(), 16);
            for q in pts {
                body.extend([q.x, q.y, q.z]);
            }
        }
    }
    let (bx, by, bz) = (v.a("body_x"), v.a("body_y"), v.a("body_z"));
    let want_body: Vec<f64> = (0..bx.len()).flat_map(|j| [bx[j], by[j], bz[j]]).collect();
    c.close("body", &body, &want_body);

    // Devices (§12.1), node id ascending; ripple off leaves Rdev = (1, 0).
    let nodes = v.a("dev_node");
    c.same(
        "dev_node",
        &sim.room()
            .nodes
            .iter()
            .map(|n| f64::from(n.id))
            .collect::<Vec<_>>(),
        &nodes,
    );
    let mut dev: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for &id in &nodes {
        let d = sim.device(id as u8).expect("a node");
        dev.entry("g_db").or_default().push(d.g_dev_db);
        dev.entry("nf").or_default().push(d.nf_dbm);
        dev.entry("tau").or_default().push(d.tau_dev);
        assert!(d.rdev.iter().all(|z| z.re == 1.0 && z.im == 0.0));
    }
    for k in ["g_db", "nf", "tau"] {
        c.same(&format!("dev_{k}"), &dev[k], &v.a(&format!("dev_{k}")));
    }

    // The quantize-false run: float buffers equal to buffer_pre, no
    // datagram, and the same packets as the main run's first 0.05 s.
    c.same("nq_quantize", &[0.0], &[v.f("nq_quantize")]);
    let mut nq_cfg = cfg.clone();
    nq_cfg.quantize = false;
    nq_cfg.duration_s = v.f("nq_duration_s");
    let nq = Simulator::new(&v.scene(), nq_cfg).unwrap();
    let nq_trs: Vec<Transmission> = nq.run().collect();
    assert!(nq_trs.len() < trs.len());
    for (a, b) in nq_trs.iter().zip(&trs) {
        assert_eq!(
            a.truth, b.truth,
            "the main run's truth, so not stored again"
        );
    }
    let nq_pkts: Vec<Packet> = nq_trs.into_iter().flat_map(|t| t.packets).collect();
    c.same(
        "nq_n_packets",
        &[nq_pkts.len() as f64],
        &[v.f("nq_n_packets")],
    );
    assert!(nq_pkts.iter().all(|p| p
        .received
        .as_ref()
        .is_none_or(|r| r.datagram.is_none() && matches!(r.buffer, Buffer::Float(_)))));
    c.same("nq_n_datagrams", &[0.0], &[v.f("nq_n_datagrams")]);
    let nq_refs: Vec<&Packet> = nq_pkts.iter().collect();
    check_switch_run(&mut c, &v, &nq, &nq_refs, "nq_");
    for (p, q) in nq_pkts.iter().zip(&pkts) {
        assert_eq!(
            (p.t, p.tx, p.rx, p.lost, p.agc_db),
            (q.t, q.tx, q.rx, q.lost, q.agc_db)
        );
        if let (Some(a), Some(b)) = (&p.received, &q.received) {
            assert_eq!(
                a.buffer_pre, b.buffer_pre,
                "quantize changes nothing before step 9"
            );
        }
    }
    c.finish();
}

/// The non-default geometry, object, tail, shadowing and people branches
/// (§14 `roomsim_physics_switches`): two runs on the hall, `impair` false,
/// each stored under its prefix (`a_`, `b_`) with its own configuration,
/// links, static paths, `H` and truth. Run `a`: `reflection_order` 1,
/// `floor_ceiling`, `object_specular` and `tail` off,
/// `object_scatter_scale` 0.5, `clutter_count` 5, `n_people` 3 (people[2]
/// enters at the door and stays, heading 2.0), the sofa's sitter standing.
/// Run `b`: `reflection_order` 0, `transmission`, `object_scatter`,
/// `object_blockage` and `shadowing` off.
#[test]
fn roomsim_physics_switches() {
    let v = Vectors::load("roomsim_physics_switches");
    let mut c = Check::new(&v.name);
    let variants: Vec<&str> = v.s("variants").split(',').collect();
    assert_eq!(variants, ["a", "b"]);
    assert_eq!(v.u("n_variants"), variants.len());
    // No run-level scalar outside a prefix: each run carries its own.
    for k in ["seed", "duration_s", "n_links"] {
        assert!(!v.kv.contains_key(k), "{}: top-level {k}", v.name);
    }
    assert!(!v.kv.keys().any(|k| k.starts_with("cfg.")));

    let a = v.config_of("a_");
    assert!(
        a.reflection_order == 1
            && !a.floor_ceiling
            && !a.object_specular
            && a.object_scatter_scale == 0.5
            && a.clutter_count == 5
            && !a.tail
            && a.n_people == Some(3)
            && a.people_mode.get(&2) == Some(&Mode::Stationary)
            && a.people_posture.get(&1) == Some(&Posture::Stand)
            && a.start_heading_rad.get(&2) == Some(&2.0)
            && a.transmission
            && a.object_scatter
            && a.object_blockage
            && a.shadowing,
        "run a's switches"
    );
    let bc = v.config_of("b_");
    assert!(
        bc.reflection_order == 0
            && !bc.transmission
            && !bc.object_scatter
            && !bc.object_blockage
            && !bc.shadowing
            && bc.floor_ceiling
            && bc.object_specular
            && bc.tail,
        "run b's switches"
    );

    let scene = v.scene();
    for name in variants {
        let pre = format!("{name}_");
        let k = |s: &str| format!("{pre}{s}");
        let cfg = v.config_of(&pre);
        assert!(!cfg.impair, "{name}: generated with impair false");
        c.same(
            &k("seed/duration_s"),
            &[cfg.seed as f64, cfg.duration_s],
            &[v.f(&k("seed")), v.f(&k("duration_s"))],
        );
        let sim = Simulator::new(&scene, cfg).unwrap();
        c.same(
            &k("carrier_hz/lambda"),
            &[sim.world().carrier_hz, sim.world().lambda],
            &[v.f("carrier_hz"), v.f("lambda")],
        );
        c.same(
            &k("counts"),
            &[
                sim.clutter().len() as f64,
                sim.world().surfaces.len() as f64,
                sim.room().walls.len() as f64,
                sim.people().len() as f64,
                sim.config().body_points as f64,
            ],
            &[
                v.f(&k("n_clutter")),
                v.f(&k("n_surfaces")),
                v.f(&k("n_walls")),
                v.f(&k("n_people")),
                v.f(&k("body_points")),
            ],
        );
        for p in sim.people() {
            let i = p.index;
            let q = |s: &str| v.f(&k(&format!("person{i}_{s}")));
            c.same(
                &k(&format!("person{i}")),
                &[
                    p.start.0,
                    p.start.1,
                    b(p.mode == Mode::Walk),
                    p.on_object.map_or(-1.0, |o| o as f64),
                    p.heading0,
                    p.body.height,
                    p.body.f_b,
                    p.body.p2p,
                    p.body.phi_b,
                    p.body.phi_g,
                ],
                &[
                    q("start_x"),
                    q("start_y"),
                    q("walking"),
                    q("on_object"),
                    q("heading0"),
                    q("height"),
                    q("f_b"),
                    q("p2p"),
                    q("phi_b"),
                    q("phi_g"),
                ],
            );
            assert_eq!(
                p.posture.name(),
                v.s(&k(&format!("person{i}_posture"))),
                "{name}: people[{i}]'s posture"
            );
        }

        let links = links_of(&v, &pre);
        assert_eq!(
            v.s(&k("links")),
            links
                .iter()
                .map(|(x, y)| format!("{x}-{y}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        check_paths(&mut c, &v, &sim, &links, &pre);

        let trs: Vec<Transmission> = sim.run().collect();
        c.same(
            &k("n_transmissions"),
            &[trs.len() as f64],
            &[v.f(&k("n_transmissions"))],
        );
        let pkts: Vec<&Packet> = trs
            .iter()
            .flat_map(|t| t.packets.iter())
            .filter(|p| links.contains(&(p.tx, p.rx)))
            .collect();
        c.same(
            &k("n_packets"),
            &[pkts.len() as f64],
            &[v.f(&k("n_packets"))],
        );
        let mut got: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for p in &pkts {
            for (key, x) in [
                ("t", p.t),
                ("m", p.m as f64),
                ("slot", p.slot as f64),
                ("tx", f64::from(p.tx)),
                ("rx", f64::from(p.rx)),
                ("link_index", p.link_index as f64),
            ] {
                got.entry(key).or_default().push(x);
            }
        }
        for key in ["m", "slot", "tx", "rx", "link_index"] {
            c.same(
                &k(key),
                got.get(key).map_or(&[][..], Vec::as_slice),
                &v.a(&k(key)),
            );
        }
        c.close(
            &k("t"),
            got.get("t").map_or(&[][..], Vec::as_slice),
            &v.a(&k("t")),
        );
        let (hre, him) = (v.a(&k("H_re")), v.a(&k("H_im")));
        c.same(
            &k("H bins"),
            &[(pkts.len() * 64) as f64],
            &[hre.len() as f64],
        );
        for (i, p) in pkts.iter().enumerate().take(hre.len() / 64) {
            let r = p.received.as_ref().expect("impair is off: nothing is lost");
            c.close_cx(
                &k(&format!("H[{i}] t={} {}->{}", p.t, p.tx, p.rx)),
                &r.h,
                &hre[i * 64..(i + 1) * 64],
                &him[i * 64..(i + 1) * 64],
            );
        }
        check_truth(&mut c, &v, &trs, &pre);
        check_body(&mut c, &v, &sim, &trs, &pre);
    }
    c.finish();
}

/// `cexpj` and `cmul` are the §1 helpers the reference uses too; a sanity
/// check that the phase convention (a longer path lags) is the reference's.
#[test]
fn phase_convention() {
    let f = carrier_hz(6);
    let l = 3.0;
    let th = saans_roomsim::numerics::theta(f, l);
    assert_eq!(th, ((-saans_roomsim::numerics::TWO_PI) * f) * l / C);
    let z = cmul(cexpj(th), cexpj(-th));
    assert!((z.re - 1.0).abs() < 1e-15 && z.im.abs() < 1e-15);
}
