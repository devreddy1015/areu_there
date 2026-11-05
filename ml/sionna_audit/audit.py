"""Phase 10a: the Sionna RT audit of Room Sim on 10 randomized SIM scenes.

    .venv/bin/python -m ml.sionna_audit.audit                       # the full audit
    .venv/bin/python -m ml.sionna_audit.audit --scenes 3 --quick    # a small run

Draws `--scenes` rooms from dataset v1's test split with a fixed seed (at
least one L-shaped room when the split has any), runs Room Sim's Python
reference and Sionna RT 2.2.0 on each, and compares them link by link
(spec/sionna_audit.md):

* `shell` — walls, floor and ceiling only (Room Sim with every object
  effect off; Sionna without furniture): the P.2040 slab and image-method
  machinery alone.
* `furnished` — the full scene, furniture as half-space boxes in Sionna
  (Room Sim's single-interface reflection; opaque).
* `furnished_panel` — the same with thin-panel furniture (sensitivity of
  the comparison to that choice).
* `full` — Room Sim's whole static model (point scatterers, clutter, tail)
  against Sionna with diffuse reflection on (`--diffuse-s`): delay spread
  and frequency selectivity only, a qualitative check.
* `slab_check` — one wall at several angles: Room Sim's §5.1 formula against
  Sionna's, as formulas and as ray-traced paths.

Writes the summary JSON (`--out`, small, every number labelled SIM) and,
with `--markdown`, the result tables. Both simulators are models: an
agreement here is an agreement between two simulators, never a property of
a real room.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml import roomsim_dataset as rd  # noqa: E402
from ml.sionna_audit import metrics as M  # noqa: E402
from ml.sionna_audit import roomsim_side as rss  # noqa: E402
from ml.sionna_audit import scene_to_sionna as s2s  # noqa: E402
from wifisense.roomsim import Config, load_scene  # noqa: E402
from wifisense.roomsim import materials as rs_materials  # noqa: E402
from wifisense.roomsim.numerics import C, carrier_hz as rs_carrier_hz  # noqa: E402

LABEL = "SIM"
SPEC = "sionna audit v1 (spec/sionna_audit.md)"
DEFAULT_DATASET = "/home/Aman/are_u_theone/datasets/roomsim-v1"
DEFAULT_OUT = os.path.join(ROOT, "ml", "results", "phase10", "sionna_audit.json")
#: Scene draw (numpy default_rng over the test split) and Sionna's sampler.
SCENE_SEED = 10
SIONNA_SEED = 42
N_SCENES = 10
#: Sionna specular runs: interactions per path (reflections + refractions).
#: Room Sim's order-2 paths, plus up to two refractions (an L's re-entrant
#: walls, a fabric box), fit in 4; 4 also bounds Sionna's "all orders"
#: reference. Depth 6 was tried: it fills Sionna's candidate hash table with
#: chains that leave the room, and real candidates were then lost at random.
SPEC_MAX_DEPTH = 4
SPEC_SAMPLES = 1_000_000
SPEC_CAP = 2_000_000
DIFFUSE_MAX_DEPTH = 4
DIFFUSE_SAMPLES = 500_000
DIFFUSE_CAP = 4_000_000
#: Sionna's scattering coefficient for the full-model check: 0.3 the main
#: setting, 0.7 a sensitivity (much more diffuse power).
DIFFUSE_S = (0.3, 0.7)

#: (name, converter options, Room Sim side) of the specular variants.
VARIANTS = (
    ("shell", {"furniture": False, "object_model": "halfspace"}, "shell"),
    ("furnished", {"furniture": True, "object_model": "halfspace"}, "furnished"),
    ("furnished_panel", {"furniture": True, "object_model": "panel"}, "furnished"),
)

#: Per-link specular metrics, in table order: key, unit, meaning.
SPEC_METRICS = (
    ("d_los_db", "dB", "LoS (direct) path power, Room Sim minus Sionna"),
    ("d_spec_db", "dB", "total specular power up to order 2, Room Sim minus Sionna"),
    ("cfr_corr", "", "correlation of |H| (dB) over the 52 LLTF subcarriers"),
    ("cfr_corr_selective", "", "the same correlation, links where both selectivities are >= 1 dB"),
    ("cfr_rms_db", "dB", "RMS of the |H| difference (dB) after removing its mean"),
    ("cfr_offset_db", "dB", "mean |H| difference (dB), Room Sim minus Sionna"),
    ("ds_rs_ns", "ns", "RMS delay spread, Room Sim"),
    ("ds_s_ns", "ns", "RMS delay spread, Sionna (order <= 2)"),
    ("d_ds_ns", "ns", "RMS delay spread, Room Sim minus Sionna"),
    ("sel_rs_db", "dB", "frequency selectivity (std of |H| dB over 52 subcarriers), Room Sim"),
    ("sel_s_db", "dB", "frequency selectivity, Sionna (order <= 2)"),
    ("d_sel_db", "dB", "frequency selectivity, Room Sim minus Sionna"),
    ("higher_order_db", "dB", "Sionna power with 3..4 reflections added, relative to order <= 2"),
    ("rs_only_frac_db", "dB", "power of Room Sim paths Sionna lacks, relative to Room Sim's specular power"),
    ("s_only_frac_db", "dB", "power of Sionna paths Room Sim lacks, relative to Sionna's order <= 2 power"),
)
FULL_METRICS = (
    ("ds_rs_full_ns", "ns", "RMS delay spread, Room Sim full model (realized tail and clutter)"),
    ("ds_rs_full_expected_ns", "ns", "RMS delay spread, Room Sim full model with the tail by expectation"),
    ("ds_s_full_ns", "ns", "RMS delay spread, Sionna with diffuse reflection"),
    ("d_ds_full_ns", "ns", "RMS delay spread, Room Sim full (realized) minus Sionna diffuse"),
    ("sel_rs_full_db", "dB", "frequency selectivity, Room Sim full model"),
    ("sel_s_full_db", "dB", "frequency selectivity, Sionna with diffuse reflection"),
    ("d_sel_full_db", "dB", "frequency selectivity, Room Sim full minus Sionna diffuse"),
    ("ds_rs_spec_ns", "ns", "RMS delay spread, Room Sim kinds 0..3 only"),
    ("ds_s_spec_ns", "ns", "RMS delay spread, the specular paths of Sionna's diffuse run (orders <= 4)"),
    ("stoch_rs_db", "dB", "Room Sim points + clutter + tail power, relative to its kinds 0..3"),
    ("diffuse_s_db", "dB", "Sionna diffuse power, relative to its specular paths in the same run"),
    ("points_rs_db", "dB", "Room Sim object point scatterers (kind 4), relative to kinds 0..3"),
    ("clutter_rs_db", "dB", "Room Sim clutter (kind 5), relative to kinds 0..3"),
    ("tail_rs_db", "dB", "Room Sim tail (kind 6, realized), relative to kinds 0..3"),
    ("ds_rs_spec_plus_points_ns", "ns", "RMS delay spread, Room Sim kinds 0..3 + point scatterers"),
    ("ds_rs_spec_plus_clutter_ns", "ns", "RMS delay spread, Room Sim kinds 0..3 + clutter"),
    ("ds_rs_spec_plus_tail_ns", "ns", "RMS delay spread, Room Sim kinds 0..3 + tail (realized)"),
    ("ds_rs_tail_at_los_expected_ns", "ns",
     "what-if, not Room Sim's model: full-model delay spread, tail by expectation but starting at the LoS delay"),
)
#: The full-model metrics also summarized per scene (the rest: overall only).
PER_SCENE_FULL = ("ds_rs_full_ns", "ds_rs_full_expected_ns", "ds_s_full_ns", "d_ds_full_ns", "sel_rs_full_db",
                  "sel_s_full_db", "d_sel_full_db", "stoch_rs_db", "diffuse_s_db")


# -- scene selection -----------------------------------------------------------


def choose_scenes(manifest: dict, n: int = N_SCENES, seed: int = SCENE_SEED) -> tuple:
    """`n` scene indices from the SIM test split, drawn with
    `numpy.random.default_rng(seed)` without replacement; when none of them
    is L-shaped but the split has L-shaped scenes, the last draw is replaced
    by one drawn from those. Returns `(sorted indices, forced_l_shape)`."""
    test = rd.splits(manifest)["test"]
    entries = {int(e["index"]): e for e in manifest["scenes"]}
    rng = np.random.default_rng(seed)
    chosen = [int(i) for i in rng.choice(np.array(test), size=min(n, len(test)), replace=False)]
    forced = False
    l_test = [i for i in test if entries[i].get("l_shape")]
    if l_test and not any(entries[i].get("l_shape") for i in chosen):
        choices = [i for i in l_test if i not in chosen]
        chosen[-1] = int(rng.choice(np.array(choices)))
        forced = True
    return sorted(chosen), forced


def load_case(dataset: str, index: int, manifest: dict) -> tuple:
    """SIM scene `index`: its Room Sim scene (sha256 and provenance checked by
    the dataset loader), its exact Room Sim run config, and its manifest
    entry."""
    rd.load_scene_json(dataset, index, manifest)          # checks sha256 + "generated"
    entry = next(e for e in manifest["scenes"] if int(e["index"]) == index)
    base = os.path.join(dataset, "scenes", entry["id"])
    scene = load_scene(base + ".json")
    with open(base + ".roomsim.json", encoding="utf-8") as f:
        cfg = Config.from_dict(json.load(f))
    return scene, cfg, entry


# -- per-link comparisons ------------------------------------------------------


def _db(p: float) -> float:
    return M.power_db(p)


def rs_key(conv: s2s.ConvertedScene, kind: int, si: int, sj: int, obj: int) -> tuple:
    """Room Sim path -> the tuple of Sionna shape names it reflects from."""
    if kind == 0:
        return ()
    if kind == 1:
        return (conv.surface_names[si],)
    if kind == 2:
        return (conv.surface_names[si], conv.surface_names[sj])
    if kind == 3:
        return (conv.object_names[obj],)
    raise ValueError(f"kind {kind} has no specular key")


def path_class(kind: int, key: tuple, refracted: bool) -> str:
    """A matched path's class for the per-path table (`+T`: it crosses a
    wall, so its phases differ by the slab's `q0`, spec/roomsim.md §5.1)."""
    fc = ("floor", "ceiling")
    if kind == 0:
        c = "los"
    elif kind == 3:
        c = "object_specular"
    elif kind == 1:
        c = "order1_floor_ceiling" if key[0] in fc else "order1_wall"
    else:
        n = sum(1 for k in key if k in fc)
        c = ("order2_wall_wall", "order2_wall_floor_ceiling", "order2_floor_ceiling")[n]
    return c + ("+T" if refracted else "")


def blocked_paths(S: s2s.SionnaLink, wavelength: float) -> np.ndarray:
    """Sionna paths more than `BLOCKED_DB` below free space at their own
    length: in practice paths through a half-space (opaque) box, which
    Sionna 2.2.0 can return because it hashes a refraction by its type only
    and its image method accepts any surface for a refraction (§3.4). They
    carry no power (about -240 dB per box) and are treated as absent."""
    gain = np.abs(S.a) * (4.0 * math.pi * (S.tau * C) / wavelength)
    return gain < 10.0 ** (BLOCKED_DB / 20.0)


#: A Sionna path weaker than free space by more than this is "blocked".
BLOCKED_DB = -150.0
#: CFR correlation is reported separately over links where both simulators'
#: selectivity reaches this (a flat |H| makes the correlation meaningless).
SELECTIVE_DB = 1.0


def specular_link(R: rss.RoomSimLink, S: s2s.SionnaLink, conv: s2s.ConvertedScene, freqs: np.ndarray,
                  rs_kinds: tuple, max_refl: int) -> tuple:
    """Metrics (a)-(e) of one link (spec/sionna_audit.md §4.1) and its path
    matching. Returns `(record, matched, rs_only, s_only)`: the record of
    per-link numbers, the matched paths as `(class, dP_dB, dphi, dL_m)` and
    the unmatched paths as `(description, power fraction)`. All SIM."""
    sel = np.isin(R.kind, np.array(rs_kinds))
    a_rs, L_rs, k_rs = R.coef[sel], R.L[sel], R.kind[sel]
    tau_rs = L_rs / C
    nref = S.n_of("specular")
    ndif = S.n_of("diffuse")
    nrefr = S.n_of("refraction")
    blocked = blocked_paths(S, conv.wavelength_m)
    spec = (ndif == 0) & ~blocked
    idx2 = np.nonzero(spec & (nref <= max_refl))[0]
    a_s, tau_s = S.a[idx2], S.tau[idx2]
    P_rs = float(np.sum(np.abs(a_rs) ** 2))
    P_s = float(np.sum(np.abs(a_s) ** 2))
    P_all = float(np.sum(np.abs(S.a[spec]) ** 2))
    P_los_rs = float(np.sum(np.abs(R.coef[R.kind == 0]) ** 2))
    P_los_s = float(np.sum(np.abs(S.a[spec & (nref == 0)]) ** 2))
    H_rs = M.cfr(a_rs, tau_rs, freqs)
    H_s = M.cfr(a_s, tau_s, freqs)
    agr = M.cfr_agreement(H_rs, H_s)
    ds_rs = M.rms_delay_spread(np.abs(a_rs) ** 2, tau_rs) * 1e9
    ds_s = M.rms_delay_spread(np.abs(a_s) ** 2, tau_s) * 1e9
    sel_rs = M.selectivity_db(H_rs)
    sel_s = M.selectivity_db(H_s)

    keys_rs = [rs_key(conv, int(k), int(i), int(j), int(o))
               for k, i, j, o in zip(k_rs, R.surf_i[sel], R.surf_j[sel], R.obj[sel])]
    keys_s = S.reflections(idx2)
    m = M.match_paths(keys_rs, L_rs, keys_s, tau_s * C)
    matched = []
    for i, j, dL in m["pairs"]:
        cls = path_class(int(k_rs[i]), keys_rs[i], bool(nrefr[idx2[j]] > 0))
        dP = 20.0 * math.log10(abs(a_rs[i]) / abs(a_s[j])) if abs(a_s[j]) > 0 else float("inf")
        dphi = float(M.wrap_phase(np.angle(a_rs[i]) - np.angle(a_s[j])))
        matched.append((cls, dP, dphi, dL))
    rs_only = [(rss.KIND_NAMES[int(k_rs[i])], float(abs(a_rs[i]) ** 2) / P_rs if P_rs > 0 else float("nan"))
               for i in m["rs_only"]]
    s_only = []
    furn = set(conv.object_names.values())
    for j in m["s_only"]:
        n = len(keys_s[j])
        what = f"order{n}" + ("_object" if any(k in furn for k in keys_s[j]) else "_surfaces")
        s_only.append((what, float(abs(a_s[j]) ** 2) / P_s if P_s > 0 else float("nan")))
    frac_rs = sum(f for _, f in rs_only)
    frac_s = sum(f for _, f in s_only)
    L_los = float(R.L[R.kind == 0][0])
    fs = (conv.wavelength_m / (4.0 * math.pi * L_los)) ** 2          # free-space LoS power
    rec = {
        "tx": R.tx, "rx": R.rx,
        "los_excess_db_rs": _db(P_los_rs) - _db(fs),
        "los_db_rs": _db(P_los_rs), "los_db_s": _db(P_los_s),
        "d_los_db": _db(P_los_rs) - _db(P_los_s) if P_los_s > 0 else float("nan"),
        "los_blocked_s": bool(P_los_s == 0.0),
        "spec_db_rs": _db(P_rs), "spec_db_s": _db(P_s), "d_spec_db": _db(P_rs) - _db(P_s),
        "higher_order_db": _db(P_all) - _db(P_s),
        "cfr_corr": agr["corr"], "cfr_rms_db": agr["rms_db"], "cfr_offset_db": agr["offset_db"],
        "cfr_corr_selective": (agr["corr"] if (sel_rs >= SELECTIVE_DB and sel_s >= SELECTIVE_DB)
                               else float("nan")),
        "n_blocked_s": int(np.sum(blocked)),
        "ds_rs_ns": ds_rs, "ds_s_ns": ds_s, "d_ds_ns": ds_rs - ds_s,
        "sel_rs_db": sel_rs, "sel_s_db": sel_s, "d_sel_db": sel_rs - sel_s,
        "n_rs": int(len(a_rs)), "n_s": int(len(a_s)), "n_matched": len(matched),
        "n_rs_only": len(rs_only), "n_s_only": len(s_only),
        "rs_only_frac_db": _db(frac_rs) if frac_rs > 0 else float("-inf"),
        "s_only_frac_db": _db(frac_s) if frac_s > 0 else float("-inf"),
    }
    return rec, matched, rs_only, s_only


def full_link(R: rss.RoomSimLink, S: s2s.SionnaLink, freqs: np.ndarray) -> dict:
    """The qualitative full-model comparison of one link (§4.2): Room Sim's
    whole static channel (kinds 0..6, no people) against Sionna with diffuse
    reflection (every path of that run). All SIM."""
    p = np.abs(R.coef) ** 2
    spec_rs = R.kind <= 3
    P_spec_rs = float(np.sum(p[spec_rs]))
    P_stoch = float(np.sum(p[R.kind >= 4]))
    H_check = M.cfr(R.coef, R.tau, freqs)
    rel_err = float(np.max(np.abs(H_check - R.H)) / np.max(np.abs(R.H)))
    _P, _m, ds_exp = rss.expected_tail_moments(R)
    _P, _m, ds_los_anchor = rss.expected_tail_moments(R, anchor="los")
    parts = {}
    for name, kind in (("points", 4), ("clutter", 5), ("tail", 6)):
        m = spec_rs | (R.kind == kind)
        Pk = float(np.sum(p[R.kind == kind]))
        parts[f"ds_rs_spec_plus_{name}_ns"] = M.rms_delay_spread(p[m], R.tau[m]) * 1e9
        parts[f"{name}_rs_db"] = _db(Pk) - _db(P_spec_rs) if Pk > 0 else float("-inf")
    ndif = S.n_of("diffuse")
    dif = ndif > 0
    pa = np.abs(S.a) ** 2
    P_spec_s = float(np.sum(pa[~dif]))
    P_dif = float(np.sum(pa[dif]))
    H_s = M.cfr(S.a, S.tau, freqs)
    ds_rs = M.rms_delay_spread(p, R.tau) * 1e9
    ds_s = M.rms_delay_spread(pa, S.tau) * 1e9
    sel_rs = M.selectivity_db(R.H)
    sel_s = M.selectivity_db(H_s)
    return {
        "tx": R.tx, "rx": R.rx,
        "ds_rs_full_ns": ds_rs, "ds_rs_full_expected_ns": ds_exp * 1e9, "ds_s_full_ns": ds_s,
        "d_ds_full_ns": ds_rs - ds_s,
        "sel_rs_full_db": sel_rs, "sel_s_full_db": sel_s, "d_sel_full_db": sel_rs - sel_s,
        "ds_rs_spec_ns": M.rms_delay_spread(p[spec_rs], R.tau[spec_rs]) * 1e9,
        "ds_s_spec_ns": M.rms_delay_spread(pa[~dif], S.tau[~dif]) * 1e9,
        "stoch_rs_db": _db(P_stoch) - _db(P_spec_rs) if P_stoch > 0 else float("-inf"),
        "diffuse_s_db": _db(P_dif) - _db(P_spec_s) if P_dif > 0 else float("-inf"),
        "ds_rs_tail_at_los_expected_ns": ds_los_anchor * 1e9,
        **parts,
        "n_diffuse_paths": int(np.sum(dif)),
        "roomsim_cfr_check_rel_err": rel_err,
    }


# -- the slab checks -------------------------------------------------------------


def slab_formula_check(carrier: float, thicknesses=(0.1, 0.2),
                       angles_deg=(0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0)) -> dict:
    """Room Sim's §5.1 slab (`wifisense.roomsim.materials`) against Sionna
    2.2.0's `itu_coefficients_single_layer_slab` with Sionna's own ITU
    permittivity, for every wall material, both polarisations. Also Room
    Sim's single-interface object `G` against Sionna's slab at the
    converter's half-space thickness. All SIM."""
    s2s.configure_backend()
    import drjit as dr
    import mitsuba as mi
    from sionna.rt.radio_materials.itu import itu_material
    from sionna.rt.utils import complex_relative_permittivity, itu_coefficients_single_layer_slab

    lam = C / carrier
    cosv = np.cos(np.radians(np.array(angles_deg)))

    def sionna_slab(itu: str, d: float):
        er, sg = itu_material(itu, mi.Float(carrier))
        eta = complex_relative_permittivity(er, sg, mi.Float(2.0 * math.pi * carrier))
        out = itu_coefficients_single_layer_slab(mi.Float(cosv), eta, mi.Float(d), mi.Float(lam))
        return [np.array(dr.real(z), dtype=np.float64) + 1j * np.array(dr.imag(z), dtype=np.float64)
                for z in out]

    rows = []
    keys = ("R_abs_err", "R_db_err", "T_db_err", "T_phase_minus_q0_err")
    worst = {k: 0.0 for k in keys}
    for mat, itu in s2s.WALL_ITU.items():
        et = rs_materials.eta(mat, carrier)
        for d in thicknesses:
            r_te, r_tm, t_te, t_tm = sionna_slab(itu, d)
            row = {"material": mat, "itu": itu, "thickness_m": d, **{k: 0.0 for k in keys},
                   "R_db_roomsim_range": [math.inf, -math.inf], "T_db_roomsim_range": [math.inf, -math.inf]}
            for ai in range(len(angles_deg)):
                c = float(cosv[ai])
                s, g_te, g_tm = rs_materials.fresnel(et, c)
                for G, rS, tS in ((g_te, r_te[ai], t_te[ai]), (g_tm, r_tm[ai], t_tm[ai])):
                    R, T, q0 = rs_materials.slab(G, s, c, d, lam)
                    Rc, Tc = complex(*R), complex(*T)
                    errs = (abs(Rc - rS), abs(20 * math.log10(abs(Rc) / abs(rS))),
                            abs(20 * math.log10(abs(Tc) / abs(tS))),
                            abs(float(M.wrap_phase(np.angle(Tc) - np.angle(tS) - q0))))
                    for k, e in zip(keys, errs):
                        row[k] = max(row[k], e)
                        worst[k] = max(worst[k], e)
                    for rk, val in (("R_db_roomsim_range", abs(Rc)), ("T_db_roomsim_range", abs(Tc))):
                        db = 20 * math.log10(val)
                        row[rk] = [min(row[rk][0], db), max(row[rk][1], db)]
            rows.append(row)
    objects = []
    for mat in ("metal", "wood", "glass"):
        d = s2s.halfspace_thickness(mat, carrier) if mat != "metal" else 0.01
        r_te, r_tm, _t_te, _t_tm = sionna_slab(mat, d)
        et = rs_materials.eta(mat, carrier)
        e = 0.0
        for ai in range(len(angles_deg)):
            _s, g_te, g_tm = rs_materials.fresnel(et, float(cosv[ai]))
            e = max(e, abs(complex(*g_te) - r_te[ai]), abs(complex(*g_tm) - r_tm[ai]))
        objects.append({"material": mat, "halfspace_thickness_m": d, "max_abs_err_G_vs_sionna_R": e})
    return {"label": LABEL, "angles_deg": list(angles_deg), "thicknesses_m": list(thicknesses),
            "worst": worst, "objects_halfspace": objects, "rows": rows}


def _single_surface_scene(kind: str, material: str, thickness: float, tx, rx, carrier: float) -> s2s.ConvertedScene:
    """A one-surface scene for the ray-traced check: the plane `y = 0`
    (`kind="wall"`) or `z = 0` (`"floor"`), 100 m square."""
    if kind == "wall":
        V = np.array([(-50.0, 0.0, -50.0), (50.0, 0.0, -50.0), (50.0, 0.0, 50.0), (-50.0, 0.0, 50.0)])
    else:
        V = np.array([(-50.0, -50.0, 0.0), (50.0, -50.0, 0.0), (50.0, 50.0, 0.0), (-50.0, 50.0, 0.0)])
    F = np.array([(0, 1, 2), (0, 2, 3)])
    mid = f"rm-{kind}-{material}-{thickness:.4f}"
    mat = s2s.MaterialSpec(mid, s2s.WALL_ITU[material], thickness, source=f"{kind} {material}")
    return s2s.ConvertedScene(
        name=f"slab-check-{kind}", carrier_hz=carrier, spacing_hz=312500.0, wavelength_m=C / carrier,
        height_m=0.0, footprint=[], nodes={0: tuple(tx), 1: tuple(rx)},
        shapes=[s2s.ShapeSpec(kind, kind, 0, 0, mid, V, F)], materials={mid: mat},
        surface_names={0: kind}, object_names={})


def raytraced_slab_check(carrier: float, material: str = "concrete", thickness: float = 0.2,
                         angles_deg=(0.0, 30.0, 60.0, 80.0), workdir: str | None = None) -> dict:
    """One surface, several angles, through the whole pipeline (converter
    material, Sionna's ray tracer, iso/V antennas): the reflected and
    transmitted paths' `a * 4 pi L / lambda` against Room Sim's slab `R` and
    `T * exp(-j q0)` (§5.1). Horizontal legs at a wall are pure TE, a floor
    bounce in a vertical plane pure TM, so these must agree to float32; one
    tilted wall bounce shows Room Sim's TE-only approximation (§16). SIM."""
    lam = C / carrier
    et = rs_materials.eta(material, carrier)
    r1, r2 = 2.0, 3.0
    cases = []
    for ang in angles_deg:
        th = math.radians(ang)
        sn, cs = math.sin(th), math.cos(th)
        cases.append(("wall", "reflection", ang, (-r1 * sn, r1 * cs, 0.0), (r2 * sn, r2 * cs, 0.0), "TE"))
        cases.append(("wall", "transmission", ang, (-r1 * sn, r1 * cs, 0.0), (r2 * sn, -r2 * cs, 0.0), "TE"))
        if ang > 0.0:
            cases.append(("floor", "reflection", ang, (-r1 * sn, 0.0, r1 * cs), (r2 * sn, 0.0, r2 * cs), "TM"))
    cases.append(("wall", "reflection_tilted", None, (-2.0, 2.0, -1.0), (2.0, 3.0, 1.5), "TE"))
    rows = []
    worst_exact = 0.0
    tmp = workdir or tempfile.mkdtemp(prefix="slabcheck-")
    try:
        for n, (kind, what, ang, tx, rx, pol) in enumerate(cases):
            conv = _single_surface_scene(kind, material, thickness, tx, rx, carrier)
            sc = s2s.load_sionna_scene(conv, os.path.join(tmp, f"case{n}"))
            links = s2s.solve_paths(sc, [0, 1], max_depth=1, samples_per_src=100_000, max_num_paths_per_src=100_000)
            S = links[(0, 1)]
            want = "specular" if what.startswith("reflection") else "refraction"
            sel = np.nonzero(S.n_of(want) == 1)[0]
            if len(sel) != 1:
                raise RuntimeError(f"slab check {kind}/{what}/{ang}: {len(sel)} {want} paths")
            a = complex(S.a[sel[0]])
            L = float(S.tau[sel[0]] * C)
            got = a * (4.0 * math.pi * L / lam)
            # Room Sim's coefficient at the incidence angle of the incoming leg.
            if what == "reflection_tilted":
                txv, rxv = np.array(tx), np.array(rx)
                img = txv * np.array([1.0, -1.0, 1.0])
                t = img[1] / (img[1] - rxv[1])
                X = img + t * (rxv - img)
                u = (X - txv) / np.linalg.norm(X - txv)
                c = abs(float(u[1]))
                ang_eff = math.degrees(math.acos(c))
            else:
                c = math.cos(math.radians(ang))
                ang_eff = ang
            s, g_te, g_tm = rs_materials.fresnel(et, c)
            G = g_te if pol == "TE" else g_tm
            R, T, q0 = rs_materials.slab(G, s, c, thickness, lam)
            ref = complex(*R) if what.startswith("reflection") else complex(*T) * complex(math.cos(-q0), math.sin(-q0))
            err_db = 20.0 * math.log10(abs(got) / abs(ref))
            err_ph = float(M.wrap_phase(np.angle(got) - np.angle(ref)))
            row = {"surface": kind, "path": what, "angle_deg": ang_eff, "pol_roomsim": pol,
                   "roomsim_db": 20.0 * math.log10(abs(ref)), "sionna_db": 20.0 * math.log10(abs(got)),
                   "diff_db": err_db, "diff_phase_rad": err_ph}
            if what == "reflection_tilted":
                _R_tm, _T, _q = rs_materials.slab(g_tm, s, c, thickness, lam)
                row["roomsim_tm_db"] = 20.0 * math.log10(abs(complex(*_R_tm)))
            else:
                worst_exact = max(worst_exact, abs(err_db))
            rows.append(row)
            del sc, links
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    return {"label": LABEL, "material": material, "thickness_m": thickness, "r1_m": r1, "r2_m": r2,
            "worst_exact_case_db": worst_exact, "rows": rows}


# -- summaries -------------------------------------------------------------------


def _summ(values, full: bool = True) -> dict:
    s = M.summarize(values)
    if not full:
        s = {k: s[k] for k in ("label", "n", "n_nonfinite", "median", "p10", "p90")}
    return s


def summarize_records(records: list, metrics, full: bool = True) -> dict:
    return {key: _summ([r[key] for r in records], full) for key, _u, _d in metrics}


def summarize_matched(matched: list) -> dict:
    """Per path class: ΔP (Room Sim minus Sionna, dB), |Δphase| and |ΔL|
    distributions, and how many matched paths differ in length by > 1 cm."""
    by: dict = {}
    for cls, dP, dphi, dL in matched:
        by.setdefault(cls, ([], [], []))
        by[cls][0].append(dP)
        by[cls][1].append(abs(dphi))
        by[cls][2].append(abs(dL))
    out = {}
    for cls in sorted(by):
        dPs, dphis, dLs = by[cls]
        out[cls] = {"label": LABEL, "n": len(dPs), "dP_db": _summ(dPs), "abs_dphi_rad": _summ(dphis),
                    "abs_dL_m": _summ(dLs), "n_dL_over_1cm": int(sum(1 for d in dLs if d > 0.01))}
    return out


def summarize_unmatched(rs_only: list, s_only: list, n_links: int, records: list) -> dict:
    def count(items):
        c: dict = {}
        for what, _f in items:
            c[what] = c.get(what, 0) + 1
        return c
    return {"label": LABEL, "links": n_links,
            "links_with_rs_only": int(sum(1 for r in records if r["n_rs_only"] > 0)),
            "links_with_s_only": int(sum(1 for r in records if r["n_s_only"] > 0)),
            "rs_only_paths_by_kind": count(rs_only), "s_only_paths_by_class": count(s_only),
            "links_with_los_blocked_in_sionna": int(sum(1 for r in records if r["los_blocked_s"])),
            "sionna_paths_dropped_as_blocked": int(sum(r["n_blocked_s"] for r in records))}


# -- the run -------------------------------------------------------------------------


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _gpu_mb_of_this_process() -> float | None:
    """GPU memory this process holds per nvidia-smi (None if unavailable)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    total = 0.0
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(os.getpid()):
            total += float(parts[1])
    return total


def _versions() -> dict:
    v = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("sionna.rt", "mitsuba", "drjit"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            v[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            v[mod] = None
    return v


def audit_scene(scene, cfg, entry: dict, opts: argparse.Namespace, log) -> dict:
    """Every comparison of one SIM scene. Returns per-link records, matched
    and unmatched paths per variant, the full-model records and timings."""
    ids = sorted(n.id for n in scene.nodes)
    freqs = s2s.lltf_frequencies(rs_carrier_hz(cfg.channel), cfg.spacing_hz)
    t0 = time.time()
    rs = {"shell": rss.roomsim_links(scene, rss.shell_config(cfg)), "furnished": rss.roomsim_links(scene, cfg)}
    timing = {"roomsim_s": time.time() - t0}
    max_refl = int(cfg.reflection_order)
    out = {"variants": {}, "full": {}, "timing": timing, "approximations": {}, "solver_stats": {}}
    for name, conv_opts, rs_side in VARIANTS:
        if name not in opts.variants:
            continue
        conv = s2s.convert(scene, cfg, **conv_opts)
        out["approximations"][name] = conv.approximations
        rs_kinds = (0, 1, 2) if rs_side == "shell" else (0, 1, 2, 3)
        t1 = time.time()
        st: dict = {}
        with tempfile.TemporaryDirectory(prefix="sionna-audit-") as d:
            sc = s2s.load_sionna_scene(conv, d)
            links = s2s.solve_paths(sc, ids, max_depth=opts.spec_depth, samples_per_src=opts.spec_samples,
                                    max_num_paths_per_src=opts.spec_cap, seed=opts.sionna_seed, stats=st)
        timing[f"sionna_{name}_s"] = time.time() - t1
        out["solver_stats"][name] = st
        recs, matched, rs_only, s_only = [], [], [], []
        for key in sorted(links):
            rec, mt, ro, so = specular_link(rs[rs_side][key], links[key], conv, freqs, rs_kinds, max_refl)
            recs.append(rec)
            matched.extend(mt)
            rs_only.extend(ro)
            s_only.extend(so)
        out["variants"][name] = {"records": recs, "matched": matched, "rs_only": rs_only, "s_only": s_only}
        log(f"  {name}: {len(recs)} links, Sionna {timing[f'sionna_{name}_s']:.1f} s, "
            f"stored candidates <= {st.get('candidates_stored_max')}")
        del sc, links
        gc.collect()
    for S_coef in opts.diffuse_s:
        conv = s2s.convert(scene, cfg, furniture=True, object_model="halfspace", scattering_coefficient=S_coef)
        t1 = time.time()
        st = {}
        with tempfile.TemporaryDirectory(prefix="sionna-audit-") as d:
            sc = s2s.load_sionna_scene(conv, d)
            links = s2s.solve_paths(sc, ids, max_depth=opts.diffuse_depth, samples_per_src=opts.diffuse_samples,
                                    max_num_paths_per_src=opts.diffuse_cap, seed=opts.sionna_seed, diffuse=True,
                                    stats=st)
        timing[f"sionna_diffuse_S{S_coef}_s"] = time.time() - t1
        out["solver_stats"][f"diffuse_S{S_coef}"] = st
        recs = [full_link(rs["furnished"][key], links[key], freqs) for key in sorted(links)]
        out["full"][f"S{S_coef}"] = recs
        log(f"  full (diffuse S = {S_coef}): Sionna {timing[f'sionna_diffuse_S{S_coef}_s']:.1f} s, "
            f"truncated={st.get('truncated')}")
        del sc, links
        gc.collect()
    return out


def reproducibility(opts: argparse.Namespace, manifest: dict, chosen: list, log) -> dict:
    """Rerun the smallest chosen scene's `furnished` comparison with another
    Sionna seed and report the largest change of each per-link specular
    metric: the specular paths Sionna finds should not depend on its ray
    sampling at these settings (spec/sionna_audit.md §3.4). SIM."""
    entries = {int(e["index"]): e for e in manifest["scenes"]}
    index = min(chosen, key=lambda i: (entries[i]["n_nodes"], i))
    scene, cfg, entry = load_case(opts.dataset, index, manifest)
    runs = []
    for seed in (opts.sionna_seed, opts.sionna_seed + 1):
        o = argparse.Namespace(**vars(opts))
        o.sionna_seed = seed
        o.variants = ["furnished"]
        o.diffuse_s = []
        runs.append(audit_scene(scene, cfg, entry, o, log=lambda *_a: None)["variants"]["furnished"]["records"])
    worst = {}
    for key, _u, _d in SPEC_METRICS:
        diffs = []
        for r1, r2 in zip(runs[0], runs[1]):
            a, b = r1[key], r2[key]
            if key.endswith("_frac_db"):        # a power fraction: compare it linearly (-inf is 0)
                a, b = 10.0 ** (a / 10.0), 10.0 ** (b / 10.0)
            if math.isfinite(a) and math.isfinite(b):
                diffs.append(abs(a - b))
            elif not (a == b or (math.isnan(a) and math.isnan(b))):
                diffs.append(float("inf"))
        worst[key] = max(diffs) if diffs else 0.0
    n_diff = sum(1 for r1, r2 in zip(runs[0], runs[1]) if r1["n_s"] != r2["n_s"])
    log(f"reproducibility ({entry['id']}, seeds {opts.sionna_seed}/{opts.sionna_seed + 1}): "
        f"max |d spec| {worst['d_spec_db']:.2e} dB, links with other path counts {n_diff}")
    return {"label": LABEL, "scene": entry["id"], "variant": "furnished",
            "seeds": [opts.sionna_seed, opts.sionna_seed + 1], "same_path_counts": n_diff == 0,
            "links": len(runs[0]), "links_with_other_path_counts": n_diff, "max_abs_change": worst}


def run(opts: argparse.Namespace, log=print) -> dict:
    """The whole audit; returns the summary document (SIM)."""
    t_start = time.time()
    variant = s2s.configure_backend()
    manifest = rd.load_manifest(opts.dataset)
    if opts.indices:
        chosen, forced = sorted(opts.indices), False
    else:
        chosen, forced = choose_scenes(manifest, opts.scenes, opts.scene_seed)
    log(f"SIM audit: scenes {chosen} from {opts.dataset} (test split, seed {opts.scene_seed}), backend {variant}")
    per_scene = {}
    agg = {name: {"records": [], "matched": [], "rs_only": [], "s_only": []} for name, _o, _s in VARIANTS}
    agg_full: dict = {}
    scenes_meta = []
    carrier = None
    for index in chosen:
        scene, cfg, entry = load_case(opts.dataset, index, manifest)
        carrier = rs_carrier_hz(cfg.channel)
        log(f"scene {entry['id']}: {len(scene.nodes)} nodes, L-shaped {entry.get('l_shape')}, "
            f"walls {entry.get('wall_material')}, {sum(1 for o in scene.objects if o.kind not in ('door', 'window'))} "
            f"furniture objects")
        res = audit_scene(scene, cfg, entry, opts, log)
        meta = {"index": index, "id": entry["id"], "n_nodes": len(scene.nodes),
                "n_links": len(scene.nodes) * (len(scene.nodes) - 1), "l_shape": bool(entry.get("l_shape")),
                "wall_material": entry.get("wall_material"),
                "wall_thickness_m": scene.walls[0].thickness_m if scene.walls else None,
                "floor_material": cfg.floor_material, "ceiling_material": cfg.ceiling_material,
                "room_m": [scene.room.w_m, scene.room.d_m, scene.room.h_m],
                "furniture": [f"{o.kind}:{o.material}" for o in scene.objects if o.kind not in ("door", "window")],
                "approximations": res["approximations"], "timing_s": res["timing"],
                "solver_stats": res["solver_stats"]}
        scenes_meta.append(meta)
        ps = {}
        for name in res["variants"]:
            v = res["variants"][name]
            for k in agg[name]:
                agg[name][k].extend(v[k])
            ps[name] = summarize_records(v["records"], SPEC_METRICS, full=False)
        for key, recs in res["full"].items():
            agg_full.setdefault(key, []).extend(recs)
            ps[f"full_{key}"] = summarize_records(recs, [m for m in FULL_METRICS if m[0] in PER_SCENE_FULL],
                                                  full=False)
        per_scene[entry["id"]] = ps
        gc.collect()

    doc = {
        "label": LABEL, "spec": SPEC,
        "note": ("Room Sim and Sionna RT are both simulators: every number here is SIM, an agreement or "
                 "disagreement between two models of the same declared rooms, never a property of a real room."),
        "dataset": {"dir": opts.dataset, "split": "test", "scene_seed": opts.scene_seed,
                    "forced_l_shape": forced, "scenes": scenes_meta},
        "settings": {"backend_variant": variant, "libllvm": s2s.backend_info().get("libllvm"),
                     "versions": _versions(),
                     "antennas": "iso pattern, V polarisation, at every node (tx and rx)",
                     "frequencies": "carrier (channel 6, 2.437 GHz) + k * 312.5 kHz, k = -26..-1, 1..26",
                     "spec_max_depth": opts.spec_depth, "spec_samples_per_src": opts.spec_samples,
                     "spec_max_num_paths_per_src": opts.spec_cap, "diffuse_max_depth": opts.diffuse_depth,
                     "diffuse_samples_per_src": opts.diffuse_samples, "diffuse_scattering_coefficients": opts.diffuse_s,
                     "diffuse_pattern": "lambertian", "sionna_seed": opts.sionna_seed,
                     "one_receiver_per_solve": True, "diffraction": False,
                     "roomsim_reflection_order": 2, "sionna_max_reflections_compared": 2},
        "metrics": {"specular": {k: {"unit": u, "meaning": d} for k, u, d in SPEC_METRICS},
                    "full": {k: {"unit": u, "meaning": d} for k, u, d in FULL_METRICS}},
        "variants": {}, "full": {}, "per_scene": per_scene,
    }
    split_keys = [m for m in SPEC_METRICS if m[0] in ("d_spec_db", "cfr_rms_db", "cfr_corr", "d_ds_ns", "d_sel_db")]
    split_keys.append(("los_excess_db_rs", "dB", "Room Sim's LoS power relative to free space"))
    for name, _o, _s in VARIANTS:
        a = agg[name]
        if not a["records"]:
            continue
        clear = [r for r in a["records"] if not r["los_blocked_s"]]
        blocked = [r for r in a["records"] if r["los_blocked_s"]]
        doc["variants"][name] = {
            "label": LABEL,
            "overall": summarize_records(a["records"], SPEC_METRICS),
            "overall_los_clear_in_sionna": summarize_records(clear, split_keys, full=False),
            "overall_los_blocked_in_sionna": summarize_records(blocked, split_keys, full=False),
            "paths_matched": summarize_matched(a["matched"]),
            "unmatched": summarize_unmatched(a["rs_only"], a["s_only"], len(a["records"]), a["records"]),
        }
    for key, recs in agg_full.items():
        doc["full"][key] = {"label": LABEL, "overall": summarize_records(recs, FULL_METRICS),
                            "roomsim_cfr_check_max_rel_err": float(max(r["roomsim_cfr_check_rel_err"] for r in recs))}
    if opts.links_out:
        with open(opts.links_out, "w", encoding="utf-8") as f:
            json.dump(_jsonable({"label": LABEL, "note": "per-link records of the audit (SIM)",
                                 "variants": {k: v["records"] for k, v in agg.items()}, "full": agg_full,
                                 "matched": {k: v["matched"] for k, v in agg.items()}}), f)
    if opts.repro and chosen:
        doc["reproducibility"] = reproducibility(opts, manifest, chosen, log)
    if not opts.no_slab_check:
        t1 = time.time()
        doc["slab_check"] = {"label": LABEL, "formula": slab_formula_check(carrier or 2.437e9),
                             "raytraced": raytraced_slab_check(carrier or 2.437e9)}
        log(f"slab check: {time.time() - t1:.1f} s")
    doc["runtime"] = {"label": LABEL, "wall_s": time.time() - t_start, "peak_rss_gb": _rss_gb(),
                      "gpu_memory_mb_this_process": _gpu_mb_of_this_process(),
                      "gpu_note": ("the audit ran on Mitsuba's LLVM (CPU) variant: on this machine the CUDA "
                                   "variant aborts in OptiX's compiler (driver 545), so no GPU memory is used")
                      if variant.startswith("llvm") else ""}
    return doc


# -- output --------------------------------------------------------------------------


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return round(v, 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


#: Per-scene summaries written to the JSON: the main comparisons; the
#: sensitivity runs (`furnished_panel`, diffuse S other than the first) are
#: summarized overall only, to keep the file small.
PER_SCENE_WRITTEN = ("shell", "furnished")


def slim(doc: dict) -> dict:
    """The document as written: per-scene summaries of the main comparisons
    only (`PER_SCENE_WRITTEN` and the first diffuse S). SIM."""
    out = dict(doc)
    full_keys = list(doc.get("full", {}).keys())
    keep = set(PER_SCENE_WRITTEN) | ({f"full_{full_keys[0]}"} if full_keys else set())
    out["per_scene"] = {sid: {k: v for k, v in ps.items() if k in keep} for sid, ps in doc["per_scene"].items()}
    out["per_scene_note"] = ("per-scene summaries of shell, furnished and the first diffuse S only; the "
                             "sensitivity runs are summarized overall (variants, full)")
    return out


def write_json(doc: dict, path: str) -> None:
    """Write a SIM audit document: indented, with every leaf dict (a dict of
    scalars, e.g. one summary) and every list of scalars on one line."""
    def leaf(x) -> bool:
        if isinstance(x, dict):
            return all(not isinstance(v, (dict, list)) for v in x.values())
        if isinstance(x, list):
            return all(not isinstance(v, (dict, list)) for v in x)
        return True

    def emit(x, ind: int) -> str:
        if leaf(x):
            return json.dumps(x, ensure_ascii=False)
        pad = " " * (ind + 1)
        if isinstance(x, dict):
            items = [f"{pad}{json.dumps(str(k), ensure_ascii=False)}: {emit(v, ind + 1)}" for k, v in x.items()]
            return "{\n" + ",\n".join(items) + "\n" + " " * ind + "}"
        items = [f"{pad}{emit(v, ind + 1)}" for v in x]
        return "[\n" + ",\n".join(items) + "\n" + " " * ind + "]"

    with open(path, "w", encoding="utf-8") as f:
        f.write(emit(_jsonable(doc), 0))
        f.write("\n")


def _f(s: dict, nd: int = 2) -> str:
    """`median [p10, p90]` of a summary."""
    if not s or s.get("n", 0) == 0:
        return "–"
    return f"{s['median']:.{nd}f} [{s['p10']:.{nd}f}, {s['p90']:.{nd}f}]"


def markdown(doc: dict) -> str:
    """The result tables of spec/sionna_audit.md §5 (every number SIM)."""
    L = []
    vs = [v for v in ("shell", "furnished", "furnished_panel") if v in doc["variants"]]
    L.append("All numbers SIM. Cells: median [p10, p90] over links.\n")
    L.append("**Table 1 — specular comparison, all links of all scenes.**\n")
    L.append("| Metric | " + " | ".join(vs) + " |")
    L.append("|---|" + "---|" * len(vs))
    for key, unit, meaning in SPEC_METRICS:
        cells = [_f(doc["variants"][v]["overall"][key], 3 if key == "cfr_corr" else 2) for v in vs]
        L.append(f"| {meaning}{f' ({unit})' if unit else ''} | " + " | ".join(cells) + " |")
    for v in vs:
        u = doc["variants"][v]["unmatched"]
        L.append(f"\n`{v}`: {u['links']} links; Sionna has no direct path on {u['links_with_los_blocked_in_sionna']}; "
                 f"Room-Sim-only paths on {u['links_with_rs_only']} links {u['rs_only_paths_by_kind']}; "
                 f"Sionna-only paths on {u['links_with_s_only']} links {u['s_only_paths_by_class']}; "
                 f"Sionna paths dropped as blocked: {u.get('sionna_paths_dropped_as_blocked', 0)}.")
    fv = [v for v in vs if v != "shell" and "overall_los_blocked_in_sionna" in doc["variants"][v]]
    if fv:
        L.append("\n**Table 1b — furnished variants, links split by whether furniture blocks Sionna's direct path.**\n")
        L.append("| Variant | links | Room Sim LoS vs free space (dB) | ΔP spec (dB) | CFR RMS (dB) | CFR corr "
                 "| ΔDS (ns) | ΔSel (dB) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for v in fv:
            for part, label in (("overall_los_clear_in_sionna", "direct path clear"),
                                ("overall_los_blocked_in_sionna", "direct path blocked")):
                s = doc["variants"][v][part]
                n = s["d_spec_db"]["n"] + s["d_spec_db"]["n_nonfinite"]
                L.append(f"| {v}, {label} | {n} | {_f(s['los_excess_db_rs'])} | {_f(s['d_spec_db'])} | "
                         f"{_f(s['cfr_rms_db'])} | {_f(s['cfr_corr'], 3)} | {_f(s['d_ds_ns'])} | {_f(s['d_sel_db'])} |")
    if "reproducibility" in doc:
        r = doc["reproducibility"]
        mx = max(v for v in r["max_abs_change"].values())
        L.append(f"\nReproducibility: `{r['variant']}` on {r['scene']} with Sionna seeds {r['seeds']}: same path "
                 f"counts {r['same_path_counts']}, largest change of any per-link metric {mx:.2e}.")
    per_scene_vs = [v for v in vs if any(v in ps for ps in doc["per_scene"].values())]
    for v in per_scene_vs:
        L.append(f"\n**Table 2{'abc'[per_scene_vs.index(v)]} — per scene, `{v}`.**\n")
        L.append("| Scene | links | ΔP LoS (dB) | ΔP spec (dB) | CFR corr | CFR RMS (dB) | DS RS (ns) | DS S (ns) "
                 "| Sel RS (dB) | Sel S (dB) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for meta in doc["dataset"]["scenes"]:
            ps = doc["per_scene"][meta["id"]].get(v)
            if ps is None:
                continue
            tag = f"{meta['id']}{' (L)' if meta['l_shape'] else ''}"
            L.append(f"| {tag} | {meta['n_links']} | {_f(ps['d_los_db'])} | {_f(ps['d_spec_db'])} | "
                     f"{_f(ps['cfr_corr'], 3)} | {_f(ps['cfr_rms_db'])} | {_f(ps['ds_rs_ns'])} | {_f(ps['ds_s_ns'])} | "
                     f"{_f(ps['sel_rs_db'])} | {_f(ps['sel_s_db'])} |")
    for v in vs:
        L.append(f"\n**Table 3{'abc'[vs.index(v)]} — matched paths by class, `{v}`** "
                 "(ΔP = Room Sim minus Sionna, per path).\n")
        L.append("| Class | paths | ΔP (dB) | max abs ΔP (dB) | abs Δphase (rad) | ΔL > 1 cm |")
        L.append("|---|---|---|---|---|---|")
        for cls, s in doc["variants"][v]["paths_matched"].items():
            mx = max(abs(s["dP_db"]["min"]), abs(s["dP_db"]["max"])) if s["dP_db"]["n"] else float("nan")
            L.append(f"| {cls} | {s['n']} | {_f(s['dP_db'], 3)} | {mx:.2f} | {_f(s['abs_dphi_rad'], 3)} | "
                     f"{s['n_dL_over_1cm']} |")
    for key, full in doc["full"].items():
        L.append(f"\n**Table 4 — full models (qualitative), Sionna diffuse {key}.**\n")
        L.append("| Metric | median [p10, p90] |")
        L.append("|---|---|")
        for k, unit, meaning in FULL_METRICS:
            L.append(f"| {meaning} ({unit}) | {_f(full['overall'][k])} |")
        if not any(f"full_{key}" in ps for ps in doc["per_scene"].values()):
            continue
        L.append(f"\n**Table 4b — full models per scene, Sionna diffuse {key}.**\n")
        L.append("| Scene | DS RS full (ns) | DS RS expected (ns) | DS S diffuse (ns) | Sel RS full (dB) "
                 "| Sel S diffuse (dB) | RS stochastic (dB) | S diffuse (dB) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for meta in doc["dataset"]["scenes"]:
            ps = doc["per_scene"][meta["id"]].get(f"full_{key}")
            if ps is None:
                continue
            tag = f"{meta['id']}{' (L)' if meta['l_shape'] else ''}"
            L.append(f"| {tag} | {_f(ps['ds_rs_full_ns'])} | {_f(ps['ds_rs_full_expected_ns'])} | "
                     f"{_f(ps['ds_s_full_ns'])} | {_f(ps['sel_rs_full_db'])} | {_f(ps['sel_s_full_db'])} | "
                     f"{_f(ps['stoch_rs_db'])} | {_f(ps['diffuse_s_db'])} |")
    if "slab_check" in doc:
        fc = doc["slab_check"]["formula"]["worst"]
        L.append("\n**Table 5 — slab coefficients, Room Sim §5.1 against Sionna 2.2.0** "
                 f"(5 wall materials × {doc['slab_check']['formula']['thicknesses_m']} m × "
                 f"{doc['slab_check']['formula']['angles_deg']}° × TE/TM): worst |ΔR| = {fc['R_abs_err']:.2e}, "
                 f"|Δ|R|| = {fc['R_db_err']:.2e} dB, |Δ|T|| = {fc['T_db_err']:.2e} dB, "
                 f"|arg T_RS − arg T_S − q0| = {fc['T_phase_minus_q0_err']:.2e} rad.\n")
        rt = doc["slab_check"]["raytraced"]
        L.append(f"Ray-traced, one {rt['material']} surface, {rt['thickness_m']} m:\n")
        L.append("| Surface | Path | Angle (°) | Room Sim (dB) | Sionna (dB) | Δ (dB) | Δphase (rad) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rt["rows"]:
            L.append(f"| {r['surface']} | {r['path']} | {r['angle_deg']:.1f} | {r['roomsim_db']:.3f} | "
                     f"{r['sionna_db']:.3f} | {r['diff_db']:+.4f} | {r['diff_phase_rad']:+.4f} |")
    rt = doc["runtime"]
    L.append(f"\nRuntime {rt['wall_s']:.0f} s, peak RSS {rt['peak_rss_gb']:.1f} GB, backend "
             f"{doc['settings']['backend_variant']}, GPU memory of this process: {rt['gpu_memory_mb_this_process']} MB.")
    return "\n".join(L) + "\n"


def _from_json(x):
    """Undo `_jsonable` for the numbers: None -> NaN, "inf"/"-inf" -> floats."""
    if isinstance(x, dict):
        return {k: _from_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_from_json(v) for v in x]
    if x is None:
        return float("nan")
    if x in ("inf", "-inf"):
        return float(x)
    return x


def markdown_from_json(path: str) -> str:
    """The result tables of a saved audit JSON (SIM)."""
    with open(path, encoding="utf-8") as f:
        doc = _from_json(json.load(f))
    if doc.get("label") != LABEL:
        raise ValueError(f"{path}: not a SIM audit document")
    return markdown(doc)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default=os.environ.get("SAANS_DATASET_DIR", DEFAULT_DATASET))
    p.add_argument("--out", default=DEFAULT_OUT, help="summary JSON (SIM)")
    p.add_argument("--markdown", default=None, help="also write the result tables here")
    p.add_argument("--scenes", type=int, default=N_SCENES)
    p.add_argument("--indices", type=int, nargs="*", default=None, help="explicit scene indices instead of a draw")
    p.add_argument("--scene-seed", type=int, default=SCENE_SEED)
    p.add_argument("--sionna-seed", type=int, default=SIONNA_SEED)
    p.add_argument("--variants", nargs="*", default=[v for v, _o, _s in VARIANTS])
    p.add_argument("--spec-depth", type=int, default=SPEC_MAX_DEPTH)
    p.add_argument("--spec-samples", type=int, default=SPEC_SAMPLES)
    p.add_argument("--spec-cap", type=int, default=SPEC_CAP)
    p.add_argument("--diffuse-s", type=float, nargs="*", default=list(DIFFUSE_S))
    p.add_argument("--diffuse-depth", type=int, default=DIFFUSE_MAX_DEPTH)
    p.add_argument("--diffuse-samples", type=int, default=DIFFUSE_SAMPLES)
    p.add_argument("--diffuse-cap", type=int, default=DIFFUSE_CAP)
    p.add_argument("--no-slab-check", action="store_true")
    p.add_argument("--quick", action="store_true", help="fewer rays (a smoke run, not the audit)")
    p.add_argument("--markdown-from", default=None, metavar="JSON",
                   help="only print the result tables of a saved audit JSON (to --markdown or stdout)")
    p.add_argument("--links-out", default=None, metavar="JSON",
                   help="also write every per-link record (large; not part of the results)")
    p.add_argument("--no-repro", dest="repro", action="store_false",
                   help="skip the reproducibility rerun of one scene with another Sionna seed")
    opts = p.parse_args(argv)
    if opts.quick:
        opts.spec_samples = min(opts.spec_samples, 100_000)
        opts.diffuse_samples = min(opts.diffuse_samples, 50_000)
    return opts


def main(argv=None) -> int:
    opts = parse_args(argv)
    if opts.markdown_from:
        text = markdown_from_json(opts.markdown_from)
        if opts.markdown:
            with open(opts.markdown, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text, end="")
        return 0
    doc = run(opts)
    os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
    write_json(slim(doc), opts.out)
    print(f"SIM: wrote {opts.out}")
    if opts.markdown:
        with open(opts.markdown, "w", encoding="utf-8") as f:
            f.write(markdown(doc))
        print(f"SIM: wrote {opts.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
