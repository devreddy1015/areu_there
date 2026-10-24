"""Plot one SIM sample of dataset v1 for the owner (spec/dataset.md §10).

    .venv/bin/python ml/plot_sample.py DIR --scene i --window w --out FILE.png [--links N]

Left: the room plan of scene `i` -- the footprint, furniture, doors and
windows (from `DIR/scenes/<id>.json`), the 3 x 3 zone grid over the
bounding box (cells with no floor hatched, occupied cells tinted), the
nodes with their ids, one line per node pair coloured by its gain-
normalised amplitude variance over the window, the people at the
window's end with their zones, and Locate's WLM estimate at that tick as a
circle of its radius with the share of floor it covers (or the text
"WLM abstained -- <reason>"). Right: the raw CSI amplitude of the `N` node
pairs that vary most (one direction each; subcarrier x time over all 64
LLTF bins, masked slots white, null and unsent subcarriers hatched) and the
labels at the window's end.

Links are ranked by the variance of **per-packet-normalised** amplitude
(each packet's active-subcarrier amplitudes divided by their mean): Room
Sim's AGC steps a link's gain by about ±3 dB uniformly across subcarriers
in about 17% of 2 s link-windows, empty room or not, and raw variance
mostly ranks those steps.

Everything drawn is SIM: a random room, simulated people, simulated CSI,
a baseline computed on it with an idealised calibration. Every panel says
so. No person estimate is ever drawn without its radius.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: this script only ever writes a file

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402

if __package__ in (None, ""):  # run as a script, even with python -I
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import roomsim_dataset as rd  # noqa: E402

WINDOW_S = 2.0
LINK_COLOURS = ("tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown")
PROFILES = {"on": "impairments on", "hardware_off": "hardware impairments off",
            "off": "all impairments and noise off"}
NOT_SENT = range(-8, 0)  # LLTF bins -8..-1: past the 56 buffer entries the firmware forwards
OPENINGS = {"door": "saddlebrown", "window": "deepskyblue"}
EMPTY_ROOM = "false alarm -- the room is empty (SIM)"


def plot_sample(dataset_dir: str, scene_index: int, window_index: int, out: str, n_links: int = 3) -> None:
    """Write the SIM sample figure for 2 s window `window_index` of scene
    `scene_index` to `out` (PNG)."""
    fig = make_figure(dataset_dir, scene_index, window_index, n_links)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def make_figure(dataset_dir: str, scene_index: int, window_index: int, n_links: int = 3):
    """The SIM sample figure (see the module docstring), not yet saved."""
    manifest = rd.load_manifest(dataset_dir)
    scene = rd.load_scene(dataset_dir, scene_index, manifest=manifest)
    try:
        geometry = rd.load_scene_json(dataset_dir, scene_index, manifest=manifest)
    except FileNotFoundError:
        geometry = None
    ends = scene["window_end_tick"]
    if not 0 <= window_index < len(ends):
        raise IndexError(f"window {window_index} outside 0..{len(ends) - 1} (SIM scene {scene_index})")
    k = int(ends[window_index])
    win = rd.window(scene, k, WINDOW_S)
    labels = rd.labels_at(scene, k)
    base = rd.baseline_at(scene, k)
    variance = link_variance(win, scene["subcarrier_active"])
    pairs = pair_variance(scene["links"], variance)
    shown = pick_links(pairs, win, variance, n_links)

    fig = plt.figure(figsize=(17, max(9.0, 2.3 * len(shown) + 2.5)), layout="constrained")
    grid = fig.add_gridspec(len(shown) + 1, 2, width_ratios=(1.15, 1), height_ratios=[1.0] * len(shown) + [0.9])
    handles = draw_plan(fig.add_subplot(grid[:, 0]), scene, geometry, labels, base, pairs, shown, window_index)
    for row, link in enumerate(shown):
        draw_link(fig.add_subplot(grid[row, 1]), scene, win, k, link, variance[link],
                  LINK_COLOURS[row % len(LINK_COLOURS)])
    draw_labels(fig.add_subplot(grid[-1, 1]), labels, base)
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    profile = PROFILES[scene["dataset"]["impairments"]]
    fig.suptitle(f"SIM -- dataset v1 scene {scene['meta']['scene_id']}, 2 s window {window_index}, {profile}: "
                 "simulated room, people and CSI (not a measurement)", fontsize=13, fontweight="bold")
    return fig


def link_variance(win: rd.Window, active: np.ndarray) -> np.ndarray:
    """Per link, the variance over the window's received slots of the
    per-packet-normalised SIM amplitude (each packet's active-subcarrier
    amplitudes divided by their mean, which removes AGC gain steps),
    averaged over active subcarriers. Dimensionless; NaN with fewer than 2
    packets."""
    amp = win.amplitude[:, active.astype(bool), :].astype(np.float64)    # [L, A, T], NaN where masked
    gain = amp.mean(axis=1)                                               # [L, T]
    ok = win.mask & (gain > 0)
    norm = np.where(ok[:, None, :], amp / np.where(ok, gain, 1.0)[:, None, :], 0.0)
    n = np.maximum(ok.sum(axis=1), 1)[:, None]
    mean = norm.sum(axis=2) / n
    var = (np.where(ok[:, None, :], norm - mean[..., None], 0.0) ** 2).sum(axis=2) / n
    out = var.mean(axis=1)
    out[ok.sum(axis=1) < 2] = np.nan
    return out


def pair_variance(links: np.ndarray, variance: np.ndarray) -> dict:
    """SIM: unordered node pair (a, b), a < b -> (its directed links, the
    mean of their finite `link_variance`, NaN if none)."""
    pairs: dict = {}
    for l, (tx, rx) in enumerate(links):
        pairs.setdefault((min(int(tx), int(rx)), max(int(tx), int(rx))), []).append(l)
    out = {}
    for pair, ls in pairs.items():
        finite = [variance[l] for l in ls if np.isfinite(variance[l])]
        out[pair] = (ls, float(np.mean(finite)) if finite else math.nan)
    return out


def pick_links(pairs: dict, win: rd.Window, variance: np.ndarray, n_links: int) -> list:
    """The SIM links to show: the `n_links` node pairs with the largest
    pair variance (pairs with no packet last, then by link index), each by
    its more variable direction, so no arrow hides another. Deterministic."""
    def rank(item):
        ls, v = item
        return (not win.link_mask[ls].any(), -v if math.isfinite(v) else math.inf, min(ls))

    def direction(l):
        return (variance[l] if np.isfinite(variance[l]) else -math.inf, -l)

    ranked = sorted(pairs.values(), key=rank)
    return [max(ls, key=direction) for ls, _ in ranked[:max(1, min(n_links, len(ranked)))]]


def floor_coverage(footprint: np.ndarray, centre, radius: float, step: float = 0.02) -> float:
    """SIM: the share of the footprint's floor within `radius` of `centre`,
    on a `step` grid (furniture counts as floor)."""
    lo, hi = footprint.min(axis=0), footprint.max(axis=0)
    xs = np.arange(lo[0] + step / 2, hi[0], step)
    ys = np.arange(lo[1] + step / 2, hi[1], step)
    x, y = (a.ravel() for a in np.meshgrid(xs, ys))
    floor = Path(footprint).contains_points(np.column_stack([x, y]))
    near = (x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= radius * radius
    return float((floor & near).sum() / max(floor.sum(), 1))


def box_corners(obj: dict) -> np.ndarray:
    """The (x, y) corners of a scene object's footprint, its size x along
    `yaw_deg` (SIM scene geometry)."""
    (cx, cy), (sx, sy) = obj["pos"][:2], obj["size"][:2]
    a = math.radians(obj.get("yaw_deg", 0.0))
    c, s = math.cos(a), math.sin(a)
    local = [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]
    return np.array([(cx + u * c - v * s, cy + u * s + v * c) for u, v in local])


def draw_plan(ax, scene: dict, geometry, labels: dict, base: dict, pairs: dict, shown: list, w: int) -> list:
    """The SIM room plan at the window's end tick; returns legend handles."""
    width, depth = (float(v) for v in scene["room_size"][:2])
    footprint = scene["footprint"].astype(np.float64)
    frac, reachable = scene["zone_floor_frac"], scene["zone_reachable"]
    handles = []
    for z in range(9):
        x0, y0 = (z % 3) * width / 3, (z // 3) * depth / 3
        if frac[z] == 0:
            ax.add_patch(Rectangle((x0, y0), width / 3, depth / 3, fill=False, hatch="///",
                                   edgecolor="0.6", linewidth=0))
        if labels["zone_occupied"][z]:
            ax.add_patch(Rectangle((x0, y0), width / 3, depth / 3, color="gold", alpha=0.25, linewidth=0))
        share = f"{frac[z]:.2f}".replace("1.00", ">0.99").replace("0.00", "<0.01")
        note = f"zone {z}" + (f" (floor {share})" if 0 < frac[z] < 1 else "")
        note += " (unreachable)" if frac[z] > 0 and not reachable[z] else ""
        ax.text(x0 + 0.06, y0 + 0.06, note, fontsize=7, color="0.45")
    for i in (1, 2):
        ax.plot([i * width / 3] * 2, [0, depth], ":", color="0.55", linewidth=0.8)
        ax.plot([0, width], [i * depth / 3] * 2, ":", color="0.55", linewidth=0.8)
    ax.add_patch(Rectangle((0, 0), width, depth, fill=False, linestyle="--", edgecolor="0.6", linewidth=0.8))
    ax.add_patch(Polygon(footprint, closed=True, fill=False, edgecolor="black", linewidth=2.0))

    if geometry is None:
        ax.text(0.5, 0.02, "scene file not found: furniture not drawn", transform=ax.transAxes,
                ha="center", fontsize=8, color="0.4")
    else:
        kinds = set()
        for obj in geometry.get("objects", []):
            kind = obj.get("type", "?")
            kinds.add(kind)
            colour = OPENINGS.get(kind)
            if colour:
                ax.add_patch(Polygon(box_corners(obj), closed=True, facecolor=colour, edgecolor=colour, zorder=3))
            else:
                ax.add_patch(Polygon(box_corners(obj), closed=True, facecolor="0.88", edgecolor="0.45", zorder=1))
                ax.text(obj["pos"][0], obj["pos"][1], kind, ha="center", va="center", fontsize=7, color="0.35")
        if kinds - set(OPENINGS):
            handles.append(Patch(facecolor="0.88", edgecolor="0.45", label="furniture (SIM scene)"))
        handles += [Patch(color=colour, label=kind) for kind, colour in OPENINGS.items() if kind in kinds]

    pos = {int(i): scene["node_pos"][j, :2] for j, i in enumerate(scene["node_ids"])}
    values = np.array([v for _, v in pairs.values()])
    lines = LineCollection([(pos[a], pos[b]) for a, b in pairs], linewidths=0.9, alpha=0.8,
                           cmap=plt.get_cmap("viridis").with_extremes(bad="0.75"))
    lines.set_array(np.ma.masked_invalid(values))
    top = np.nanmax(values) if np.isfinite(values).any() else 0.0
    lines.set_clim(0.0, top if top > 0 else 1.0)
    ax.add_collection(lines)
    plt.colorbar(lines, ax=ax, shrink=0.6, pad=0.01).set_label(
        "SIM node-pair variance over the window of per-packet-normalised amplitude\n"
        "(amplitude / packet mean over active subcarriers; dimensionless)", fontsize=8)
    for row, l in enumerate(shown):
        tx, rx = (int(v) for v in scene["links"][l])
        colour = LINK_COLOURS[row % len(LINK_COLOURS)]
        ax.annotate("", xy=pos[rx], xytext=pos[tx], arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.8})
        handles.append(Line2D([], [], color=colour, lw=1.8, label=f"link {l}: {tx} -> {rx} (right)"))

    for i, p in pos.items():
        ax.plot(p[0], p[1], "s", color="black", markersize=7, zorder=4)
        ax.annotate(f"node {i}", p, xytext=(5, 5), textcoords="offset points", fontsize=8)
    for person in labels["people"]:
        ax.plot(person["x"], person["y"], "o", color="crimson", markersize=9, zorder=5)
        what = "walking" if person["walking"] else person["posture"]
        ax.annotate(f"P{person['index']} zone {person['zone']}, {what}", (person["x"], person["y"]),
                    xytext=(6, -12), textcoords="offset points", fontsize=8, color="crimson", zorder=5)
    handles.append(Line2D([], [], marker="o", color="crimson", linestyle="", label="person (SIM truth)"))

    lo, hi = np.array([-0.5, -0.5]), np.array([width + 0.5, depth + 0.5])
    wlm = base["wlm"]
    if not base["valid"]:
        banner = "baseline not valid for this dataset (SIM)"
    elif wlm["published"]:
        banner = None
        (cx, cy), r = wlm["xy"], wlm["radius_m"]
        ax.add_patch(Circle((cx, cy), r, fill=False, linestyle="--", edgecolor="darkviolet", linewidth=1.6))
        ax.plot(cx, cy, "+", color="darkviolet", markersize=12, mew=2)
        lo = np.minimum(lo, (cx - r - 0.2, cy - r - 0.2))
        hi = np.maximum(hi, (cx + r + 0.2, cy + r + 0.2))
        label = (f"WLM estimate (SIM): radius {r:.2f} m (uncalibrated), "
                 f"covers {100 * floor_coverage(footprint, (cx, cy), r):.0f}% of floor (SIM)")
        if labels["count"] == 0:
            label += f"\n{EMPTY_ROOM}"
        handles.append(Line2D([], [], color="darkviolet", linestyle="--", marker="+", label=label))
    else:
        banner = f"WLM abstained -- {wlm['reason']} (SIM)"
    if banner:
        ax.text(0.5, 0.985, banner, transform=ax.transAxes, ha="center", va="top", fontsize=10,
                color="darkviolet", bbox={"facecolor": "white", "edgecolor": "darkviolet"})

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, SIM scene geometry)")
    ax.set_ylabel("y (m, SIM scene geometry)")
    ax.set_title(f"SIM room plan at t = {labels['t']:.1f} s (end of window {w}, label tick {labels['k']})")
    return handles


def draw_link(ax, scene: dict, win: rd.Window, k: int, link: int, variance: float, colour: str) -> None:
    """One link's raw SIM CSI amplitude over the window: one row per LLTF
    bin k = -32..31 (the 56 sent entries; -8..-1 are not sent), one column
    per slot. White: a lost packet; grey dots: a slot outside the window."""
    image = np.full((64, win.amplitude.shape[2]), np.nan, np.float32)
    image[scene["subcarrier_k"].astype(int) + 32] = win.amplitude[link]
    n_tx = int(scene["meta"]["n_tx"])
    t_end = scene["t_label"][k]
    t = win.amplitude.shape[2]
    slot_s = n_tx / rd.TX_PER_S
    t_start = t_end - t * slot_s
    img = ax.imshow(image, aspect="auto", origin="lower", interpolation="nearest",
                    cmap=plt.get_cmap("viridis").with_extremes(bad="white"), extent=(t_start, t_end, -32.5, 31.5))
    for kk, active in zip(scene["subcarrier_k"].astype(int), scene["subcarrier_active"]):
        if not active:
            ax.axhspan(kk - 0.5, kk + 0.5, facecolor="0.88", edgecolor="0.55", hatch="xxx", linewidth=0)
    ax.axhspan(NOT_SENT[0] - 0.5, NOT_SENT[-1] + 0.5, facecolor="0.95", edgecolor="0.6", hatch="//", linewidth=0)
    ax.text((t_start + t_end) / 2, -4.5, "k -8..-1 not sent", ha="center", va="center", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    for j in np.flatnonzero(~win.inside[link]):
        ax.axvspan(t_start + j * slot_s, t_start + (j + 1) * slot_s, facecolor="0.8", hatch="..", linewidth=0)
    ax.axvline(t_end - WINDOW_S, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks([-32, -24, -16, -8, 0, 8, 16, 24, 31])
    ax.tick_params(axis="y", labelsize=7)
    tx, rx = (int(v) for v in scene["links"][link])
    got, sent = int(win.mask[link].sum()), int(win.inside[link].sum())
    ax.set_title(f"SIM raw CSI amplitude, link {link} ({tx} -> {rx}): {got} of {sent} in-window packets "
                 f"received; normalised variance {variance:.2g}", fontsize=9, color=colour)
    ax.set_ylabel("subcarrier k", fontsize=8)
    ax.set_xlabel("t (s, SIM). Hatched x: null subcarrier; hatched /: not sent; white: lost packet; "
                  "grey dots: outside the window; dashed: window start", fontsize=7)
    plt.colorbar(img, ax=ax, fraction=0.04, pad=0.01).set_label("SIM raw |CSI| (int8 units)", fontsize=7)


def draw_labels(ax, labels: dict, base: dict) -> None:
    """The SIM labels and baseline at the window's end, as text."""
    ax.axis("off")
    cells = [z for z in range(9) if labels["zone_occupied"][z]]
    lines = [f"SIM labels at t = {labels['t']:.1f} s (tick {labels['k']})",
             f"presence {int(labels['presence'])}, count {labels['count']}, "
             f"zone_occupied cells {cells or 'none'}, zone_single {labels['zone_single']}"]
    for p in labels["people"]:
        lines.append(f"P{p['index']}: xy ({p['x']:.2f}, {p['y']:.2f}) m, zone {p['zone']}, "
                     f"walking {int(p['walking'])}, posture {p['posture']}")
    if not base["valid"]:
        lines.append("WLM, RTI (SIM baseline): baseline not valid for this dataset")
    for name in ("wlm", "rti") if base["valid"] else ():
        e = base[name]
        if e["published"]:
            lines.append(f"{name.upper()} (SIM baseline): centre ({e['xy'][0]:.2f}, {e['xy'][1]:.2f}) m, "
                         f"radius {e['radius_m']:.2f} m (uncalibrated)")
            if labels["count"] == 0:
                lines.append(f"    {EMPTY_ROOM}")
        else:
            lines.append(f"{name.upper()} (SIM baseline): abstained -- {e['reason']}")
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", family="monospace",
            fontsize=8.5, bbox={"facecolor": "0.96", "edgecolor": "0.6", "boxstyle": "round"})


def main(argv=None) -> int:
    """The command line: one SIM sample plot."""
    p = argparse.ArgumentParser(description="Plot one SIM sample of dataset v1 (spec/dataset.md §10).")
    p.add_argument("dir", help="the dataset directory (holds manifest.json)")
    p.add_argument("--scene", type=int, required=True, help="scene index")
    p.add_argument("--window", type=int, required=True, help="2 s window index w (ends at tick 4w+3)")
    p.add_argument("--out", required=True, help="the PNG to write")
    p.add_argument("--links", type=int, default=3, help="node pairs to show on the right (default 3)")
    args = p.parse_args(argv)
    plot_sample(args.dir, args.scene, args.window, args.out, args.links)
    print(f"SIM sample plot written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
