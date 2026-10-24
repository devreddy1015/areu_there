"""Plot one SIM sample of dataset v1 for the owner (spec/dataset.md §10).

    .venv/bin/python ml/plot_sample.py DIR --scene i --window w --out FILE.png [--links N]

Left: the room plan of scene `i` -- the footprint, the 3 x 3 zone grid over
the bounding box (cells with no floor hatched, occupied cells tinted),
the nodes with their ids, every link as a thin line coloured by its
amplitude variance over the window, the people at the window's end with
their zones, and Locate's WLM estimate at that tick drawn as a circle of
its radius (or the text "WLM abstained -- <reason>"). Right: the CSI
amplitude of the `N` links that vary most over 2 s window `w` (subcarrier
x time; masked slots are white gaps, null subcarriers hatched), and the
labels at the window's end.

Everything drawn is SIM: a random room, simulated people, simulated CSI,
a baseline computed on it. Every panel says so. No person estimate is
ever drawn without its radius.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: this script only ever writes a file

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

if __package__ in (None, ""):  # run as a script, even with python -I
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import roomsim_dataset as rd  # noqa: E402

WINDOW_S = 2.0
LINK_COLOURS = ("tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown")


def plot_sample(dataset_dir: str, scene_index: int, window_index: int, out: str, n_links: int = 3) -> None:
    """Write the SIM sample figure for 2 s window `window_index` of scene
    `scene_index` to `out` (PNG)."""
    manifest = rd.load_manifest(dataset_dir)
    scene = rd.load_scene(dataset_dir, scene_index, manifest=manifest)
    ends = scene["window_end_tick"]
    if not 0 <= window_index < len(ends):
        raise IndexError(f"window {window_index} outside 0..{len(ends) - 1} (SIM scene {scene_index})")
    k = int(ends[window_index])
    win = rd.window(scene, k, WINDOW_S)
    labels = rd.labels_at(scene, k)
    base = rd.baseline_at(scene, k)
    variance = link_variance(win, scene["subcarrier_active"])
    shown = pick_links(win, variance, n_links)

    fig = plt.figure(figsize=(17, max(9.0, 2.3 * len(shown) + 2.5)), layout="constrained")
    grid = fig.add_gridspec(len(shown) + 1, 2, width_ratios=(1.15, 1), height_ratios=[1.0] * len(shown) + [0.9])
    draw_plan(fig.add_subplot(grid[:, 0]), scene, labels, base, variance, shown, window_index)
    for row, link in enumerate(shown):
        draw_link(fig.add_subplot(grid[row, 1]), scene, win, k, link, LINK_COLOURS[row % len(LINK_COLOURS)])
    draw_labels(fig.add_subplot(grid[-1, 1]), labels, base)
    fig.suptitle(f"SIM -- dataset v1 scene {scene['meta']['scene_id']}, 2 s window {window_index}: "
                 "simulated room, people and CSI (not a measurement)", fontsize=13, fontweight="bold")
    fig.savefig(out, dpi=100)
    plt.close(fig)


def link_variance(win: rd.Window, active: np.ndarray) -> np.ndarray:
    """Per link, the SIM amplitude's variance over the window's received
    slots, averaged over active subcarriers; NaN with fewer than 2 slots."""
    amp = win.amplitude[:, active.astype(bool), :].astype(np.float64)
    m = win.mask[:, None, :]
    n = win.mask.sum(axis=1)
    mean = np.where(m, amp, 0.0).sum(axis=2) / np.maximum(n, 1)[:, None]
    var = (np.where(m, amp - mean[..., None], 0.0) ** 2).sum(axis=2) / np.maximum(n, 1)[:, None]
    out = var.mean(axis=1)
    out[n < 2] = np.nan
    return out


def pick_links(win: rd.Window, variance: np.ndarray, n_links: int) -> list:
    """The `n_links` links with the largest SIM variance (links with no
    packet last, then by index): deterministic."""
    key = np.where(np.isfinite(variance), -variance, np.inf)
    order = sorted(range(len(key)), key=lambda l: (not win.link_mask[l], key[l], l))
    return order[:max(1, min(n_links, len(order)))]


def draw_plan(ax, scene: dict, labels: dict, base: dict, variance: np.ndarray, shown: list, w: int) -> None:
    """The SIM room plan at the window's end tick."""
    width, depth = (float(v) for v in scene["room_size"][:2])
    frac = scene["zone_floor_frac"]
    for z in range(9):
        x0, y0 = (z % 3) * width / 3, (z // 3) * depth / 3
        if frac[z] == 0:
            ax.add_patch(Rectangle((x0, y0), width / 3, depth / 3, fill=False, hatch="///",
                                   edgecolor="0.6", linewidth=0))
        if labels["zone_occupied"][z]:
            ax.add_patch(Rectangle((x0, y0), width / 3, depth / 3, color="gold", alpha=0.25, linewidth=0))
        note = f"zone {z}" + (f" (floor {frac[z]:.2f})" if 0 < frac[z] < 1 else "")
        ax.text(x0 + 0.06, y0 + 0.06, note, fontsize=7, color="0.45")
    for i in (1, 2):
        ax.plot([i * width / 3] * 2, [0, depth], ":", color="0.55", linewidth=0.8)
        ax.plot([0, width], [i * depth / 3] * 2, ":", color="0.55", linewidth=0.8)
    ax.add_patch(Rectangle((0, 0), width, depth, fill=False, linestyle="--", edgecolor="0.6", linewidth=0.8))
    ax.add_patch(Polygon(scene["footprint"], closed=True, fill=False, edgecolor="black", linewidth=2.0))

    pos = {int(i): scene["node_pos"][j, :2] for j, i in enumerate(scene["node_ids"])}
    pairs = {}
    for l, (tx, rx) in enumerate(scene["links"]):
        pairs.setdefault(tuple(sorted((int(tx), int(rx)))), []).append(variance[l])
    segments = [(pos[a], pos[b]) for a, b in pairs]
    values = np.array([np.nan if np.isnan(v).all() else np.nanmean(v) for v in pairs.values()])
    lines = LineCollection(segments, cmap=plt.get_cmap("viridis").with_extremes(bad="0.75"),
                           linewidths=0.9, alpha=0.8)
    lines.set_array(np.ma.masked_invalid(values))
    ax.add_collection(lines)
    plt.colorbar(lines, ax=ax, shrink=0.6, pad=0.01).set_label(
        "SIM link amplitude variance over the window (both directions)", fontsize=8)
    handles = []
    for row, l in enumerate(shown):
        tx, rx = (int(v) for v in scene["links"][l])
        colour = LINK_COLOURS[row % len(LINK_COLOURS)]
        ax.annotate("", xy=pos[rx], xytext=pos[tx], arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.8})
        handles.append(Line2D([], [], color=colour, lw=1.8, label=f"link {l}: {tx} -> {rx} (right)"))

    for i, p in pos.items():
        ax.plot(p[0], p[1], "s", color="black", markersize=7)
        ax.annotate(f"node {i}", p, xytext=(5, 5), textcoords="offset points", fontsize=8)
    for person in labels["people"]:
        ax.plot(person["x"], person["y"], "o", color="crimson", markersize=9)
        what = "walking" if person["walking"] else person["posture"]
        ax.annotate(f"P{person['index']} zone {person['zone']}, {what}", (person["x"], person["y"]),
                    xytext=(6, -12), textcoords="offset points", fontsize=8, color="crimson")
    handles.append(Line2D([], [], marker="o", color="crimson", linestyle="", label="person (SIM truth)"))

    wlm = base["wlm"]
    if wlm["published"]:
        (cx, cy), r = wlm["xy"], wlm["radius_m"]
        ax.add_patch(Circle((cx, cy), r, fill=False, linestyle="--", edgecolor="darkviolet", linewidth=1.6))
        ax.plot(cx, cy, "+", color="darkviolet", markersize=12, mew=2)
        handles.append(Line2D([], [], color="darkviolet", linestyle="--", marker="+",
                              label=f"WLM estimate (SIM), radius {r:.2f} m (uncalibrated)"))
    else:
        ax.text(0.5, 0.985, f"WLM abstained -- {wlm['reason']} (SIM)", transform=ax.transAxes, ha="center",
                va="top", fontsize=10, color="darkviolet", bbox={"facecolor": "white", "edgecolor": "darkviolet"})
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=3, frameon=False)

    ax.set_xlim(-0.5, width + 0.5)
    ax.set_ylim(-0.5, depth + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, SIM scene geometry)")
    ax.set_ylabel("y (m, SIM scene geometry)")
    ax.set_title(f"SIM room plan at t = {labels['t']:.1f} s (end of window {w}, label tick {labels['k']})")


def draw_link(ax, scene: dict, win: rd.Window, k: int, link: int, colour: str) -> None:
    """One link's SIM CSI amplitude over the window: rows by subcarrier
    index k ascending, one column per slot, masked slots white."""
    order = np.argsort(scene["subcarrier_k"], kind="stable")
    sub_k = scene["subcarrier_k"][order]
    active = scene["subcarrier_active"][order].astype(bool)
    n_tx = int(scene["meta"]["n_tx"])
    t_end = scene["t_label"][k]
    t = win.amplitude.shape[2]
    t_start = t_end - t * n_tx / rd.TX_PER_S
    img = ax.imshow(win.amplitude[link][order], aspect="auto", origin="lower", interpolation="nearest",
                    cmap=plt.get_cmap("viridis").with_extremes(bad="white"),
                    extent=(t_start, t_end, -0.5, len(order) - 0.5))
    for row in np.flatnonzero(~active):
        ax.axhspan(row - 0.5, row + 0.5, facecolor="0.88", edgecolor="0.55", hatch="xxx", linewidth=0)
    ax.axvline(t_end - WINDOW_S, color="black", linewidth=0.8, linestyle="--")
    ticks = list(range(0, len(order), 8)) + [len(order) - 1]
    ax.set_yticks(ticks, [str(int(sub_k[r])) for r in ticks], fontsize=7)
    tx, rx = (int(v) for v in scene["links"][link])
    got = int(win.mask[link].sum())
    ax.set_title(f"SIM CSI amplitude, link {link} ({tx} -> {rx}): {got}/{t} slots received",
                 fontsize=9, color=colour)
    ax.set_ylabel("subcarrier k", fontsize=8)
    ax.set_xlabel("t (s, SIM); hatched rows: null subcarriers; white: no packet; dashed: window start",
                  fontsize=7)
    plt.colorbar(img, ax=ax, fraction=0.04, pad=0.01).set_label("SIM |CSI| (int8 units)", fontsize=7)


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
    for name in ("wlm", "rti"):
        e = base[name]
        if e["published"]:
            lines.append(f"{name.upper()} (SIM baseline): centre ({e['xy'][0]:.2f}, {e['xy'][1]:.2f}) m, "
                         f"radius {e['radius_m']:.2f} m (uncalibrated)")
        else:
            lines.append(f"{name.upper()} (SIM baseline): abstained -- {e['reason']}")
    if not base["valid"]:
        lines.append("impairments off: the stored baseline is not valid (SIM, §8)")
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", family="monospace",
            fontsize=8.5, bbox={"facecolor": "0.96", "edgecolor": "0.6", "boxstyle": "round"})


def main(argv=None) -> int:
    """The command line: one SIM sample plot."""
    p = argparse.ArgumentParser(description="Plot one SIM sample of dataset v1 (spec/dataset.md §10).")
    p.add_argument("dir", help="the dataset directory (holds manifest.json)")
    p.add_argument("--scene", type=int, required=True, help="scene index")
    p.add_argument("--window", type=int, required=True, help="2 s window index w (ends at tick 4w+3)")
    p.add_argument("--out", required=True, help="the PNG to write")
    p.add_argument("--links", type=int, default=3, help="links to show on the right (default 3)")
    args = p.parse_args(argv)
    plot_sample(args.dir, args.scene, args.window, args.out, args.links)
    print(f"SIM sample plot written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
