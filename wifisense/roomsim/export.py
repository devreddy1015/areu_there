"""Export a Room Sim run (spec/roomsim.md §13): `<prefix>.sim.pcap`,
`<prefix>.truth.jsonl` and `<prefix>.manifest.json`.

The pcap is the `wifisense.sources.recorder` format (ns pcap, LINKTYPE_RAW,
IPv4+UDP), written by its own `PcapWriter`: source `10.1.<rx>.1:5566`,
destination `10.0.0.1:5566`, timestamp `1_790_000_000 s + t`. The source
address and port are a convention, never the gateway's own or a network
address (a real node sends from an ephemeral port). Truth goes only into the
truth file, never the pcap. Everything is labelled SIM. `saans
roomsim-export` (Rust) writes the same three files; its pcap is
byte-identical to this one for the same scene and configuration.

`export_run` refuses to start when any of the three files exists, unless
`overwrite=True`, which first deletes the old manifest; the manifest is
written last, so a run that fails part-way never leaves a manifest that
describes other files.
"""

from __future__ import annotations

import errno
import json
import os

from ..sources.recorder import PcapWriter
from .sim import LABEL, RoomSim

__all__ = ["export_run", "manifest", "source_address", "T0_NS", "PORT", "GATEWAY"]

T0_NS = 1_790_000_000 * 1_000_000_000
PORT = 5566
GATEWAY = ("10.0.0.1", PORT)


def source_address(rx: int) -> tuple[str, int]:
    """§13: `10.1.<rx>.1:5566` for node `rx` (0..254), a convention that is
    never the gateway's address or a network address."""
    return (f"10.1.{rx}.1", PORT)


def receiver_rate_hz(sim: RoomSim) -> dict:
    """§13: receiver (ascending, every `rx`/`txrx` node) -> the plain left
    to right sum, over its links in link order, of their per-link rates."""
    out = {}
    for rx in sorted(sim.receivers):
        acc = 0.0
        for a, b in sim.links:
            if b == rx:
                acc = acc + sim.per_link_rate_hz(a)
        out[str(rx)] = acc
    return out


def manifest(sim: RoomSim, counts: dict | None = None) -> dict:
    """The §13 manifest of a run. `counts`, from `export_run`, adds each
    link's `packets`, `lost` and `datagrams` and the run's totals."""
    scene = sim.scene
    rates = {f"{a},{b}": sim.per_link_rate_hz(a) for (a, b) in sim.links}
    links = []
    for li, (a, b) in enumerate(sim.links):
        rec = {"link_index": li, "tx": a, "rx": b, "per_link_rate_hz": sim.per_link_rate_hz(a)}
        if counts is not None:
            rec.update(counts["links"][li])
        links.append(rec)
    out = {
        "label": LABEL,
        "spec": "roomsim v1",
        "scene_name": scene.name,
        "scene_sha256": scene.sha256,
        "provenance": scene.provenance,
        "config": sim.cfg.to_dict(),
        "per_link_rate_hz": rates,
        "links": links,
    }
    if counts is not None:
        out["counts"] = dict(counts["totals"])
    out["receiver_rate_hz"] = receiver_rate_hz(sim)
    # §12.4: only one transmitter's stream can be keyed by rx alone.
    out["gateway_faithful"] = sim.n_transmitters == 1
    if sim.world.implied_walls:
        cfg = sim.cfg
        out["assumptions"] = [
            f"SIM: the scene lists no walls; the {len(sim.world.walls)} footprint edges are walls of "
            f"{cfg.implied_wall_material}, {float(cfg.implied_wall_thickness_m)!r} m thick, full height"
        ]
    return out


def export_run(sim: RoomSim, out_prefix: str, *, overwrite: bool = False) -> dict:
    """Run `sim` and write `<out_prefix>.sim.pcap`, `<out_prefix>.truth.jsonl`
    and `<out_prefix>.manifest.json` (§13). Returns the paths and counts.

    Without `overwrite`, raises FileExistsError naming the first of the
    three (pcap, truth, manifest) that exists, before opening any. With it,
    the old manifest is deleted first; the manifest is always written last.
    """
    if not sim.cfg.quantize:
        raise ValueError("quantize: an export needs datagrams, and quantize = false emits none")
    out_prefix = os.fspath(out_prefix)
    pcap_path = out_prefix + ".sim.pcap"
    truth_path = out_prefix + ".truth.jsonl"
    man_path = out_prefix + ".manifest.json"
    if overwrite:
        try:
            os.remove(man_path)
        except FileNotFoundError:
            pass
    else:
        for path in (pcap_path, truth_path, man_path):
            if os.path.lexists(path):
                raise FileExistsError(errno.EEXIST, "already exists (pass overwrite=True to replace the export)",
                                      path)
    mode = "w" if overwrite else "x"
    per_link = [{"packets": 0, "lost": 0, "datagrams": 0} for _ in sim.links]
    n_dgram = 0
    n_tr = 0
    with PcapWriter(pcap_path, overwrite=overwrite) as pw, open(truth_path, mode) as tf:
        for tr in sim.transmissions():
            n_tr += 1
            tr_rec = {
                "label": LABEL, "m": tr.m, "slot": tr.slot, "t": tr.t, "tx": tr.tx,
                "people": [
                    {"name": f"people[{i}]", "x": p.x, "y": p.y, "heading": p.heading,
                     "walking": p.walking, "posture": p.posture}
                    for i, p in enumerate(tr.truth.people)
                ],
                "phone": None if tr.truth.phone is None else list(tr.truth.phone),
                "packets": [
                    {"rx": pk.rx, "link_index": pk.link_index, "lost": pk.lost, "seq": pk.seq,
                     "rssi": pk.rssi, "rssi_true": pk.rssi_true}
                    for pk in tr.packets
                ],
            }
            tf.write(json.dumps(tr_rec) + "\n")
            for pk in tr.packets:
                c = per_link[pk.link_index]
                c["packets"] += 1
                c["lost"] += int(pk.lost)
                if pk.datagram is None:
                    continue
                t_ns = T0_NS + round(pk.t * 1e9)
                pw.write(t_ns, source_address(pk.rx), GATEWAY, pk.datagram)
                c["datagrams"] += 1
                n_dgram += 1
    totals = {
        "transmissions": n_tr,
        "packets": sum(c["packets"] for c in per_link),
        "lost": sum(c["lost"] for c in per_link),
        "datagrams": n_dgram,
    }
    with open(man_path, mode) as fh:
        json.dump(manifest(sim, {"links": per_link, "totals": totals}), fh, indent=1)
        fh.write("\n")
    return {"pcap": pcap_path, "truth": truth_path, "manifest": man_path,
            "datagrams": n_dgram, "transmissions": n_tr}
