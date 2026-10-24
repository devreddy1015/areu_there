"""Command line entry point.

    python3 -m wifisense demo              # synthetic end-to-end, no hardware
    python3 -m wifisense listen --port 5566 --rate 20   # live ESP32 nodes
    python3 -m wifisense record --out run.pcap          # raw datagrams to disk
    python3 -m wifisense stats run.pcap                 # per-node rate/loss/jitter
    python3 -m wifisense replay run.pcap                # pipeline over a recording
    python3 -m wifisense replay run.pcap --send 127.0.0.1:5566   # re-send over UDP
"""

from __future__ import annotations

import argparse
import errno
import math
import signal
import sys
import threading

from .pipeline import SensingPipeline
from .sources import SyntheticCsiSource, SyntheticConfig, listen_udp
from .sources.esp32_udp import SequenceGapFiller
from .sources.recorder import (
    LinkStats,
    ReadStats,
    RecordStats,
    iter_frames,
    link_stats,
    record_udp,
    replay_udp,
)
from .types import VitalStatus

_STATUS_MARK = {
    VitalStatus.VALID: "ok",
    VitalStatus.DEGRADED: "~",
    VitalStatus.UNRELIABLE: "?",
    VitalStatus.UNAVAILABLE: "-",
}


def _format(reading) -> str:
    rr, hr = reading.respiratory_rate, reading.heart_rate
    return (
        f"t={reading.timestamp_s:7.1f}s  "
        f"{'OCCUPIED' if reading.occupied else 'empty   '}  "
        f"energy={reading.motion_energy:7.4f}  "
        f"breathing={rr.value_bpm:6.2f} BPM [{_STATUS_MARK[rr.status]:>2}] "
        f"c={rr.confidence:.2f}  "
        f"heart={hr.value_bpm:6.2f} BPM [{_STATUS_MARK[hr.status]:>2}] "
        f"c={hr.confidence:.2f}"
    )


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = SyntheticConfig(
        breathing_bpm=args.breathing_bpm,
        heart_bpm=args.heart_bpm,
        sample_rate_hz=args.rate,
    )
    source = SyntheticCsiSource(cfg)
    pipeline = SensingPipeline(
        n_subcarriers=cfg.n_subcarriers,
        sample_rate_hz=cfg.sample_rate_hz,
        calibration_s=args.calibration_s,
    )

    # The empty-room phase has to cover filter settling *and* the
    # calibration window, or the threshold is never established.
    warmup_s = pipeline.presence.warmup_s
    empty_phase_s = warmup_s + args.calibration_s + 10.0

    print(f"ground truth: breathing {cfg.breathing_bpm} BPM, heart {cfg.heart_bpm} BPM")
    print(
        f"phase 1: empty room, {empty_phase_s:g}s "
        f"({warmup_s:g}s filter warmup + {args.calibration_s:g}s calibration)"
    )
    cfg.occupied = False
    for frame in source.seconds(empty_phase_s):
        pipeline.process(frame)
    print(f"  calibrated: {pipeline.calibrated}  threshold={pipeline.presence.threshold:.4f}")

    print(f"phase 2: subject present, {args.occupied_s:g}s")
    cfg.occupied = True
    every = int(cfg.sample_rate_hz * 10)
    for i, frame in enumerate(source.seconds(args.occupied_s)):
        reading = pipeline.process(frame)
        if i % every == 0:
            print("  " + _format(reading))

    print(f"phase 3: subject leaves, {args.empty_s:g}s")
    cfg.occupied = False
    for i, frame in enumerate(source.seconds(args.empty_s)):
        reading = pipeline.process(frame)
        if i % every == 0:
            print("  " + _format(reading))

    print()
    print("Synthetic data. Validates the DSP, says nothing about real hardware.")
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    pipeline = SensingPipeline(
        n_subcarriers=args.subcarriers,
        sample_rate_hz=args.rate,
        calibration_s=args.calibration_s,
        subcarrier_selection=args.subcarrier_selection,
    )
    print(f"listening on {args.host}:{args.port}, assuming {args.rate:g} Hz")
    print(f"keep the space EMPTY for the first {args.calibration_s:g}s")

    every = max(int(args.rate), 1)
    filler = SequenceGapFiller(max_gap=int(args.max_gap_s * args.rate))
    count = 0
    try:
        for esp_frame in listen_udp(port=args.port, host=args.host):
            for frame in filler.push(esp_frame.sequence, esp_frame.to_csi_frame(args.rate)):
                reading = pipeline.process(frame)
                if count % every == 0:
                    print(_format(reading), flush=True)
                count += 1
    except KeyboardInterrupt:
        print(
            f"\nstopped after {count} frames ({filler.filled} interpolated across lost packets, "
            f"{filler.discontinuities} discontinuities, {filler.duplicates} duplicates dropped)"
        )
    return 0


def _node_counts(nodes: dict[int, int]) -> str:
    return " ".join(f"{n}:{c}" for n, c in sorted(nodes.items())) or "-"


def cmd_record(args: argparse.Namespace) -> int:
    # Ctrl-C and SIGTERM (systemd stop, kill) set an event the recorder polls,
    # so it finishes the file and returns its stats instead of unwinding
    # mid-write. A second Ctrl-C interrupts for real.
    stop = threading.Event()

    def on_signal(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, on_signal)
        except ValueError:  # not the main thread; rely on KeyboardInterrupt
            pass

    last = {"packets": 0, "elapsed": 0.0}

    def on_ready(addr: tuple[str, int]) -> None:
        print(f"recording UDP {addr[0]}:{addr[1]} -> {args.out}", flush=True)
        print("Ctrl-C to stop; progress every 5 s (packets/s over the last interval)")

    def on_progress(st: RecordStats) -> None:
        dt = st.elapsed_s - last["elapsed"]
        rate = (st.packets - last["packets"]) / dt if dt > 0 else 0.0
        last.update(packets=st.packets, elapsed=st.elapsed_s)
        drops = "" if not st.kernel_drops else f"  KERNEL DROPS {st.kernel_drops}"
        print(
            f"  {st.elapsed_s:7.0f}s  {st.packets:8d} packets  {rate:7.1f}/s  "
            f"malformed {st.malformed}  nodes {_node_counts(st.nodes)}{drops}",
            flush=True,
        )

    try:
        st = record_udp(
            args.out,
            port=args.port,
            host=args.host,
            duration_s=args.duration_s,
            stop_event=stop,
            on_ready=on_ready,
            on_progress=on_progress,
            overwrite=args.overwrite,
        )
    except FileExistsError:
        print(f"{args.out} exists; pass --overwrite to replace it", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\ninterrupted twice; {args.out} was closed cleanly, no summary")
        return 130
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"UDP port {args.port} is already bound "
                f"(is `wifisense listen` or another recorder running?)",
                file=sys.stderr,
            )
            return 1
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    print(
        f"\nstopped: {st.packets} packets ({st.payload_bytes} payload bytes) "
        f"in {st.elapsed_s:.1f}s -> {st.path}"
    )
    print(f"  valid frames per node: {_node_counts(st.nodes)}   malformed: {st.malformed}")
    if st.kernel_drops is None:
        print("  kernel socket drops: unknown on this platform")
    elif st.kernel_drops:
        print(
            f"  kernel socket drops: {st.kernel_drops}  <- the GATEWAY lost these, "
            f"not the radio; loss in `stats` is overstated by this much"
        )
    else:
        print("  kernel socket drops: 0")
    print(f"  next: python3 -m wifisense stats {st.path}")
    return 0


def _ms(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.1f}"


def _read_summary(read: ReadStats) -> str:
    parts = [f"{read.packets} UDP datagrams", f"{read.malformed} malformed (not raw CSI)"]
    if read.skipped:
        parts.append(
            f"{read.skipped} records skipped (non-IPv4 {read.not_ipv4}, "
            f"non-UDP {read.not_udp}, fragments {read.fragments}, "
            f"other port {read.other_port}, snapped {read.snapped})"
        )
    tail = f"YES, {read.truncated_bytes} bytes" if read.truncated else "no"
    parts.append(f"truncated tail: {tail}")
    return ", ".join(parts)


def cmd_stats(args: argparse.Namespace) -> int:
    read = ReadStats()
    links = link_stats(args.file, port=args.port, stats=read)
    print(f"per-node delivery measured at the gateway: {args.file}")
    print(
        "  rate = received packets / time span of receive timestamps; "
        "loss = raw-CSI sequence gaps;"
    )
    print(
        "  resets = sequence jumped backwards (node reboot), not counted as loss; "
        "dt = inter-arrival time"
    )
    print()
    header = (
        f"{'node':>4} {'packets':>8} {'span_s':>8} {'rate_Hz':>8} {'lost':>6} "
        f"{'loss_%':>6} {'resets':>6} {'dt_med_ms':>9} {'dt_p95_ms':>9} "
        f"{'dt_max_ms':>9} {'subc':>5}"
    )
    print(header)
    print("-" * len(header))
    for ls in links.values():
        print(
            f"{ls.node_id:>4} {ls.packets:>8} {ls.duration_s:>8.1f} "
            f"{ls.rate_hz:>8.2f} {ls.lost:>6} {ls.loss_pct:>6.2f} {ls.resets:>6} "
            f"{_ms(ls.dt_median_ms):>9} {_ms(ls.dt_p95_ms):>9} "
            f"{_ms(ls.dt_max_ms):>9} {'/'.join(map(str, ls.subcarriers)):>5}"
        )
    if not links:
        print("  (no valid raw-CSI frames)")
    print()
    for ls in links.values():
        if ls.duplicates or ls.reordered:
            print(
                f"node {ls.node_id}: {ls.duplicates} duplicate and "
                f"{ls.reordered} reordered packets"
            )
        if len(ls.sources) > 1:
            print(
                f"WARNING node {ls.node_id} arrived from {len(ls.sources)} addresses "
                f"({', '.join(ls.sources)}): two boards sharing a node_id?"
            )
    # PROTOCOL_S5 §1.1: how often the receiver changed its gain, from the
    # raw header (spec/wire.md §1). On an idle node in an empty room this
    # is the fraction to report; above ~0.1% presence is at risk.
    for ls in links.values():
        if not ls.gain_frames:
            print(
                f"node {ls.node_id} gain: not in this recording "
                "(header bytes 18-19 are zero: firmware without gain logging, or not an ESP32-S3)"
            )
            continue
        pairs = ls.packets - 1
        pct = 100.0 * ls.agc_moves / pairs if pairs else 0.0
        print(
            f"node {ls.node_id} gain: agc {ls.agc_min}-{ls.agc_max}, moved on "
            f"{ls.agc_moves} of {pairs} frame pairs ({pct:.3f}%); fft moved on {ls.fft_moves}"
        )
    print(f"file: {_read_summary(read)}")
    return 0


def _pick_rate(ls: LinkStats, requested: str) -> float:
    if requested != "auto":
        rate = float(requested)
        if rate <= 0:
            raise ValueError(f"--rate must be positive, got {requested}")
        print(f"rate: {rate:g} Hz (from --rate; measured median would be "
              f"{_hz(ls.dt_median_ms)})")
        return rate
    if math.isnan(ls.dt_median_ms) or ls.dt_median_ms <= 0:
        raise ValueError(
            f"node {ls.node_id}: cannot measure a rate from {ls.packets} packet(s); "
            f"pass --rate"
        )
    rate = 1000.0 / ls.dt_median_ms
    print(
        f"rate: {rate:.3f} Hz measured (1 / median inter-arrival "
        f"{ls.dt_median_ms:.2f} ms; mean over the recording {ls.rate_hz:.2f} Hz, "
        f"loss {ls.loss_pct:.2f}%)"
    )
    return rate


def _hz(dt_ms: float) -> str:
    return "-" if math.isnan(dt_ms) or dt_ms <= 0 else f"{1000.0 / dt_ms:.3f} Hz"


def _replay_send(args: argparse.Namespace) -> int:
    host, sep, port = args.send.rpartition(":")
    if not sep or not host or not port.isdigit():
        print(f"--send expects HOST:PORT, got {args.send!r}", file=sys.stderr)
        return 2
    read = ReadStats()
    pace = "as fast as possible" if args.speed == 0 else f"at {args.speed:g}x recorded timing"
    print(f"re-sending {args.file} to {host}:{port} {pace}", flush=True)
    try:
        sent = replay_udp(
            args.file, host, int(port), speed=args.speed, filter_port=args.port, stats=read
        )
    except KeyboardInterrupt:
        print(f"\ninterrupted after {read.packets} datagrams")
        return 130
    print(f"sent {sent} datagrams (malformed ones included, as recorded)")
    if read.truncated:
        print(f"file ended in a truncated record ({read.truncated_bytes} bytes ignored)")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    if args.speed < 0:
        print("--speed must be >= 0", file=sys.stderr)
        return 2
    if args.send:
        return _replay_send(args)

    links = link_stats(args.file, port=args.port)
    if not links:
        print(f"{args.file}: no valid raw-CSI frames", file=sys.stderr)
        return 1
    if args.node is None:
        # The pipeline is single-link: take the busiest node, lowest id on ties.
        node = max(sorted(links), key=lambda n: links[n].packets)
        others = ", ".join(f"{n}:{links[n].packets}" for n in links if n != node)
        print(
            f"node {node} ({links[node].packets} frames; "
            f"others: {others or 'none'}; choose with --node)"
        )
    elif args.node in links:
        node = args.node
        print(f"node {node} ({links[node].packets} frames)")
    else:
        print(
            f"node {args.node} not in {args.file}; nodes present: "
            f"{', '.join(map(str, links))}",
            file=sys.stderr,
        )
        return 1

    try:
        rate = _pick_rate(links[node], args.rate)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    n_subcarriers = args.subcarriers or links[node].subcarriers[0]
    if len(links[node].subcarriers) > 1:
        print(
            f"note: node {node} changed subcarrier count mid-recording "
            f"{links[node].subcarriers}; the pipeline uses {n_subcarriers}"
        )
    pipeline = SensingPipeline(
        n_subcarriers=n_subcarriers,
        sample_rate_hz=rate,
        calibration_s=args.calibration_s,
        subcarrier_selection=args.subcarrier_selection,
    )
    print(
        f"subcarriers: {n_subcarriers}; calibrating on the first "
        f"{args.calibration_s:g}s after filter warmup (the space must have been EMPTY then)"
    )
    print("rec = seconds into the recording; t = pipeline time (frames / rate)")

    read = ReadStats()
    filler = SequenceGapFiller(max_gap=int(args.max_gap_s * rate))
    t0 = None
    next_print = 0.0
    count = 0
    for t_s, esp_frame in iter_frames(args.file, port=args.port, stats=read):
        if esp_frame.node_id != node:
            continue
        for frame in filler.push(esp_frame.sequence, esp_frame.to_csi_frame(rate)):
            reading = pipeline.process(frame)
        count += 1
        if t0 is None:
            t0 = t_s
        rel = t_s - t0
        if rel >= next_print:
            print(f"rec={rel:7.1f}s  " + _format(reading))
            next_print = math.floor(rel) + 1.0

    ls = links[node]
    print()
    print(
        f"replayed {count} frames from node {node} spanning {ls.duration_s:.1f}s; "
        f"{ls.lost} lost and {ls.resets} resets by sequence; "
        f"{filler.filled} frames interpolated across gaps <= {args.max_gap_s:g}s, "
        f"{filler.discontinuities} longer discontinuities left as-is"
    )
    print(f"file: {_read_summary(read)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wifisense",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the pipeline on synthetic CSI")
    demo.add_argument("--breathing-bpm", type=float, default=15.0)
    demo.add_argument("--heart-bpm", type=float, default=68.0)
    demo.add_argument("--rate", type=float, default=20.0)
    demo.add_argument("--calibration-s", type=float, default=20.0)
    demo.add_argument("--occupied-s", type=float, default=150.0)
    demo.add_argument("--empty-s", type=float, default=40.0)
    demo.set_defaults(func=cmd_demo)

    listen = sub.add_parser("listen", help="consume live CSI from ESP32 nodes over UDP")
    listen.add_argument("--port", type=int, default=5566)
    listen.add_argument("--host", default="0.0.0.0",
                        help="address to receive on. All interfaces by default, because the nodes are on the network; pass the address of the interface the nodes reach to narrow it. Datagrams are not authenticated (see README, \"Operating requirements\")")
    listen.add_argument("--rate", type=float, default=20.0, help="CSI frames/sec")
    listen.add_argument("--subcarriers", type=int, default=56)
    listen.add_argument("--calibration-s", type=float, default=30.0)
    listen.add_argument("--max-gap-s", type=float, default=1.0,
                        help="interpolate across lost packets up to this long (by raw-CSI sequence)")
    listen.add_argument("--subcarrier-selection", choices=("variance", "respirfi"), default="variance",
                        help="how subcarriers are combined for vitals (see evaluate.py --suite nulls)")
    listen.set_defaults(func=cmd_listen)

    record = sub.add_parser(
        "record", help="record raw UDP datagrams (malformed ones too) to a pcap file"
    )
    record.add_argument("--port", type=int, default=5566)
    record.add_argument("--host", default="0.0.0.0",
                        help="address to receive on. All interfaces by default, because the nodes are on the network; pass the address of the interface the nodes reach to narrow it. Datagrams are not authenticated (see README, \"Operating requirements\")")
    record.add_argument("--out", required=True, help="pcap file to create")
    record.add_argument("--duration-s", type=float, default=None)
    record.add_argument(
        "--overwrite", action="store_true", help="replace --out if it already exists"
    )
    record.set_defaults(func=cmd_record)

    replay = sub.add_parser(
        "replay",
        help="run the pipeline over a recording offline, or re-send it over UDP",
    )
    replay.add_argument("file", help="pcap from `record` or `tcpdump -w`")
    replay.add_argument(
        "--port", type=int, default=None,
        help="only datagrams originally sent to this UDP port (default: all)",
    )
    replay.add_argument("--node", type=int, default=None,
                        help="node_id to process (default: the most frequent)")
    replay.add_argument("--rate", default="auto",
                        help="CSI frames/sec, or 'auto' = 1 / median inter-arrival")
    replay.add_argument("--subcarriers", type=int, default=None,
                        help="default: as recorded")
    replay.add_argument("--calibration-s", type=float, default=30.0)
    replay.add_argument("--max-gap-s", type=float, default=1.0,
                        help="interpolate across lost packets up to this long (by raw-CSI sequence)")
    replay.add_argument("--subcarrier-selection", choices=("variance", "respirfi"), default="variance",
                        help="how subcarriers are combined for vitals (see evaluate.py --suite nulls)")
    replay.add_argument("--send", metavar="HOST:PORT", default=None,
                        help="re-send the datagrams over UDP instead of processing")
    replay.add_argument("--speed", type=float, default=1.0,
                        help="with --send: timing multiplier, 0 = as fast as possible")
    replay.set_defaults(func=cmd_replay)

    stats = sub.add_parser("stats", help="per-node rate, loss and jitter of a recording")
    stats.add_argument("file")
    stats.add_argument("--port", type=int, default=None,
                       help="only datagrams sent to this UDP port (default: all)")
    stats.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
