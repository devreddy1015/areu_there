#!/usr/bin/env python3
"""Three baselines, measured before anything is optimised (issue #43).

    python3 tools/baselines.py            all three, as docs/RESULTS.md has them
    python3 tools/baselines.py ci         CI wall time from GitHub's records
    python3 tools/baselines.py gateway    gateway CPU per link, on this machine
    python3 tools/baselines.py node       node DSP budget: host count, S3 estimate

Every figure is labelled with where it came from:

    MEASURED       read from GitHub's run records (CI wall time)
    MEASURED-HOST  measured on the machine printed at the top, on SYNTHETIC input
    ESTIMATE       a MEASURED-HOST figure carried to other hardware under the
                   assumptions printed beside it; nothing has measured it

The gateway figure is the marginal CPU of the REAL binary: `saans-gateway
--replay --serve` on a SIM pcap (`examples/link_load.rs`), run at two lengths
so that startup cancels, pinned to one core, timed by the kernel's rusage.
The node figure is retired x86-64 user-space instructions per `push` of the
node's own extractor type (`edge/dsp/bench/node_budget.cpp`), built scalar.

Standard library only, like the rest of tools/. Needs gh (authenticated) for
`ci`, cargo for `gateway`, g++ for `node`, and Linux for the pinning and the
instruction counter.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUST = os.path.join(ROOT, "rust")

# One P-core of a hybrid laptop CPU: the E-cores run the same code ~2x slower,
# and a measurement that migrates between them measures the scheduler.
PIN_CPU = 2

CI_RUNS = 10
SLOWEST_STEPS = 8

LINKS = (1, 3, 6, 12)
SHORT_MIN, LONG_MIN = 10, 40
REPEATS = 5
# A row whose long runs spread by more than this is measured once more, and
# flagged `!` if it still does: on a shared laptop that is another process on
# the pinned core's sibling, not the gateway. (The first full run of this
# script, made while a build ran on CPU 3, read 3x high and non-monotonic.)
NOISY = 0.20
# ESTIMATE: a Raspberry Pi 5 core (Cortex-A76, 2.4 GHz) against one Alder Lake
# P-core, from public single-core benchmark scores. Not measured here.
PI5_SLOWER = (2.5, 3.0, 4.0)

S3_HZ = 240e6
FRAME_HZ = 20.0
# ESTIMATE: S3 cycles per x86-64 instruction for scalar float code. The LX7
# issues one instruction a cycle in order; an x86 instruction with a memory
# operand is a load plus an op there, and FP divide and sqrt are multi-
# instruction sequences. Bracketed, not known: issue #45 measures it.
S3_CPI = (1.5, 3.0)
DIRAM_BYTES = 171 * 1024
# The ESP32-S3 sizes, read out of the cross-compiled object's .bss by
# `edge/dsp/bench/run_size_report.sh --target` (node_config.hpp, at
# SaansNodeExtractor). Not re-measured by this script: that needs the pinned
# ESP-IDF container. The host sizes beside them are measured on every run.
S3_BYTES_LINK = 45_408
S3_BYTES_LINK_NOBNR = 16_584

CXXFLAGS = ["-std=c++17", "-O2", "-fno-tree-vectorize", "-ffp-contract=off",
            "-fno-exceptions", "-fno-rtti", "-Wall", "-Wextra", "-Wpedantic",
            "-Wconversion", "-Wdouble-promotion"]


# --------------------------------------------------------------- arithmetic

def median(xs):
    v = sorted(xs)
    if not v:
        raise ValueError("median of nothing")
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def cpu_per_link_s(cpu_short, cpu_long, links, short_s, long_s):
    """CPU seconds per link-second from two replays that differ only in length.

    Whatever both runs pay once (process start, config, the first ticks)
    cancels in the difference.
    """
    if long_s <= short_s:
        raise ValueError("the long replay must be longer than the short one")
    return (cpu_long - cpu_short) / (links * (long_s - short_s))


def parse_kv(line):
    """`NAME k=v k=v ...` -> {k: int | float | str}."""
    out = {}
    for tok in line.split()[1:]:
        k, _, v = tok.partition("=")
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = v
    return out


def s3_core_share(instr_per_frame, frame_hz, cpi):
    return instr_per_frame * frame_hz * cpi / S3_HZ


def s3_ms(instr, cpi):
    return instr * cpi / S3_HZ * 1e3


def links_in_diram(bytes_per_link, arena_bytes):
    return (DIRAM_BYTES - arena_bytes) // bytes_per_link


def _t(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _secs(a, b):
    return int((_t(b) - _t(a)).total_seconds())


def job_seconds(run):
    return {j["name"]: _secs(j["startedAt"], j["completedAt"]) for j in run["jobs"]}


def run_wall_seconds(run):
    return _secs(min(j["startedAt"] for j in run["jobs"]),
                 max(j["completedAt"] for j in run["jobs"]))


def runner_seconds(run):
    return sum(job_seconds(run).values())


def step_seconds(run):
    out = {}
    for j in run["jobs"]:
        for s in j.get("steps", []):
            if s.get("startedAt") and s.get("completedAt"):
                out[(j["name"], s["name"])] = _secs(s["startedAt"], s["completedAt"])
    return out


def mmss(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


# ------------------------------------------------------------------ helpers

def sh(cmd, cwd=ROOT):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
        raise SystemExit(f"`{' '.join(cmd)}` exited {p.returncode}")
    return p.stdout


def first_line(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        return "not found"


def pin():
    os.sched_setaffinity(0, {PIN_CPU})


def machine():
    model = "unknown CPU"
    with open("/proc/cpuinfo") as fh:
        for line in fh:
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    with open("/proc/meminfo") as fh:
        mem_kib = int(fh.readline().split()[1])
    print("MACHINE (every MEASURED-HOST row below ran here)")
    print(f"  cpu       {model}, {os.cpu_count()} logical CPUs; timed runs pinned to "
          f"CPU {PIN_CPU} (a P-core)")
    print(f"  memory    {mem_kib / 2**20:.1f} GiB")
    print(f"  os        {platform.system()} {platform.release()}")
    print(f"  rustc     {first_line(['rustc', '--version'])}")
    print(f"  g++       {first_line(['g++', '--version'])}")
    print("  load      a working laptop, not an idle bench machine: medians of "
          f"{REPEATS} runs, and the spread is printed")


# ----------------------------------------------------------------------- ci

def ci_runs():
    listing = json.loads(sh(["gh", "run", "list", "--workflow", "ci", "--branch", "main",
                             "--status", "success", "--limit", str(CI_RUNS), "--json",
                             "databaseId,headSha,createdAt,displayTitle"]))
    for r in listing:
        r["jobs"] = json.loads(sh(["gh", "run", "view", str(r["databaseId"]),
                                   "--json", "jobs"]))["jobs"]
    return listing


def on_current_workflow(sha, since):
    """True if `sha` already has the last commit that changed ci.yml."""
    p = subprocess.run(["git", "merge-base", "--is-ancestor", since, sha], cwd=ROOT,
                       capture_output=True)
    return {0: True, 1: False}.get(p.returncode)


def ci():
    runs = ci_runs()
    since = sh(["git", "log", "-1", "--format=%H", "--", ".github/workflows/ci.yml"]).strip()
    cur = [r for r in runs if on_current_workflow(r["headSha"], since)]
    print()
    print("1. CI WALL TIME -- MEASURED, GitHub-hosted ubuntu-latest runners")
    print(f"   the last {len(runs)} successful `ci` runs on main; {len(cur)} of them ran the "
          f"current ci.yml (last changed in {since[:7]})")
    print("   `gh run list --workflow ci --branch main --status success --limit "
          f"{CI_RUNS}` and `gh run view <id> --json jobs`")
    print()
    for r in runs:
        mark = "*" if r in cur else " "
        print(f"   {mark} {r['databaseId']}  {r['createdAt'][:10]}  {r['displayTitle'][:52]}")
    print("   (* = current ci.yml)")
    print()
    _ci_table(runs, cur)
    _ci_steps(runs)


def _ci_row(label, vals, cur_vals):
    a = f"{mmss(median(vals)):>7} {mmss(max(vals)):>7}" if vals else f"{'-':>7} {'-':>7}"
    b = (f"{mmss(median(cur_vals)):>7} {mmss(max(cur_vals)):>7}" if cur_vals
         else f"{'-':>7} {'-':>7}")
    print(f"   {label:<46} {len(vals):>2}  {a}   {len(cur_vals):>2}  {b}")


def _ci_table(runs, cur):
    print(f"   {'':<46} {'last ' + str(len(runs)):^19}   {'current ci.yml':^19}")
    print(f"   {'job (m:ss, job start to job end)':<46} {'n':>2}  {'median':>7} {'max':>7}"
          f"   {'n':>2}  {'median':>7} {'max':>7}")
    names = sorted({j for r in runs for j in job_seconds(r)},
                   key=lambda n: -median([job_seconds(r)[n] for r in runs
                                          if n in job_seconds(r)]))
    for n in names:
        _ci_row(n, [job_seconds(r)[n] for r in runs if n in job_seconds(r)],
                [job_seconds(r)[n] for r in cur if n in job_seconds(r)])
    _ci_row("TOTAL wall, first job start to last job end",
            [run_wall_seconds(r) for r in runs], [run_wall_seconds(r) for r in cur])
    _ci_row("TOTAL runner time, sum of jobs",
            [runner_seconds(r) for r in runs], [runner_seconds(r) for r in cur])


def _ci_steps(runs):
    per = {}
    for r in runs:
        for k, s in step_seconds(r).items():
            per.setdefault(k, []).append(s)
    top = sorted(per.items(), key=lambda kv: -median(kv[1]))[:SLOWEST_STEPS]
    print()
    print(f"   the {SLOWEST_STEPS} slowest steps, by median over the runs that have them")
    print(f"   {'job / step':<66} {'n':>2}  {'median':>7} {'max':>7}")
    for (job, step), v in top:
        label = f"{job.split(' (')[0]} / {step}"
        print(f"   {label[:66]:<66} {len(v):>2}  {mmss(median(v)):>7} {mmss(max(v)):>7}")


# ------------------------------------------------------------------ gateway

def _build_gateway():
    sh(["cargo", "build", "--release", "-q", "-p", "saans-gateway", "--bin", "saans-gateway",
        "--example", "link_load"], cwd=RUST)
    target = os.path.join(RUST, "target", "release")
    return os.path.join(target, "saans-gateway"), os.path.join(target, "examples", "link_load")


def _replay(gw, d):
    """One timed replay with publishing on. -> (cpu s, peak RSS KiB, summary)."""
    p = subprocess.Popen([gw, "--config", os.path.join(d, "saans.toml"), "--replay",
                          os.path.join(d, "load.pcap"), "--serve"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         preexec_fn=pin)
    finished = False
    for line in p.stderr:
        if "replay finished" in line:
            finished = True
            break
    p.send_signal(signal.SIGTERM)
    # Not communicate(): it reaps the child, and wait4 is what returns rusage.
    out = p.stdout.read()
    err = p.stderr.read()
    _, status, ru = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    if not finished:
        raise SystemExit(f"saans-gateway stopped before the replay finished:\n{out}{err}")
    return ru.ru_utime + ru.ru_stime, ru.ru_maxrss, out


def _check_summary(out, mode):
    """Every datagram ingested, none refused, none unassigned. -> seconds replayed."""
    pk = re.search(r"pcap: \d+ records, (\d+) UDP datagrams", out)
    ing = re.search(r"ingest: (\d+) FeatureFrames, (\d+) raw CSI -> \d+ computed frames "
                    r"\(\d+ shadowed\), (\d+) decode errors, (\d+) unassigned", out)
    dur = re.search(r"\(([\d.]+) s\)", out)
    if not (pk and ing and dur):
        raise SystemExit("replay summary not understood:\n" + out)
    datagrams = int(pk.group(1))
    taken = int(ing.group(1) if mode == "features" else ing.group(2))
    if taken != datagrams or int(ing.group(3)) or int(ing.group(4)):
        raise SystemExit(f"the replay did not ingest the whole load:\n{out}")
    return float(dur.group(1))


def _measure(gw, gen, tmp, mode, links, minutes):
    d = os.path.join(tmp, f"{mode}-{links}-{minutes}")
    if not os.path.exists(d):
        sh([gen, "--links", str(links), "--minutes", str(minutes), "--mode", mode, "--out", d])
    runs = [_replay(gw, d) for _ in range(REPEATS)]
    secs = _check_summary(runs[0][2], mode)
    cpu = [r[0] for r in runs]
    return {"secs": secs, "cpu": median(cpu), "spread": max(cpu) - min(cpu),
            "rss": median([r[1] for r in runs])}


def noisy(m):
    return m["spread"] > NOISY * m["cpu"]


def _measure_pair(gw, gen, tmp, mode, n):
    """Short and long replays of N links; once more if either is noisy."""
    for _ in range(2):
        s = _measure(gw, gen, tmp, mode, n, SHORT_MIN)
        l = _measure(gw, gen, tmp, mode, n, LONG_MIN)
        if not (noisy(s) or noisy(l)):
            break
    return s, l, noisy(s) or noisy(l)


def load_average():
    return " ".join(f"{x:.1f}" for x in os.getloadavg())


def gateway():
    gw, gen = _build_gateway()
    print()
    print("2. GATEWAY CPU PER LINK -- MEASURED-HOST on SYNTHETIC input; Pi 5 is an ESTIMATE")
    print("   `cd rust && cargo run --release -p saans-gateway --example link_load -- "
          "--links N --minutes M --mode MODE --out DIR`")
    print("   `rust/target/release/saans-gateway --config DIR/saans.toml --replay "
          "DIR/load.pcap --serve`, then SIGTERM")
    print(f"   each N replayed at {SHORT_MIN} and {LONG_MIN} min, {REPEATS} times each; CPU "
          "per link-second is the difference over the extra link-seconds, so startup cancels")
    print("   included: decode, frame checks, routing, fusion, detectors, privacy, events, "
          "the dashboard's per-tick publish")
    print("   excluded: UDP receive, connected dashboard clients, MQTT and webhooks, the "
          "event log and history on disk (replay never writes data_dir)")
    print(f"   load average (1, 5, 15 min) at the start: {load_average()}")
    lo, mid, hi = PI5_SLOWER
    with tempfile.TemporaryDirectory(prefix="saans-baselines-") as tmp:
        for mode, what in (("features", "FeatureFrames at 2 Hz a link: the product"),
                           ("raw", "raw CSI at 20 Hz a link, extractor in the gateway: "
                                   "bench and debug mode")):
            print()
            print(f"   {mode}: {what}")
            print(f"   {'links':>5} {'CPU s, 10 / 40 min':>19} {'spread':>7} "
                  f"{'us CPU per':>11} {'% of one core':>14} {'peak RSS':>9}"
                  f"   {'ESTIMATE % of one Pi 5 core':>27}")
            print(f"   {'':>5} {'(median of ' + str(REPEATS) + ')':>19} {'(long)':>7} "
                  f"{'link-second':>11} {'all N links':>14} {'MiB':>9}"
                  f"   {f'x{mid:g} (x{lo:g} .. x{hi:g})':>27}")
            for n in LINKS:
                s, l, bad = _measure_pair(gw, gen, tmp, mode, n)
                per = cpu_per_link_s(s["cpu"], l["cpu"], n, s["secs"], l["secs"])
                core = per * n * 100
                flag = "!" if bad else " "
                print(f"  {flag}{n:>5} {s['cpu']:>9.3f} / {l['cpu']:<7.3f} {l['spread']:>7.3f} "
                      f"{per * 1e6:>11.1f} {core:>13.2f}% {l['rss'] / 1024:>9.1f}"
                      f"   {core * mid:>9.2f}% ({core * lo:.2f} .. {core * hi:.2f}%)")
    print()
    print("   us per link-second falls as N grows because most of the work is per space and "
          "per 0.5 s tick, not per frame: read the % of one core")
    print(f"   load average at the end: {load_average()}; ! = runs still spread more than "
          f"{NOISY:.0%} after a second try")
    print(f"   Pi 5: ESTIMATE, x{lo:g}..x{hi:g} the CPU time of this P-core (Cortex-A76 at "
          "2.4 GHz, from public single-core scores); not measured")


# --------------------------------------------------------------------- node

def _node_run(tmp):
    exe = os.path.join(tmp, "node_budget")
    sh(["g++", *CXXFLAGS, "-I", os.path.join(ROOT, "edge", "dsp"),
        os.path.join(ROOT, "edge", "dsp", "bench", "node_budget.cpp"), "-o", exe])
    p = subprocess.run([exe, "--runs", str(REPEATS)], capture_output=True, text=True,
                       preexec_fn=pin, check=True)
    lines = p.stdout.splitlines()
    shape = parse_kv(next(x for x in lines if x.startswith("NODE_SHAPE ")))
    rows = [parse_kv(x) for x in lines if x.startswith("NODE ")]
    return shape, rows


def _node_cpu(shape, rows):
    lo, hi = S3_CPI
    print(f"   {'config':<20} {'x86 instr per frame':^31}   {'host us per frame':^25}")
    print(f"   {'':<20} {'mean':>9} {'2 Hz frame':>10} {'worst':>10}   "
          f"{'mean':>7} {'2 Hz':>8} {'worst':>8}")
    for r in rows:
        print(f"   {r['config']:<20} {r['instr_mean']:>9,} {r['instr_emit']:>10,} "
              f"{r['instr_max']:>10,}   {r['mean_us']:>7.1f} {r['emit_us']:>8.1f} "
              f"{r['max_us']:>8.1f}")
    for r in rows:
        share = r["instr_emit"] * r["emitted"] / (r["instr_mean"] * r["frames"]) * 100
        print(f"   {r['config']:<20} {share:.0f}% of the instructions are in the 2 Hz frames")
    print()
    print(f"   ESTIMATE on the S3: cycles = x86 instructions x CPI, CPI {lo:g} .. {hi:g} "
          f"(assumed), at {S3_HZ / 1e6:.0f} MHz")
    print(f"   {'config':<20} {'% of one core':>16} {'2 Hz frame, ms':>16} "
          f"{'worst frame, ms':>16} {'links to fill':>14}")
    print(f"   {'':<20} {'per link':>16} {'(budget 50)':>16} {'(budget 50)':>16} "
          f"{'one core':>14}")
    for r in rows:
        a, b = (s3_core_share(r["instr_mean"], FRAME_HZ, c) * 100 for c in (lo, hi))
        e = (s3_ms(r["instr_emit"], c) for c in (lo, hi))
        w = (s3_ms(r["instr_max"], c) for c in (lo, hi))
        fill = f"{int(100 // b)} .. {int(100 // a)}"
        print(f"   {r['config']:<20} {f'{a:.1f} .. {b:.1f}%':>16} "
              f"{'{:.0f} .. {:.0f}'.format(*e):>16} {'{:.0f} .. {:.0f}'.format(*w):>16} "
              f"{fill:>14}")


def _node_diram(shape):
    arena = shape["arena_bytes"]
    print(f"   {'extractor type':<36} {'host B/link':>11} {'S3 B/link':>10} "
          f"{'links in 171 KB':>16}")
    for name, host, s3 in (("SaansNodeExtractor (best_bnr on)", shape["bytes_link"],
                            S3_BYTES_LINK),
                           ("SaansNodeExtractorNoBnr", shape["bytes_link_nobnr"],
                            S3_BYTES_LINK_NOBNR)):
        print(f"   {name:<36} {host:>11,} {s3:>10,} {links_in_diram(s3, arena):>16}")
    print(f"   plus one {arena:,} B spectral arena per node; S3 bytes are the cross-compiled "
          ".bss (run_size_report.sh --target, not re-run here)")


def node():
    with tempfile.TemporaryDirectory(prefix="saans-node-") as tmp:
        shape, rows = _node_run(tmp)
    print()
    print("3. NODE DSP BUDGET -- MEASURED-HOST counts on SYNTHETIC input; S3 figures are "
          "ESTIMATES")
    print(f"   `g++ {' '.join(CXXFLAGS)} -I edge/dsp edge/dsp/bench/node_budget.cpp`, run "
          f"with --runs {REPEATS}")
    print(f"   node::SaansNodeExtractor, {shape['subcarriers']} subcarriers at "
          f"{shape['rate_hz']} Hz, FeatureFrames at 2 Hz; {shape['measured_s']} s of a still "
          "breather after a 120 s empty room")
    occ = rows[-1]
    print(f"   workload check: {occ['occupied']}/{occ['emitted']} frames occupied, "
          f"{occ['breathing']} breathing-valid, {occ['heart']} heart-valid "
          f"(vitals+heart rows)")
    print()
    if shape["counting"]:
        _node_cpu(shape, rows)
    else:
        print("   no instruction counter (perf_event_paranoid > 2?): S3 estimate skipped")
    print()
    _node_diram(shape)
    print()
    print("   TODO(#45): replace every S3 ESTIMATE here with the bench session's on-board "
          "cycle counts and the firmware's boot-time sizeof log")


def main(argv):
    parts = argv[1:] or ["ci", "gateway", "node"]
    unknown = set(parts) - {"ci", "gateway", "node"}
    if unknown:
        raise SystemExit(f"unknown part(s) {sorted(unknown)}; choose ci, gateway, node")
    machine()
    for part in parts:
        {"ci": ci, "gateway": gateway, "node": node}[part]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
