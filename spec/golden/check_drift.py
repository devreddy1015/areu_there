#!/usr/bin/env python3
"""Regenerate the golden vectors and check they still say the same thing.

WHY THIS IS NOT `git diff`
--------------------------
It was, for one push. `git diff --exit-code -- spec/golden/` is the
obvious check and it fails on a clean checkout: regenerating on Ubuntu
with CPython 3.12 produces `sim_geometric__amplitudes.f64`,
`__phases.f64` and `__sensitivity.f64` byte-different from the committed
files, which were generated on Arch with CPython 3.14. Same length,
different bytes. The files that did *not* move are the ones that come
out of the Mersenne Twister (`scatterer_x`, `scatterer_y`,
`scatterer_rcs`), which is bit-reproducible across versions and
platforms by specification.

What moved is everything downstream of `math.sin`, `math.cos` and
friends. Those are thin wrappers over the platform's libm, and libm is
not required to be correctly rounded -- different implementations, and
different versions of the same implementation, are allowed to differ in
the last place or two. Nothing is wrong; two machines simply do not
agree bit-for-bit on a long chain of transcendentals.

So byte-equality was never the contract, and asking for it was a
mistake in the check rather than a fact about the vectors. ADR-002's
contract is that the ports reproduce the reference within a stated
tolerance: 1e-9 relative for Rust, 1e-4 for C++. A reference that wobbles
in its sixteenth significant figure between machines satisfies that
completely.

This script therefore asks the question the contract actually asks: has
anything moved by an amount that *matters*?

THE TOLERANCE, AND WHY IT IS NOT A LOOSENING
--------------------------------------------
`DRIFT_TOL = 1e-11` relative.

Two orders of magnitude tighter than the 1e-9 the Rust port is held to,
so nothing this check tolerates can be large enough to affect the
contract it exists to protect. A DSP change big enough to matter is
nowhere near this small: perturbing the Goertzel coefficient by one part
in 1e9 -- itself an absurdly small edit, made as a test of this gate --
moved 26 vector files far past this bound.

This is not "loosen the tolerance until it passes", which this project
does not do. There was no tolerance before: there was a byte comparison
that encoded a requirement nobody had. The bound here is set by what the
check must catch, and the observed drift is printed on every run so the
margin between the two stays visible rather than assumed.

MEASURED AGAINST WHAT
---------------------
For an array, against the array's own scale (its largest magnitude), not
each element's. The first version divided every difference by the
element itself, and failed on Ubuntu 26.04: `extractors__input.f64[761]`
is 1.05e-5, a unit-scale test signal caught near a zero crossing, and it
moved by 6.8e-16 -- three ULPs of 1.0 -- which divided by 1.05e-5 reads as
6.4e-11. SAANS_PLAN.md §4 says it in so many words: "a relative tolerance
is meaningless near 0". Against the array's scale that same value moved by
7e-16, and the negative control below still fails by orders of
magnitude. The tolerance is unchanged; the metric is now the one the spec
mandates. Scalars in .kv and .json have no array to scale by and are still
compared each to itself.

USAGE
-----
    python3 spec/golden/check_drift.py

Exits 0 if every value agrees within tolerance, 1 otherwise. The working
tree is left exactly as it was found, pass or fail: the committed
vectors are restored before exit, so a failing CI job does not also
leave a dirty checkout behind.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# See the module docstring. Two orders tighter than ADR-002's Rust bound.
DRIFT_TOL = 1e-11


def rel(a: float, b: float) -> float:
    """Relative difference, falling back to absolute near zero.

    A relative tolerance is meaningless at zero -- the same reason
    SAANS_PLAN.md §4 compares confidences absolutely.
    """
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b))
    return abs(a - b) if scale < 1e-300 else abs(a - b) / scale


def read_f64(path: str) -> list[float]:
    with open(path, "rb") as fh:
        raw = fh.read()
    return list(struct.unpack(f"<{len(raw) // 8}d", raw))


def numeric(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def compare_f64(name: str, old: str, new: str, bad: list, worst: dict) -> None:
    a, b = read_f64(old), read_f64(new)
    if len(a) != len(b):
        bad.append(f"{name}: length {len(a)} -> {len(b)}")
        return
    finite = [abs(v) for v in a + b if v == v and abs(v) != float("inf")]
    scale = max(finite) if finite else 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y or (x != x and y != y):
            continue
        if (x != x) or (y != y) or abs(x) == float("inf") or abs(y) == float("inf"):
            # A NaN or an infinity on one side only. Without this branch the
            # difference is NaN, NaN compares false against the tolerance,
            # and the value would pass.
            bad.append(f"{name}[{i}]: {x!r} -> {y!r}")
            continue
        d = abs(x - y) / scale if scale > 1e-300 else abs(x - y)
        if d > worst.get(name, 0.0):
            worst[name] = d
        if d > DRIFT_TOL:
            bad.append(f"{name}[{i}]: {x!r} -> {y!r}  (relative {d:.3e})")


def compare_kv(name: str, old: str, new: str, bad: list, worst: dict) -> None:
    def load(p):
        out = {}
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    k, _, v = line.partition(" ")
                    out[k] = v.strip()
        return out

    a, b = load(old), load(new)
    if a.keys() != b.keys():
        bad.append(f"{name}: keys changed ({sorted(set(a) ^ set(b))})")
        return
    for k in a:
        x, y = numeric(a[k]), numeric(b[k])
        if x is None or y is None:
            if a[k] != b[k]:
                bad.append(f"{name}:{k}: {a[k]!r} -> {b[k]!r}")
            continue
        d = rel(x, y)
        if d > worst.get(name, 0.0):
            worst[name] = d
        if d > DRIFT_TOL:
            bad.append(f"{name}:{k}: {x!r} -> {y!r}  (relative {d:.3e})")


def compare_json(name: str, old: str, new: str, bad: list, worst: dict) -> None:
    with open(old) as fh:
        a = json.load(fh)
    with open(new) as fh:
        b = json.load(fh)

    def walk(path, x, y):
        if isinstance(x, dict) and isinstance(y, dict):
            if x.keys() != y.keys():
                bad.append(f"{name}{path}: keys changed")
                return
            for k in x:
                walk(f"{path}.{k}", x[k], y[k])
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                bad.append(f"{name}{path}: length {len(x)} -> {len(y)}")
                return
            for i, (p, q) in enumerate(zip(x, y)):
                walk(f"{path}[{i}]", p, q)
        elif isinstance(x, bool) or isinstance(y, bool):
            if x != y:
                bad.append(f"{name}{path}: {x!r} -> {y!r}")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            d = rel(float(x), float(y))
            if d > worst.get(name, 0.0):
                worst[name] = d
            if d > DRIFT_TOL:
                bad.append(f"{name}{path}: {x!r} -> {y!r}  (relative {d:.3e})")
        elif x != y:
            bad.append(f"{name}{path}: {x!r} -> {y!r}")

    walk("", a, b)


def main() -> int:
    names = sorted(
        f
        for f in os.listdir(HERE)
        if f.endswith((".f64", ".kv", ".json")) and not f.startswith(".")
    )
    if not names:
        print("no golden vectors found", file=sys.stderr)
        return 1

    snapshot = tempfile.mkdtemp(prefix="golden-committed-")
    try:
        for f in names:
            shutil.copy2(os.path.join(HERE, f), os.path.join(snapshot, f))

        print(f"regenerating {len(names)} files ...", flush=True)
        run = subprocess.run(
            [sys.executable, os.path.join(HERE, "generate.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            print(run.stdout)
            print(run.stderr, file=sys.stderr)
            print("generate.py failed", file=sys.stderr)
            return 1

        bad: list[str] = []
        worst: dict[str, float] = {}
        after = set(os.listdir(HERE))
        for f in names:
            if f not in after:
                bad.append(f"{f}: no longer generated")
                continue
            old, new = os.path.join(snapshot, f), os.path.join(HERE, f)
            if f.endswith(".f64"):
                compare_f64(f, old, new, bad, worst)
            elif f.endswith(".kv"):
                compare_kv(f, old, new, bad, worst)
            else:
                compare_json(f, old, new, bad, worst)
        for f in sorted(after - set(names)):
            if f.endswith((".f64", ".kv", ".json")):
                bad.append(f"{f}: newly generated and not committed")
    finally:
        # Restore whatever was committed, pass or fail.
        for f in names:
            src = os.path.join(snapshot, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(HERE, f))
        shutil.rmtree(snapshot, ignore_errors=True)

    moved = {k: v for k, v in worst.items() if v > 0.0}
    if moved:
        top = sorted(moved.items(), key=lambda kv: -kv[1])[:8]
        print(f"\n{len(moved)} of {len(names)} files moved at all. Largest:")
        for f, d in top:
            print(f"  {d:.3e}  {f}")
        print(f"\nworst relative difference: {max(moved.values()):.3e}")
    else:
        print("\nbit-identical: no value moved at all")
    print(f"tolerance:                 {DRIFT_TOL:.3e}")

    if bad:
        print(f"\nDRIFT: {len(bad)} value(s) past tolerance\n", file=sys.stderr)
        for line in bad[:40]:
            print(f"  {line}", file=sys.stderr)
        if len(bad) > 40:
            print(f"  ... and {len(bad) - 40} more", file=sys.stderr)
        print(
            "\nEither the Python reference changed and the vectors were not "
            "regenerated\n(commit them), or the vectors were edited by hand "
            "(do not).",
            file=sys.stderr,
        )
        return 1

    print("\nno drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
