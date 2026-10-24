"""Locate configuration: every spec/locate.md §9 parameter and its validation.

One object for the person pipeline (§4, §5, §10), the RTI it ports (§3)
and the simulated phone (§6, §7). Every default is SIM unless §9 says
otherwise; the three `*_radius_scale` are uncalibrated (§8).

Validation names the parameter it rejects and is NaN-safe: every test is
written as the positive condition (`not (x > 0.0)`), so a NaN is refused
rather than slipping through a comparison that is False for it.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from ..roomsim.numerics import C, FOUR_PI

__all__ = [
    "LocateConfig",
    "LocateError",
    "LINK_SIGNALS",
    "LABELS",
    "PERSON_LABELS",
    "check_person_label",
    "check_roomsim_label",
    "check_sim_label",
    "default_rssi_p0_dbm",
]

# The link signals `LinkMeasurement` can use, by their `LinkSignal.name`.
LINK_SIGNALS = ("rssi_dbm", "total_power_db", "static_suppressed_amplitude")

# §1 source labels.
LABELS = ("SCENE", "SIM", "REAL", "SIM->REAL")

# §1: the labels a person estimate can carry. SCENE labels geometry, never
# an estimate.
PERSON_LABELS = ("SIM", "REAL", "SIM->REAL")

_U64 = 1 << 64


class LocateError(ValueError):
    """A configuration or run Locate refuses; the message names the parameter.

    `param` is the parameter's name (a §9 name, a Room Sim one such as
    `roles`, `nodes`, `quantize`, `tx_period`, `calib_s`, or an input:
    `label`, `reports`, `t`, `phone`, `walls`, `objects`, `cylinders`). No
    scene id and no refused label appears in any message.
    """

    def __init__(self, param: str, msg: str) -> None:
        self.param = param
        super().__init__(f"{param}: {msg}")


# Every label refusal names `label` and never echoes the value (§1).


def check_person_label(label) -> str:
    """A person pipeline fed by a caller: "SIM", "REAL" or "SIM->REAL"
    (§1; SCENE labels geometry, never an estimate), else
    `LocateError('label')`."""
    if not (isinstance(label, str) and label in PERSON_LABELS):
        raise LocateError("label", f"a person estimate is labelled one of {', '.join(PERSON_LABELS)}")
    return label


def check_roomsim_label(label) -> str:
    """The Room Sim pipeline (§10): its input is simulated, so only "SIM",
    else `LocateError('label')`."""
    if not (isinstance(label, str) and label == "SIM"):
        raise LocateError("label", 'Room Sim input is simulated (ADR-008 d16); the only label is "SIM"')
    return label


def check_sim_label(label) -> str:
    """The phone's label: only "SIM" (ADR-008 d16), else `LocateError('label')`."""
    if not (isinstance(label, str) and label == "SIM"):
        raise LocateError("label", 'the phone is simulated only (ADR-008 d16); the only label is "SIM"')
    return label


# §9 groups, for validation.
# A zero `min_plausible_radius_m` or `*_radius_scale` would publish a bare
# point (ADR-008 d5); `carrier_hz` guards `C / carrier_hz`.
_POSITIVE = (
    "rtt_sigma_m", "rssi_shadow_sigma_db", "rssi_shadow_corr_m", "rssi_breakpoint_m",
    "phone_rate_hz", "huber_k", "rssi_far_exponent", "rssi_est_exponent",
    "phone_accel_psd", "phone_init_speed_std", "tick_hz", "calib_s", "window_s",
    "carrier_hz", "min_plausible_radius_m", "wlm_radius_scale", "rti_radius_scale",
    "phone_radius_scale",
)
_NON_NEGATIVE = (
    "rtt_nlos_sigma_scale", "rtt_nlos_bias_m", "rtt_offset_sigma_m", "rtt_persist_sigma_m",
    "rssi_nlos_loss_db", "rssi_device_sigma_db", "rssi_persist_sigma_db", "rssi_jitter_db",
    "phone_max_coast_s", "wlm_gate_z", "wlm_floor",
)
_PROBABILITY = ("rtt_outlier_p", "rtt_fail_p", "rssi_fail_p")
# Every binary64 parameter; the optional ones (None until resolved) are
# checked only when set.
_NUMBERS = (
    "lambda_m", "pitch_m", "alpha", "sigma", "delta_c", "z_threshold", "window_s",
    "calib_s", "tick_hz", "wlm_gate_z", "wlm_floor", "wlm_radius_scale",
    "rti_radius_scale", "phone_radius_scale", "min_plausible_radius_m", "phone_rate_hz",
    "rtt_sigma_m", "rtt_nlos_sigma_scale", "rtt_nlos_bias_m", "rtt_outlier_p",
    "rtt_outlier_lo_m", "rtt_outlier_hi_m", "rtt_offset_sigma_m", "rtt_persist_sigma_m",
    "rtt_fail_p", "rssi_fail_p", "rssi_breakpoint_m", "rssi_far_exponent",
    "rssi_shadow_sigma_db", "rssi_persist_sigma_db", "rssi_shadow_corr_m",
    "rssi_nlos_loss_db", "rssi_device_sigma_db", "rssi_jitter_db", "rssi_est_exponent",
    "rssi_p0_dbm", "phone_assumed_z_m", "huber_k", "phone_accel_psd",
    "phone_init_speed_std", "phone_max_coast_s", "tx_power_dbm", "carrier_hz",
)
_OPTIONAL = ("wlm_floor", "phone_seed", "rssi_p0_dbm", "tx_power_dbm", "carrier_hz")


@dataclass(frozen=True)
class LocateConfig:
    """Every spec/locate.md §9 parameter, with its default.

    `None` means "derived" (unset): `wlm_floor` from `link_signal` (0.25
    for `rssi_dbm`, else 0.0), `phone_seed` from the run's seed,
    `tx_power_dbm` and `carrier_hz` from Room Sim, `rssi_p0_dbm` from those
    two (`rssi_p0_value`, wherever it is used). On a Room Sim run (§10.2) a
    `phone_seed`, `tx_power_dbm` or `carrier_hz` that is set must equal
    Room Sim's or the run is refused naming it; a plain-data phone run
    keeps a set `phone_seed`. The measurement's `sample_rate_hz` is not a
    parameter: it is Room Sim's per-link rate (§4.3).
    """

    # RTI (ported; the Python defaults)
    lambda_m: float = 0.3
    pitch_m: float = 0.5
    alpha: float = 1.0
    sigma: float = 1.0
    delta_c: float = 0.5
    z_threshold: float = 5.0
    # measurement and ticks
    window_s: float = 1.0
    min_windows: int = 10
    link_signal: str = "rssi_dbm"
    calib_s: float = 15.0
    tick_hz: float = 2.0
    # weighted-link-midpoint baseline (§5)
    wlm_gate_z: float = 3.0
    wlm_floor: float | None = None
    # radius (§8)
    wlm_radius_scale: float = 1.0
    rti_radius_scale: float = 1.0
    phone_radius_scale: float = 1.0
    min_plausible_radius_m: float = 0.3
    # phone radio (§6)
    phone_rate_hz: float = 5.0
    phone_seed: int | None = None
    rtt_sigma_m: float = 1.0
    rtt_nlos_sigma_scale: float = 1.5
    rtt_nlos_bias_m: float = 1.0
    rtt_outlier_p: float = 0.05
    rtt_outlier_lo_m: float = 2.0
    rtt_outlier_hi_m: float = 8.0
    rtt_offset_sigma_m: float = 0.5
    rtt_persist_sigma_m: float = 0.5
    rtt_fail_p: float = 0.02
    rssi_fail_p: float = 0.02
    rssi_breakpoint_m: float = 5.0
    rssi_far_exponent: float = 3.5
    rssi_shadow_sigma_db: float = 4.0
    rssi_persist_sigma_db: float = 4.0
    rssi_shadow_corr_m: float = 6.0
    rssi_nlos_loss_db: float = 5.0
    rssi_device_sigma_db: float = 3.0
    rssi_jitter_db: float = 1.0
    # phone estimators (§7)
    rssi_est_exponent: float = 2.0
    rssi_p0_dbm: float | None = None
    phone_assumed_z_m: float = 1.0
    huber_k: float = 1.345
    phone_accel_psd: float = 1.0
    phone_init_speed_std: float = 1.0
    phone_max_coast_s: float = 3.0
    # Room Sim's
    tx_power_dbm: float | None = None
    carrier_hz: float | None = None

    # -- validation ---------------------------------------------------------

    def validate(self) -> "LocateConfig":
        """§9 validation; raises `LocateError` naming the parameter.
        Returns `self` so a call can be chained."""

        def fail(param: str, msg: str) -> None:
            raise LocateError(param, msg)

        for p in _NUMBERS:
            v = getattr(self, p)
            if v is None and p in _OPTIONAL:
                continue
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            if ok:
                try:
                    ok = math.isfinite(float(v))
                except OverflowError:
                    ok = False
            if not ok:
                fail(p, f"{v!r} is not a finite number")
        for p in _POSITIVE:
            v = getattr(self, p)
            if v is None and p in _OPTIONAL:
                continue
            if not (v > 0.0):
                fail(p, f"{v!r} must be > 0")
        for p in _NON_NEGATIVE:
            v = getattr(self, p)
            if v is None and p in _OPTIONAL:
                continue
            if not (v >= 0.0):
                fail(p, f"{v!r} must be >= 0")
        for p in _PROBABILITY:
            v = getattr(self, p)
            if not (0.0 <= v <= 1.0):
                fail(p, f"{v!r} is not a probability in 0..1")
        if not (self.rtt_outlier_lo_m <= self.rtt_outlier_hi_m):
            fail("rtt_outlier_lo_m", f"{self.rtt_outlier_lo_m!r} > rtt_outlier_hi_m {self.rtt_outlier_hi_m!r}")
        v = self.min_windows
        if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
            fail("min_windows", f"{v!r} is not an integer >= 1")
        if not (isinstance(self.link_signal, str) and self.link_signal in LINK_SIGNALS):
            fail("link_signal", f"{self.link_signal!r} is not one of {', '.join(LINK_SIGNALS)}")
        v = self.phone_seed
        if v is not None and not (isinstance(v, int) and not isinstance(v, bool) and 0 <= v < _U64):
            fail("phone_seed", f"{v!r} is not an integer in 0..2^64-1")
        return self

    # -- derived values -------------------------------------------------------

    def wlm_floor_value(self) -> float:
        """`wlm_floor`, or its default: 0.25 for `rssi_dbm`, 0.0 otherwise."""
        if self.wlm_floor is not None:
            return self.wlm_floor
        return 0.25 if self.link_signal == "rssi_dbm" else 0.0

    def resolved(self, seed: int, tx_power_dbm: float, carrier_hz: float) -> "LocateConfig":
        """A validated copy with the derived values filled: `phone_seed`
        (`seed` when unset), `tx_power_dbm`, `carrier_hz` (Room Sim's),
        `wlm_floor` and `rssi_p0_dbm` (§9, `rssi_p0_value`). A
        `tx_power_dbm` or `carrier_hz` already set must agree with Room
        Sim's. A set `phone_seed` is kept here (plain-data phone runs choose
        their own; the Rust `Config::resolved` keeps it too); `run_locate`
        refuses one that is not the Room Sim run's seed."""
        self.validate()
        for p, v in (("tx_power_dbm", tx_power_dbm), ("carrier_hz", carrier_hz)):
            mine = getattr(self, p)
            if mine is not None and mine != v:
                raise LocateError(p, f"{mine!r} is not Room Sim's {v!r}")
        out = dataclasses.replace(
            self,
            phone_seed=seed if self.phone_seed is None else self.phone_seed,
            tx_power_dbm=float(tx_power_dbm),
            carrier_hz=float(carrier_hz),
            wlm_floor=self.wlm_floor_value(),
        )
        out = out.validate()
        if out.rssi_p0_dbm is None:
            out = dataclasses.replace(out, rssi_p0_dbm=out.rssi_p0_value())
        return out.validate()

    def rssi_p0_value(self) -> float:
        """`rssi_p0_dbm`, or when it is unset its §9 default
        `default_rssi_p0_dbm(tx_power_dbm, carrier_hz)` (§7.2, the true 1 m
        level); `LocateError('rssi_p0_dbm')` when it is unset and
        `tx_power_dbm` or `carrier_hz` is too."""
        if self.rssi_p0_dbm is not None:
            return float(self.rssi_p0_dbm)
        if self.tx_power_dbm is None or self.carrier_hz is None:
            raise LocateError(
                "rssi_p0_dbm",
                "not set, and tx_power_dbm or carrier_hz is unset to derive it; "
                "use LocateConfig.resolved(...) first",
            )
        return default_rssi_p0_dbm(self.tx_power_dbm, self.carrier_hz)

    def require_resolved(self) -> "LocateConfig":
        """Validate, and refuse a config whose Room Sim values are not set
        (`phone_seed`, `tx_power_dbm`, `carrier_hz`, the first unset is
        named). `rssi_p0_dbm` may stay unset: it is derived where it is
        used (`rssi_p0_value`)."""
        self.validate()
        for p in ("phone_seed", "tx_power_dbm", "carrier_hz"):
            if getattr(self, p) is None:
                raise LocateError(p, "not set; use LocateConfig.resolved(...) first")
        return self


def default_rssi_p0_dbm(tx_power_dbm: float, carrier_hz: float) -> float:
    """§7.2: `tx_power_dbm - 20.0*log10(FOUR_PI / LAMBDA)`, `LAMBDA = C / carrier_hz`
    (the true 1 m level). `carrier_hz > 0` is validated before the division."""
    if not (carrier_hz > 0.0):
        raise LocateError("carrier_hz", f"{carrier_hz!r} must be > 0")
    lam = C / carrier_hz
    return tx_power_dbm - 20.0 * math.log10(FOUR_PI / lam)

