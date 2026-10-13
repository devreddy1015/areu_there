"""wifisense — a minimal, honest WiFi CSI sensing pipeline.

The reference implementation of the Saans sensing stack. Pure stdlib: every stage is simple float arithmetic
that ports directly to C or Rust for on-device use.

    from wifisense import SensingPipeline
    from wifisense.sources import SyntheticCsiSource

    source = SyntheticCsiSource()
    pipeline = SensingPipeline(sample_rate_hz=source.config.sample_rate_hz)
    reading = pipeline.process_all(source.seconds(90))
    print(reading.respiratory_rate.value_bpm)
"""

from .pipeline import SensingPipeline
from .preprocess import StaticSuppressor, corner_to_alpha, fuse
from .presence import PresenceDetector, PresenceState
from .types import CsiFrame, VitalEstimate, VitalReading, VitalStatus
from .vitals import BreathingExtractor, HeartRateExtractor

__version__ = "0.1.0"

__all__ = [
    "SensingPipeline",
    "StaticSuppressor",
    "corner_to_alpha",
    "fuse",
    "PresenceDetector",
    "PresenceState",
    "CsiFrame",
    "VitalEstimate",
    "VitalReading",
    "VitalStatus",
    "BreathingExtractor",
    "HeartRateExtractor",
]
