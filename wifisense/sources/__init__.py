"""CSI sources: synthetic generation now, real hardware later."""

from .synthetic import SyntheticCsiSource, SyntheticConfig
from .esp32_udp import (
    CSI_MAGIC,
    CSI_HEADER_SIZE,
    Esp32Frame,
    parse_esp32_frame,
    listen_udp,
)
from .recorder import (
    RecordedPacket,
    ReadStats,
    PcapWriter,
    read_pcap,
    iter_frames,
    RecordStats,
    record_udp,
    replay_udp,
    LinkStats,
    link_stats,
)

__all__ = [
    "SyntheticCsiSource",
    "SyntheticConfig",
    "CSI_MAGIC",
    "CSI_HEADER_SIZE",
    "Esp32Frame",
    "parse_esp32_frame",
    "listen_udp",
    "RecordedPacket",
    "ReadStats",
    "PcapWriter",
    "read_pcap",
    "iter_frames",
    "RecordStats",
    "record_udp",
    "replay_udp",
    "LinkStats",
    "link_stats",
]
