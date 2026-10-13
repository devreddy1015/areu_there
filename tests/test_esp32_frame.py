import unittest

from wifisense.sources.esp32_udp import (
    CSI_HEADER_SIZE,
    CSI_MAGIC,
    build_esp32_frame,
    parse_esp32_frame,
)


class TestEsp32Frame(unittest.TestCase):
    def test_roundtrip(self):
        iq = [(1, 2), (-3, 4), (127, -128), (0, 0)]
        decoded = parse_esp32_frame(
            build_esp32_frame(iq, node_id=7, sequence=99, freq_mhz=2437, rssi_dbm=-45)
        )
        self.assertEqual(decoded.iq, iq)
        self.assertEqual(decoded.node_id, 7)
        self.assertEqual(decoded.sequence, 99)
        self.assertEqual(decoded.freq_mhz, 2437)
        self.assertEqual(decoded.rssi_dbm, -45)
        self.assertEqual(decoded.n_subcarriers, 4)

    def test_signed_values_survive(self):
        """int8 range must round-trip; an unsigned read turns -1 into 255."""
        decoded = parse_esp32_frame(build_esp32_frame([(-128, -1), (127, -100)]))
        self.assertEqual(decoded.iq, [(-128, -1), (127, -100)])

    def test_imaginary_component_comes_first(self):
        """ESP-IDF emits [imag, real] pairs. Reversing them leaves amplitude
        correct and mirrors every phase, which is invisible to amplitude-only
        tests and then silently breaks phase-based work."""
        decoded = parse_esp32_frame(build_esp32_frame([(3, 4)]))
        imag, real = decoded.iq[0]
        self.assertEqual((imag, real), (3, 4))

        csi = decoded.to_csi_frame(sample_rate_hz=20.0)
        self.assertAlmostEqual(csi.amplitudes[0], 5.0)          # hypot(3, 4)
        self.assertAlmostEqual(csi.phases[0], 0.6435011, places=6)  # atan2(3, 4)

    def test_header_size_constant(self):
        self.assertEqual(len(build_esp32_frame([])), CSI_HEADER_SIZE)

    def test_rejects_bad_magic(self):
        data = bytearray(build_esp32_frame([(1, 2)]))
        data[0] ^= 0xFF
        with self.assertRaises(ValueError) as ctx:
            parse_esp32_frame(bytes(data))
        self.assertIn("magic", str(ctx.exception).lower())

    def test_rejects_short_header(self):
        with self.assertRaises(ValueError):
            parse_esp32_frame(b"\x01\x02\x03")

    def test_rejects_truncated_payload(self):
        full = build_esp32_frame([(1, 2), (3, 4), (5, 6)])
        with self.assertRaises(ValueError) as ctx:
            parse_esp32_frame(full[:-2])
        self.assertIn("truncated", str(ctx.exception).lower())

    def test_magic_value_matches_firmware(self):
        self.assertEqual(CSI_MAGIC, 0xC5110001)


if __name__ == "__main__":
    unittest.main()
