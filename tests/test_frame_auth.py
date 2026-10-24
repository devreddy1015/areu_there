"""FeatureFrame v2: signed frames and the replay floor (ADR-012, spec/wire.md §2b)."""

import hashlib
import hmac
import unittest

from wifisense.features import (
    FEATURE_FRAME_SIZE,
    FEATURE_MAGIC,
    FEATURE_VERSION_SIGNED,
    NODE_KEY_LABEL,
    SIGNED_FRAME_SIZE,
    TAG_SIZE,
    FeatureFrame,
    FrameAuthError,
    ReplayGuard,
    decode_feature_frame,
    decode_signed,
    derive_node_key,
    encode_signed,
)

SITE = bytes(range(32))
OTHER_SITE = bytes(range(1, 33))


def frame(**kw) -> FeatureFrame:
    base = dict(node_id=12, tx_id=3, flags=0b0011_0111, seq=41, uptime_ms=123456789,
                csi_rate_chz=1987, loss_permille=31, motion_energy=0.15625,
                presence_threshold=0.05, gross_motion=0.0625, breath_rms=0.2,
                breathing_bpm=15.25, breathing_conf=0.93, heart_bpm=68.5, heart_conf=0.41,
                best_bnr=42.0, calib_age_s=3600, rssi_dbm=-61, noise_floor_dbm=-95,
                n_subcarriers=56)
    base.update(kw)
    return FeatureFrame(**base)


def resolver(keys: dict[int, bytes]):
    """The gateway's lookup: key_id -> site key -> this node's key."""
    def node_key_for(key_id: int, node_id: int) -> bytes | None:
        site = keys.get(key_id)
        return None if site is None else derive_node_key(site, node_id)
    return node_key_for


class TestPrimitive(unittest.TestCase):
    def test_hmac_sha256_is_the_rfc_4231_function(self):
        # RFC 4231 test case 2. The C++ port implements SHA-256 itself and
        # is held to the same vectors; this pins what "HMAC-SHA256" means.
        tag = hmac.new(b"Jefe", b"what do ya want for nothing?", hashlib.sha256).hexdigest()
        self.assertEqual(tag, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843")

    def test_node_keys_are_derived_per_node_and_per_site(self):
        k12 = derive_node_key(SITE, 12)
        self.assertEqual(len(k12), 32)
        self.assertEqual(k12, hmac.new(SITE, NODE_KEY_LABEL + bytes([12]), hashlib.sha256).digest())
        self.assertNotEqual(k12, derive_node_key(SITE, 13))
        self.assertNotEqual(k12, derive_node_key(OTHER_SITE, 12))

    def test_site_key_and_node_id_are_checked(self):
        with self.assertRaises(ValueError):
            derive_node_key(b"short", 1)
        with self.assertRaises(ValueError):
            derive_node_key(SITE, 256)


class TestSignedFrame(unittest.TestCase):
    def setUp(self):
        self.key = derive_node_key(SITE, 12)
        self.f = frame()
        self.data = encode_signed(self.f, node_key=self.key, key_id=3, boot=7)

    def test_layout(self):
        d = self.data
        self.assertEqual(len(d), SIGNED_FRAME_SIZE)
        self.assertEqual(SIGNED_FRAME_SIZE, 84)
        self.assertEqual(int.from_bytes(d[0:4], "little"), FEATURE_MAGIC)
        self.assertEqual(d[4], FEATURE_VERSION_SIGNED)
        self.assertEqual(d[63], 3, "key_id replaces v1's reserved byte")
        self.assertEqual(int.from_bytes(d[64:68], "little"), 7)
        expected = hmac.new(self.key, d[:68], hashlib.sha256).digest()[:TAG_SIZE]
        self.assertEqual(d[68:], expected)

    def test_every_v1_field_keeps_its_offset(self):
        v1 = self.f.encode()
        self.assertEqual(self.data[:4], v1[:4])
        self.assertEqual(self.data[5:63], v1[5:63])

    def test_round_trip(self):
        s = decode_signed(self.data, resolver({3: SITE}))
        self.assertEqual((s.key_id, s.boot), (3, 7))
        # Exactly what v1 round-trips to: the f32 fields are rounded on the
        # wire in both versions.
        want = decode_feature_frame(self.f.encode()).as_dict()
        want["version"] = FEATURE_VERSION_SIGNED
        self.assertEqual(s.frame.as_dict(), want)

    def test_any_changed_byte_is_refused(self):
        lookup = resolver({3: SITE})
        for i in range(SIGNED_FRAME_SIZE):
            bad = bytearray(self.data)
            bad[i] ^= 0x01
            with self.assertRaises(FrameAuthError, msg=f"byte {i}") as cm:
                decode_signed(bytes(bad), lookup)
            # Header bytes fail before the tag is checked; everything else
            # is the tag's job.
            expected = {0: "magic", 1: "magic", 2: "magic", 3: "magic", 4: "version", 63: "key"}
            self.assertEqual(cm.exception.reason, expected.get(i, "tag"), f"byte {i}")

    def test_another_nodes_key_cannot_sign_for_this_node(self):
        forged = encode_signed(self.f, node_key=derive_node_key(SITE, 13), key_id=3, boot=7)
        with self.assertRaises(FrameAuthError) as cm:
            decode_signed(forged, resolver({3: SITE}))
        self.assertEqual(cm.exception.reason, "tag")

    def test_a_retired_or_unknown_site_key_is_refused(self):
        with self.assertRaises(FrameAuthError) as cm:
            decode_signed(self.data, resolver({4: SITE}))
        self.assertEqual(cm.exception.reason, "key")
        with self.assertRaises(FrameAuthError) as cm:
            decode_signed(self.data, resolver({3: OTHER_SITE}))
        self.assertEqual(cm.exception.reason, "tag")

    def test_v1_and_v2_do_not_cross(self):
        with self.assertRaises(FrameAuthError) as cm:
            decode_signed(self.f.encode(), resolver({3: SITE}))
        self.assertEqual(cm.exception.reason, "size")
        with self.assertRaises(ValueError):
            decode_feature_frame(self.data)
        self.assertEqual(FEATURE_FRAME_SIZE, 68, "v1 is unchanged")

    def test_out_of_range_header_values_are_refused_on_encode(self):
        for kw in ({"key_id": 256}, {"key_id": -1}, {"boot": 1 << 32}, {"boot": -1}):
            args = {"node_key": self.key, "key_id": 3, "boot": 7, **kw}
            with self.assertRaises(ValueError, msg=str(kw)):
                encode_signed(self.f, **args)
        with self.assertRaises(ValueError):
            encode_signed(self.f, node_key=b"short", key_id=3, boot=7)


class TestReplayGuard(unittest.TestCase):
    def test_strictly_increasing_per_link(self):
        g = ReplayGuard()
        self.assertTrue(g.accept(1, 2, boot=5, seq=10))
        self.assertFalse(g.accept(1, 2, boot=5, seq=10), "the same frame twice")
        self.assertFalse(g.accept(1, 2, boot=5, seq=9), "an older frame")
        self.assertTrue(g.accept(1, 2, boot=5, seq=12), "a gap is fine")
        self.assertTrue(g.accept(1, 3, boot=5, seq=0), "another link has its own floor")

    def test_a_reboot_restarts_seq_and_retires_the_old_boot(self):
        g = ReplayGuard()
        g.accept(1, 2, boot=5, seq=1000)
        self.assertTrue(g.accept(1, 2, boot=6, seq=0))
        self.assertFalse(g.accept(1, 2, boot=5, seq=1001), "the previous boot is over")

    def test_a_refused_frame_does_not_move_the_floor(self):
        g = ReplayGuard()
        g.accept(1, 2, boot=5, seq=10)
        g.accept(1, 2, boot=4, seq=99)
        self.assertEqual(g.floors(), {(1, 2): (5, 10)})

    def test_floors_survive_a_round_trip(self):
        g = ReplayGuard()
        g.accept(1, 2, boot=5, seq=10)
        g.accept(7, 255, boot=1, seq=3)
        h = ReplayGuard.from_json(g.to_json())
        self.assertEqual(h.floors(), g.floors())
        self.assertFalse(h.accept(1, 2, boot=5, seq=10), "a restart does not reopen the past")


if __name__ == "__main__":
    unittest.main()
