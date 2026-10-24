# ADR-012: FeatureFrame v2 — authenticated frames

**Status:** accepted, 2026-10-24; **built** 2026-10-24 in all three
implementations:
- the Python reference and golden vectors;
- the gateway (`saans-gateway/src/frameauth.rs`);
- the node (`edge/dsp/sha256.hpp`, `feature_frame.hpp`,
  `edge/firmware/main/frame_auth.cpp`, and `edge/build.sh provision`).

The firmware compiles for the ESP32-S3 but has not run on a board.

## Context
FeatureFrame v1 (ADR-004) ends in a CRC-32. A CRC catches a flipped bit;
it does nothing against someone who means it, because anyone can compute
one. Debug raw CSI has no check at all. The gateway believes any datagram
that reaches its UDP port, for any link it names. This was found
triaging two CodeQL alerts about the listener binding every interface
(README, "Operating requirements"; PR #18).

What a forged or replayed frame can do:

- **Fake occupancy, or hide a person.** Occupancy drives MQTT, and from
  there lights and air conditioning (SAANS_PLAN §10.2). Hiding a person
  is worse than faking one.
- **For a consented space, fake breathing.** That can raise a health
  alert that wakes a warden, or worse, keep breathing looking normal
  through a real pause, so an alert that should fire never does. Every
  health output is advisory (ADR-005), but an advisory that an attacker
  controls is not advice.
- **Replay.** A frame recorded at 3 a.m. last week is indistinguishable
  from one sent now. Signing alone does not stop this.

The mitigation until now is a trusted network: a sensor VLAN, a narrowed
`udp_bind`, a firewall. That stays necessary (see "What this does not
do"), but it should not be the only thing between an attacker and a
health alert.

## Decision
**FeatureFrame v2: 84 bytes, signed with a per-node key, with a counter
the gateway will not accept twice.**

| Offset | Size | Field | |
|---|---|---|---|
| 0–62 | 63 | as v1 | magic "SANS", `version` = **2**, then every v1 field at its v1 offset |
| 63 | 1 | `key_id` | which site key signed it (v1's `reserved` byte) |
| 64 | 4 | `boot` | u32: the node's boot counter, +1 on every boot, persisted |
| 68 | 16 | `tag` | the first 16 bytes of HMAC-SHA256(node key, bytes 0..67) |

- **No CRC.** The tag detects any change to bytes 0..67, accidental or
  not, and checking both would only be a second way to disagree.
- **Per-node keys, derived.** `node_key = HMAC-SHA256(site_key,
  "saans frame key v1" ‖ node_id)`, where `site_key` is 32 random
  bytes. The gateway holds only the site key. Each node is flashed with
  its own derived key, so a node taken off the wall gives up only the
  ability to sign as that one node, and the gateway never needs a
  per-node key file.
- **`key_id`** lets the gateway hold an old and a new site key at once,
  so keys can be rotated one node at a time.
- **Replay.** The gateway keeps, per link (`node_id`, `tx_id`), the
  highest (`boot`, `seq`) it has accepted. It accepts a frame only if
  the frame's pair is strictly greater, comparing `boot` first. A replayed
  frame is never greater than one already accepted. A reboot raises
  `boot`, so `seq` restarting from 0 is still fresh. The floors are
  persisted in `data_dir`, so a gateway restart does not reopen the past.
  `seq` is per link and per boot, so at 2 Hz it wraps after 68 years,
  and the counter needs no wrap handling.
- **Verify before parse.** Size, magic, version and `key_id` are read
  first, then the tag is checked in constant time over bytes 0..67.
  Fields are read only from a frame that passed; the replay check comes
  last, and only a fresh frame updates the floor.
- **Policy.** Once a site key is configured, the gateway accepts v2 only.
  v1 frames and raw CSI datagrams are refused and counted, because
  accepting them would leave the old door open beside the new one.
  Without a site key the gateway behaves as today: it accepts v1 and
  refuses v2, which it cannot check. It is **off by default** because no
  node sends v2 yet. Whether it becomes mandatory on a non-loopback
  `udp_bind`, the way HTTP authentication is, is decided when the
  firmware ships v2.

### Why HMAC-SHA256, and not an AEAD
HMAC-SHA256 is in Python's standard library, and the reference
implementation is standard-library only (ADR-002). In C++ it is a few
hundred lines of SHA-256 that can be tested against published vectors.
AES-GCM and ChaCha20-Poly1305 would also encrypt, but AES is not in
Python's standard library, and both need a nonce that never repeats
under one key. A node that loses its boot counter (erased flash, a
replaced board flashed with the old key) would then repeat nonces, which
for those modes is catastrophic. With HMAC it fails closed instead: the
gateway refuses that link's frames as replays and counts them in
`/api/health`, until an operator clears that link's floor. Encryption is a later decision if eavesdropping
on the sensor network becomes the threat; see below.

### Why truncate to 16 bytes
128 bits is the usual floor for an authentication tag, and at 2 Hz
there is no online forgery budget that 2^-128 does not cover. The
remaining 16 bytes would cost bandwidth and buy nothing.

## What this does not do
- **No confidentiality.** Frames are still readable on the wire,
  including vitals-gated fields for consented spaces. Anyone on the
  sensor network can read occupancy and breathing. The trusted-network
  requirement stays for that reason, and it is why the README still lists
  it.
- **A node in an attacker's hands.** Its flash holds its key. That
  compromises that node's link and nothing else.
- **Jamming.** A missing frame looks like a missing frame; the gateway
  reports the link degraded or down, as now.
- **Frames the gateway never saw.** If the gateway was down while a node
  kept sending, someone who recorded those frames can replay them later,
  in order, as long as no newer frame from that link has arrived since.
  The window closes at the node's next real frame (0.5 s if it is up).
  A freshness check against the gateway's clock could narrow it, and is
  not part of v2.
- **Raw CSI.** It stays unauthenticated debug output (ADR-001). With
  authentication on, the gateway refuses it.

## Consequences
- Every implementation grows: the Python reference (`wifisense/features.py`,
  the reference for the other two), golden vectors
  (`spec/golden/feature_codec_signed`), the Rust codec and the gateway's
  verification, replay floors and key configuration, the C++ codec with a
  SHA-256 of its own, and the firmware's key and boot counter in NVS.
- Provisioning becomes a step: generate a site key once, keep it in an
  environment variable on the gateway, derive each node's key and flash
  it. A node without a key keeps sending v1, which a gateway with
  authentication on refuses, and says so in `/api/health`.
- A node whose boot counter goes backwards is refused until its floor is
  cleared, deliberately: from the gateway's side that is
  indistinguishable from a replay. Clearing it is an operator action.
- Frames grow from 68 to 84 bytes: at 12 links and 2 Hz, 2.0 kB/s
  instead of 1.6.
