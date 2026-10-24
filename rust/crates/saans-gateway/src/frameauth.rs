//! FeatureFrame v2 (ADR-012, `spec/wire.md` §2b): site and node keys,
//! signing, checking, and the replay floor. The byte layout is
//! `saans-dsp`'s; the reference is `wifisense/features.py`.
//!
//! The keys are the one secret the gateway must hold for as long as it
//! runs: every frame is checked with them. So they live in types whose
//! `Debug` prints nothing, have no `Display`, and are read from the
//! environment once, at startup ([`load_site_keys`]). A node key leaves
//! this module only through [`NodeKey::to_hex`], which exists to print a
//! key for flashing a node, and is called from nowhere else.

use crate::config::GatewayConfig;
use hmac::{Hmac, Mac};
use saans_dsp::features::{
    FeatureFrame, FEATURE_MAGIC, FEATURE_VERSION_SIGNED, SIGNED_BODY_SIZE, SIGNED_FRAME_SIZE,
    TAG_SIZE,
};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::BTreeMap;
use std::fmt;
use std::io;

type HmacSha256 = Hmac<Sha256>;

/// Prefix of the message a node key is derived from (then the node id).
pub const NODE_KEY_LABEL: &[u8] = b"saans frame key v1";

fn hmac(key: &[u8], parts: &[&[u8]]) -> HmacSha256 {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC takes any key length");
    for p in parts {
        mac.update(p);
    }
    mac
}

fn from_hex(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 || !s.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

fn to_hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// 32 random bytes the gateway holds; nodes never see one.
#[derive(Clone, PartialEq, Eq)]
pub struct SiteKey([u8; 32]);

impl fmt::Debug for SiteKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SiteKey(..)")
    }
}

impl SiteKey {
    pub fn from_bytes(b: [u8; 32]) -> Self {
        Self(b)
    }

    /// 64 hex digits, surrounding whitespace ignored.
    pub fn from_hex(s: &str) -> Option<Self> {
        from_hex(s).map(Self)
    }

    /// The key node `node_id` is flashed with: HMAC-SHA256(site key,
    /// label || node_id). It signs for that node and no other.
    pub fn node_key(&self, node_id: u8) -> NodeKey {
        let out = hmac(&self.0, &[NODE_KEY_LABEL, &[node_id]])
            .finalize()
            .into_bytes();
        NodeKey(out.into())
    }
}

/// One node's key, as flashed into it.
#[derive(Clone, PartialEq, Eq)]
pub struct NodeKey([u8; 32]);

impl fmt::Debug for NodeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NodeKey(..)")
    }
}

impl NodeKey {
    pub fn from_bytes(b: [u8; 32]) -> Self {
        Self(b)
    }

    /// For flashing a node (`saans-gateway --node-key`), and nothing else.
    pub fn to_hex(&self) -> String {
        to_hex(&self.0)
    }

    fn tag(&self, body: &[u8]) -> HmacSha256 {
        hmac(&self.0, &[body])
    }
}

/// FeatureFrame v2: `frame`'s fields, `key_id`, `boot`, and the first 16
/// bytes of HMAC-SHA256 over everything before the tag. What a node sends;
/// here for tests, fixtures and the reference parity check.
pub fn sign(frame: &FeatureFrame, key: &NodeKey, key_id: u8, boot: u32) -> [u8; SIGNED_FRAME_SIZE] {
    let body = frame.encode_signed_body(key_id, boot);
    let tag = key.tag(&body).finalize().into_bytes();
    let mut out = [0u8; SIGNED_FRAME_SIZE];
    out[..SIGNED_BODY_SIZE].copy_from_slice(&body);
    out[SIGNED_BODY_SIZE..].copy_from_slice(&tag[..TAG_SIZE]);
    out
}

/// Why a v2 datagram was refused. Counted separately in `/api/health`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    Size(usize),
    Magic,
    Version(u8),
    /// No site key with this `key_id` is held.
    UnknownKey(u8),
    /// Forged, corrupted, or signed with another key.
    Tag,
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Size(n) => write!(
                f,
                "a signed FeatureFrame is {SIGNED_FRAME_SIZE} bytes, got {n}"
            ),
            Refused::Magic => f.write_str("bad magic"),
            Refused::Version(v) => write!(
                f,
                "expected FeatureFrame version {FEATURE_VERSION_SIGNED}, got {v}"
            ),
            Refused::UnknownKey(k) => write!(f, "no site key with key_id {k}"),
            Refused::Tag => f.write_str("the tag does not match"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verified {
    pub frame: FeatureFrame,
    pub key_id: u8,
    pub boot: u32,
}

/// Check, then parse, one v2 datagram: length, magic, version, a held key,
/// then the tag in constant time, and only then the fields. The replay
/// floor is the caller's ([`ReplayGuard`]), applied after this.
pub fn verify(data: &[u8], keys: &BTreeMap<u8, SiteKey>) -> Result<Verified, Refused> {
    if data.len() != SIGNED_FRAME_SIZE {
        return Err(Refused::Size(data.len()));
    }
    if data[..4] != FEATURE_MAGIC.to_le_bytes() {
        return Err(Refused::Magic);
    }
    if data[4] != FEATURE_VERSION_SIGNED {
        return Err(Refused::Version(data[4]));
    }
    let (key_id, node_id) = (data[63], data[5]);
    let site = keys.get(&key_id).ok_or(Refused::UnknownKey(key_id))?;
    site.node_key(node_id)
        .tag(&data[..SIGNED_BODY_SIZE])
        .verify_truncated_left(&data[SIGNED_BODY_SIZE..])
        .map_err(|_| Refused::Tag)?;
    let (frame, key_id, boot) =
        FeatureFrame::decode_signed_body(data).expect("length, magic and version checked above");
    Ok(Verified {
        frame,
        key_id,
        boot,
    })
}

/// The highest (boot, seq) accepted per link (node_id, tx_id).
///
/// A frame is fresh only if its pair is strictly greater, boot first: a
/// replay is never greater than a frame already accepted, and a reboot
/// raises `boot`, so `seq` restarting from 0 is fresh. A node whose boot
/// counter goes backwards is refused until its floor is cleared; from
/// here that is indistinguishable from a replay.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReplayGuard {
    floors: BTreeMap<(u8, u8), (u32, u32)>,
    /// Set by every accepted frame, cleared by [`ReplayGuard::take_dirty`].
    dirty: bool,
}

/// One floor, as persisted in `data_dir/frame_counters.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Floor {
    pub node_id: u8,
    pub tx_id: u8,
    pub boot: u32,
    pub seq: u32,
}

impl ReplayGuard {
    pub fn accept(&mut self, node_id: u8, tx_id: u8, boot: u32, seq: u32) -> bool {
        let link = (node_id, tx_id);
        let pair = (boot, seq);
        if self.floors.get(&link).is_some_and(|&floor| pair <= floor) {
            return false;
        }
        self.floors.insert(link, pair);
        self.dirty = true;
        true
    }

    pub fn floors(&self) -> Vec<Floor> {
        self.floors
            .iter()
            .map(|(&(node_id, tx_id), &(boot, seq))| Floor {
                node_id,
                tx_id,
                boot,
                seq,
            })
            .collect()
    }

    pub fn from_floors(floors: &[Floor]) -> Self {
        Self {
            floors: floors
                .iter()
                .map(|f| ((f.node_id, f.tx_id), (f.boot, f.seq)))
                .collect(),
            dirty: false,
        }
    }

    pub fn links(&self) -> usize {
        self.floors.len()
    }

    /// True once after any accepted frame: time to persist the floors.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }
}

/// Counted in `ingest.frame_auth` in `/api/health`.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct AuthStats {
    /// v2 frames whose tag matched and whose counter was fresh.
    pub accepted: u64,
    /// Forged, corrupted, or signed with another key.
    pub bad_tag: u64,
    /// A `key_id` no `[[frame_key]]` provides.
    pub unknown_key: u64,
    /// Authentic, but not newer than one already accepted on that link.
    pub replayed: u64,
    /// v1 frames, refused because authentication is on.
    pub unsigned_refused: u64,
    /// Raw-CSI datagrams, refused because authentication is on.
    pub raw_refused: u64,
    /// v2 frames, refused because no site key is configured to check them.
    pub no_key: u64,
}

/// Read every `[[frame_key]]`'s environment variable. Called once at
/// startup; an unset or malformed variable is an error naming the
/// variable and never its value.
pub fn load_site_keys(cfg: &GatewayConfig) -> io::Result<BTreeMap<u8, SiteKey>> {
    let mut keys = BTreeMap::new();
    for k in &cfg.frame_keys {
        let value = std::env::var(&k.env).map_err(|_| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "frame_key {}: environment variable {} is not set",
                    k.id, k.env
                ),
            )
        })?;
        let key = SiteKey::from_hex(&value).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "frame_key {}: {} must hold 64 hex digits (32 bytes)",
                    k.id, k.env
                ),
            )
        })?;
        keys.insert(k.id, key);
    }
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site() -> SiteKey {
        SiteKey::from_bytes(std::array::from_fn(|i| i as u8))
    }

    fn frame() -> FeatureFrame {
        FeatureFrame {
            node_id: 12,
            tx_id: 3,
            seq: 41,
            motion_energy: 0.15625,
            breathing_bpm: 15.25,
            ..Default::default()
        }
    }

    fn keys(id: u8, k: SiteKey) -> BTreeMap<u8, SiteKey> {
        BTreeMap::from([(id, k)])
    }

    #[test]
    fn hmac_sha256_is_the_rfc_4231_function() {
        // RFC 4231 test case 2, as the Python reference pins it.
        let out = hmac(b"Jefe", &[b"what do ya want for nothing?"])
            .finalize()
            .into_bytes();
        assert_eq!(
            to_hex(&out),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn a_signed_frame_round_trips_and_every_changed_byte_is_refused() {
        let k = keys(3, site());
        let data = sign(&frame(), &site().node_key(12), 3, 7);
        let v = verify(&data, &k).unwrap();
        assert_eq!((v.key_id, v.boot, v.frame.version), (3, 7, 2));
        assert_eq!(v.frame.encode_signed_body(3, 7), data[..68]);
        for i in 0..SIGNED_FRAME_SIZE {
            let mut bad = data;
            bad[i] ^= 1;
            let want = match i {
                0..=3 => Refused::Magic,
                4 => Refused::Version(3),
                63 => Refused::UnknownKey(2),
                _ => Refused::Tag,
            };
            assert_eq!(verify(&bad, &k), Err(want), "byte {i}");
        }
    }

    #[test]
    fn another_nodes_key_or_another_site_cannot_sign() {
        let k = keys(3, site());
        let forged = sign(&frame(), &site().node_key(13), 3, 7);
        assert_eq!(verify(&forged, &k), Err(Refused::Tag));
        let other = SiteKey::from_bytes([9; 32]);
        assert_eq!(
            verify(&sign(&frame(), &other.node_key(12), 3, 7), &k),
            Err(Refused::Tag)
        );
        assert_eq!(verify(&frame().encode(), &k), Err(Refused::Size(68)));
    }

    #[test]
    fn the_replay_floor_is_per_link_and_boot_first() {
        let mut g = ReplayGuard::default();
        assert!(g.accept(1, 2, 0, 0), "a link's first frame, even (0, 0)");
        assert!(!g.accept(1, 2, 0, 0), "the same frame twice");
        assert!(g.accept(1, 2, 5, 10));
        assert!(!g.accept(1, 2, 5, 9));
        assert!(g.accept(1, 3, 0, 0), "another link has its own floor");
        assert!(g.accept(1, 2, 6, 0), "a reboot restarts seq");
        assert!(!g.accept(1, 2, 5, 11), "the previous boot is over");
        let h = ReplayGuard::from_floors(&g.floors());
        assert_eq!(h.floors(), g.floors());
        let mut h = h;
        assert!(!h.accept(1, 2, 6, 0), "a restart does not reopen the past");
        assert!(!h.accept(1, 3, 0, 0), "nor a link's (0, 0)");
    }

    #[test]
    fn keys_never_print() {
        assert_eq!(format!("{:?}", site()), "SiteKey(..)");
        assert_eq!(format!("{:?}", site().node_key(1)), "NodeKey(..)");
        assert_eq!(SiteKey::from_hex("00"), None);
        let hex = "ab".repeat(32);
        assert_eq!(
            SiteKey::from_hex(&hex),
            Some(SiteKey::from_bytes([0xab; 32]))
        );
        assert_eq!(SiteKey::from_hex(&"zz".repeat(32)), None);
    }
}
