//! Replay a pcap recording through exactly the live ingest path, with
//! `t_s` taken from the packet timestamps. At speed 0 it runs as fast as
//! possible and is deterministic: the same file and config give the same
//! events, byte for byte. This is what makes a real-hardware recording
//! reproducible in Rust.
//!
//! Replay never reads or writes `data_dir` (no history import, no event
//! logs): its output is the summary and, optionally, `--events-out`.

use crate::config::GatewayConfig;
use crate::engine::{IngestStats, UnknownLink, TICK_S};
use crate::hub::{local_epoch_s, Hub, Wall};
use crate::pcap::{PcapReader, ReadStats};
use crate::shared::Shared;
use crate::timefmt::iso8601;
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ReplayOptions {
    pub pcap: PathBuf,
    /// 0 = as fast as possible; 1 = real time; 10 = ten times real time.
    pub speed: f64,
    pub events_out: Option<PathBuf>,
    /// Keep only datagrams sent to this UDP port.
    pub port: Option<u16>,
}

#[derive(Debug, Clone)]
pub struct ReplaySummary {
    pub pcap: PathBuf,
    pub read: ReadStats,
    pub ingest: IngestStats,
    pub first_unix_ms: Option<i64>,
    pub duration_s: f64,
    /// Packets whose timestamp went backwards (clamped to keep t monotonic).
    pub clock_steps_back: u64,
    pub events: u64,
    /// Per space: event type -> count.
    pub per_space: Vec<(String, BTreeMap<String, u64>)>,
    pub unknown_links: Vec<UnknownLink>,
    pub utc_offset_minutes: i32,
}

impl fmt::Display for ReplaySummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = &self.read;
        let i = &self.ingest;
        writeln!(f, "replay of {}", self.pcap.display())?;
        if let Some(ms) = self.first_unix_ms {
            let end = ms + (self.duration_s * 1000.0) as i64;
            writeln!(
                f,
                "  recorded {} .. {} ({:.1} s)",
                iso8601(ms, self.utc_offset_minutes),
                iso8601(end, self.utc_offset_minutes),
                self.duration_s
            )?;
        }
        writeln!(
            f,
            "  pcap: {} records, {} UDP datagrams, skipped {} (not IPv4 {}, not UDP {}, fragments {}, other port {}, snapped {}){}",
            r.records,
            r.packets,
            r.not_ipv4 + r.not_udp + r.fragments + r.other_port + r.snapped,
            r.not_ipv4,
            r.not_udp,
            r.fragments,
            r.other_port,
            r.snapped,
            if r.truncated { ", truncated final record" } else { "" }
        )?;
        writeln!(
            f,
            "  ingest: {} FeatureFrames, {} raw CSI -> {} computed frames ({} shadowed), {} decode errors, {} unassigned{}",
            i.feature_frames,
            i.raw_datagrams,
            i.raw_feature_frames,
            i.raw_shadowed,
            i.errors.total(),
            i.unassigned,
            if self.clock_steps_back > 0 { format!(", {} timestamps stepped back", self.clock_steps_back) } else { String::new() }
        )?;
        let a = &i.frame_auth;
        let refused =
            a.bad_tag + a.unknown_key + a.replayed + a.unsigned_refused + a.raw_refused + a.no_key;
        if a.accepted + refused > 0 {
            writeln!(
                f,
                "  signed frames: {} accepted; refused {} bad tag, {} unknown key, {} replayed, {} unsigned, {} raw, {} with no key configured",
                a.accepted, a.bad_tag, a.unknown_key, a.replayed, a.unsigned_refused, a.raw_refused, a.no_key
            )?;
        }
        for u in &self.unknown_links {
            writeln!(
                f,
                "  unknown link {}<-{} ({}): {} frames, last from {}",
                u.rx,
                u.tx,
                u.kind,
                u.frames,
                u.last_src.as_deref().unwrap_or("?")
            )?;
        }
        writeln!(f, "events: {}", self.events)?;
        for (space, by) in &self.per_space {
            let n: u64 = by.values().sum();
            writeln!(f, "  {space}: {n}")?;
            for (ty, c) in by {
                writeln!(f, "    {ty:<22} {c}")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ReplayError(pub String);

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ReplayError {}

/// Run a replay. With `shared`, every tick is published for the HTTP side
/// (watch a recording on the dashboard, best with `speed` > 0).
pub async fn replay(
    cfg: GatewayConfig,
    opts: &ReplayOptions,
    shared: Option<Arc<Shared>>,
) -> Result<ReplaySummary, ReplayError> {
    let path = opts.pcap.display().to_string();
    let file = File::open(&opts.pcap).map_err(|e| ReplayError(format!("{path}: {e}")))?;
    let mut reader = PcapReader::new(BufReader::new(file), opts.port)
        .map_err(|e| ReplayError(format!("{path}: {e}")))?;
    let read_err = |e: crate::pcap::PcapError| ReplayError(format!("{path}: {e}"));
    let offset = cfg.utc_offset_minutes;
    let mut summary = ReplaySummary {
        pcap: opts.pcap.clone(),
        read: ReadStats::default(),
        ingest: IngestStats::default(),
        first_unix_ms: None,
        duration_s: 0.0,
        clock_steps_back: 0,
        events: 0,
        per_space: Vec::new(),
        unknown_links: Vec::new(),
        utc_offset_minutes: offset,
    };
    let Some(first) = reader.next_datagram().map_err(read_err)? else {
        summary.read = reader.stats.clone();
        return Ok(summary);
    };
    // t_s = local-epoch seconds of the packet (see hub.rs, "Time").
    let t_of = |ns: i64| {
        local_epoch_s(ns.div_euclid(1_000_000), offset) + ns.rem_euclid(1_000_000) as f64 / 1e9
    };
    let t0 = t_of(first.t_ns);
    let wall = Wall::Anchored {
        t0,
        unix_ms_at_t0: first.t_ns.div_euclid(1_000_000),
    };
    let frame_keys =
        crate::frameauth::load_site_keys(&cfg).map_err(|e| ReplayError(e.to_string()))?;
    let mut hub = Hub::new(cfg, shared, false, wall, t0);
    hub.engine.set_frame_keys(frame_keys);
    if let Some(out) = &opts.events_out {
        hub.set_events_out(out)
            .map_err(|e| ReplayError(format!("{}: {e}", out.display())))?;
    }
    let started = Instant::now();
    let pace = |t: f64| {
        let speed = opts.speed;
        async move {
            if speed > 0.0 {
                let target = started + Duration::from_secs_f64(((t - t0) / speed).max(0.0));
                tokio::time::sleep_until(target.into()).await;
            }
        }
    };
    let mut next_tick = t0 + TICK_S;
    let mut last_t = t0;
    let mut datagram = Some(first);
    while let Some(d) = datagram {
        let mut t = t_of(d.t_ns);
        if t < last_t {
            summary.clock_steps_back += 1;
            t = last_t;
        }
        last_t = t;
        while next_tick <= t {
            pace(next_tick).await;
            hub.tick(next_tick);
            next_tick += TICK_S;
        }
        pace(t).await;
        hub.ingest(&d.payload, t, Some(SocketAddr::V4(d.src)));
        datagram = reader.next_datagram().map_err(read_err)?;
    }
    // One more tick so the last frames are sampled.
    hub.tick(next_tick);
    hub.shutdown(last_t);
    summary.read = reader.stats.clone();
    summary.ingest = hub.engine.stats.clone();
    summary.first_unix_ms = Some(wall.unix_ms(t0, t0));
    summary.duration_s = last_t - t0;
    summary.events = hub.events_emitted;
    summary.per_space = hub.counts_by_type();
    summary.unknown_links = hub.engine.unknown.values().cloned().collect();
    Ok(summary)
}
