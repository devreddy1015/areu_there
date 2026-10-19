//! Keep the time axis uniform across lost packets
//! (port of `SequenceGapFiller` in `wifisense/sources/esp32_udp.py`).
//!
//! Feeding survivors of 15% packet loss straight in made a 15 BPM breather
//! read 17.8 BPM at confidence 0.98 (NOTES.md §6.6).

use crate::types::CsiFrame;

const SEQ_MOD: u64 = 1 << 32;

#[derive(Debug, Clone)]
pub struct SequenceGapFiller {
    pub max_gap: u64,
    last_seq: Option<u64>,
    last: Option<CsiFrame>,
    pub filled: u64,
    pub discontinuities: u64,
    pub duplicates: u64,
}

impl SequenceGapFiller {
    pub fn new(max_gap: u64) -> Self {
        Self {
            max_gap,
            last_seq: None,
            last: None,
            filled: 0,
            discontinuities: 0,
            duplicates: 0,
        }
    }

    /// Frames to process, in order: interpolated fill, then `frame`.
    pub fn push(&mut self, sequence: u64, frame: CsiFrame) -> Vec<CsiFrame> {
        let sequence = sequence % SEQ_MOD;
        let (Some(last_seq), Some(last)) = (self.last_seq, self.last.as_ref()) else {
            self.last_seq = Some(sequence);
            self.last = Some(frame.clone());
            return vec![frame];
        };
        let step = (sequence + SEQ_MOD - last_seq) % SEQ_MOD;
        if step == 0 {
            self.duplicates += 1;
            return Vec::new();
        }
        let mut out = Vec::new();
        let missing = step - 1;
        if step > SEQ_MOD / 2 || missing > self.max_gap {
            self.discontinuities += 1;
        } else if missing > 0 && last.n_subcarriers() == frame.n_subcarriers() {
            for j in 1..=missing {
                let w = j as f64 / step as f64;
                out.push(CsiFrame {
                    amplitudes: last
                        .amplitudes
                        .iter()
                        .zip(&frame.amplitudes)
                        .map(|(x, y)| x + (y - x) * w)
                        .collect(),
                    phases: last.phases.clone(),
                    sample_index: (last_seq + j) % SEQ_MOD,
                    sample_rate_hz: frame.sample_rate_hz,
                    rssi_dbm: frame.rssi_dbm,
                    node_id: frame.node_id,
                });
            }
            self.filled += missing;
        } else if missing > 0 {
            self.discontinuities += 1;
        }
        self.last_seq = Some(sequence);
        self.last = Some(frame.clone());
        out.push(frame);
        out
    }
}
