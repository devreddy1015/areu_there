//! raw CSI datagrams (byte-exact port of
//! `wifisense/sources/esp32_udp.py`). Debug / data collection only
//! (ADR-001); the gateway turns them into FeatureFrames itself.
//!
//! ```text
//! [0..3]   magic 0xC5110001        [12..15] sequence (u32)
//! [4]      node id                 [16]     RSSI (i8, dBm)
//! [5]      antenna count           [17]     noise floor (i8, dBm)
//! [6..7]   subcarrier count (u16)  [18]     AGC gain (u8, ESP32-S3)
//! [8..11]  centre freq MHz (u32)   [19]     FFT gain (i8, ESP32-S3)
//! [20..]   int8 I/Q pairs, IMAGINARY FIRST
//! ```

use std::fmt;

pub const CSI_MAGIC: u32 = 0xC511_0001;
pub const CSI_HEADER_SIZE: usize = 20;

/// True if `data` starts with the raw-CSI magic.
pub fn sniff(data: &[u8]) -> bool {
    data.len() >= 4 && data[..4] == CSI_MAGIC.to_le_bytes()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Esp32Frame {
    pub node_id: u8,
    pub n_antennas: u8,
    pub n_subcarriers: u16,
    pub freq_mhz: u32,
    pub sequence: u32,
    pub rssi_dbm: i8,
    pub noise_floor_dbm: i8,
    pub ppdu_type: u8,
    pub flags: u8,
    /// (imaginary, real) of the first antenna, one pair per subcarrier.
    pub iq: Vec<(i8, i8)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawError {
    Short(usize),
    Magic(u32),
    Truncated { have: usize, need: usize },
}

impl fmt::Display for RawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RawError::Short(n) => {
                write!(f, "short frame: {n} bytes, need at least {CSI_HEADER_SIZE}")
            }
            RawError::Magic(m) => write!(f, "bad magic 0x{m:08X}, expected 0x{CSI_MAGIC:08X}"),
            RawError::Truncated { have, need } => {
                write!(f, "truncated payload: {have} bytes, expected {need}")
            }
        }
    }
}

impl std::error::Error for RawError {}

pub fn parse(data: &[u8]) -> Result<Esp32Frame, RawError> {
    if data.len() < CSI_HEADER_SIZE {
        return Err(RawError::Short(data.len()));
    }
    let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let magic = u32_at(0);
    if magic != CSI_MAGIC {
        return Err(RawError::Magic(magic));
    }
    let n_antennas = data[5];
    let n_subcarriers = u16::from_le_bytes([data[6], data[7]]);
    let payload = &data[CSI_HEADER_SIZE..];
    let need = n_subcarriers as usize * n_antennas as usize * 2;
    if payload.len() < need {
        return Err(RawError::Truncated {
            have: payload.len(),
            need,
        });
    }
    // Imaginary first: swapping these keeps amplitude right and mirrors
    // every phase, which survives amplitude-only testing.
    let iq = payload
        .iter()
        .step_by(2)
        .zip(payload.iter().skip(1).step_by(2))
        .take(n_subcarriers as usize)
        .map(|(&im, &re)| (im as i8, re as i8))
        .collect();
    Ok(Esp32Frame {
        node_id: data[4],
        n_antennas,
        n_subcarriers,
        freq_mhz: u32_at(8),
        sequence: u32_at(12),
        rssi_dbm: data[16] as i8,
        noise_floor_dbm: data[17] as i8,
        ppdu_type: data[18],
        flags: data[19],
        iq,
    })
}

/// Encode a datagram (single antenna), as `build_esp32_frame` does.
pub fn build(
    iq: &[(i8, i8)],
    node_id: u8,
    sequence: u32,
    rssi_dbm: i8,
    noise_floor_dbm: i8,
) -> Vec<u8> {
    let mut b = Vec::with_capacity(CSI_HEADER_SIZE + 2 * iq.len());
    b.extend_from_slice(&CSI_MAGIC.to_le_bytes());
    b.push(node_id);
    b.push(1);
    b.extend_from_slice(&(iq.len() as u16).to_le_bytes());
    b.extend_from_slice(&2437u32.to_le_bytes());
    b.extend_from_slice(&sequence.to_le_bytes());
    b.push(rssi_dbm as u8);
    b.push(noise_floor_dbm as u8);
    b.push(0);
    b.push(0);
    for &(im, re) in iq {
        b.push(im as u8);
        b.push(re as u8);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_imaginary_first() {
        let iq = [(3i8, -4i8), (-128, 127), (0, 1)];
        let d = build(&iq, 7, 0xDEAD_BEEF, -45, -92);
        assert_eq!(d.len(), 26);
        assert_eq!(&d[..4], &[0x01, 0x00, 0x11, 0xC5]);
        assert_eq!(d[20], 3, "imaginary byte comes first");
        let f = parse(&d).unwrap();
        assert_eq!(f.iq, iq);
        assert_eq!(
            (f.node_id, f.sequence, f.rssi_dbm, f.noise_floor_dbm),
            (7, 0xDEAD_BEEF, -45, -92)
        );
        assert_eq!(f.n_subcarriers, 3);
    }

    #[test]
    fn rejects_malformed() {
        assert_eq!(parse(&[0u8; 5]), Err(RawError::Short(5)));
        let mut d = build(&[(1, 1); 4], 1, 0, 0, 0);
        d.truncate(25);
        assert!(matches!(
            parse(&d),
            Err(RawError::Truncated { have: 5, need: 8 })
        ));
        d[0] = 0;
        assert!(matches!(parse(&d), Err(RawError::Magic(_))));
    }
}
