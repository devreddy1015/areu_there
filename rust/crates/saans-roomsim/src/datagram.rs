//! The raw CSI datagram of spec/roomsim.md §12.4, byte for byte as
//! `edge/firmware/main/csi_capture.cpp::serialize_raw` writes it, so that
//! `wifisense.sources.esp32_udp.parse_esp32_frame` and the gateway's
//! `raw.rs` parse it unchanged.
//!
//! ```text
//! [0..3]   magic 0xC5110001 (u32 LE)   [12..15] sequence (u32 LE)
//! [4]      node id (rx)                [16]     RSSI (i8, dBm)
//! [5]      n_antennas = 1              [17]     noise floor (i8, dBm)
//! [6..7]   n_subcarriers (u16 LE)      [18]     0 (PPDU type)
//! [8..11]  freq MHz (u32 LE)           [19]     0 (flags)
//! [20..]   n_subcarriers pairs of (i8 imag, i8 real)
//! ```

/// `0xC5110001`.
pub const CSI_MAGIC: u32 = 0xC511_0001;
/// Bytes before the I/Q payload.
pub const CSI_HEADER_SIZE: usize = 20;

/// `freq_mhz = 2412 + 5*(channel-1)`.
pub fn freq_mhz(channel: u32) -> u32 {
    2412 + 5 * (channel - 1)
}

/// Serialize a datagram. `iq` is the quantized buffer as `(re, im)` pairs in
/// buffer order; its first `fw_subcarriers` entries are written, each
/// imaginary first.
pub fn serialize(
    iq: &[(i8, i8)],
    fw_subcarriers: usize,
    rx: u8,
    channel: u32,
    sequence: u32,
    rssi: i8,
    nf_dbm: i8,
) -> Vec<u8> {
    let n = fw_subcarriers.min(iq.len());
    let mut b = Vec::with_capacity(CSI_HEADER_SIZE + 2 * n);
    b.extend_from_slice(&CSI_MAGIC.to_le_bytes());
    b.push(rx);
    b.push(1);
    b.extend_from_slice(&(n as u16).to_le_bytes());
    b.extend_from_slice(&freq_mhz(channel).to_le_bytes());
    b.extend_from_slice(&sequence.to_le_bytes());
    b.push(rssi as u8);
    b.push(nf_dbm as u8);
    b.push(0);
    b.push(0);
    for &(re, im) in &iq[..n] {
        b.push(im as u8);
        b.push(re as u8);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_serialize_raw() {
        let iq = [(1i8, -2i8), (-128, 127), (5, 6)];
        let d = serialize(&iq, 2, 3, 6, 0x0102_0304, -45, -95);
        assert_eq!(d.len(), 20 + 4);
        assert_eq!(&d[0..4], &[0x01, 0x00, 0x11, 0xC5]);
        assert_eq!(d[4], 3);
        assert_eq!(d[5], 1);
        assert_eq!(u16::from_le_bytes([d[6], d[7]]), 2);
        assert_eq!(u32::from_le_bytes(d[8..12].try_into().unwrap()), 2437);
        assert_eq!(
            u32::from_le_bytes(d[12..16].try_into().unwrap()),
            0x0102_0304
        );
        assert_eq!(d[16] as i8, -45);
        assert_eq!(d[17] as i8, -95);
        assert_eq!((d[18], d[19]), (0, 0));
        assert_eq!(
            &d[20..],
            &[(-2i8) as u8, 1, 127, (-128i8) as u8],
            "imaginary first"
        );
    }
}
