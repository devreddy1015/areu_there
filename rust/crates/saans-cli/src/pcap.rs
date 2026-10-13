//! Minimal pcap writer (nanosecond, LINKTYPE_RAW, synthesized IPv4/UDP),
//! byte-compatible with `wifisense record` and readable by
//! `saans-gateway --replay` and Wireshark. Used to turn a simulated
//! scenario into a "recording" the gateway and dashboard can play back.

use std::io::{self, Write};

pub struct PcapWriter<W: Write> {
    out: W,
}

const MAGIC_NS: u32 = 0xA1B2_3C4D;
const LINKTYPE_RAW: u32 = 101;

fn ipv4_checksum(header: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    for chunk in header.chunks(2) {
        sum += u16::from_be_bytes([chunk[0], *chunk.get(1).unwrap_or(&0)]) as u32;
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

impl<W: Write> PcapWriter<W> {
    pub fn new(mut out: W) -> io::Result<Self> {
        out.write_all(&MAGIC_NS.to_le_bytes())?;
        out.write_all(&2u16.to_le_bytes())?;
        out.write_all(&4u16.to_le_bytes())?;
        out.write_all(&0i32.to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?;
        out.write_all(&65535u32.to_le_bytes())?;
        out.write_all(&LINKTYPE_RAW.to_le_bytes())?;
        Ok(Self { out })
    }

    /// One UDP datagram from `src` to `dst` at `t_ns` since the epoch.
    pub fn write(&mut self, t_ns: u64, src: ([u8; 4], u16), dst: ([u8; 4], u16), payload: &[u8]) -> io::Result<()> {
        let total = 20 + 8 + payload.len();
        let mut ip = [0u8; 20];
        ip[0] = 0x45;
        ip[2..4].copy_from_slice(&(total as u16).to_be_bytes());
        ip[8] = 64;
        ip[9] = 17;
        ip[12..16].copy_from_slice(&src.0);
        ip[16..20].copy_from_slice(&dst.0);
        let csum = ipv4_checksum(&ip);
        ip[10..12].copy_from_slice(&csum.to_be_bytes());
        let mut udp = [0u8; 8];
        udp[0..2].copy_from_slice(&src.1.to_be_bytes());
        udp[2..4].copy_from_slice(&dst.1.to_be_bytes());
        udp[4..6].copy_from_slice(&((8 + payload.len()) as u16).to_be_bytes());
        let secs = (t_ns / 1_000_000_000) as u32;
        let frac = (t_ns % 1_000_000_000) as u32;
        self.out.write_all(&secs.to_le_bytes())?;
        self.out.write_all(&frac.to_le_bytes())?;
        self.out.write_all(&(total as u32).to_le_bytes())?;
        self.out.write_all(&(total as u32).to_le_bytes())?;
        self.out.write_all(&ip)?;
        self.out.write_all(&udp)?;
        self.out.write_all(payload)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_checksum_verifies() {
        let mut ip = [0u8; 20];
        ip[0] = 0x45;
        ip[2..4].copy_from_slice(&96u16.to_be_bytes());
        ip[8] = 64;
        ip[9] = 17;
        let c = ipv4_checksum(&ip);
        ip[10..12].copy_from_slice(&c.to_be_bytes());
        assert_eq!(ipv4_checksum(&ip), 0);
    }
}
