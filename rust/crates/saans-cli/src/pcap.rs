//! Minimal pcap writers (nanosecond, LINKTYPE_RAW, synthesized IPv4/UDP),
//! readable by `wifisense`, `saans-gateway --replay` and Wireshark. Used to
//! turn a simulated scenario into a "recording" the gateway and dashboard
//! can play back.
//!
//! [`PcapWriter`] leaves the IPv4 identification and flags at zero.
//! [`RecorderPcapWriter`] writes exactly the bytes of
//! `wifisense.sources.recorder.PcapWriter` (identification counting from 0
//! per file, don't-fragment set), which `saans roomsim-export` needs so its
//! `*.sim.pcap` is byte-identical to the Python reference's export
//! (spec/roomsim.md §13).

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
        write_global_header(&mut out)?;
        Ok(Self { out })
    }

    /// One UDP datagram from `src` to `dst` at `t_ns` since the epoch.
    pub fn write(
        &mut self,
        t_ns: u64,
        src: ([u8; 4], u16),
        dst: ([u8; 4], u16),
        payload: &[u8],
    ) -> io::Result<()> {
        let secs = (t_ns / 1_000_000_000) as u32;
        let frac = (t_ns % 1_000_000_000) as u32;
        write_record(&mut self.out, secs, frac, 0, 0, src, dst, payload)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

fn write_global_header<W: Write>(out: &mut W) -> io::Result<()> {
    out.write_all(&MAGIC_NS.to_le_bytes())?;
    out.write_all(&2u16.to_le_bytes())?;
    out.write_all(&4u16.to_le_bytes())?;
    out.write_all(&0i32.to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())?;
    out.write_all(&65535u32.to_le_bytes())?;
    out.write_all(&LINKTYPE_RAW.to_le_bytes())
}

/// One record: pcap record header, IPv4 header (identification `ip_id`,
/// flags/fragment word `flags`, TTL 64, UDP, correct checksum), UDP header
/// (checksum 0 = not computed, legal for IPv4) and the payload.
#[allow(clippy::too_many_arguments)]
fn write_record<W: Write>(
    out: &mut W,
    secs: u32,
    nsecs: u32,
    ip_id: u16,
    flags: u16,
    src: ([u8; 4], u16),
    dst: ([u8; 4], u16),
    payload: &[u8],
) -> io::Result<()> {
    let total = 20 + 8 + payload.len();
    let mut ip = [0u8; 20];
    ip[0] = 0x45;
    ip[2..4].copy_from_slice(&(total as u16).to_be_bytes());
    ip[4..6].copy_from_slice(&ip_id.to_be_bytes());
    ip[6..8].copy_from_slice(&flags.to_be_bytes());
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
    out.write_all(&secs.to_le_bytes())?;
    out.write_all(&nsecs.to_le_bytes())?;
    out.write_all(&(total as u32).to_le_bytes())?;
    out.write_all(&(total as u32).to_le_bytes())?;
    out.write_all(&ip)?;
    out.write_all(&udp)?;
    out.write_all(payload)
}

/// The largest payload one IPv4 datagram can carry (the recorder's
/// `MAX_UDP_PAYLOAD`).
pub const MAX_UDP_PAYLOAD: usize = 65535 - 20 - 8;

/// Byte for byte `wifisense.sources.recorder.PcapWriter`: the same global
/// header, and per record the IPv4 identification counting 0, 1, 2, ...
/// (wrapping at 2^16) and the don't-fragment flag (`0x4000`). Like the
/// recorder it refuses a payload that does not fit one IPv4 datagram and a
/// timestamp whose seconds do not fit the record's u32, rather than
/// truncating either.
pub struct RecorderPcapWriter<W: Write> {
    out: W,
    ip_id: u16,
}

impl<W: Write> RecorderPcapWriter<W> {
    pub fn new(mut out: W) -> io::Result<Self> {
        write_global_header(&mut out)?;
        Ok(Self { out, ip_id: 0 })
    }

    /// Append one datagram received at epoch time `t_ns`.
    pub fn write(
        &mut self,
        t_ns: u64,
        src: ([u8; 4], u16),
        dst: ([u8; 4], u16),
        payload: &[u8],
    ) -> io::Result<()> {
        if payload.len() > MAX_UDP_PAYLOAD {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "payload of {} bytes does not fit one IPv4 datagram",
                    payload.len()
                ),
            ));
        }
        let secs = u32::try_from(t_ns / 1_000_000_000).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("timestamp {t_ns} ns does not fit a pcap record"),
            )
        })?;
        let nsecs = (t_ns % 1_000_000_000) as u32;
        write_record(
            &mut self.out,
            secs,
            nsecs,
            self.ip_id,
            0x4000,
            src,
            dst,
            payload,
        )?;
        self.ip_id = self.ip_id.wrapping_add(1);
        Ok(())
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

    #[test]
    fn the_plain_writer_keeps_its_bytes() {
        // `saans export` output must not change: id and flags stay zero.
        let mut w = PcapWriter::new(Vec::new()).unwrap();
        w.write(
            1_500_000_000,
            ([10, 0, 0, 101], 5566),
            ([10, 0, 0, 1], 5566),
            b"abc",
        )
        .unwrap();
        let b = w.out;
        assert_eq!(b.len(), 24 + 16 + 28 + 3);
        let ip = &b[40..60];
        assert_eq!(&ip[4..8], &[0, 0, 0, 0]);
        assert_eq!(ipv4_checksum(ip), 0);
        assert_eq!(&b[24..28], &1u32.to_le_bytes());
        assert_eq!(&b[28..32], &500_000_000u32.to_le_bytes());
    }

    /// The bytes `wifisense.sources.recorder.PcapWriter` writes for the same
    /// two datagrams, computed by hand from its struct formats.
    #[test]
    fn the_recorder_writer_matches_the_python_recorder() {
        let mut w = RecorderPcapWriter::new(Vec::new()).unwrap();
        let t0 = 1_790_000_000u64 * 1_000_000_000;
        w.write(
            t0 + 12_500_000,
            ([10, 0, 0, 3], 5566),
            ([10, 0, 0, 1], 5566),
            &[1, 2],
        )
        .unwrap();
        w.write(
            t0 + 25_000_000,
            ([10, 0, 0, 2], 5566),
            ([10, 0, 0, 1], 5566),
            &[3],
        )
        .unwrap();
        let b = w.out;
        // "<IHHiIII": magic, v2.4, thiszone 0, sigfigs 0, snaplen, LINKTYPE_RAW.
        let mut want = 0xA1B2_3C4Du32.to_le_bytes().to_vec();
        want.extend_from_slice(&2u16.to_le_bytes());
        want.extend_from_slice(&4u16.to_le_bytes());
        for v in [0u32, 0, 65535, 101] {
            want.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(&b[..24], &want[..]);
        // Record 1: "<IIII" header, then "!BBHHHBBH4s4sHHHH".
        let r1 = &b[24..];
        assert_eq!(&r1[0..4], &1_790_000_000u32.to_le_bytes());
        assert_eq!(&r1[4..8], &12_500_000u32.to_le_bytes());
        assert_eq!(&r1[8..12], &30u32.to_le_bytes());
        assert_eq!(&r1[12..16], &30u32.to_le_bytes());
        let ip = &r1[16..36];
        assert_eq!(&ip[0..2], &[0x45, 0]);
        assert_eq!(&ip[2..4], &30u16.to_be_bytes());
        assert_eq!(&ip[4..6], &0u16.to_be_bytes(), "first identification is 0");
        assert_eq!(&ip[6..8], &0x4000u16.to_be_bytes(), "don't-fragment");
        assert_eq!(&ip[8..10], &[64, 17]);
        assert_eq!(&ip[12..20], &[10, 0, 0, 3, 10, 0, 0, 1]);
        assert_eq!(ipv4_checksum(ip), 0);
        assert_eq!(&r1[36..44], &[0x15, 0xBE, 0x15, 0xBE, 0, 10, 0, 0]);
        assert_eq!(&r1[44..46], &[1, 2]);
        // Record 2: the identification has counted on.
        let r2 = &r1[46..];
        assert_eq!(&r2[16 + 4..16 + 6], &1u16.to_be_bytes());
        assert_eq!(r2.len(), 16 + 28 + 1);
    }

    #[test]
    fn the_recorder_writer_refuses_what_the_recorder_refuses() {
        let mut w = RecorderPcapWriter::new(Vec::new()).unwrap();
        let src = ([10, 0, 0, 1], 5566);
        assert!(w
            .write(0, src, src, &vec![0u8; MAX_UDP_PAYLOAD + 1])
            .is_err());
        assert!(w
            .write((u32::MAX as u64 + 1) * 1_000_000_000, src, src, b"x")
            .is_err());
        assert_eq!(w.out.len(), 24, "nothing written");
    }
}
