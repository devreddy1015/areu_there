//! Classic libpcap reading and writing, compatible with
//! `wifisense/sources/recorder.py`.
//!
//! The recorder writes nanosecond little-endian pcap (magic 0xA1B23C4D)
//! with LINKTYPE_RAW (101): a synthesized IPv4 + UDP header around each
//! datagram. The reader also takes microsecond files, either byte order,
//! and the link types a `tcpdump -w` capture on a gateway is likely to
//! have: Ethernet (with VLAN tags), raw IPv4 (101, 228), Linux cooked
//! v1/v2 (113, 276) and BSD loopback (0). Anything that is not an
//! unfragmented IPv4 UDP datagram is skipped and counted; a half-written
//! final record ends the file cleanly.

use std::fmt;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4};

pub const PCAP_MAGIC_US: u32 = 0xA1B2_C3D4;
pub const PCAP_MAGIC_NS: u32 = 0xA1B2_3C4D;
const PCAPNG_MAGIC: u32 = 0x0A0D_0D0A;
pub const LINKTYPE_NULL: u32 = 0;
pub const LINKTYPE_ETHERNET: u32 = 1;
pub const LINKTYPE_RAW: u32 = 101;
pub const LINKTYPE_LINUX_SLL: u32 = 113;
pub const LINKTYPE_IPV4: u32 = 228;
pub const LINKTYPE_LINUX_SLL2: u32 = 276;
const MAX_RECORD: u32 = 262_144;
const ETHERTYPE_IPV4: u16 = 0x0800;
const VLAN_TPIDS: [u16; 3] = [0x8100, 0x88A8, 0x9100];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Datagram {
    /// Capture time, Unix nanoseconds.
    pub t_ns: i64,
    pub src: SocketAddrV4,
    pub dst: SocketAddrV4,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReadStats {
    pub records: u64,
    pub packets: u64,
    pub not_ipv4: u64,
    pub not_udp: u64,
    pub fragments: u64,
    pub other_port: u64,
    pub snapped: u64,
    pub truncated: bool,
    pub linktype: u32,
    pub nanosecond: bool,
}

#[derive(Debug)]
pub enum PcapError {
    Io(io::Error),
    Format(String),
}

impl fmt::Display for PcapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PcapError::Io(e) => write!(f, "{e}"),
            PcapError::Format(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for PcapError {}

impl From<io::Error> for PcapError {
    fn from(e: io::Error) -> Self {
        PcapError::Io(e)
    }
}

/// Reads UDP datagrams from a pcap stream, one at a time.
pub struct PcapReader<R: Read> {
    r: R,
    big_endian: bool,
    frac_ns: i64,
    linktype: u32,
    port: Option<u16>,
    pub stats: ReadStats,
}

/// Fill `buf` completely; Ok(n < len) only at end of stream.
fn read_full<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(n)
}

impl<R: Read> PcapReader<R> {
    /// `port`: keep only datagrams sent to this UDP port.
    pub fn new(mut r: R, port: Option<u16>) -> Result<Self, PcapError> {
        let mut head = [0u8; 24];
        let n = read_full(&mut r, &mut head)?;
        if n < 24 {
            return Err(PcapError::Format(format!(
                "not a pcap file (only {n} bytes)"
            )));
        }
        let le = u32::from_le_bytes(head[0..4].try_into().unwrap());
        let (big_endian, frac_ns) = match le {
            PCAP_MAGIC_US => (false, 1000),
            PCAP_MAGIC_NS => (false, 1),
            _ => match u32::from_be_bytes(head[0..4].try_into().unwrap()) {
                PCAP_MAGIC_US => (true, 1000),
                PCAP_MAGIC_NS => (true, 1),
                _ if le == PCAPNG_MAGIC => return Err(PcapError::Format(
                    "pcapng is not supported; convert with `editcap -F pcap in.pcapng out.pcap`"
                        .into(),
                )),
                _ => {
                    return Err(PcapError::Format(format!(
                        "not a pcap file (magic 0x{le:08X})"
                    )))
                }
            },
        };
        let u32_at = |o: usize| {
            let b: [u8; 4] = head[o..o + 4].try_into().unwrap();
            if big_endian {
                u32::from_be_bytes(b)
            } else {
                u32::from_le_bytes(b)
            }
        };
        // The top bits of the link-type field can carry FCS information.
        let linktype = u32_at(20) & 0x03FF_FFFF;
        if ![
            LINKTYPE_NULL,
            LINKTYPE_ETHERNET,
            LINKTYPE_RAW,
            LINKTYPE_LINUX_SLL,
            LINKTYPE_IPV4,
            LINKTYPE_LINUX_SLL2,
        ]
        .contains(&linktype)
        {
            return Err(PcapError::Format(format!(
                "unsupported pcap link type {linktype}"
            )));
        }
        let stats = ReadStats {
            linktype,
            nanosecond: frac_ns == 1,
            ..Default::default()
        };
        Ok(Self {
            r,
            big_endian,
            frac_ns,
            linktype,
            port,
            stats,
        })
    }

    fn u32_of(&self, b: &[u8]) -> u32 {
        let b: [u8; 4] = b.try_into().unwrap();
        if self.big_endian {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        }
    }

    /// The next UDP datagram, or None at the end of the capture.
    pub fn next_datagram(&mut self) -> Result<Option<Datagram>, PcapError> {
        loop {
            let mut hdr = [0u8; 16];
            let n = read_full(&mut self.r, &mut hdr)?;
            if n == 0 {
                return Ok(None);
            }
            if n < 16 {
                self.stats.truncated = true;
                return Ok(None);
            }
            let (sec, frac, incl) = (
                self.u32_of(&hdr[0..4]),
                self.u32_of(&hdr[4..8]),
                self.u32_of(&hdr[8..12]),
            );
            if incl > MAX_RECORD {
                return Err(PcapError::Format(format!(
                    "corrupt record {}: length {incl} exceeds {MAX_RECORD}",
                    self.stats.records + 1
                )));
            }
            let mut data = vec![0u8; incl as usize];
            if read_full(&mut self.r, &mut data)? < data.len() {
                self.stats.truncated = true;
                return Ok(None);
            }
            self.stats.records += 1;
            if let Some(d) = self.udp_of(
                &data,
                sec as i64 * 1_000_000_000 + frac as i64 * self.frac_ns,
            ) {
                self.stats.packets += 1;
                return Ok(Some(d));
            }
        }
    }

    fn udp_of(&mut self, data: &[u8], t_ns: i64) -> Option<Datagram> {
        let be16 = |o: usize| u16::from_be_bytes([data[o], data[o + 1]]);
        let st = &mut self.stats;
        let Some(ip) = ipv4_offset(self.linktype, data)
            .filter(|&ip| data.len() >= ip + 20 && data[ip] >> 4 == 4)
        else {
            st.not_ipv4 += 1;
            return None;
        };
        let ihl = (data[ip] & 0x0F) as usize * 4;
        if ihl < 20 || data.len() < ip + ihl {
            st.not_ipv4 += 1;
            return None;
        }
        if be16(ip + 6) & 0x3FFF != 0 {
            st.fragments += 1;
            return None;
        }
        if data[ip + 9] != 17 {
            st.not_udp += 1;
            return None;
        }
        // Ethernet pads short frames and may carry an FCS: trust the IP
        // total length over the record length where it is set.
        let ip_total = be16(ip + 2) as usize;
        let ip_end = if ip_total == 0 {
            data.len()
        } else {
            data.len().min(ip + ip_total)
        };
        let udp = ip + ihl;
        if ip_end < udp + 8 {
            st.snapped += 1;
            return None;
        }
        let (sport, dport, ulen) = (be16(udp), be16(udp + 2), be16(udp + 4) as usize);
        if self.port.is_some_and(|p| p != dport) {
            st.other_port += 1;
            return None;
        }
        if ulen < 8 || udp + ulen > ip_end {
            st.snapped += 1;
            return None;
        }
        let addr = |o: usize| Ipv4Addr::new(data[o], data[o + 1], data[o + 2], data[o + 3]);
        Some(Datagram {
            t_ns,
            src: SocketAddrV4::new(addr(ip + 12), sport),
            dst: SocketAddrV4::new(addr(ip + 16), dport),
            payload: data[udp + 8..udp + ulen].to_vec(),
        })
    }
}

/// Offset of the IPv4 header inside one record, or None if not IPv4.
fn ipv4_offset(linktype: u32, data: &[u8]) -> Option<usize> {
    let be16 = |o: usize| (data.len() >= o + 2).then(|| u16::from_be_bytes([data[o], data[o + 1]]));
    match linktype {
        LINKTYPE_RAW | LINKTYPE_IPV4 => Some(0),
        LINKTYPE_ETHERNET => {
            let mut off = 12;
            for _ in 0..3 {
                let et = be16(off)?;
                if !VLAN_TPIDS.contains(&et) {
                    return (et == ETHERTYPE_IPV4).then_some(off + 2);
                }
                off += 4;
            }
            None
        }
        LINKTYPE_LINUX_SLL => (be16(14)? == ETHERTYPE_IPV4).then_some(16),
        LINKTYPE_LINUX_SLL2 => (data.len() >= 20 && be16(0)? == ETHERTYPE_IPV4).then_some(20),
        LINKTYPE_NULL => {
            let fam: [u8; 4] = data.get(..4)?.try_into().ok()?;
            (u32::from_le_bytes(fam) == 2 || u32::from_be_bytes(fam) == 2).then_some(4)
        }
        _ => None,
    }
}

/// Writes the recorder's format: ns, little-endian, LINKTYPE_RAW, with a
/// synthesized IPv4 + UDP header (IP checksum correct, UDP checksum 0).
pub struct PcapWriter<W: Write> {
    w: W,
    ip_id: u16,
}

impl<W: Write> PcapWriter<W> {
    pub fn new(mut w: W) -> io::Result<Self> {
        let mut h = Vec::with_capacity(24);
        h.extend_from_slice(&PCAP_MAGIC_NS.to_le_bytes());
        h.extend_from_slice(&2u16.to_le_bytes());
        h.extend_from_slice(&4u16.to_le_bytes());
        h.extend_from_slice(&0i32.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes());
        h.extend_from_slice(&65_535u32.to_le_bytes());
        h.extend_from_slice(&LINKTYPE_RAW.to_le_bytes());
        w.write_all(&h)?;
        Ok(Self { w, ip_id: 0 })
    }

    pub fn write(
        &mut self,
        t_ns: i64,
        src: SocketAddrV4,
        dst: SocketAddrV4,
        payload: &[u8],
    ) -> io::Result<()> {
        if payload.len() > 65_535 - 28 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "payload does not fit one IPv4 datagram",
            ));
        }
        let total = 28 + payload.len();
        let mut ip = [0u8; 28];
        ip[0] = 0x45;
        ip[2..4].copy_from_slice(&(total as u16).to_be_bytes());
        ip[4..6].copy_from_slice(&self.ip_id.to_be_bytes());
        ip[6..8].copy_from_slice(&0x4000u16.to_be_bytes());
        ip[8] = 64;
        ip[9] = 17;
        ip[12..16].copy_from_slice(&src.ip().octets());
        ip[16..20].copy_from_slice(&dst.ip().octets());
        let mut sum: u32 = ip[..20]
            .chunks(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]) as u32)
            .sum();
        while sum >> 16 != 0 {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        ip[10..12].copy_from_slice(&(!(sum as u16)).to_be_bytes());
        ip[20..22].copy_from_slice(&src.port().to_be_bytes());
        ip[22..24].copy_from_slice(&dst.port().to_be_bytes());
        ip[24..26].copy_from_slice(&((8 + payload.len()) as u16).to_be_bytes());
        let (sec, nsec) = (
            t_ns.div_euclid(1_000_000_000),
            t_ns.rem_euclid(1_000_000_000),
        );
        let mut rec = Vec::with_capacity(16);
        rec.extend_from_slice(&(sec as u32).to_le_bytes());
        rec.extend_from_slice(&(nsec as u32).to_le_bytes());
        rec.extend_from_slice(&(total as u32).to_le_bytes());
        rec.extend_from_slice(&(total as u32).to_le_bytes());
        self.w.write_all(&rec)?;
        self.w.write_all(&ip)?;
        self.w.write_all(payload)?;
        self.ip_id = self.ip_id.wrapping_add(1);
        Ok(())
    }

    pub fn into_inner(self) -> W {
        self.w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(last: u8, port: u16) -> SocketAddrV4 {
        SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, last), port)
    }

    #[test]
    fn writer_reader_round_trip() {
        let mut w = PcapWriter::new(Vec::new()).unwrap();
        w.write(
            1_700_000_000_123_456_789,
            addr(7, 4000),
            addr(1, 5566),
            b"hello",
        )
        .unwrap();
        w.write(
            1_700_000_000_623_456_789,
            addr(8, 4000),
            addr(1, 9999),
            b"other port",
        )
        .unwrap();
        let mut bytes = w.into_inner();
        bytes.extend_from_slice(&[1, 2, 3]); // half-written final record
        let mut r = PcapReader::new(&bytes[..], Some(5566)).unwrap();
        let d = r.next_datagram().unwrap().unwrap();
        assert_eq!(d.t_ns, 1_700_000_000_123_456_789);
        assert_eq!(d.payload, b"hello");
        assert_eq!(d.src, addr(7, 4000));
        assert!(r.next_datagram().unwrap().is_none());
        assert_eq!(r.stats.other_port, 1);
        assert!(r.stats.truncated);
        assert!(r.stats.nanosecond);
    }

    #[test]
    fn reads_microsecond_ethernet_big_endian() {
        // Hand-built: BE µs header, Ethernet + one VLAN tag + IPv4 + UDP.
        let mut f = Vec::new();
        f.extend_from_slice(&PCAP_MAGIC_US.to_be_bytes());
        f.extend_from_slice(&[
            0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 1,
        ]);
        let payload = b"xyz";
        let mut pkt = vec![0u8; 12];
        pkt.extend_from_slice(&[0x81, 0x00, 0x00, 0x05, 0x08, 0x00]);
        let mut ip = vec![
            0x45,
            0,
            0,
            (28 + payload.len()) as u8,
            0,
            0,
            0,
            0,
            64,
            17,
            0,
            0,
            10,
            0,
            0,
            9,
            10,
            0,
            0,
            1,
        ];
        ip.extend_from_slice(&[0x10, 0x00, 0x15, 0xBE, 0, (8 + payload.len()) as u8, 0, 0]);
        pkt.extend_from_slice(&ip);
        pkt.extend_from_slice(payload);
        pkt.extend_from_slice(&[0; 4]); // FCS / padding beyond the IP length
        f.extend_from_slice(&5u32.to_be_bytes());
        f.extend_from_slice(&250_000u32.to_be_bytes());
        f.extend_from_slice(&(pkt.len() as u32).to_be_bytes());
        f.extend_from_slice(&(pkt.len() as u32).to_be_bytes());
        f.extend_from_slice(&pkt);
        let mut r = PcapReader::new(&f[..], None).unwrap();
        let d = r.next_datagram().unwrap().unwrap();
        assert_eq!(d.t_ns, 5_250_000_000);
        assert_eq!(d.payload, payload);
        assert_eq!(d.dst, addr(1, 5566));
        assert_eq!(r.stats.linktype, LINKTYPE_ETHERNET);
    }

    #[test]
    fn rejects_non_pcap() {
        assert!(PcapReader::new(&[0u8; 10][..], None).is_err());
        let mut ng = [0u8; 24];
        ng[..4].copy_from_slice(&PCAPNG_MAGIC.to_le_bytes());
        assert!(PcapReader::new(&ng[..], None)
            .err()
            .unwrap()
            .to_string()
            .contains("pcapng"));
    }
}
