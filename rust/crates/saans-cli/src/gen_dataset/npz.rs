//! The shard's bytes (spec/dataset.md §7.2): `.npy` members in a STORED
//! zip, written by hand so every byte is fixed and the shards are
//! byte-identical for any thread count (§9).
//!
//! - **`.npy`**: magic `\x93NUMPY`, version 1.0, u16 LE header length,
//!   then `{'descr': '<f4', 'fortran_order': False, 'shape': (60,), }`
//!   (a 1-tuple keeps its comma), space-padded and ended by `\n` so the
//!   data starts at a multiple of 64 bytes; little-endian C-order data.
//! - **zip**: members in the order given; each local header version 20,
//!   flags 0, method 0 (STORED), DOS time 0, date `0x0021` (1980-01-01),
//!   IEEE CRC-32, sizes in the local header (no data descriptor), no extra
//!   field, no comment; a central directory and end record likewise; no
//!   Zip64, so a shard of 4 GiB or more is refused before a byte is
//!   written.

use std::borrow::Cow;
use std::io::{self, Write};

use sha2::{Digest, Sha256};

/// The CRC-32 table of the IEEE polynomial (reflected `0xEDB88320`).
const CRC_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// A running IEEE CRC-32 (zip's, zlib's `crc32`).
#[derive(Debug, Clone, Copy)]
pub struct Crc32(u32);

impl Crc32 {
    pub fn new() -> Crc32 {
        Crc32(0xFFFF_FFFF)
    }

    pub fn update(&mut self, bytes: &[u8]) {
        let mut c = self.0;
        for &b in bytes {
            c = CRC_TABLE[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
        }
        self.0 = c;
    }

    pub fn value(self) -> u32 {
        self.0 ^ 0xFFFF_FFFF
    }
}

/// The CRC-32 of `bytes`.
#[cfg(test)]
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(bytes);
    c.value()
}

/// An array's elements, in C order, of one of the five dtypes §7.2
/// allows.
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    U8(Vec<u8>),
    I8(Vec<i8>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Data {
    /// numpy's `descr`: `|u1 |i1 <i4 <f4 <f8`.
    pub fn descr(&self) -> &'static str {
        match self {
            Data::U8(_) => "|u1",
            Data::I8(_) => "|i1",
            Data::I32(_) => "<i4",
            Data::F32(_) => "<f4",
            Data::F64(_) => "<f8",
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Data::U8(v) => v.len(),
            Data::I8(v) => v.len(),
            Data::I32(v) => v.len(),
            Data::F32(v) => v.len(),
            Data::F64(v) => v.len(),
        }
    }

    /// Bytes per element.
    pub fn itemsize(&self) -> usize {
        match self {
            Data::U8(_) | Data::I8(_) => 1,
            Data::I32(_) | Data::F32(_) => 4,
            Data::F64(_) => 8,
        }
    }

    /// The little-endian bytes of every element, in order.
    pub fn le_bytes(&self) -> Cow<'_, [u8]> {
        match self {
            Data::U8(v) => Cow::Borrowed(v),
            Data::I8(v) => Cow::Owned(v.iter().map(|&x| x as u8).collect()),
            Data::I32(v) => Cow::Owned(v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            Data::F32(v) => Cow::Owned(v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            Data::F64(v) => Cow::Owned(v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        }
    }
}

/// One `.npy` member: its name without `.npy`, its shape and its data
/// (`shape`'s product elements).
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub data: Data,
}

impl Array {
    /// An array, checking that `shape` holds exactly the data's elements
    /// (a mismatch is a bug in the caller, not data).
    pub fn new(name: &'static str, shape: &[usize], data: Data) -> Array {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "{name}: shape {shape:?} does not hold {} elements",
            data.len()
        );
        Array {
            name,
            shape: shape.to_vec(),
            data,
        }
    }
}

/// The `.npy` header of §7.2, from the magic through the `\n`: its length
/// is a multiple of 64.
pub fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape = match dims.as_slice() {
        [one] => format!("({one},)"),
        _ => format!("({})", dims.join(", ")),
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    // Magic (6) + version (2) + length (2) + dict + padding + "\n".
    let unpadded = 10 + dict.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    let header_len = dict.len() + pad + 1;
    let mut out = Vec::with_capacity(10 + header_len);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.resize(out.len() + pad, b' ');
    out.push(b'\n');
    debug_assert_eq!(out.len() % 64, 0);
    out
}

/// The member's file name in the zip, `NAME.npy`.
fn member_name(a: &Array) -> String {
    format!("{}.npy", a.name)
}

/// The bytes the zip of `arrays` will take, without writing it: the
/// local headers and members, the central directory and the end record.
pub fn zip_len(arrays: &[Array]) -> u64 {
    let mut n = 22u64;
    for a in arrays {
        let name = member_name(a).len() as u64;
        let member = npy_header(a.data.descr(), &a.shape).len() as u64
            + (a.data.len() * a.data.itemsize()) as u64;
        n += 30 + name + member + 46 + name;
    }
    n
}

/// A writer that counts and hashes what goes through it.
struct Hashing<W: Write> {
    inner: W,
    sha: Sha256,
    n: u64,
}

impl<W: Write> Write for Hashing<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let k = self.inner.write(buf)?;
        self.sha.update(&buf[..k]);
        self.n += k as u64;
        Ok(k)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// The DOS date of 1980-01-01 (§7.2), the earliest a zip can carry.
const DOS_DATE: u16 = 0x0021;
/// "Version needed to extract" (and "made by", MS-DOS): 2.0, STORED.
const ZIP_VERSION: u16 = 20;

/// Write `arrays` as one STORED zip of `.npy` members, in order, to `w`.
/// Returns the bytes written and their lower-case hex sha256. A zip of
/// 4 GiB or more, or of more than 65535 members, is refused before
/// anything is written (no Zip64).
pub fn write_npz(w: impl Write, arrays: &[Array]) -> io::Result<(u64, String)> {
    let total = zip_len(arrays);
    if total >= 1u64 << 32 || arrays.len() > usize::from(u16::MAX) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("a shard of {total} bytes and {} members needs Zip64, which spec/dataset.md §7.2 refuses", arrays.len()),
        ));
    }
    let mut w = Hashing {
        inner: w,
        sha: Sha256::new(),
        n: 0,
    };
    // (name, crc, size, local header offset) per member.
    let mut central: Vec<(String, u32, u32, u32)> = Vec::with_capacity(arrays.len());
    for a in arrays {
        let name = member_name(a);
        let header = npy_header(a.data.descr(), &a.shape);
        let data = a.data.le_bytes();
        let mut crc = Crc32::new();
        crc.update(&header);
        crc.update(&data);
        let crc = crc.value();
        let size = (header.len() + data.len()) as u32;
        let offset = w.n as u32;
        let mut local = Vec::with_capacity(30 + name.len());
        local.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        local.extend_from_slice(&ZIP_VERSION.to_le_bytes());
        local.extend_from_slice(&0u16.to_le_bytes()); // flags
        local.extend_from_slice(&0u16.to_le_bytes()); // method: STORED
        local.extend_from_slice(&0u16.to_le_bytes()); // DOS time
        local.extend_from_slice(&DOS_DATE.to_le_bytes());
        local.extend_from_slice(&crc.to_le_bytes());
        local.extend_from_slice(&size.to_le_bytes()); // compressed
        local.extend_from_slice(&size.to_le_bytes()); // uncompressed
        local.extend_from_slice(&(name.len() as u16).to_le_bytes());
        local.extend_from_slice(&0u16.to_le_bytes()); // extra field
        local.extend_from_slice(name.as_bytes());
        w.write_all(&local)?;
        w.write_all(&header)?;
        w.write_all(&data)?;
        central.push((name, crc, size, offset));
    }
    let cd_offset = w.n as u32;
    let mut cd = Vec::new();
    for (name, crc, size, offset) in &central {
        cd.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        cd.extend_from_slice(&ZIP_VERSION.to_le_bytes()); // made by
        cd.extend_from_slice(&ZIP_VERSION.to_le_bytes()); // needed
        cd.extend_from_slice(&0u16.to_le_bytes()); // flags
        cd.extend_from_slice(&0u16.to_le_bytes()); // method
        cd.extend_from_slice(&0u16.to_le_bytes()); // DOS time
        cd.extend_from_slice(&DOS_DATE.to_le_bytes());
        cd.extend_from_slice(&crc.to_le_bytes());
        cd.extend_from_slice(&size.to_le_bytes());
        cd.extend_from_slice(&size.to_le_bytes());
        cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes()); // extra field
        cd.extend_from_slice(&0u16.to_le_bytes()); // comment
        cd.extend_from_slice(&0u16.to_le_bytes()); // disk
        cd.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
        cd.extend_from_slice(&0u32.to_le_bytes()); // external attributes
        cd.extend_from_slice(&offset.to_le_bytes());
        cd.extend_from_slice(name.as_bytes());
    }
    w.write_all(&cd)?;
    let n = central.len() as u16;
    let mut end = Vec::with_capacity(22);
    end.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    end.extend_from_slice(&0u16.to_le_bytes()); // this disk
    end.extend_from_slice(&0u16.to_le_bytes()); // the central directory's disk
    end.extend_from_slice(&n.to_le_bytes());
    end.extend_from_slice(&n.to_le_bytes());
    end.extend_from_slice(&(cd.len() as u32).to_le_bytes());
    end.extend_from_slice(&cd_offset.to_le_bytes());
    end.extend_from_slice(&0u16.to_le_bytes()); // comment
    w.write_all(&end)?;
    w.flush()?;
    debug_assert_eq!(w.n, total);
    let sha = w
        .sha
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok((w.n, sha))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_is_the_ieee_one() {
        // The standard check value, and zlib.crc32 of a short text.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
        let mut c = Crc32::new();
        c.update(b"12345");
        c.update(b"6789");
        assert_eq!(c.value(), 0xCBF4_3926);
    }

    #[test]
    fn npy_headers_are_numpy_dicts_padded_to_64() {
        let h = npy_header("<f4", &[60]);
        assert_eq!(&h[..8], b"\x93NUMPY\x01\x00");
        assert_eq!(h.len() % 64, 0);
        let len = u16::from_le_bytes([h[8], h[9]]) as usize;
        assert_eq!(10 + len, h.len());
        let text = std::str::from_utf8(&h[10..]).unwrap();
        assert!(text.starts_with("{'descr': '<f4', 'fortran_order': False, 'shape': (60,), }"));
        assert!(text.ends_with('\n'));
        let h = npy_header("|u1", &[3, 4, 2]);
        assert!(std::str::from_utf8(&h[10..])
            .unwrap()
            .starts_with("{'descr': '|u1', 'fortran_order': False, 'shape': (3, 4, 2), }"));
        assert_eq!(h.len() % 64, 0);
        let h = npy_header("<f8", &[]);
        assert!(std::str::from_utf8(&h[10..])
            .unwrap()
            .contains("'shape': (), }"));
    }

    #[test]
    fn a_zip_has_the_fixed_bytes_and_its_predicted_length() {
        let arrays = vec![
            Array::new("a", &[3], Data::U8(vec![1, 0, 1])),
            Array::new("b", &[2, 2], Data::F32(vec![1.0, -2.0, f32::NAN, 0.5])),
            Array::new("c", &[2], Data::I8(vec![-1, 127])),
        ];
        let mut bytes = Vec::new();
        let (n, sha) = write_npz(&mut bytes, &arrays).unwrap();
        assert_eq!(n as usize, bytes.len());
        assert_eq!(n, zip_len(&arrays));
        assert_eq!(sha.len(), 64);
        // First local header: signature, version 20, flags 0, STORED,
        // time 0, date 0x0021, then the CRC of the member.
        assert_eq!(&bytes[..4], &[0x50, 0x4b, 0x03, 0x04]);
        assert_eq!(&bytes[4..14], &[20, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]);
        let size = u32::from_le_bytes(bytes[18..22].try_into().unwrap()) as usize;
        let member = &bytes[30 + 5..30 + 5 + size];
        assert_eq!(
            u32::from_le_bytes(bytes[14..18].try_into().unwrap()),
            crc32(member)
        );
        assert_eq!(&bytes[30..35], b"a.npy");
        // The dict is longer than 53 bytes, so the header pads to 128.
        assert_eq!(member.len(), 128 + 3);
        assert_eq!(&member[128..], &[1, 0, 1]);
        // The end record names 3 members and comes last.
        let end = &bytes[bytes.len() - 22..];
        assert_eq!(&end[..4], &[0x50, 0x4b, 0x05, 0x06]);
        assert_eq!(u16::from_le_bytes([end[10], end[11]]), 3);
        // The same arrays give the same bytes.
        let mut again = Vec::new();
        write_npz(&mut again, &arrays).unwrap();
        assert_eq!(again, bytes);
    }

    #[test]
    #[should_panic(expected = "does not hold")]
    fn a_shape_that_does_not_hold_the_data_is_a_bug() {
        Array::new("x", &[2, 2], Data::U8(vec![0; 3]));
    }
}
