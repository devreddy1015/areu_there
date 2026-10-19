//! Wall-clock formatting without a date library: civil-date conversion
//! (H. Hinnant's algorithms), ISO 8601 with a fixed UTC offset, and a
//! parser for the subset the API accepts.

use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_unix_ms() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_millis() as i64,
        Err(e) => -(e.duration().as_millis() as i64),
    }
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// (year, month, day) of a day count since 1970-01-01.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Local day number (days since 1970-01-01 in the given offset).
pub fn local_day(unix_ms: i64, offset_min: i32) -> i64 {
    (unix_ms + offset_min as i64 * 60_000).div_euclid(86_400_000)
}

pub fn date_string(day: i64) -> String {
    let (y, m, d) = civil_from_days(day);
    format!("{y:04}-{m:02}-{d:02}")
}

/// `2026-10-13T23:05:01.500+05:30` (or `...Z` for offset 0).
pub fn iso8601(unix_ms: i64, offset_min: i32) -> String {
    let local = unix_ms + offset_min as i64 * 60_000;
    let day = local.div_euclid(86_400_000);
    let ms_of_day = local.rem_euclid(86_400_000);
    let (y, m, d) = civil_from_days(day);
    let (h, mi, s, ms) = (
        ms_of_day / 3_600_000,
        ms_of_day / 60_000 % 60,
        ms_of_day / 1000 % 60,
        ms_of_day % 1000,
    );
    let off = if offset_min == 0 {
        "Z".to_string()
    } else {
        let sign = if offset_min < 0 { '-' } else { '+' };
        let a = offset_min.unsigned_abs();
        format!("{sign}{:02}:{:02}", a / 60, a % 60)
    };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{ms:03}{off}")
}

/// Parse `YYYY-MM-DD`, or `YYYY-MM-DDTHH:MM[:SS[.fff]]` followed by `Z`,
/// `±HH:MM` or nothing (nothing = UTC). Returns Unix milliseconds.
pub fn parse_iso8601(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    let num = |from: usize, len: usize| -> Option<i64> {
        let part = s.get(from..from + len)?;
        if part.bytes().all(|c| c.is_ascii_digit()) {
            part.parse().ok()
        } else {
            None
        }
    };
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let (y, mo, d) = (num(0, 4)?, num(5, 2)?, num(8, 2)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let day = days_from_civil(y, mo as u32, d as u32);
    if b.len() == 10 {
        return Some(day * 86_400_000);
    }
    if !matches!(b[10], b'T' | b't' | b' ') || b.len() < 16 || b[13] != b':' {
        return None;
    }
    let (h, mi) = (num(11, 2)?, num(14, 2)?);
    let mut i = 16;
    let mut sec = 0i64;
    let mut ms = 0i64;
    if b.get(i) == Some(&b':') {
        sec = num(i + 1, 2)?;
        i += 3;
        if b.get(i) == Some(&b'.') {
            let start = i + 1;
            let mut j = start;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j == start {
                return None;
            }
            let frac = &s[start..j.min(start + 3)];
            ms = frac.parse::<i64>().ok()? * 10i64.pow(3 - frac.len() as u32);
            i = j;
        }
    }
    if h > 23 || mi > 59 || sec > 60 {
        return None;
    }
    let offset_min = match &s[i..] {
        "" | "Z" | "z" => 0,
        rest if rest.len() == 6
            && (rest.starts_with('+') || rest.starts_with('-'))
            && rest.as_bytes()[3] == b':' =>
        {
            let oh = num(i + 1, 2)?;
            let om = num(i + 4, 2)?;
            let v = oh * 60 + om;
            if rest.starts_with('-') {
                -v
            } else {
                v
            }
        }
        _ => return None,
    };
    Some(day * 86_400_000 + ((h * 60 + mi) * 60 + sec) * 1000 + ms - offset_min * 60_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_round_trip() {
        for z in [-800_000, -1, 0, 1, 19_000, 20_739, 2_000_000] {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
    }

    #[test]
    fn iso_format_and_parse() {
        let t = parse_iso8601("2026-10-13T23:05:01.5Z").unwrap();
        assert_eq!(iso8601(t, 0), "2026-10-13T23:05:01.500Z");
        assert_eq!(iso8601(t, 330), "2026-10-14T04:35:01.500+05:30");
        assert_eq!(parse_iso8601("2026-10-14T04:35:01.500+05:30"), Some(t));
        assert_eq!(iso8601(t, -90), "2026-10-13T21:35:01.500-01:30");
        assert_eq!(
            parse_iso8601("2026-10-13"),
            Some(days_from_civil(2026, 10, 13) * 86_400_000)
        );
        assert_eq!(parse_iso8601("2026-10-13T23:05"), Some(t - 1500));
        assert!(parse_iso8601("yesterday").is_none());
        assert!(parse_iso8601("2026-13-01").is_none());
        assert_eq!(date_string(local_day(t, 330)), "2026-10-14");
    }
}
