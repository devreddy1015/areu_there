//! The query of `GET /api/roomsim` (spec/livedemo.md §3.1, §6.3) and the
//! commands a client sends (§3.3). Pure: no socket, no clock.
//!
//! A refusal names the parameter and never echoes the value given. Its
//! message is a fixed text, so a client's value — a scene id, or a
//! person's id typed into the URL — cannot come back in it.

use std::sync::Arc;

use saans_locate::variant::Variant;
use serde_json::Value;

use super::replay::Recorded;
use super::Embedded;
use crate::shared::Mode;

/// Why a query or a command was refused: the parameter it names and a
/// fixed message (§3.4 `error`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Refusal {
    pub(crate) param: &'static str,
    pub(crate) message: &'static str,
}

/// The query's parameters, in the order they are checked after the keys.
const KEYS: [&str; 4] = ["scene", "variant", "seed", "replay"];

/// The query as given: each parameter once at most, values decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Query {
    pub(crate) scene: String,
    pub(crate) variant: Option<Variant>,
    pub(crate) seed: Option<u64>,
    pub(crate) replay: bool,
}

/// What a session runs: a simulation of an embedded scene (SIM), or the
/// recorded run of one (REPLAY).
pub(crate) enum Job {
    Sim {
        scene: Embedded,
        variant: Variant,
        seed: u64,
    },
    Replay {
        recorded: Arc<Recorded>,
    },
}

/// `application/x-www-form-urlencoded` decoding: `+` is a space and
/// `%XX` a byte; `None` for a broken escape or bytes that are not UTF-8.
fn decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' => {
                let hex = b
                    .get(i + 1..i + 3)
                    .filter(|h| h.iter().all(u8::is_ascii_hexdigit))?;
                out.push(u8::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok()?);
                i += 3;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

/// The query's `key=value` pairs; a key without `=` has the value `""`.
fn pairs(raw: Option<&str>) -> impl Iterator<Item = (&str, &str)> {
    raw.unwrap_or("")
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| p.split_once('=').unwrap_or((p, "")))
}

/// The mode an `error` message names before the query is known to be
/// valid (§3.4): `replay` when `replay=1` was given, else `sim`. Never
/// `live`.
pub(crate) fn requested_mode(raw: Option<&str>) -> Mode {
    let replay = pairs(raw)
        .any(|(k, v)| decode(k).as_deref() == Some("replay") && decode(v).as_deref() == Some("1"));
    if replay {
        Mode::Replay
    } else {
        Mode::Sim
    }
}

/// The query's syntax (§3.1): only the four known keys, each once, each
/// value well formed. An unknown key is named as `query`.
pub(crate) fn parse_query(raw: Option<&str>) -> Result<Query, Refusal> {
    let mut given: [Option<String>; 4] = Default::default();
    for (k, v) in pairs(raw) {
        let key = decode(k).ok_or(Refusal {
            param: "query",
            message: "the query is not valid URL encoding",
        })?;
        let Some(i) = KEYS.iter().position(|known| *known == key) else {
            return Err(Refusal {
                param: "query",
                message:
                    "unknown query parameter; the parameters are scene, variant, seed and replay",
            });
        };
        let value = decode(v).ok_or(Refusal {
            param: KEYS[i],
            message: "is not valid URL encoding",
        })?;
        if given[i].replace(value).is_some() {
            return Err(Refusal {
                param: KEYS[i],
                message: "is given more than once",
            });
        }
    }
    let [scene, variant, seed, replay] = given;
    let scene = scene.ok_or(Refusal {
        param: "scene",
        message: "scene is required: the id of an embedded example scene",
    })?;
    let variant = match variant {
        None => None,
        Some(v) => Some(Variant::parse(&v).ok_or(Refusal {
            param: "variant",
            message: "variant must be declared, with_walker or empty",
        })?),
    };
    let seed = match seed {
        None => None,
        Some(s) => Some(
            Some(s.as_str())
                .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or(Refusal {
                    param: "seed",
                    message: "seed must be an integer from 0 to 18446744073709551615",
                })?,
        ),
    };
    let replay = match replay.as_deref() {
        None => false,
        Some("1") => true,
        Some(_) => {
            return Err(Refusal {
                param: "replay",
                message: "replay must be 1 when it is given",
            })
        }
    };
    Ok(Query {
        scene,
        variant,
        seed,
        replay,
    })
}

/// The session a valid query asks for (§3.1, §6.3). `recorded` gives the
/// recorded run of an embedded scene, if `--room-replay` loaded one.
pub(crate) fn resolve(
    q: Query,
    recorded: impl Fn(Embedded) -> Option<Arc<Recorded>>,
) -> Result<Job, Refusal> {
    let scene = Embedded::find(&q.scene).ok_or(Refusal {
        param: "scene",
        message:
            "scene must be the id of an embedded example scene (GET /roomsim/scenes lists them)",
    })?;
    if !q.replay {
        return Ok(Job::Sim {
            scene,
            variant: q.variant.unwrap_or(Variant::Declared),
            seed: q.seed.unwrap_or(0),
        });
    }
    if q.variant.is_some() {
        return Err(Refusal {
            param: "variant",
            message: "the recorded run fixes the variant; leave it out with replay=1",
        });
    }
    if q.seed.is_some() {
        return Err(Refusal {
            param: "seed",
            message: "the recorded run fixes the seed; leave it out with replay=1",
        });
    }
    let recorded = recorded(scene).ok_or(Refusal {
        param: "replay",
        message: "no recorded run for this scene",
    })?;
    Ok(Job::Replay { recorded })
}

// ===================================================================
// Commands (§3.3)
// ===================================================================

/// A speed of §3.2, held as the quanta of 0.025 s one frame adds: 1, 2,
/// 4, 8, 16 or 32 for 0.25x to 8x.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Speed(u64);

impl Speed {
    pub(crate) const ALL: [Speed; 6] =
        [Speed(1), Speed(2), Speed(4), Speed(8), Speed(16), Speed(32)];
    /// 1x, the speed a session starts at.
    pub(crate) const DEFAULT: Speed = Speed(4);

    /// Quanta per frame, `4 * speed`.
    pub(crate) fn quanta(self) -> u64 {
        self.0
    }

    /// The speed on the wire: 0.25, 0.5, 1, 2, 4 or 8 (exact in binary).
    pub(crate) fn value(self) -> f64 {
        self.0 as f64 / 4.0
    }

    fn from_value(v: f64) -> Option<Speed> {
        Self::ALL.into_iter().find(|s| s.value() == v)
    }
}

/// A client's command, applied in arrival order before the next frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Command {
    Play,
    Pause,
    Speed(Speed),
    Restart,
}

/// The refusal of anything that is not one of the four commands.
const NOT_A_COMMAND: Refusal = Refusal {
    param: "cmd",
    message: "a command is {\"cmd\":\"play\"}, {\"cmd\":\"pause\"}, {\"cmd\":\"restart\"} or {\"cmd\":\"speed\",\"value\":v}",
};

/// One command (§3.3): exactly `{"cmd":"play"}`, `{"cmd":"pause"}`,
/// `{"cmd":"restart"}` or `{"cmd":"speed","value":v}` with `v` one of the
/// six speeds. Anything else is refused naming `cmd`, or `value` for a
/// bad speed.
pub(crate) fn parse_command(text: &str) -> Result<Command, Refusal> {
    let v: Value = serde_json::from_str(text).map_err(|_| NOT_A_COMMAND)?;
    let o = v.as_object().ok_or(NOT_A_COMMAND)?;
    let only = |keys: &[&str]| o.keys().all(|k| keys.contains(&k.as_str()));
    match o.get("cmd").and_then(Value::as_str) {
        Some("play") if only(&["cmd"]) => Ok(Command::Play),
        Some("pause") if only(&["cmd"]) => Ok(Command::Pause),
        Some("restart") if only(&["cmd"]) => Ok(Command::Restart),
        Some("speed") if only(&["cmd", "value"]) => o
            .get("value")
            .and_then(Value::as_f64)
            .and_then(Speed::from_value)
            .map(Command::Speed)
            .ok_or(Refusal {
                param: "value",
                message: "value must be one of the speeds 0.25, 0.5, 1, 2, 4 and 8",
            }),
        _ => Err(NOT_A_COMMAND),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refusal(raw: &str) -> Refusal {
        parse_query(Some(raw))
            .and_then(|q| resolve(q, |_| None))
            .err()
            .unwrap_or_else(|| panic!("{raw} was accepted"))
    }

    #[test]
    fn the_query_takes_the_defaults_of_section_3_1() {
        let q = parse_query(Some("scene=bedroom")).unwrap();
        assert_eq!(
            q,
            Query {
                scene: "bedroom".into(),
                variant: None,
                seed: None,
                replay: false
            }
        );
        match resolve(q, |_| None).ok().unwrap() {
            Job::Sim {
                scene,
                variant,
                seed,
            } => {
                assert_eq!(scene.id, "bedroom");
                assert_eq!(variant, Variant::Declared);
                assert_eq!(seed, 0);
            }
            Job::Replay { .. } => panic!("not a replay"),
        }
        let q = parse_query(Some(
            "seed=18446744073709551615&variant=with_walker&scene=hall-l-shaped",
        ))
        .unwrap();
        assert_eq!(q.variant, Some(Variant::WithWalker));
        assert_eq!(q.seed, Some(u64::MAX));
        for v in Variant::ALL {
            assert_eq!(
                parse_query(Some(&format!("scene=office&variant={}", v.as_str())))
                    .unwrap()
                    .variant,
                Some(v)
            );
        }
        // Form encoding is decoded.
        assert_eq!(
            parse_query(Some("scene=hall%2Dl-shaped&seed=%37"))
                .unwrap()
                .seed,
            Some(7)
        );
        assert_eq!(
            parse_query(Some("scene=hall%2Dl-shaped")).unwrap().scene,
            "hall-l-shaped"
        );
    }

    #[test]
    fn each_refusal_names_its_parameter_and_never_the_value() {
        let cases = [
            ("", "scene"),
            ("variant=empty", "scene"),
            ("scene=walker", "scene"),
            ("scene=bedroom&variant=walker", "variant"),
            ("scene=bedroom&variant=scene", "variant"),
            ("scene=bedroom&seed=-1", "seed"),
            ("scene=bedroom&seed=+1", "seed"),
            ("scene=bedroom&seed=18446744073709551616", "seed"),
            ("scene=bedroom&seed=", "seed"),
            ("scene=bedroom&replay=yes", "replay"),
            ("scene=bedroom&replay", "replay"),
            ("scene=bedroom&walker=1", "query"),
            ("scene=bedroom&%ZZ=1", "query"),
            ("scene=bedroom&scene=office", "scene"),
            ("scene=bedroom&seed=1&seed=2", "seed"),
            ("scene=be%FFdroom", "scene"),
            ("scene=bedroom&replay=1&variant=declared", "variant"),
            ("scene=bedroom&replay=1&seed=0", "seed"),
            ("scene=bedroom&replay=1", "replay"),
        ];
        for (raw, param) in cases {
            assert_eq!(refusal(raw).param, param, "{raw}");
        }
        assert_eq!(
            refusal("scene=bedroom&replay=1").message,
            "no recorded run for this scene"
        );
        // A given value never comes back (the messages are fixed texts).
        for (raw, param) in [
            ("scene=zzq", "scene"),
            ("scene=bedroom&variant=zzq", "variant"),
            ("scene=bedroom&seed=zzq", "seed"),
            ("scene=bedroom&replay=zzq", "replay"),
            ("scene=bedroom&zzq=1", "query"),
            ("scene=bedroom&scene=zzq", "scene"),
        ] {
            let r = refusal(raw);
            assert_eq!(r.param, param, "{raw}");
            assert!(!r.message.contains("zzq"), "{raw}: {r:?}");
        }
    }

    #[test]
    fn the_mode_of_an_early_error_is_the_requested_one() {
        assert_eq!(requested_mode(None), Mode::Sim);
        assert_eq!(requested_mode(Some("scene=bedroom")), Mode::Sim);
        assert_eq!(requested_mode(Some("scene=nope&replay=1")), Mode::Replay);
        assert_eq!(requested_mode(Some("replay=%31&bad=1")), Mode::Replay);
        assert_eq!(requested_mode(Some("replay=2")), Mode::Sim);
    }

    #[test]
    fn commands_parse_strictly() {
        assert_eq!(parse_command(r#"{"cmd":"play"}"#), Ok(Command::Play));
        assert_eq!(parse_command(r#"{"cmd":"pause"}"#), Ok(Command::Pause));
        assert_eq!(parse_command(r#"{"cmd":"restart"}"#), Ok(Command::Restart));
        let speeds: Vec<f64> = Speed::ALL.iter().map(|s| s.value()).collect();
        assert_eq!(speeds, [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]);
        assert_eq!(Speed::DEFAULT.value(), 1.0);
        for (s, text) in Speed::ALL.iter().zip(["0.25", "0.5", "1", "2.0", "4", "8"]) {
            assert_eq!(
                parse_command(&format!(r#"{{"cmd":"speed","value":{text}}}"#)),
                Ok(Command::Speed(*s))
            );
            assert_eq!(s.quanta() as f64, 4.0 * s.value());
        }
        for bad in ["0", "3", "16", "-1", "0.3", "\"2\"", "null"] {
            let r = parse_command(&format!(r#"{{"cmd":"speed","value":{bad}}}"#)).unwrap_err();
            assert_eq!(r.param, "value", "{bad}");
        }
        assert_eq!(
            parse_command(r#"{"cmd":"speed"}"#).unwrap_err().param,
            "value"
        );
        for bad in [
            "",
            "play",
            "[]",
            r#"{"cmd":"stop"}"#,
            r#"{"cmd":"PLAY"}"#,
            r#"{"cmd":1}"#,
            r#"{"value":2}"#,
            r#"{"cmd":"play","value":2}"#,
            r#"{"cmd":"speed","value":2,"x":1}"#,
        ] {
            assert_eq!(parse_command(bad), Err(NOT_A_COMMAND), "{bad}");
        }
    }
}
