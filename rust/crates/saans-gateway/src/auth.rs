//! Authentication: sessions for people, bearer tokens for machines
//! (SAANS_PLAN.md §10.1).
//!
//! The gateway serves health data about identifiable spaces, so the
//! question "who is asking?" has to be answered before any of it is
//! copied into a response. [`Auth::principal`] answers it once, from the
//! request headers, and everything downstream works from the resulting
//! [`Principal`] and the predicates on it in [`crate::roles`].
//!
//! Two credentials, because there are two kinds of caller:
//!
//! - **Bearer token** (`Authorization: Bearer <token>`) for machines. Only
//!   the token's SHA-256 is configured, so the config file never holds a
//!   usable credential.
//! - **Session cookie** (`saans_session`) for people, issued by
//!   `POST /api/login` against an Argon2id hash and held in memory only.
//!   Sessions therefore end when the gateway restarts, which is the
//!   behaviour you want from a box that is restarted to change consent.
//!
//! When `auth_required()` is false — which the config permits only on a
//! loopback `http_addr` — every request is [`Principal::local_admin`].
//! That is the single-operator default and it is checked, not assumed:
//! binding the API to a network address without a `[[user]]` or
//! `[[token]]` is a configuration error (`config::validate_outputs`).

use crate::config::GatewayConfig;
use crate::roles::{Principal, Role};
use argon2::password_hash::{PasswordHash, PasswordVerifier};
use argon2::Argon2;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Cookie the browser session is carried in.
pub const SESSION_COOKIE: &str = "saans_session";
/// Sessions are swept whenever one is created, and at most this often, so
/// an idle gateway does not hold expired entries forever.
const SWEEP_EVERY: Duration = Duration::from_secs(60);
/// Failed logins sleep this long before answering. Argon2 already costs
/// far more than this; the point is that a *missing* user costs the same
/// as a wrong password, so the response time does not enumerate names.
const FAIL_DELAY: Duration = Duration::from_millis(250);

/// Why a request has no principal. Both are 401 to the caller; they are
/// distinguished for the audit log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denied {
    /// No `Authorization` header and no session cookie.
    Missing,
    /// A credential was presented and did not match.
    Invalid,
}

struct Session {
    principal: Principal,
    expires: Instant,
}

pub struct Auth {
    required: bool,
    /// (name, role, spaces, argon2 PHC string)
    users: Vec<(String, Role, Option<BTreeSet<String>>, String)>,
    /// (name, role, spaces, lowercase hex SHA-256)
    tokens: Vec<(String, Role, Option<BTreeSet<String>>, String)>,
    sessions: Mutex<HashMap<String, Session>>,
    last_sweep: Mutex<Instant>,
    session_ttl: Duration,
}

fn scope(spaces: &[String]) -> Option<BTreeSet<String>> {
    // Empty means "every space", which is not the same as "no spaces".
    (!spaces.is_empty()).then(|| spaces.iter().cloned().collect())
}

/// Length-independent comparison. Both arguments here are fixed-length
/// hex digests, so this only has to avoid an early-exit byte loop.
fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut diff = (a.len() ^ b.len()) as u8;
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    diff == 0
}

pub fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// 256 bits from the OS. A session id is a bearer credential for its
/// lifetime, so it must not come from a seeded PRNG.
fn random_id() -> Result<String, getrandom::Error> {
    let mut buf = [0u8; 32];
    getrandom::getrandom(&mut buf)?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

impl Auth {
    pub fn new(cfg: &GatewayConfig) -> Self {
        Self {
            required: cfg.auth_required(),
            users: cfg
                .users
                .iter()
                .map(|u| (u.name.clone(), u.role, scope(&u.spaces), u.password_hash.clone()))
                .collect(),
            tokens: cfg
                .tokens
                .iter()
                .map(|t| (t.name.clone(), t.role, scope(&t.spaces), t.sha256.to_ascii_lowercase()))
                .collect(),
            sessions: Mutex::new(HashMap::new()),
            last_sweep: Mutex::new(Instant::now()),
            session_ttl: Duration::from_secs_f64(cfg.auth.session_hours * 3600.0),
        }
    }

    pub fn required(&self) -> bool {
        self.required
    }

    pub fn sessions_open(&self) -> usize {
        self.sessions.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// The principal for a request, or why there is none.
    ///
    /// `cookies` is the raw `Cookie` header; `authorization` the raw
    /// `Authorization` header. Both are optional and either may be
    /// attacker-controlled, so neither is trusted beyond an exact match
    /// against configured material.
    pub fn principal(&self, authorization: Option<&str>, cookies: Option<&str>) -> Result<Principal, Denied> {
        if !self.required {
            return Ok(Principal::local_admin());
        }
        if let Some(raw) = authorization {
            let Some(token) = raw.strip_prefix("Bearer ").or_else(|| raw.strip_prefix("bearer ")) else {
                return Err(Denied::Invalid);
            };
            let presented = sha256_hex(token.trim());
            return self
                .tokens
                .iter()
                .find(|(_, _, _, sha)| ct_eq(sha, &presented))
                .map(|(name, role, spaces, _)| Principal { name: name.clone(), role: *role, spaces: spaces.clone() })
                .ok_or(Denied::Invalid);
        }
        let Some(id) = cookies.and_then(cookie_value) else {
            return Err(Denied::Missing);
        };
        let now = Instant::now();
        let sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        match sessions.get(&id) {
            Some(s) if s.expires > now => Ok(s.principal.clone()),
            _ => Err(Denied::Invalid),
        }
    }

    /// Verify a password and open a session. Returns the session id to
    /// set as a cookie. A wrong password and an unknown user are
    /// deliberately indistinguishable, in both answer and timing.
    pub fn login(&self, name: &str, password: &str) -> Option<(String, Principal)> {
        let found = self.users.iter().find(|(n, _, _, _)| n == name);
        let ok = found.is_some_and(|(_, _, _, phc)| {
            PasswordHash::new(phc).is_ok_and(|parsed| Argon2::default().verify_password(password.as_bytes(), &parsed).is_ok())
        });
        let Some((name, role, spaces, _)) = found.filter(|_| ok) else {
            // Hash a dummy so an unknown user costs roughly what a known
            // one does, then add the same fixed delay either way.
            if found.is_none() {
                let _ = PasswordHash::new(DUMMY_PHC)
                    .map(|p| Argon2::default().verify_password(password.as_bytes(), &p).is_ok());
            }
            std::thread::sleep(FAIL_DELAY);
            return None;
        };
        let principal = Principal { name: name.clone(), role: *role, spaces: spaces.clone() };
        let id = random_id().ok()?;
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        self.sweep(&mut sessions);
        sessions.insert(id.clone(), Session { principal: principal.clone(), expires: Instant::now() + self.session_ttl });
        Some((id, principal))
    }

    pub fn logout(&self, cookies: Option<&str>) {
        if let Some(id) = cookies.and_then(cookie_value) {
            self.sessions.lock().unwrap_or_else(|p| p.into_inner()).remove(&id);
        }
    }

    fn sweep(&self, sessions: &mut HashMap<String, Session>) {
        let now = Instant::now();
        let mut last = self.last_sweep.lock().unwrap_or_else(|p| p.into_inner());
        if now.duration_since(*last) < SWEEP_EVERY {
            return;
        }
        *last = now;
        sessions.retain(|_, s| s.expires > now);
    }
}

/// Argon2id hash of a value no password equals, used to keep the cost of
/// an unknown user close to the cost of a known one.
const DUMMY_PHC: &str = "$argon2id$v=19$m=19456,t=2,p=1$O044JbqQ2K4/IkWa+3MfeA$pj+Uh0i0ESgOZy6widkgctkj+GeYJfILyR8a+iiwlRc";

/// The `saans_session` value out of a `Cookie` header.
fn cookie_value(header: &str) -> Option<String> {
    header.split(';').find_map(|part| {
        let (k, v) = part.split_once('=')?;
        (k.trim() == SESSION_COOKIE).then(|| v.trim().to_string())
    })
}

/// `Set-Cookie` for a new session. `HttpOnly` keeps it out of reach of
/// page script, `SameSite=Strict` stops another site driving the API
/// with it, and `Secure` is set when the dashboard was loaded over TLS.
pub fn session_cookie(id: &str, max_age_s: u64, secure: bool) -> String {
    format!(
        "{SESSION_COOKIE}={id}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age_s}{}",
        if secure { "; Secure" } else { "" }
    )
}

pub fn clear_cookie(secure: bool) -> String {
    session_cookie("", 0, secure)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sha256("abc"), the FIPS 180-2 test vector: ground truth that does not
    /// come from this crate. Token hashes and the room demo's scene digests
    /// both rest on this function.
    #[test]
    fn sha256_hex_matches_the_fips_vector() {
        assert_eq!(sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    fn cfg(toml: &str) -> GatewayConfig {
        GatewayConfig::from_toml(toml).expect("config")
    }

    /// Argon2id hash of "correct horse", generated by `--hash-password`.
    const PHC: &str = "$argon2id$v=19$m=19456,t=2,p=1$rjAhW5UtYRHkEv+PZqEwzA$nnxjNuRf8MDhos1tvWzgI2VFTttDr2FWt2DxKWuCCVc";

    fn authed() -> Auth {
        Auth::new(&cfg(&format!(
            "http_addr = \"0.0.0.0:8080\"\n\
             [[space]]\nid = \"lab\"\nlinks = [{{ rx = 1, tx = 255 }}]\n\
             [[space]]\nid = \"rest\"\nlinks = [{{ rx = 2, tx = 255 }}]\n\
             [[user]]\nname = \"ops\"\nrole = \"ops\"\npassword_hash = \"{PHC}\"\n\
             [[token]]\nname = \"bms\"\nrole = \"occupancy\"\nspaces = [\"lab\"]\nsha256 = \"{}\"\n",
            sha256_hex("s3cret")
        )))
    }

    #[test]
    fn loopback_without_auth_is_local_admin() {
        let a = Auth::new(&cfg("http_addr = \"127.0.0.1:8080\"\n[[space]]\nid = \"lab\"\nlinks = [{ rx = 1, tx = 255 }]\n"));
        assert!(!a.required());
        assert_eq!(a.principal(None, None).unwrap(), Principal::local_admin());
    }

    #[test]
    fn a_network_address_rejects_anonymous_requests() {
        let a = authed();
        assert!(a.required());
        assert_eq!(a.principal(None, None), Err(Denied::Missing));
        assert_eq!(a.principal(Some("Bearer wrong"), None), Err(Denied::Invalid));
        assert_eq!(a.principal(Some("Basic abc"), None), Err(Denied::Invalid), "only Bearer is accepted");
        assert_eq!(a.principal(None, Some("saans_session=nope")), Err(Denied::Invalid));
        assert_eq!(a.principal(None, Some("other=1")), Err(Denied::Missing));
    }

    #[test]
    fn a_token_carries_its_role_and_space_scope() {
        let p = authed().principal(Some("Bearer s3cret"), None).expect("token accepted");
        assert_eq!((p.name.as_str(), p.role), ("bms", Role::Occupancy));
        assert!(p.can_see_space("lab") && !p.can_see_space("rest"));
        assert!(!p.can_see_vitals("lab", true), "occupancy never sees vitals");
    }

    #[test]
    fn login_opens_a_session_and_logout_closes_it() {
        let a = authed();
        assert!(a.login("ops", "wrong").is_none());
        assert!(a.login("nobody", "correct horse").is_none());
        let (id, p) = a.login("ops", "correct horse").expect("login");
        assert_eq!(p.role, Role::Ops);
        assert_eq!(a.sessions_open(), 1);
        let cookie = format!("{SESSION_COOKIE}={id}");
        assert_eq!(a.principal(None, Some(&cookie)).unwrap().name, "ops");
        // A header with other cookies around it still parses.
        assert!(a.principal(None, Some(&format!("theme=dark; {cookie}; x=1"))).is_ok());
        a.logout(Some(&cookie));
        assert_eq!(a.principal(None, Some(&cookie)), Err(Denied::Invalid));
        assert_eq!(a.sessions_open(), 0);
    }

    #[test]
    fn an_expired_session_is_not_a_principal() {
        let a = Auth::new(&cfg(&format!(
            "http_addr = \"0.0.0.0:8080\"\n[auth]\nsession_hours = 0.0000001\n\
             [[space]]\nid = \"lab\"\nlinks = [{{ rx = 1, tx = 255 }}]\n\
             [[user]]\nname = \"ops\"\nrole = \"ops\"\npassword_hash = \"{PHC}\"\n"
        )));
        let (id, _) = a.login("ops", "correct horse").expect("login");
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(a.principal(None, Some(&format!("{SESSION_COOKIE}={id}"))), Err(Denied::Invalid));
    }

    #[test]
    fn cookie_attributes_are_set() {
        let c = session_cookie("abc", 3600, true);
        for want in ["saans_session=abc", "HttpOnly", "SameSite=Strict", "Max-Age=3600", "Secure", "Path=/"] {
            assert!(c.contains(want), "{c:?} missing {want}");
        }
        assert!(!session_cookie("abc", 3600, false).contains("Secure"));
        assert!(clear_cookie(false).contains("Max-Age=0"));
    }

    #[test]
    fn ct_eq_matches_str_eq() {
        assert!(ct_eq("abc", "abc") && !ct_eq("abc", "abd") && !ct_eq("abc", "abcd") && !ct_eq("", "a"));
        assert!(ct_eq("", ""));
    }
}
