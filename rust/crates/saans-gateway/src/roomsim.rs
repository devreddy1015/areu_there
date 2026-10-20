//! The static room demo (ADR-008): a three.js view of a declared scene.
//!
//! Compiled only with the `roomsim` cargo feature, which is off by
//! default (ADR-008 decision 13): a production gateway does not carry a
//! room view nobody asked for. The decision's other half — refusing at
//! config load to serve the view LIVE without the feature — arrives with
//! the Phase 5 live mode; today there is nothing live to refuse.
//!
//! Everything this module serves is SCENE data — room geometry written by
//! hand or imported from a scan, never inferred from WiFi — plus the code
//! that draws it. It serves no sensor data, no estimate and no person:
//! a scene is served through an allowlist of keys, so the people and
//! phone a scene file may carry for the simulator never leave the gateway
//! (decision 3), and neither does any key added to the format later until
//! someone decides it should. The page draws no one.
//!
//! The scenes are the hand-written examples in `spec/scenes/`, embedded at
//! build time, so the demo needs no configuration and no file system.
//! Binding a real, surveyed scene to a configured space is deliberately
//! not here: that is the Phase 5 plumbing, and such a scene describes a
//! real room and belongs behind the authenticated `/api` routes, not
//! these public ones.
//!
//! The page gets its own CSP, stricter than the dashboard's: no inline
//! script or style at all, `'self'` only, and Trusted Types, so the
//! browser itself refuses any `innerHTML`-style sink rather than relying
//! on a code comment. three.js is vendored (`assets/vendor/three-0.186.1/`,
//! see VENDOR.md there) so the gateway keeps working with no route out.

use crate::auth::sha256_hex;
use axum::extract::Path;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Map, Value};

pub(crate) const ROOMSIM_HTML: &str = include_str!("../assets/roomsim.html");
pub(crate) const ROOMSIM_JS: &str = include_str!("../assets/roomsim.js");
pub(crate) const ROOMSIM_CSS: &str = include_str!("../assets/roomsim.css");

/// The vendored bytes' identity in the served path: the upstream three.js
/// version plus a local revision. The files are cached as immutable, so
/// any change to a vendored byte — including the one-line local patch to
/// OrbitControls — must change this, or browsers keep the old file for a
/// year. VENDOR.md records the rule.
pub(crate) const VENDOR_REV: &str = "0.186.1-saans1";
const THREE_CORE: &str = include_str!("../assets/vendor/three-0.186.1/three.core.js");
const THREE_MODULE: &str = include_str!("../assets/vendor/three-0.186.1/three.module.js");
const ORBIT_CONTROLS: &str = include_str!("../assets/vendor/three-0.186.1/OrbitControls.js");

/// The example scenes: `(id, file contents)`. All three declare
/// `"provenance": "hand_written"` — they are illustrations, not surveys.
/// A test keeps this list equal to the files in `spec/scenes/`, every one
/// of which saans-scene validates.
pub(crate) const SCENES: &[(&str, &str)] = &[
    ("bedroom", include_str!("../../../../spec/scenes/bedroom.json")),
    ("office", include_str!("../../../../spec/scenes/office.json")),
    ("hall-l-shaped", include_str!("../../../../spec/scenes/hall-l-shaped.json")),
];

/// The only top-level keys a served scene carries. An allowlist rather
/// than a list of keys to strip: `people` and `phone` exist for the
/// simulator and must not be published, and a key added to the format
/// later stays private until someone adds it here on purpose.
pub(crate) const SERVED_KEYS: [&str; 7] = ["schema_version", "name", "provenance", "room", "walls", "objects", "nodes"];

/// The demo's policy. Compared with the dashboard's (`http.rs::CSP`) it
/// drops `'unsafe-inline'` from both `script-src` and `style-src` (every
/// script and stylesheet is a same-origin file), and adds Trusted Types
/// with no policies allowed, which makes `innerHTML`, `outerHTML`,
/// `insertAdjacentHTML`, `document.write` and string-to-script sinks throw
/// in the browser.
pub(crate) const ROOMSIM_CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; img-src data:; \
                                      connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; \
                                      require-trusted-types-for 'script'; trusted-types 'none'";

/// Routes for the demo, merged into the gateway's router. Stateless, so
/// they fit whatever state the main router carries.
pub(crate) fn routes<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/roomsim", get(page))
        .route("/roomsim/", get(page))
        .route("/roomsim/roomsim.js", get(|| async { asset(ROOMSIM_JS, "text/javascript; charset=utf-8", false) }))
        .route("/roomsim/roomsim.css", get(|| async { asset(ROOMSIM_CSS, "text/css; charset=utf-8", false) }))
        .route("/roomsim/vendor/{rev}/{file}", get(vendor))
        .route("/roomsim/scenes", get(scene_list))
        .route("/roomsim/scenes/{id}", get(scene))
}

fn common_headers(r: &mut Response) {
    let h = r.headers_mut();
    h.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    h.insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
}

async fn page() -> Response {
    let mut r = ROOMSIM_HTML.into_response();
    common_headers(&mut r);
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
    h.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static(ROOMSIM_CSP));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    r
}

/// A static file. Vendored files are named by `VENDOR_REV` in their path,
/// so they may be cached for good; this page's own files change with the
/// binary.
fn asset(body: &'static str, content_type: &'static str, immutable: bool) -> Response {
    let mut r = body.into_response();
    common_headers(&mut r);
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    let cache = if immutable { "public, max-age=31536000, immutable" } else { "no-store" };
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
    r
}

fn not_found() -> Response {
    let mut r = crate::http::error(StatusCode::NOT_FOUND, "no such endpoint");
    common_headers(&mut r);
    crate::http::no_store(r)
}

async fn vendor(Path((rev, file)): Path<(String, String)>) -> Response {
    if rev != VENDOR_REV {
        return not_found();
    }
    // An allowlist of three names: LICENSE, VENDOR.md and anything a path
    // trick might reach are not served.
    let body = match file.as_str() {
        "three.core.js" => THREE_CORE,
        "three.module.js" => THREE_MODULE,
        "OrbitControls.js" => ORBIT_CONTROLS,
        _ => return not_found(),
    };
    asset(body, "text/javascript; charset=utf-8", true)
}

/// A scene as served: only the `SERVED_KEYS` of the file. (Key order in
/// the response is serde_json's, alphabetical; nothing depends on it.)
pub(crate) fn served_scene(text: &str) -> Option<Value> {
    let v: Value = serde_json::from_str(text).ok()?;
    let obj = v.as_object()?;
    let kept: Map<String, Value> =
        SERVED_KEYS.iter().filter_map(|k| obj.get(*k).map(|v| ((*k).to_string(), v.clone()))).collect();
    Some(Value::Object(kept))
}

/// The list the page's scene picker shows. `sha256` is the digest of the
/// scene FILE as embedded — before the allowlist is applied — because its
/// job is to name the file in `spec/scenes/` that the view was drawn from
/// (`sha256sum spec/scenes/<id>.json` gives the same value), not to
/// checksum the response.
async fn scene_list() -> Response {
    let list: Vec<Value> = SCENES
        .iter()
        .filter_map(|(id, text)| {
            let v: Value = serde_json::from_str(text).ok()?;
            Some(json!({
                "id": id,
                "name": v.get("name")?,
                "provenance": v.get("provenance")?,
                "sha256": sha256_hex(text),
            }))
        })
        .collect();
    let mut r = Json(json!({ "label": "SCENE", "scenes": list })).into_response();
    common_headers(&mut r);
    crate::http::no_store(r)
}

async fn scene(Path(id): Path<String>) -> Response {
    let Some((_, text)) = SCENES.iter().find(|(sid, _)| *sid == id) else { return not_found() };
    let Some(v) = served_scene(text) else { return not_found() };
    let mut r = Json(v).into_response();
    common_headers(&mut r);
    crate::http::no_store(r)
}

/// Every module specifier a JavaScript file names in code — `from '…'`,
/// `from "…"`, `import('…')` and side-effect `import '…'` — skipping
/// comment lines, which in three.js carry usage examples.
#[cfg(test)]
pub(crate) fn module_specifiers(js: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in js.lines() {
        let mut t = line.trim_start();
        // A line that opens with a closed block comment may still carry
        // code after it: `/* note */ import x from './x.js';`.
        if t.starts_with("/*") {
            match t.find("*/") {
                Some(end) => t = t[end + 2..].trim_start(),
                None => continue,
            }
        }
        if t.starts_with('*') || t.starts_with("//") {
            continue;
        }
        for pat in [" from '", " from \"", "import('", "import(\"", "import '", "import \""] {
            let mut rest = t;
            while let Some(i) = rest.find(pat) {
                let after = &rest[i + pat.len()..];
                let quote = pat.chars().last().unwrap();
                if let Some(end) = after.find(quote) {
                    out.push(after[..end].to_string());
                }
                rest = after;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vendored files are pinned. A changed byte fails this test, so
    /// an upgrade or an accidental edit cannot happen silently; update
    /// VENDOR.md, these hashes and `VENDOR_REV` together.
    #[test]
    fn vendored_three_js_is_exactly_what_vendor_md_records() {
        let pinned = [
            ("three.core.js", THREE_CORE, "9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6"),
            ("three.module.js", THREE_MODULE, "9052042d676cb0fdc1ddfefe193053f34b7ac0513a616fdac4535d49987812ea"),
            ("OrbitControls.js", ORBIT_CONTROLS, "30386f7141d5ce5d5e0a3ab643e4eb16265f4564466108105ce449857f62fdda"),
        ];
        for (name, body, want) in pinned {
            assert_eq!(sha256_hex(body), want, "{name} differs from the version recorded in VENDOR.md");
        }
        let vendor_md = include_str!("../assets/vendor/three-0.186.1/VENDOR.md");
        for (_, _, want) in pinned {
            assert!(vendor_md.contains(want), "VENDOR.md does not record {want}");
        }
        assert!(vendor_md.contains(VENDOR_REV), "VENDOR.md does not name the served revision {VENDOR_REV}");
    }

    #[test]
    fn the_vendored_files_need_no_eval_and_reach_nothing_but_each_other() {
        for (name, body) in [("three.core.js", THREE_CORE), ("three.module.js", THREE_MODULE), ("OrbitControls.js", ORBIT_CONTROLS)] {
            // `import(` too: none of these files loads code dynamically, and
            // a dynamic import with a computed argument could not be checked.
            for banned in ["eval(", "new Function", "new Worker", "importScripts", "import("] {
                assert!(!body.contains(banned), "{name} contains {banned:?}, which the CSP would have to allow");
            }
            for spec in module_specifiers(body) {
                assert!(spec.starts_with("./"), "{name} imports {spec:?}, which is not same-origin relative");
            }
        }
        // Not vacuous: the files do link to each other, including through
        // the `export { … } from` line an import-only scan would miss.
        assert_eq!(module_specifiers(THREE_MODULE), ["./three.core.js", "./three.core.js"]);
        assert_eq!(module_specifiers(ORBIT_CONTROLS), ["./three.module.js"]);
        assert!(module_specifiers(THREE_CORE).is_empty());
    }

    #[test]
    fn the_page_imports_exactly_the_vendored_revision_the_gateway_serves() {
        let specs = module_specifiers(ROOMSIM_JS);
        assert_eq!(
            specs,
            [format!("./vendor/{VENDOR_REV}/three.module.js"), format!("./vendor/{VENDOR_REV}/OrbitControls.js")],
            "roomsim.js and VENDOR_REV disagree; the page would 404 on its own library"
        );
    }

    #[test]
    fn the_specifier_scan_finds_every_form_and_ignores_comments() {
        let js = "import { a } from './a.js';\nexport { b } from \"./b.js\";\nconst c = await import('./c.js');\nimport './d.js';\n * import { e } from 'three';\n// import x from 'y';\n/* note */ import { f } from './f.js';\n/* unclosed\n";
        assert_eq!(module_specifiers(js), ["./a.js", "./b.js", "./c.js", "./d.js", "./f.js"]);
    }

    /// The page's policy must be at least as strict as the dashboard's, and
    /// stricter where it can be: no inline script or style at all, and the
    /// browser enforcing the no-innerHTML rule.
    #[test]
    fn the_demo_csp_allows_nothing_inline_and_nothing_foreign() {
        let csp = ROOMSIM_CSP;
        assert!(csp.contains("default-src 'none'"));
        for loose in ["'unsafe-inline'", "'unsafe-eval'", "'unsafe-hashes'", "*", "http:", "https:", "blob:"] {
            assert!(!csp.contains(loose), "demo CSP contains {loose}");
        }
        for directive in csp.split(';') {
            let mut parts = directive.split_whitespace();
            let Some(name) = parts.next() else { continue };
            for source in parts {
                let ok = match name {
                    "require-trusted-types-for" => source == "'script'",
                    _ => matches!(source, "'none'" | "'self'" | "data:"),
                };
                assert!(ok, "unexpected CSP source {source:?} in {directive:?}");
            }
        }
        for required in ["frame-ancestors 'none'", "base-uri 'none'", "require-trusted-types-for 'script'", "trusted-types 'none'"] {
            assert!(csp.contains(required), "demo CSP lacks {required}");
        }
    }

    #[test]
    fn every_embedded_scene_is_served_through_the_allowlist() {
        for (id, text) in SCENES {
            let raw: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("{id}: {e}"));
            // The fixtures do carry people and a phone, so this is not vacuous.
            assert!(raw.get("people").is_some() && raw.get("phone").is_some(), "{id} fixture lost its people/phone");
            let served = served_scene(text).expect("parses");
            for k in served.as_object().unwrap().keys() {
                assert!(SERVED_KEYS.contains(&k.as_str()), "{id}: {k} reached the served scene");
            }
            // And nothing on the allowlist was lost: the geometry is intact.
            for k in SERVED_KEYS {
                assert_eq!(served.get(k), raw.get(k), "{id}: {k} changed on the way out");
            }
        }
    }

    #[test]
    fn every_embedded_scene_says_it_is_hand_written() {
        for (id, text) in SCENES {
            let v: Value = serde_json::from_str(text).unwrap();
            assert_eq!(v["provenance"], "hand_written", "{id}: the demo would print a false provenance");
        }
    }

    /// Nothing is served that saans-scene's tests never validated: those
    /// tests load every file in spec/scenes/, and this list is that set.
    #[test]
    fn embedded_scenes_are_exactly_the_files_in_spec_scenes() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes");
        let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let p = e.ok()?.path();
                (p.extension()? == "json").then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
            })
            .collect();
        on_disk.sort();
        let mut embedded: Vec<String> = SCENES.iter().map(|(id, _)| id.to_string()).collect();
        embedded.sort();
        assert_eq!(embedded, on_disk, "SCENES and spec/scenes/ disagree");
        for (id, text) in SCENES {
            let file = std::fs::read_to_string(dir.join(format!("{id}.json"))).unwrap();
            assert_eq!(file, *text, "{id}: embedded copy is stale; rebuild");
        }
    }
}
