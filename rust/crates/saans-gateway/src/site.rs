//! The public site: the pages a visitor reads before they ever sign in.
//!
//! These used to be a separate `python3 -m http.server` on another port,
//! which meant the project had two front doors and neither one mentioned
//! the other. They are served from this binary now, on the same origin as
//! the API, so that `/demo` can call `/api/sim` without CORS, a bookmark
//! keeps working, and deploying the product is still copying one file
//! onto a box.
//!
//! Every page is embedded with `include_str!`. There is no build step and
//! no bundler: what is in `assets/site/` is exactly what is served. That
//! is also what makes the offline guarantee checkable — [`tests`] greps
//! the embedded bytes rather than trusting a pipeline to have inlined
//! everything.
//!
//! The dashboard moved from `/` to `/app` to make room for the landing
//! page. `assets/site/site.js` forwards the old tab fragments, so a
//! `/#system` link someone saved still arrives where it meant to.

use axum::http::{header, HeaderValue};
use axum::response::{IntoResponse, Response};

/// The dashboard's policy, plus the three things a multi-file site needs
/// that a single inlined document did not.
///
/// `'self'` for scripts and styles: the stylesheet and the script are
/// real files shared by seven pages, rather than the same few thousand
/// lines pasted into each one.
///
/// `font-src 'self'`: without it the self-hosted Geist files fall through
/// to `default-src 'none'` and the site renders in a system font. The
/// dashboard never noticed because it asks for no fonts at all.
///
/// `'unsafe-inline'` stays on both, for two narrow reasons that are worth
/// naming rather than hiding: the one-line script in each `<head>` that
/// sets the `js` class before first paint, and the `style="--i:1"`
/// attributes that carry per-element animation offsets. Hashes would be
/// stricter, but a hash that has to be regenerated whenever a page is
/// edited is a hash that eventually gets loosened instead of updated.
pub const SITE_CSP: &str = "default-src 'none'; script-src 'self' 'unsafe-inline'; \
                            style-src 'self' 'unsafe-inline'; img-src data:; font-src 'self'; \
                            connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

/// One readable page at one path.
pub struct Page {
    /// The route. No trailing slash, no `.html`: the extension is an
    /// implementation detail of how the file is stored, not part of the
    /// address someone types or shares.
    pub path: &'static str,
    pub html: &'static str,
}

/// The site, in reading order. The router walks this, so adding a page is
/// adding a line here and a file beside the others — there is no second
/// place to register it and therefore no second place to forget.
pub const PAGES: &[Page] = &[
    Page { path: "/", html: include_str!("../assets/site/index.html") },
    Page { path: "/how-it-works", html: include_str!("../assets/site/how-it-works.html") },
    Page { path: "/demo", html: include_str!("../assets/site/demo.html") },
    Page { path: "/room-sim", html: include_str!("../assets/site/room-sim.html") },
    Page { path: "/evidence", html: include_str!("../assets/site/evidence.html") },
    Page { path: "/privacy", html: include_str!("../assets/site/privacy.html") },
    Page { path: "/hardware", html: include_str!("../assets/site/hardware.html") },
    Page { path: "/start", html: include_str!("../assets/site/start.html") },
];

/// A file the pages reference. Text and binary both, because the fonts
/// are the reason `font-src` exists in [`SITE_CSP`] at all.
pub struct Asset {
    pub path: &'static str,
    pub bytes: &'static [u8],
    pub content_type: &'static str,
    /// Fonts are named after their content and change only when the
    /// subset is regenerated, so they may be held for a day. The
    /// stylesheet and the script share a name across every edit, so they
    /// are revalidated every time: a stale `site.js` against a fresh page
    /// is the kind of bug that wastes an afternoon.
    pub cache: &'static str,
    /// Third-party code committed under `assets/site/vendor/` exactly as
    /// it was published, and never edited here. The offline test reads
    /// this flag, and nothing else, to decide whether a bare URL sitting
    /// in a licence header or a comment is allowed; every form that
    /// would actually fetch something stays banned either way. It is
    /// written out per file rather than derived from the bytes, so
    /// setting it on a file we wrote would be writing ourselves a
    /// permission slip in plain sight.
    pub vendored: bool,
}

pub const ASSETS: &[Asset] = &[
    Asset {
        path: "/site.css",
        bytes: include_bytes!("../assets/site/site.css"),
        content_type: "text/css; charset=utf-8",
        cache: "no-cache",
        vendored: false,
    },
    Asset {
        path: "/site.js",
        bytes: include_bytes!("../assets/site/site.js"),
        content_type: "text/javascript; charset=utf-8",
        cache: "no-cache",
        vendored: false,
    },
    Asset {
        path: "/demo.js",
        bytes: include_bytes!("../assets/site/demo.js"),
        content_type: "text/javascript; charset=utf-8",
        cache: "no-cache",
        vendored: false,
    },
    // The room viewer. A module, because it imports three.js; the
    // `type="module"` is on the tag in the page, not here. Revalidated
    // every time, like the other scripts we write.
    Asset {
        path: "/room-sim.js",
        bytes: include_bytes!("../assets/site/room-sim.js"),
        content_type: "text/javascript; charset=utf-8",
        cache: "no-cache",
        vendored: false,
    },
    Asset {
        path: "/fonts/geist-latin.woff2",
        bytes: include_bytes!("../assets/site/fonts/geist-latin.woff2"),
        content_type: "font/woff2",
        cache: "public, max-age=86400",
        vendored: false,
    },
    Asset {
        path: "/fonts/geist-mono-latin.woff2",
        bytes: include_bytes!("../assets/site/fonts/geist-mono-latin.woff2"),
        content_type: "font/woff2",
        cache: "public, max-age=86400",
        vendored: false,
    },
    // Served because the OFL requires the licence to travel with the
    // fonts, and a licence nobody can reach is not travelling with them.
    Asset {
        path: "/fonts/OFL.txt",
        bytes: include_bytes!("../assets/site/fonts/OFL.txt"),
        content_type: "text/plain; charset=utf-8",
        cache: "public, max-age=86400",
        vendored: false,
    },
    // three.js 0.185.1, unmodified, MIT. `three.module.min.js` imports
    // `./three.core.min.js` by relative path, which the browser resolves
    // against the directory it was served from: the two have to sit
    // under the same served directory or the import 404s and the page
    // renders nothing. `vendored_relative_imports_resolve` holds them
    // together. Held for a day, like the fonts — the bytes change only
    // when the pinned version does, and the version is changed by hand.
    Asset {
        path: "/vendor/three.module.min.js",
        bytes: include_bytes!("../assets/site/vendor/three.module.min.js"),
        content_type: "text/javascript; charset=utf-8",
        cache: "public, max-age=86400",
        vendored: true,
    },
    Asset {
        path: "/vendor/three.core.min.js",
        bytes: include_bytes!("../assets/site/vendor/three.core.min.js"),
        content_type: "text/javascript; charset=utf-8",
        cache: "public, max-age=86400",
        vendored: true,
    },
    // Served for the reason the OFL is: the MIT licence asks that the
    // notice travel with the code, and a licence nobody can reach is not
    // travelling with it. It is in the repository's NOTICE as well; this
    // is the copy a reader of the running site can open.
    Asset {
        path: "/vendor/three-LICENSE.txt",
        bytes: include_bytes!("../assets/site/vendor/three-LICENSE.txt"),
        content_type: "text/plain; charset=utf-8",
        cache: "public, max-age=86400",
        vendored: true,
    },
    // The example scenes, served from the same files `saans-sim` and the
    // Python reader compile in, so a room cannot mean one thing to the
    // page and another to the simulator. A scene is typed in: every
    // dimension and every label in these files was written by a person,
    // and none of it was measured by a radio. Revalidated every time,
    // like the stylesheet, because they keep their names while they are
    // edited.
    Asset {
        path: "/scenes/bedroom.json",
        bytes: include_bytes!("../../../../scenes/bedroom.json"),
        content_type: "application/json",
        cache: "no-cache",
        vendored: false,
    },
    Asset {
        path: "/scenes/office.json",
        bytes: include_bytes!("../../../../scenes/office.json"),
        content_type: "application/json",
        cache: "no-cache",
        vendored: false,
    },
    Asset {
        path: "/scenes/hall-l.json",
        bytes: include_bytes!("../../../../scenes/hall-l.json"),
        content_type: "application/json",
        cache: "no-cache",
        vendored: false,
    },
    // The material table the scenes name by string. Served because
    // `/room-sim` shows the electrical constants of whatever object a
    // reader clicks, and a table it cannot fetch is a panel that would
    // have to either invent a permittivity or go blank. `fabric` has a
    // null row on purpose — ITU-R P.2040 has no fabric — and the page
    // says so rather than leaving a gap.
    Asset {
        path: "/scenes/materials.json",
        bytes: include_bytes!("../../../../scenes/materials.json"),
        content_type: "application/json",
        cache: "no-cache",
        vendored: false,
    },
];

/// Shared hardening for anything the site serves. The dashboard sets the
/// same three headers; they are repeated rather than shared because the
/// CSP differs and a helper that took the CSP as an argument would be the
/// same number of lines with one more place to pass the wrong one.
fn harden(mut r: Response, csp: &'static str, cache: &'static str) -> Response {
    let h = r.headers_mut();
    h.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static(csp));
    h.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    h.insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
    r
}

pub fn page_response(html: &'static str) -> Response {
    let mut r = html.into_response();
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
    harden(r, SITE_CSP, "no-cache")
}

pub fn asset_response(a: &'static Asset) -> Response {
    let mut r = a.bytes.into_response();
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(a.content_type));
    harden(r, SITE_CSP, a.cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The site must render with no route to the internet. A sensor
    /// gateway lives in a building that may have no way out, and a
    /// stylesheet that fails to load is a page nobody can read.
    ///
    /// The distinction this test draws, which the dashboard's equivalent
    /// did not have to: a *subresource* is fetched to render the page and
    /// must be local, while an `<a href>` is somewhere a reader chooses
    /// to go. Links to the repository are the whole point of publishing
    /// the work — they are how someone checks a claim against the code —
    /// and forbidding them would be enforcing offline-ness against the
    /// wrong thing.
    #[test]
    fn pages_fetch_nothing_remote() {
        const SUBRESOURCE: [&str; 6] =
            ["src=\"http", "src=\"//", "@import", "url(http", "url(\"http", "url('http"];
        for p in PAGES {
            for bad in SUBRESOURCE {
                assert!(!p.html.contains(bad), "{}: remote subresource {bad:?}", p.path);
            }
            // A page can fetch too: /room-sim carries an inline script,
            // and ASSETS never sees it. Same normalised patterns as the
            // asset check, so an inline `fetch("https://...")` cannot
            // slip through the gap between the two tests.
            let flat = normalise(p.html);
            for bad in FETCHES {
                assert!(!flat.contains(bad), "{}: fetches from elsewhere, {bad:?}", p.path);
            }
            // `<link rel=stylesheet href=...>` is a subresource even
            // though it spells its address `href`. Check every <link>
            // separately rather than banning remote `href=` outright.
            for (i, _) in p.html.match_indices("<link ") {
                let tag = &p.html[i..p.html[i..].find('>').map_or(p.html.len(), |e| i + e)];
                assert!(
                    !(tag.contains("href=\"http") || tag.contains("href=\"//")),
                    "{}: remote <link>: {tag}",
                    p.path
                );
            }
        }
    }

    /// Stylesheets and scripts get the same treatment: no `@import` from
    /// elsewhere, no font pulled off a CDN, no module fetched from
    /// somewhere else at import time.
    ///
    /// What is banned is *fetching*, not the characters `http`. The list
    /// below is every spelling that actually retrieves something while
    /// the page is running, and it applies to every text asset we serve,
    /// vendored included: a vendored file that pulled a module off a CDN
    /// would break the offline guarantee exactly as one of ours would.
    ///
    /// A bare `https://` retrieves nothing — it sits in a licence header
    /// or a comment, and three.js carries two of them. So the bare-URL
    /// rule is kept where it was before three.js arrived: on the
    /// stylesheets and scripts we write, which are the files that can
    /// regress into fetching when someone edits the syntax around a URL
    /// that was "only a comment". The allowance is per file and
    /// explicit. It is claimed by `Asset::vendored`, which is typed out
    /// in the table above rather than inferred from a path at run time
    /// or from the bytes, so granting it to one of our own files is a
    /// visible edit to the table — and `vendored_assets_are_served`
    /// checks that the files claiming it are real and reachable.
    ///
    /// Two kinds of file are therefore outside the bare-URL rule and
    /// inside the fetch ban: anything vendored, and the licences, which
    /// are text we did not write and cannot run. `/fonts/OFL.txt` was
    /// already in that position before three.js arrived — the rule has
    /// never reached it — and `/vendor/three-LICENSE.txt` joins it.
    /// The fetching forms, written once against NORMALISED text.
    ///
    /// Listing spellings by hand is how this kind of check rots: the
    /// first version missed `from  "http` and `` fetch(`http ``, which
    /// is exactly where a vendored file would hide one, because
    /// vendored files are where the bare-URL backstop is switched off.
    /// [`normalise`] removes the spacing and collapses the three quote
    /// characters, so each form below needs exactly one spelling.
    const FETCHES: [&str; 9] =
        ["from\"http", "import(\"http", "fetch(\"http", "newurl(\"http", "src=\"http", "src=\"//", "@import", "url(\"http", "//cdn"];

    /// Lower-cases, drops whitespace, and turns `'` and a backtick into
    /// `"`, so one pattern matches every way a fetch can be written.
    /// It is deliberately lossy: it is a tripwire, not a parser.
    fn normalise(s: &str) -> String {
        s.chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| match c {
                '\'' | '`' => '"',
                c => c.to_ascii_lowercase(),
            })
            .collect()
    }

    #[test]
    fn assets_fetch_nothing_remote() {
        const BARE_URL: [&str; 2] = ["https://", "http://"];
        for a in ASSETS {
            // JSON counts: the scene files are fetched at runtime by the
            // Room Sim page, so they can carry a URL like anything else.
            if !a.content_type.starts_with("text/") && !a.content_type.starts_with("application/json") {
                continue;
            }
            let s = std::str::from_utf8(a.bytes).expect("text asset is utf-8");
            let flat = normalise(s);
            for bad in FETCHES {
                assert!(!flat.contains(bad), "{}: fetches from elsewhere, {bad:?}", a.path);
            }
            let ours_and_executable = !a.vendored
                && (a.content_type.starts_with("text/css")
                    || a.content_type.starts_with("text/javascript"));
            if ours_and_executable {
                for bad in BARE_URL {
                    assert!(!s.contains(bad), "{}: remote reference {bad:?}", a.path);
                }
            }
        }
    }

    /// The allowance above is only honest if the files claiming it are
    /// files somebody can actually load. An empty vendor list would mean
    /// the exemption has no subject and the rule was widened for
    /// nothing; a `vendored: true` on a path the router never serves
    /// would be a permission slip written for a file that does not
    /// exist. Both are checked here, through the real handler rather
    /// than by reading the table twice.
    #[test]
    fn vendored_assets_are_served() {
        let vendored: Vec<&'static Asset> = ASSETS.iter().filter(|a| a.vendored).collect();
        assert!(!vendored.is_empty(), "nothing is vendored: the bare-URL allowance has no subject");
        for a in vendored {
            assert!(a.path.starts_with("/vendor/"), "{}: vendored but not under /vendor/", a.path);
            assert!(!a.bytes.is_empty(), "{}: vendored and empty", a.path);
            let r = asset_response(a);
            assert_eq!(r.status(), axum::http::StatusCode::OK, "{}: not served", a.path);
            assert_eq!(
                r.headers().get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()),
                Some(a.content_type),
                "{}: served as something else",
                a.path
            );
        }
        // The licence has to be reachable, not merely present: MIT asks
        // for the notice to travel with the code it covers.
        assert!(
            ASSETS.iter().any(|a| a.path == "/vendor/three-LICENSE.txt"),
            "the vendored licence is not served"
        );
    }

    /// A relative import resolves against the directory the importing
    /// file was served from, not against the repository. So a vendored
    /// module that imports `./something.js` needs that sibling served
    /// from the same directory, or the import 404s at run time and the
    /// page fails with nothing in the log but a missing file.
    #[test]
    fn vendored_relative_imports_resolve() {
        for a in ASSETS.iter().filter(|a| a.vendored && a.content_type.starts_with("text/javascript")) {
            let s = std::str::from_utf8(a.bytes).expect("text asset is utf-8");
            let dir = &a.path[..=a.path.rfind('/').expect("asset paths are absolute")];
            for pat in ["from\"", "from '", "from \"", "from'", "import(\"", "import('"] {
                let quote = pat.chars().last().expect("pattern ends in a quote");
                for (i, _) in s.match_indices(pat) {
                    let rest = &s[i + pat.len()..];
                    let end = rest.find(quote).expect("closing quote");
                    let spec = &rest[..end];
                    let target = match spec.strip_prefix("./") {
                        Some(name) => format!("{dir}{name}"),
                        None if spec.starts_with('/') => spec.to_string(),
                        // A bare specifier (`from"three"`) needs an
                        // import map to resolve, and we ship none.
                        None => panic!("{}: unresolvable module specifier {spec:?}", a.path),
                    };
                    assert!(
                        ASSETS.iter().any(|b| b.path == target),
                        "{}: imports {spec:?}, which no asset serves at {target}",
                        a.path
                    );
                }
            }
        }
    }

    /// Every asset a page asks for is an asset the router serves. A 404
    /// on the stylesheet is a site that looks broken in exactly the
    /// deployment where nobody is watching the logs.
    #[test]
    fn every_referenced_asset_exists() {
        for p in PAGES {
            for (i, _) in p.html.match_indices("href=\"/").chain(p.html.match_indices("src=\"/")) {
                let rest = &p.html[i..];
                let start = rest.find('"').expect("attribute quote") + 1;
                let end = start + rest[start..].find('"').expect("closing quote");
                let target = &rest[start..end];
                // Pages are checked by the route table; only files are
                // our business here.
                if !target.contains('.') {
                    continue;
                }
                assert!(
                    ASSETS.iter().any(|a| a.path == target),
                    "{} references {target}, which no asset serves",
                    p.path
                );
            }
        }
    }

    /// Links between pages have to land somewhere. A dead internal link
    /// is the one kind of breakage a reader blames on the product.
    #[test]
    fn internal_links_resolve() {
        for p in PAGES {
            for (i, _) in p.html.match_indices("href=\"/") {
                let rest = &p.html[i..];
                let start = rest.find('"').expect("attribute quote") + 1;
                let end = start + rest[start..].find('"').expect("closing quote");
                let target = rest[start..end].split('#').next().unwrap_or("/");
                if target.contains('.') || target == "/app" {
                    continue; // assets are checked above; /app is the dashboard
                }
                assert!(
                    PAGES.iter().any(|q| q.path == target),
                    "{p} links to {target}, which is not a page",
                    p = p.path
                );
            }
        }
    }

    /// The disclaimer is not decoration. Every page carries a number that
    /// came out of a simulator, so every page says so where it can be
    /// read, not only in a footer somebody scrolled past.
    #[test]
    fn every_page_says_the_numbers_are_simulated() {
        for p in PAGES {
            let t = p.html.to_ascii_lowercase();
            assert!(
                t.contains("simulat") || t.contains("synthetic"),
                "{}: no visible word about where its numbers come from",
                p.path
            );
            assert!(p.html.contains("Not a medical device"), "{}: missing the medical-device line", p.path);
        }
    }

    /// Each page marks itself, and the marker matches a nav entry — this
    /// is what puts the underline under the right word.
    #[test]
    fn every_page_identifies_itself() {
        for p in PAGES {
            assert!(p.html.contains("<body data-page=\""), "{}: no data-page on <body>", p.path);
            assert!(p.html.contains("<main id=\"main\">"), "{}: no <main id=main> for the skip link", p.path);
        }
    }
}
