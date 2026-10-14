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
}

pub const ASSETS: &[Asset] = &[
    Asset {
        path: "/site.css",
        bytes: include_bytes!("../assets/site/site.css"),
        content_type: "text/css; charset=utf-8",
        cache: "no-cache",
    },
    Asset {
        path: "/site.js",
        bytes: include_bytes!("../assets/site/site.js"),
        content_type: "text/javascript; charset=utf-8",
        cache: "no-cache",
    },
    Asset {
        path: "/demo.js",
        bytes: include_bytes!("../assets/site/demo.js"),
        content_type: "text/javascript; charset=utf-8",
        cache: "no-cache",
    },
    Asset {
        path: "/fonts/geist-latin.woff2",
        bytes: include_bytes!("../assets/site/fonts/geist-latin.woff2"),
        content_type: "font/woff2",
        cache: "public, max-age=86400",
    },
    Asset {
        path: "/fonts/geist-mono-latin.woff2",
        bytes: include_bytes!("../assets/site/fonts/geist-mono-latin.woff2"),
        content_type: "font/woff2",
        cache: "public, max-age=86400",
    },
    // Served because the OFL requires the licence to travel with the
    // fonts, and a licence nobody can reach is not travelling with them.
    Asset {
        path: "/fonts/OFL.txt",
        bytes: include_bytes!("../assets/site/fonts/OFL.txt"),
        content_type: "text/plain; charset=utf-8",
        cache: "public, max-age=86400",
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
    /// elsewhere, no font pulled off a CDN.
    #[test]
    fn assets_fetch_nothing_remote() {
        for a in ASSETS {
            if !a.content_type.starts_with("text/css") && !a.content_type.starts_with("text/javascript") {
                continue;
            }
            let s = std::str::from_utf8(a.bytes).expect("text asset is utf-8");
            for bad in ["url(http", "url(\"http", "url('http", "@import url(h", "//cdn", "https://", "http://"] {
                assert!(!s.contains(bad), "{}: remote reference {bad:?}", a.path);
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
