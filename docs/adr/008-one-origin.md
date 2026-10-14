# ADR-008: The site and the dashboard are one server

**Status:** accepted, 2026-10-14

## Context
The project had two front doors. A static marketing site was served by
`python3 -m http.server` out of `site/`, and the gateway served its
dashboard from its own binary on another port. Neither linked to the
other, so there was no path from "what is this" to "watch it work", and
the only runnable thing the site could offer was a GitHub quick-start
anchor.

The site was also a single document: fourteen sections and a floating
scroll-spy rail, which meant every visitor paid for all of it to read any
of it, and a link to one section was a fragment that scrolled.

A demo page is the part that makes this concrete, and a demo page has to
reach `/api/sim` and `/api/spaces`. Across two origins that means CORS, a
second configuration surface, and a credential story for a page that is
meant to be public. Across one origin it means a relative `fetch`.

## Decision
The gateway serves both. `assets/site/` holds seven pages, a stylesheet,
two scripts and the fonts; `src/site.rs` holds one table that maps each to
a path, and the router walks that table. Everything is `include_str!` /
`include_bytes!`, so there is still no build step and deployment is still
copying one file onto a box.

`/` is the site. The dashboard moved to `/app`, and `site.js` forwards the
old `/#overview`-style fragments so saved links still arrive.

The pages are separate documents rather than one scrolling one. The rail
went with them: on a page you can read in a minute, a second floating nav
is furniture. "Where am I" is answered from `location.pathname`, which is
correct before any script has run.

`/demo` drives the gateway's own simulate mode. It shows the simulator's
ground truth beside what the real DSP recovered from the synthetic
channel, including the half-metre move that blinds it. Where the API is
absent — a gateway not in simulate mode, or one that requires sign-in —
the page says so and draws nothing.

## Consequences
- The offline guarantee needed a distinction the dashboard never did. A
  *subresource* (stylesheet, script, font, `@import`, `url()`) is fetched
  to render the page and must be local. An `<a href>` to the repository is
  somewhere a reader chooses to go, it breaks nothing when there is no
  route out, and it is how someone checks a claim against the code.
  `site::tests::pages_fetch_nothing_remote` bans the first and permits the
  second; the dashboard keeps its stricter grep, because it has no
  outbound links at all.
- The CSP had to gain `'self'` for scripts and styles, and `font-src
  'self'` — without the last one the self-hosted Geist files fall through
  to `default-src 'none'` and the site renders in a system font. The
  dashboard never noticed because it asks for no fonts.
- Seven pages share their chrome by duplication, not by a template. That
  is deliberate: it keeps every page readable with JavaScript off, and the
  build step at zero. Three tests hold the duplication honest — every page
  identifies itself, every internal link resolves, and every asset a page
  references is one the router serves.
- One port to open, one process to supervise, one place to put TLS.
- `--simulate` is now a user-facing mode, not only a developer one. It
  remains the only way `/api/sim` exists at all; a live gateway cannot
  serve a simulated number through those routes.

## What this does not change
The API is exactly as authenticated as it was. Every route that carries
data still goes through `principal()`, and `/demo` is public only in the
sense that the dashboard already was: it is a file with no data in it,
which then asks for data as whoever is asking. On a loopback gateway that
is the local admin; on a gateway that requires sign-in, the page says so
instead of inventing a room.
