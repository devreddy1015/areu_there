# ADR-009: The site is also published as a static copy

**Status:** accepted, 2026-10-14

## Context
ADR-008 made the gateway the one server for the site and the dashboard.
That is still the product, but it means nobody can read the site without
building and running a Rust binary first. A public copy on GitHub Pages
fixes that, and Pages serves a project under a path prefix
(`/areu_there/`), where every absolute in-site URL — `/site.css`,
`/scenes/bedroom.json` — points at the host's root instead of ours.

## Decision
Every in-site URL the site writes is relative (`site.css`, `./`, `api/sim`).
Every route is one segment deep, so on the gateway a relative URL names
exactly what the absolute one did, and under any prefix it still names the
right file. `saans-gateway --export-site DIR` writes the copy from the same
`PAGES` and `ASSETS` table the router walks: no second list, no build step,
the same bytes. `.github/workflows/pages.yml` runs it on every push to
`main` that touches the site and deploys the result.

What a file host cannot carry, the copy says rather than imitates. `/demo`
asks `/api/sim`; on a 404 it asks `/api/health`, which every gateway routes
in every mode, and if that is missing too it says there is no gateway
behind this address and stops polling. `/app` is a page that exists only in
the copy (`site::STATIC_ONLY`) and says the dashboard is not on it.

## Consequences
- `site::tests::the_site_names_no_absolute_paths` and
  `routes_are_one_segment_deep` hold the two halves of that argument;
  `the_static_copy_is_complete` checks every link in the copy lands on a
  file in it.
- The copy is served without the gateway's CSP and other headers: a file
  host does not send them. It fetches nothing remote either way.
- A nested route (`/a/b`) would break relative URLs, and the test above
  says so rather than letting the copy break quietly.
