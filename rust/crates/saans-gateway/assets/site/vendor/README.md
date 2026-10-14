# Vendored third-party code

Committed here rather than fetched at runtime, because the gateway is
served from a box in a building that may have no route to the internet.
Nothing in this directory is ours; nothing in it has been modified.

## three.js

| | |
| --- | --- |
| Version | 0.185.1 (REVISION 185) |
| Licence | MIT — `three-LICENSE.txt`, also recorded in the repository's `NOTICE` |
| Fetched | 2026-10-14, from `https://unpkg.com/three@0.185.1/build/` |

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `three.module.min.js` | 365,552 | `86bcee248b64f44bcfc23c331ae74619061957d59cab040171dcb6fb5900beb6` |
| `three.core.min.js` | 385,386 | `05b2609338c76cd65daf74f3ac515bc9a5045e1b3b33edc07d8c9bd55250fa90` |

`three.module.min.js` imports `./three.core.min.js` by relative path, so
both are served from `/vendor/` and the pair must stay together.

The minified builds are used rather than the 2.1 MB unminified pair
because the whole site is embedded in the binary with `include_bytes!`.

### Re-fetching

    curl -L -o three.module.min.js https://unpkg.com/three@0.185.1/build/three.module.min.js
    curl -L -o three.core.min.js   https://unpkg.com/three@0.185.1/build/three.core.min.js
    sha256sum three.module.min.js three.core.min.js   # must match the table above

### Why these files are exempt from part of the offline test

`site::tests::assets_fetch_nothing_remote` forbids a bare `https://` in
the text assets we write, so our own code cannot regress into fetching
something. three.js carries URLs in its licence header and comments, so
that rule cannot apply here unchanged.

The exemption is narrow and is about *fetching*, not about the
characters `http`. Every form that would actually retrieve something at
runtime — `from "http`, `import("http`, `fetch("http`, `new URL("http`,
`src="http`, `src="//`, `@import`, `url(http`, `//cdn` — stays banned in
**every** text asset, vendored included. Only a bare URL in prose is
allowed, and only in a file explicitly listed as vendored.
