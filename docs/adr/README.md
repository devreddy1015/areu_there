# Decision records

One file per decision, numbered in the order they were made. A decision
that changes an earlier one supersedes it in a new file; the old file is
not edited except to add a dated amendment, because an ADR whose content
was quietly corrected afterwards is a worse record than one that is
dated (see the amendment at the end of ADR-008 for what that looks like).

"Built" means the code does what the ADR says. Two are accepted and not
built, and say so in their status line; an ADR in that state for long is
how ADRs stop meaning anything, so `docs/NEXT.md` carries each as an item.

| ADR | Decision | Accepted | Built |
|---|---|---|---|
| [001](001-edge-private-features.md) | Raw CSI stays on the sensor; only the 68-byte FeatureFrame leaves it | 2026-10-13 | yes |
| [002](002-one-spec-three-implementations.md) | One spec, three implementations, held to each other by golden vectors; Python is the reference | 2026-10-13 | yes |
| [003](003-float32-edge-f64-gateway.md) | float32 on the edge, f64 on the gateway, with the tolerances that follow | 2026-10-13 | yes |
| [004](004-featureframe-v1.md) | FeatureFrame v1 wire format | 2026-10-13 | yes |
| [005](005-advisory-consented-abstaining.md) | Health outputs are advisory, consented per space, allowed to abstain, and carry no identities | 2026-10-13 | yes |
| [006](006-link-diversity.md) | Link diversity over subcarrier selection for blind spots; amended 2026-10-24 with the octave rule | 2026-10-13 | in code; its own consequences require a 1/2/3-link measurement, sim then hardware, not yet done |
| [007](007-no-learned-models-yet.md) | No learned models until the DSP ceiling is measured on real data | 2026-10-13 | n/a — a prohibition |
| [008](008-one-origin.md) | The site and the dashboard are one server | 2026-10-14 | yes |
| [009](009-static-copy.md) | The site is also published as a static copy that says it has no gateway behind it | 2026-10-14 | yes |
| [010](010-audit-retention.md) | The audit log is kept for a year, then deleted | 2026-10-19 | **no** |
| [011](011-multi-site.md) | One gateway per site; a hosted control plane is ruled out | 2026-10-19 | **no** — and nothing is scheduled |

Writing a new one: copy the shape of the existing files — title line,
`**Status:**` with a date, then Context, Decision, Consequences — and
argue from a specific cause. If it is not built when it is accepted, the
status line says **Decided, not built**, and this table says no.
