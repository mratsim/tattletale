# Writing Docs — Examples

Banned vocabulary with replacements, and before/after pairs from actual slop
seen in this repo.

## Banned vocabulary (hard blocklist)

Campaign terms and test-theater metaphors that must NEVER appear in any
committed doc comment, module header, or prose. A reviewer greps for
these, so even a domain-legitimate use should be rephrased to the
replacement below. Terms ruled acceptable by the operator and removed
from this list: fingerprint (format names and prose), sentinel (the CS
term of art), harness, smoke.

| Banned | Replacement |
|---|---|
| `pin` / `pinned` | "verified against", "checked by", "locked" (never "pins behavior"). CUDA page-locked memory: write "page-locked", never "pinned" |
| `gate` / `gates` / `gating` (for tests) | "test", "check", "assert" — model-architecture names (gate.weight, sigmoid gate) are exempt |
| `draw` / `draws` (test/harness verb) | "read", "take", "use" |
| `bite` | "chunk", "step", "case" |
| `mission` (campaign phase label) | the module's real name or path ("ex02a", never "Mission 02") |
| `digest` (test-run summary sense) | "summary", "report" (the cryptographic sense stays: "Returns the SHA-256 digest" is canonical) |
| `mutation` (test-theater sense) | "change", "variation" |
| `RED` / `GREEN` (test-status theater) | never. State the invariant or behavior in present tense |
| `oracle` | "reference implementation" |
| `probe` | "test". Test files are `test_*.nim`, never "probes" |
| `deviation class` | describe the actual difference (ulps, bytes, layout) |
| `load-bearing` | "essential", "critical", "necessary" ("the load-bearing guard" → "the essential guard") |
| `seam` | "boundary", "interface", "edge" ("the seam between tiles" → "the boundary between tiles") |
| `physics` / `physics-bearing` (for numerical behavior) | "honest rounding", "rounding", "drift behavior", "value-bearing" ("the physics-bearing slices" → "the value-bearing slices") |
| `committed bytes` / `committed blob hashes` (invented compounds) | "recorded inputs", "recorded files", "recorded checksums" ("pure functions of committed bytes" → "deterministic ops on recorded inputs"). The plain word blob is fine, "frozen blobs" reads correctly |
| `tail-mass` / `tail mass` | "tail probability" (the standard statistics term for the scalar, probability beyond the top ranks; "tail distribution" would suggest a shape comparison). Applied to the on-disk key and identifiers too: "tail_probability", tailProbability, fkTailProbability |
| `arm` (test-variant sense: "CPU arm", "MPS arm", "forced-first-step arm") | "variant" or rephrase ("the CPU variant (reference device)", "the forced first step") |
| `rail` / `rails` (reference-path sense: "sequential rail", "KV page rails") | "reference", "reference path", "boundary reference" |
| `battery` (set-of-checks sense: "model battery", "battery log") | "checks", "suite"; "battery log results" → "the re-run results" |
| `wave` (campaign-phase sense: "re-record wave", "device-matrix wave") | "pass" ("re-record pass"), "work" ("device-matrix work") |
| `donor` (fixture-source sense: "fixture donor", "the same donor") | "recorded family", "recorded source" |
| `law` / `laws` (set-of-rules sense: "harness law", "fixture law") | "rule", "rule set", "contract" ("harness law" → "harness contract") |

## Line-end hazards from the reflow scanner

The line-break checker rejects lines ending on more words than the doc
skill lists. Words seen flagged in review, in addition to the documented
stopwords: `both`, `own`. When a line would end on one, reflow it so the
line ends on a content noun or verb.

| Rejected line end | House style |
|---|---|
| "...the subject and committer date survive both" | "...the subject and committer date survive both events" |
| "...the report readable on its own" | "...the report self-contained" |

## Justification prose (lawyer speak)

A doc line that argues for a design choice, tombstones a deleted check,
or prefaces a skip with its justification is rejected on review. State
what the code does, never what it refuses, lacks, or why something is
absent. Real lines from this repo, all rejected:

| Rejected justification | House style |
|---|---|
| "The per-token top-8 routing index equality case drops here, because neither assert expresses a top-8 id list, and the routing weights record carries the routing coverage." | the tombstone comment is deleted outright, the kept check carries the coverage |
| "The stored K must differ from the raw k_proj output. That negative contrast case drops here, because neither assert expresses a must-differ comparison." | deleted, the suite states the cases it runs |
| "The decode fixture carries no stats sidecar, so the record comes off the fixture payload in-suite." | deleted, the in-suite recording call shows it |
| "The layer-3 payload carries no stats sidecar, its records come off the fixture." | deleted from the module header, no absent-capability notes |
| "The recorded intermediates accept zero drift, so no cross-device drift case applies and the suite skips." | the echo line alone: "cross-device replay is out of scope for this suite, skipping on <device>" |

## Before → after

From actual slop seen in this repo:

| Sloppy (banned) | House style |
|---|---|
| "pinned by the probe's host mirror" | "verified against the host reference" |
| "probe gates cb2 with that deviation class" | "cb2 matches the CUDA-faithful rounding to within a few fp16 ulps" |
| "Missions 02/03 import this module" | "The ex02a microkernel examples import this module" |
| "the harness draws a sample from the fixture" | "the test reads a sample from the fixture" |
| "gates this check with RED until the fix" | "checks that the packed layout round-trips" |
| "the oracle emits the reference layout" | "the reference implementation emits the expected layout" |
| "smoke test covers the load path" | "sanity test covers the load path" |
| "a sentinel row marks the tile end" | "a marker row ends the tile" |
| "the tail-mass check catches the drift" | "the tail-probability check catches the drift" |
| "a pure function of committed bytes" | "a deterministic op on recorded inputs" |
| "the physics-bearing slices of the family" | "the value-bearing slices of the family" |
| "record-time drift physics differ per row class" | "drift behavior differs per row class" |
