---
name: writing-docs
description: "Repository documentation contract for the Tattletale monorepo: the house style for doc comments, module headers, inline comments, and any committed prose (what-over-how, contracts over narration, banned-vocabulary blocklist, format rules, seven canonical reference files). Use when writing or updating doc comments, module headers, inline comments, or any prose in this repo, or when de-sloping existing comments."
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: documentation
---

# Writing Docs (Tattletale House Style)

## What I do

This skill is the repository documentation contract. It distills the style of
the seven canonical files (REFERENCE.md) plus the Write Without Hidden Context
framing below into one enforceable standard: comments for a developer who just
cloned the repo, no history, no pipeline labels, no campaign vocabulary.

## When to use me

Use when writing or updating any committed prose (doc comments, module
headers, inline comments, type docs, kernel docs, Lean comments, test file
headers, commit messages) or when de-sloping existing comments.

## Golden rules

1. **Think who is your audience.** A *user* of the API gets `##` docs on how
   to use it correctly, never a drowning in how it is done. A *maintainer*
   gets `#` comments; use industry-standard jargon, never invented jargon.
2. **Write without hidden context.** If someone who pulls the repo cannot
   understand a sentence, rewrite it.
3. **A wall of text is hostile to the reader.** Strongly prefer diagrams and
   charts for lifecycles, dataflow, multidimensional algorithms and data
   structures, and bullet points for contracts.

## Quick start

1. Write the doc as a contract: what the code does, its preconditions,
   postconditions, invariants, shapes, layout facts. Never narrate the
   journey.
2. Open a module header with a noun phrase (`PagedRadixTrie — a compressed
   Radix/Patricia trie ...`), never with `The`.
3. Module headers 3-8 tight lines; test headers carry the run command only;
   test docs scale with test complexity.
4. Before committing: grep the banned vocabulary (EXAMPLES.md) and run the
   self-check (REFERENCE.md).

## Write Without Hidden Context

Applies to all repository prose: documentation, comments, docstrings, commit
and PR text, plans, reports, names, errors, technical summaries. Everything
must make sense to a technically capable reader who has the repository but
none of the conversation or development history.

- Describe the system as it exists: purpose, behavior, invariants,
  interfaces, evidence, limitations. Do not narrate the journey.
- No lifecycle labels as identities: `Phase 2`, `pilot`, `next`, `current`,
  `new`, `old`, `latest` are not technical names.
- No false definite references: `the 1M-token capture`, `the experiment`,
  `this approach` are invalid unless the exact object was introduced locally
  and unambiguously. Counts, dates, and versions are attributes, not
  identities.
- On first reference, give the object's semantic role and its durable
  identifier: artifact name, path, schema, revision, manifest, or hash.
- Explain concepts before identifiers. Codenames, experiment labels, and
  implementation shorthand such as `XOR-Cheb-T12` are never the vocabulary of
  the design; mention literal identifiers only after describing what they
  mean, and only when the reader must use them.
- Canonical documentation is a present-state specification, not a changelog.
  Replace stale claims instead of layering history on top. Chronology,
  rejected attempts, and retrospectives belong only in explicitly historical
  documents.
- Label status explicitly: `implemented`, `qualified`, `research-only`,
  `unsupported`. State evidence as conditions, measurement, result,
  conclusion, not as a story.
- Comments explain invariants, intent, and non-obvious constraints, never
  change history. TODOs name the missing condition and removal criterion.
- Commits and PRs state the resulting behavior, technical reason,
  compatibility impact, and validation. They do not recount attempts or
  pivots.

Final test: if understanding any sentence requires "you had to be there,"
rewrite it.

## The contract

1. **What, not how.** State what the code does and the contract it upholds.
   Narration of an *invisible* strategy, dataflow, or lifecycle is legitimate
   (a scheduler's strategy, a sync path whose failure mode is deadlock).
2. **Contracts over narration.** Preconditions, postconditions, invariants,
   ownership, shapes, lifecycle. Enumerations and layout facts are bullets or
   tables, one concept per bullet.
3. **Never open a description with `The`.** Start with what the thing is
   (`Compile-time record: ...`). A value-returning function opens with
   `Returns ...`.
4. **Depth scales with reader need.** One line for self-evident failure
   modes; full explanation for invisible ones (concurrency, memory ordering,
   asm correctness). State and prove non-obvious math.
5. **No hidden context.** No finding IDs, iteration labels, history or
   journey narration, temporal words, or unverifiable claims (perf numbers
   need a referenced benchmark).
6. **Audience syntax.** `##`/`///` is for API users, `#`/`//` for maintainers
   and auditors. A `##` comment never points at test files or unrelated
   subsystems; a contract reference to a paired module (pack ↔ ukernel) is
   legitimate.
7. **The name carries the doc.** When the identifier says what it is, drop
   the comment that restates it. Public items still need *a* doc comment
   stating the contract; this rule only bans re-explaining visible structure.
8. **Banned vocabulary.** The hard blocklist (EXAMPLES.md) is absolute for
   consumer-authored prose; this skill's own rule definitions and examples
   are exempt.
9. **Format rules.** Parentheses stay whole, no `;` or em-dashes in prose,
   lines break at phrase boundaries (cap 128 chars), bullets for
   enumerations. Details in REFERENCE.md.
10. **Test docs scale with test complexity.** An element-wise tile op or a
    kernel-vs-reference match needs setup, the reference call, the tolerance,
    done. Elaborate docs only for genuinely intricate machinery. When in
    doubt, cut.

## Advanced features

- Per-domain patterns (module headers, SME2 kernels, tensor ops, transformer
  layers, stateful modules + Lean), format rules, canonical references, and
  the full self-check: See [REFERENCE.md](REFERENCE.md)
- Banned-vocabulary replacement table and before/after examples: See
  [EXAMPLES.md](EXAMPLES.md)
