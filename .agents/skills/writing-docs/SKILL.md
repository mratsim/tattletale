---
name: writing-docs
description: "Repository documentation contract for the Tattletale monorepo: the house style for doc comments, module headers, inline comments, and any committed prose (what-over-how, contracts over narration, banned-vocabulary blocklist, format rules, eight canonical reference files). Use when writing or updating doc comments, module headers, inline comments, or any prose in this repo, or when de-sloping existing comments."
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: documentation
---

# Writing Docs (Tattletale House Style)

## What I do

This skill is the repository documentation contract.
It distills the eight canonical files' style (REFERENCE.md) plus the Write
Without Hidden Context framing below into one enforceable standard.
- the reader is a developer who just cloned the repo
- the reader carries no history, no pipeline labels, no campaign vocabulary

## When to use me

- writing or updating any committed prose
- de-sloping existing comments

The prose spans doc comments, module headers, inline comments, type docs,
kernel docs, Lean comments, test file headers, and commit messages.

## Golden rules

1. **Think who is your audience.** A *user* gets `##` docs on correct use, never a drowning in how it is done.
2. **Think who is your audience, part two.** A *maintainer* gets `#` comments in industry-standard jargon.
3. **Write without hidden context.** Rewrite any sentence a repo puller cannot understand.
4. **A wall of text is hostile.** Prefer diagrams for lifecycles, dataflow, and multidimensional structures.
5. **A wall of text is hostile, part two.** Prefer bullet points for contracts.

## Quick start

1. Write the doc as a contract covering behavior, preconditions, postconditions, invariants, and shapes.
2. Never narrate the journey.
3. Open a module header with a noun phrase, `PagedRadixTrie` (a compressed Radix/Patricia trie).
4. Module headers run 3-8 tight lines, test headers carry the run command only.
5. Before committing, run the doc linter and read the banned-vocabulary table ([EXAMPLES.md](EXAMPLES.md)).

## Write Without Hidden Context

Applies to all repository prose.
- documentation, comments, docstrings
- commit and PR text, plans, reports
- names, errors, technical summaries

Everything must make sense to a technically capable reader.
The reader has the repository.
The reader carries none of the conversation or development history.

- Describe the system as it exists with purpose, behavior, invariants, interfaces, evidence, and limits.
- Never narrate the journey.
- No lifecycle labels act as identities (`Phase 2`, `pilot`, `next`, `current`, `new`, `old`, `latest`).
- No false definite references. A bare `the experiment` is invalid unless the antecedent is locally clear.
- Counts, dates, and versions are attributes, not identities.
- On first reference, give the semantic role and the durable identifier (artifact, path, schema, revision, hash).
- Explain concepts before identifiers. Codenames and experiment labels never enter the design vocabulary.
- Mention literal identifiers such as `XOR-Cheb-T12` only after describing what they mean.
- Canonical documentation is a present-state specification, not a changelog.
- Replace stale claims instead of layering history on top.
- Chronology, rejected attempts, and retrospectives belong only in explicitly historical documents.
- Label status explicitly (`implemented`, `qualified`, `research-only`, `unsupported`).
- State evidence as conditions, measurement, result, and conclusion, not as a story.
- Comments explain invariants, intent, and non-obvious constraints, never change history.
- TODOs name the missing condition and the removal criterion.
- Commits and PRs state the resulting behavior, the technical reason, the compatibility impact, and the validation.
- Commits and PRs do not recount attempts or pivots.

The final test applies to every sentence, if understanding it requires
"you had to be there," rewrite it.

## Contract

6. **What, not how.** State what the code does and the contract it upholds.
7. **What, not how, the exception.** Narration of an *invisible* strategy, dataflow, or lifecycle is legitimate.
8. **What, not how, the exception in detail.** A scheduler's strategy or a sync path with a deadlock failure mode qualifies.
9. **Contracts over narration.** Preconditions, postconditions, invariants, ownership, shapes, lifecycle.
10. **Contracts over narration, the form.** Enumerations and layout facts are bullets or tables, one concept per bullet.
11. **Never open a description with `The`.** Start with what the thing is (`Compile-time record: ...`).
12. **Never open a description with `The`, the exception.** A value-returning function opens with `Returns ...`.
13. **Depth scales with reader need.** One line for self-evident failure modes, full explanation for invisible ones.
14. **Depth scales with reader need, the invisible set.** Concurrency, memory ordering, asm correctness.
15. **Depth scales with reader need, the math.** State and prove non-obvious math.
16. **No hidden context.** No finding IDs, iteration labels, history narration, temporal words, or unverifiable claims.
17. **No hidden context, the perf clause.** Perf numbers need a referenced benchmark.
18. **Audience syntax.** `##`/`///` is for API users, `#`/`//` for maintainers and auditors.
19. **Audience syntax, the boundary.** A `##` comment never points at test files or unrelated subsystems.
20. **Audience syntax, the legitimate case.** A contract reference to a paired module (pack ↔ ukernel) is legitimate.
21. **The name carries the doc.** When the identifier says what it is, drop the comment that restates it.
22. **The name carries the doc, the floor.** Public items still need a doc comment stating the contract.
23. **Doc comments sit in the body.** The doc comment of a proc or func is its first body line, a ## block above the declaration is banned.
24. **Banned vocabulary.** The hard blocklist ([EXAMPLES.md](EXAMPLES.md)) is absolute for every committed prose.
25. **Banned vocabulary, the scope.** The blocklist binds this skill's own files too.
26. **Banned vocabulary, the single exception.** EXAMPLES.md quotes banned forms as teaching material.
27. **Format rules.** Parentheses stay whole, no `;` and no em-dashes in prose, lines break at phrase boundaries.
28. **Format rules, the cap.** 140 chars per prose line, bullets for enumerations.
29. **Format rules, the colon.** A prose colon ends its line, what it introduces goes on the next lines.
30. **Format rules, the severed unit.** No line opens on a severed one-word continuation ("apply,").
31. **Format rules, the reference.** Details in [REFERENCE.md](REFERENCE.md).
32. **Fix in batches.** One scripted replace pass per file, then one rescan, never a lint-edit-recheck cycle per finding.
33. **Write in passing form.** Compose every comment block in the passing form the first time, a block written clean costs no fix tokens.
34. **Test docs scale with test complexity.** A tile op or kernel-vs-reference match needs setup, reference, tolerance.
33. **Test docs scale with test complexity, the floor.** Elaborate docs only for genuinely intricate machinery.
34. **Test docs scale with test complexity, the doubt clause.** When in doubt, cut.
35. **No justification prose.** State what the code does, never a refusal or lack.
36. **No justification prose, the tombstone clause.** A deleted check carries no tombstone comment.

## Advanced features

- Per-domain patterns (module headers, SME2 kernels, tensor ops, transformer layers, stateful modules + Lean)
- Format rules, canonical references, and the full self-check live in [REFERENCE.md](REFERENCE.md)
- The banned-vocabulary replacement table and before/after examples live in [EXAMPLES.md](EXAMPLES.md)
