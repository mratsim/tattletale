---
name: engineering-discipline
description: "Recurring engineering rulings for the Tattletale monorepo: comment budget (docs scale with need, paraphrase comments on short internals banned) and naming rules (simple getters carry a get prefix). Use when writing or reviewing code."
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: engineering
---

# Engineering Discipline

Recurring engineering rulings for this repo.
Cross-reference `.agents/skills/writing-docs/SKILL.md` for linter mechanics.
This skill covers comment budget and change verification.

## Comment budget scales with need

An internal proc with a self-explanatory name and a short body (1-7 lines)
gets **no comment**. Name plus body explains everything. A comment there is
pure noise and will rot.

A comment earns its place by giving exactly one of these three things:

| kind | example |
|---|---|
| usage contract for a public proc | returns, requires, raises |
| background context | design ruling, external constraint, bug avoided |
| non-trivial control flow | state machines, lifecycles |

For the third kind prefer a lifecycle or dataflow diagram over prose once
the flow exceeds a handful of states.

Explaining the how of ordinary code is banned.
Narration, temporal prose and artifact references are banned.
The writing-docs linter holds the blocklist.

Litmus test: delete the comment. If code plus name still tells the whole
story to a competent reader, the comment stays deleted.

## No drive-by comment edits

Do not touch comments in code the change does not otherwise modify.
Rewording that swaps a concrete example for vague prose is a regression, so revert it.
Comment shape rules never justify deleting information.

## One statement per line

One statement per line. The exception is case, if, else and when used as
expressions (a value is produced). A statement form that introduces a block
(loop, block, a statement-position if or case) never shares its line with its inner statements:

```nim
# wrong
for i in 0 ..< n: total += weights[i]
if done: return

# right
for i in 0 ..< n:
  total += weights[i]
if done:
  return

# allowed: value-producing expressions
let sign = if x < 0: -1 else: 1
let label = case k
  of ka: "a"
  else: "?"
```

## Branch comments indent under the branch

A comment on a branch of a statement-position if, case, when or except goes
indented inside the branch body, never above the branch keyword:

```nim
# wrong
# scratch pad only
if local:
  emit scratch

# right
if local:
  # scratch pad only
  emit scratch
```

## Naming: getters

Simple getters carry a `get` prefix.
The prefix signals a small helper free of side effects.
A proc that transforms, records, or rewrites never takes the prefix,
its name names the effect instead.
