---
name: engineering-discipline
description: "Engineering rulings for the Tattletale monorepo: golden rules (no speculative code, no overdefensive code, no guard without a failing input, check preexisting primitives first), naming rules (action verbs, the get prefix implies a cheap read-only lookup) and comment budget (comments scale with need). Use when writing or reviewing code."
license: MIT
metadata:
  audience: developers
  workflow: engineering
---

# Engineering Discipline

Recurring engineering rulings for this repo.
Reading `.agents/skills/writing-docs/SKILL.md` is mandatory alongside this skill.
Golden rules first, then naming and comment budget.

## Golden rules

- No speculative code. A feature nobody asked for, an option nobody sets,
  an abstraction with one implementation dies before it enters.
- No overdefensive code. A covered failure gets no second guard.
  The Nim typechecker is one such covering layer.
  An internal authoring mistake is caught by the first compile.
  The fix is writing the data right.
- No guard without a failing input. A check names the input that fails it,
  or it is a deletion candidate. An untestable platform limit is recorded
  in a comment. The code never asserts against it.
- Check the preexisting primitives first. The stdlib, the sibling module
  and the helper one directory over come first.
  A new dependency or a new name for an existing mechanism needs a reason.


## Package guidance

- Check the AGENTS.md of the subpackage being edited before writing.
- It carries the package's specific rules: naming, verification, refusals.


## Writing new code

Scan before adding: grep the module and its siblings for the concept.
The primitive usually exists under another name, a near-copy forks the logic.
Write the smallest complete change: the minimum code that works.
It reaches every caller, test and fixture the change breaks.

No option, wrapper, config or flexibility nobody asked for.
Deletion beats addition. The codebase's existing structure survives.
New non-trivial logic carries one test that fails when it breaks.


## Verification

- A test asserts behavior through the public surface or the data delta,
  never a copy of its own output.


## Naming: getters

A function name starts with an action verb.
Simple getters carry a `get` prefix.
The prefix implies cheapness: a read-only lookup, no side effects, no refusal.

A proc that transforms, records or rewrites never takes the prefix.
Its name states the effect instead.


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


## Audience: CS undergraduate, no hidden context

Every comment addresses a reader fluent in the language, data structures and algorithms.

The reader knows nothing about this codebase beyond what the same file says:

- no knowledge of internal jargon or team shorthand
- no knowledge of the history behind the code
- no memory of the conversation that produced the change

Two rules follow:

- A concept named but never stated, like "per the callee's param kind",
  hides context the reader cannot recover.
- State the concept in full where the file first uses it.
- A comment that only makes sense after reading the surrounding 200 lines
  carries hidden context. Rewrite it until it stands alone.

### Coursebook sentences

Write like a coursebook, not like a spec lawyer or a press release.
The style exemplars are Physically-Based Rendering, Crafting Interpreters,
The Art of Multiprocessor Programming, and Programming Massively Parallel
Processors (PMPP): plain subjects, active verbs, mechanisms over labels.

- Every prose sentence carries a subject and a verb. A noun pile that only names
  ("one global ordinal across all resources in declaration order") is not a sentence.
  Titles, headers, and label bullets are exempt: a noun phrase is the correct
  form for a name (module headers open on one).
- Code and data never act. "The resource identity flows from the call site" is banned.
  Name the actor: "each call passes the caller's buffer as the argument".
- No marketing speech ("seamlessly", "elegant", "powerful"), no lawyer
  speech ("it shall be noted", "as per"). Prefer a mechanism sentence
  over a category label: say what happens, not what kind of thing it is.
- No reduced relative clauses: the connective `that`/`which`/`who` is never omitted.
  Broken: "the register slots its thread coordinates own" is a telegraphic noun pile.
  Fix: restore the connective ("the register slots that the thread coordinates own") or split into plain subject-verb-object sentences.


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

