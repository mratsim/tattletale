---
name: engineering-discipline
description: "MANDATORY READING BEFORE ANY EDIT in this monorepo. Engineering rulings for the Tattletale workspace: golden rules (no speculative code, no overdefensive code, no guard without a failing input, no paraphrase comments, no how-narration, check preexisting primitives first), naming rules (action verbs, the get prefix implies a cheap read-only lookup, a better name deletes a comment) and comment budget (comments scale with need). Use when writing or reviewing code."
license: MIT
metadata:
  audience: developers
  workflow: engineering
---

# Engineering Discipline

Recurring engineering rulings for this repo.

- Reading `.agents/skills/writing-docs/SKILL.md` is mandatory alongside this skill.
- Reading this skill is mandatory before any edit in any directory.

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

Comment golden rules:

- No paraphrase comments. A comment that translates visible code into English
  words ("sets the counter to zero" over `ctr = 0`) is deleted, not reworded.
- No how-narration. A comment that narrates the steps of the code below it
  ("first we X, then Y, finally Z", "this call does X") dies.
  The code is the narration.
- The mathematics survives. The equation behind the code is content code
  alone cannot show: a sum identity like `n*(n+1)/2`, a sigmoid derivative
  `s*(1-s)` in a backward pass.
  Math the code shows is narration. Math the code hides is content.


## Package guidance

- Check the AGENTS.md of the subpackage being edited before writing.
- The package's specific rules live there: naming, verification, refusals.


## Writing new code

Scan before adding: grep the module and its siblings for the concept.
The primitive usually exists under another name, a near-copy forks the logic.
Write the smallest complete change: the minimum code that works.
It reaches every caller, test and fixture the change breaks.

No option, wrapper, config or flexibility nobody asked for.
Deletion beats addition. The codebase's existing structure survives.
New non-trivial logic gets one test that fails when it breaks.


## Verification

- A test asserts behavior through the public surface or the data delta,
  never a copy of its own output.


## Naming: getters

A function name starts with an action verb.
Simple getters carry a `get` prefix.
The prefix implies cheapness: a read-only lookup, no side effects, no refusal.

A proc that transforms, records or rewrites never takes the prefix.
Its name states the effect instead.


## Naming: a better name deletes a comment

A comment that explains what an identifier failed to say is a finding:
rename the identifier so it states the meaning, then delete the comment.

The bar: a competent reader recovers the meaning from the identifier alone
after the rename. Precedent: `calleeRefC` became `calleeSymbolC`,
stating that the callee resolves to a symbol. The comment that explained the old identifier died with the rename.

- Rename domain: private and internal identifiers of the files under change.
  Exported identifiers stay frozen without an operator ruling.
- The commit applies the rename and deletes the comment together.
- When no identifier can state the constraint, the comment survives:
  an external contract, a backend behavior, a why-not-else.
  A name that would need a sentence is not a name.


## Comment budget scales with need

An internal proc with a self-explanatory name and a short body (1-7 lines)
gets **no comment**. Name plus body explains everything. A comment there is
pure noise and will rot.

A comment earns its place by giving exactly one of these four things:

| kind | example |
|---|---|
| usage contract for a public proc | returns, requires, raises |
| background context | design ruling, external constraint, bug avoided |
| the mathematics | the equation or identity the code implements, notation conventions |
| non-trivial control flow | state machines, lifecycles |

For the non-trivial control flow kind prefer a lifecycle or dataflow diagram over prose once
the flow exceeds a handful of states.

## Big comments that earn their size

Density scales with complexity in BOTH directions: a deslop or review must
not flatten a big comment that earns its size.
The keep-classes:

- **The mathematics**: derivations, identities and the equations behind the code. Never traded for a one-liner.
- **Lifecycle and dataflow diagrams** for complex interactions: multi-pass algorithms,
  producer/consumer chains across passes, state machines.
  A diagram is the correct form and it is allowed to be big.
- **Spec notes documented nowhere else**: ABI contracts, hardware behavior, protocol excerpts.
- **Usage examples on public API**: a worked example in a public proc's
  doc block is contract, not decoration. The example shows the call shape,
  the argument order and the do-form alternates in a way no signature can.
  Kept verbatim, never traded for prose.

A deslop's job is to kill paraphrase and narration, not to shrink the file.
Flattening an essential diagram or derivation to hit a line count is a regression, not a cleanup.

Explaining the how of ordinary code is banned.
Narration, temporal prose and artifact references are banned.
The writing-docs linter checks part of the blocklist. Read the banned-vocabulary
table in EXAMPLES.md for the rest.

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
  hides context the reader cannot recover. Rewrite it until it stands alone.

### Coursebook sentences

Write like a coursebook, not like a spec lawyer or a press release.
The style exemplars are Physically-Based Rendering, Crafting Interpreters,
The Art of Multiprocessor Programming, and Programming Massively Parallel
Processors (PMPP): direct subjects, active verbs, mechanisms over labels.

- Every prose sentence needs a subject and a verb. A noun pile that only names
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
  Fix: restore the connective ("the register slots that the thread coordinates own") or split into subject-verb-object sentences.


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

