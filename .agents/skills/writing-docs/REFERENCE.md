# Writing Docs — Reference

Detailed patterns per domain, format rules, the canonical files, and the
self-check. Read SKILL.md first (contract + Write Without Hidden Context).

## Module headers

Header depth scales with reader need. Three tiers: title-only (noun-phrase
banner), lifecycle (banner plus phase sequence), definition (banner plus
math, invariants, references). Escalate only when the reader needs what the
lower tier omits.

The shape, from the canonical files:

1. **Purpose line.** A noun phrase naming what the module provides. Never a
   `The`-opening story, never narrative history, never temporal words.
2. **Contract facts.** The layouts consumed and produced, hardware
   requirements, streaming-mode rules, conventions a fresh developer must
   know.
3. **Pointer to formalization** where one exists (`DESIGN (see kvcache.lean
   for formalization)`).
4. **3-8 tight lines max.** A facts-dense header may run to ~12 lines only
   when every line carries a fact (gemm_packing:1-12). A 30-line essay header
   is slop. Diagrams and decision tables may extend beyond.

Exemplar (gemm_packing_arm64_sme2.nim:1-12):

```nim
## SIMD-accelerated pack kernels for the ex02a hand-tuned GEMM example.
##
## Packs transpose or copy row-major A/B panels into the `(ir, kc, 32)` layout
## consumed by the SME2 ukernels in `gemm_ukernel_arm64_sme2.nim`.
## `sme_` packs run inside an smstart/smstop bracket (streaming mode, SVE/SME2 only).
## NEON packs run outside the bracket, since NEON faults
## inside streaming mode. Requires ARMv9.2 with FEAT_SME2 and assumes
## SVL = 64 B (16 f32 lanes per vector).
```

Exemplar (kvcache.nim:9-12):

```nim
## PagedRadixTrie — a compressed Radix/Patricia trie keyed by token sequences,
## where split/graft operations are at page granularity (256 tokens).
##
## DESIGN (see kvcache.lean for formalization):
```

Exemplar (kvcache.lean:1-10, the Lean twin):

```lean
/-
# Tattletale KVCache — Formal specification in Lean 4

## Design rationale

The KVCache is a **PagedRadixTrie**: a compressed Radix/Patricia trie over token
sequences, where each node holds a contiguous prefix of tokens and the trie
operates at page (256-token) granularity. GPU pages are immutable once grafted.

See kvcache.nim for the full implementation (with WAVL acceleration).
-/
```

Never write: "This implements mission 02", "Now that the previous bug is
fixed", "Once X lands", "as of", "currently". Those go stale the instant the
code changes.

## Kernel docs (SME2 asm, the packing and ukernel files)

Hardware mapping + data layout + the math contract in tight bullets, then the
before/after ASCII diagram. Structure for every exported kernel proc:

1. **One-line purpose with the math contract.**
   `## 16×16 f32 SME outer-product micro-kernel: AB[i][j] = Σ_k packA[k*16+i] * packB[k*16+j].`
   (gemm_ukernel_arm64_sme2.nim:74)
2. **`Expected input:`** one bullet per parameter: pointer role, stride,
   count, validity (gemm_packing_arm64_sme2.nim:107).
3. **`Output:`** the write formula. `## Output: one transposed 8×8 block per
   column group, columns 32 f32 apart: dst[c*32 + r] = (r < validRows) ? src[r*rs + c] : 0`
   (gemm_packing_arm64_sme2.nim:115).
4. **Before/after ASCII diagram** for any layout transform. Two fixed-width
   blocks, cell labels that make the mapping visible, dst offsets where the
   packed stride matters (gemm_packing_arm64_sme2.nim:120). Diagram lines are
   exempt from the 128-char prose cap.
5. **Contract lines** the caller must uphold, as separate sentences:
   "A zero group count packs nothing." / "Caller guarantees 32 valid lanes
   per source row and routes panels with `eff < 32` lanes to the scalar
   copyMem path." (gemm_packing_arm64_sme2.nim:410)
6. **Divergence notes as bullets** when the kernel differs from a scalar or
   sibling path: "ReLU `fclamp` maps NaN to 0 on M4, like the scalar
   epilogue." / "`beta == 0` stores `alpha*f(AB)`, possibly `-0.0` where the
   scalar epilogue stores `+0.0`." (gemm_ukernel_arm64_sme2.nim:464)

The file header additionally records hardware mapping and conventions:
requires ARMv9.2 with FEAT_SME2, SVL = 64 B, the streaming-mode rules, and
the fmopa operand-order convention: "Two fmopa conventions live in this
file: AB-store kernels (16×16, 32×32): operands (B, A) ... Swapping the
order silently transposes every tile. Only asymmetric data detects it."
(gemm_ukernel_arm64_sme2.nim:16-19)

The `build*` procs get one terse line recording what the instruction stream
is, in source order. The `gen*` macros get one line: "Expands to the
`{.emit: ...}` pragma for ...". The dispatch proc gets the same kernel
contract plus which asm kernel backs which tile shape
(gemm_ukernel_arm64_sme2.nim:733).

## Tensor ops (tensors_nn.nim)

1. **Simple ops get one compact line.** `## SiLU (Sigmoid Linear Unit)
   activation function: ``x / (1 + exp(-x))`` Also known as Swish.`
   (tensors_nn.nim:37)
2. **Complex ops get the labeled-shape contract.** Purpose line with the math
   (`## SDPA — Transformers' attention` / `## Computes softmax(Q @ K^T / scale) @ V ...`),
   then:
   - `## Input shapes:` one bullet per tensor: `(B, H_q, L, d_k) or (B, L, H_q * d_k)`
   - `## Output shape:` the result shape
   - `## Parameters (forwarded to C++ std::optional):` one bullet per
     parameter: role, shape, default (tensors_nn.nim:143-164)
3. **Backend selection as a table** plus numbered priority lists and a
   constraints bullet list (tensors_nn.nim:93-133).
4. **`#` for maintainer-level mechanics.** Compiler quirks and destructor
   notes are `#` comments, never `##` (they would leak into generated user
   docs).

## Transformer layers (attn.nim and rope.nim)

1. **Type doc = what + lifetime + invariants.** rope.nim:13-55: purpose line,
   then `LIFETIME`, `DATA FLOW` (ASCII diagram), `INVARIANTS` (bullets, each
   with shapes or formulas), `USAGE` (a code block).
2. **Data flow as an ASCII diagram**, not prose: the `# Data flow through
   RopeGQAttention` block with `x → q_proj → reshape → q_norm → applyRope →
   q_rot ...` (attn.nim:46-72).
3. **Function docs: `Args:` / `Returns:` with shapes**, then `Computes:` as a
   numbered sequence when the steps are the contract (attn.nim:171-190).
4. **Algorithm steps as a numbered list**: `new()` documents the NEOX-style
   freq computation as 4 numbered steps plus a one-line complexity note
   (rope.nim:126-140).
5. **Contracts stated explicitly**: `**Contract:** cos and sin MUST be 2D
   (seq, head_dim). Shape normalization is the caller's responsibility.`
   (rope.nim:97)
6. **Inline `#` carries rationale**: the batch-size guard explains why ("The
   paged KV cache write/gather path indexes with [0, ...] throughout") and
   the page-chunk write explains the win ("Reduces GPU kernel launches from
   O(seq_len) to O(num_pages)") (attn.nim:191-196).
7. **Disambiguate confusing pairs**: the `input_ids vs position_ids` section
   (rope.nim:164) states what each is, when they coincide, and when they
   diverge.

## Stateful modules and Lean (kvcache.nim + kvcache.lean)

The Nim/Lean pair documents the same design twice. Each side points at the
other. This is the pattern for stateful structures with invisible invariants:

1. **Module header**: purpose + `DESIGN` rationale (numbered, argumentative,
   with measurements where claimed) + `Usage:` code block + `Invariants:`
   bullets (kvcache.nim:9-130).
2. **Invariants carry names and one-line statements** that a verifier can
   check: `A1. Prefix entropy (children diverge within first page)`,
   `C2. subtreeSumLocked partition`, `E1. No single-child nodes`
   (kvcache.nim:745-755). The Lean side states each as an inductive predicate
   with the same one-liner (kvcache.lean:496).
3. **Decision tables as audit contracts**: the classifyGraft table maps every
   condition to its branch and action, so a reviewer can check every row has
   a branch and tests cover every row (kvcache.nim:685-700).
4. **Lifecycle as bullets**, as in kvcache.nim:677-681:
   - `lpm()` → inc `subtree_sum_locked` on each visited node
   - `graftPages()` → the branch proc decrements `subtree_sum_locked` via
     `walkUpUpdate`
   - Locks are always released by `graftPages`. There is no separate unlock
5. **Lean docs stay compact**: one-line purpose, "Mirrors Nim's X" pointer
   for twins, field lists as bullets, `CONTRACT:` bullets for the public API
   (kvcache.lean:404-412). Theorems get a one-line statement of what they
   guarantee.

## Format rules

1. **Header length.** 3-8 tight lines. A 30-line essay header is slop. Only a
   layout/math diagram or a decision table may extend beyond.
2. **Parenthesis = one semantic unit.** Never break `f(a, b)` so a paren
   lands on the next line. When a parenthetical cannot fit, end the line
   before the `(` so it opens the next line.
3. **No `;` in prose.** A bullet, an arrow (`→`), a comma, a colon, or
   parentheses express the relationship. Nim's code-level `;` (generic
   separators) is exempt. The ban is prose-only.
4. **No em-dashes in prose.** Restructure with commas, colons, or
   parentheses.
5. **No line-break prose.** Reflow to fill lines. Break at phrase boundaries,
   never inside a unit (an article, possessive, stranded connective, or bare
   verb at line end is a broken break). Prose lines cap at 128 chars. Tables,
   diagrams, and URLs are exempt.
6. **Bullets and tables for enumerations and layout facts.** One concept per
   bullet. Never a colon mid-sentence introducing an explanation. End the
   line after the colon and start the unit on the next line.
7. **Test file headers = the run command only**, nothing else. No prose about
   what the suite covers (test names say that), no status markers.
8. **Test documentation scales with test complexity.** A test that runs an
   element-wise tile op or matches a kernel against a reference needs no
   essay: a few lines of setup, the reference call, the tolerance, done.
   Elaborate docs only for genuinely intricate machinery (a scheduler, a
   page-boundary crossing, a lock lifecycle). When in doubt, cut.
9. **No transient markers.** No `RED`/`GREEN`, no "fails now", no "turns
   green after the fix", no "currently", no "as of". State the invariant the
   test prevents, in present tense.
10. **No comment restates the code.** A comment that only re-explains visible
    structure is narration. Delete, don't rephrase.

## Canonical references

When in doubt, match these files exactly. They are the operator's definition
of the house style:

1. `workspace/libtorch/src/tensors_nn.nim` (SDPA + nn functional API: tensor
   op docs, `Input shapes:` / `Output shape:` / `Parameters:` contracts,
   backend tables, compact one-liners)
2. `workspace/ceramic/examples/ex02_matmul_microkernels/gemm_packing_arm64_sme2.nim`
   (pack kernels: hardware mapping, `Expected input:` / `Output:` contracts,
   before/after diagrams, caller-contract lines)
3. `workspace/ceramic/examples/ex02_matmul_microkernels/gemm_ukernel_arm64_sme2.nim`
   (ukernels + dispatch: math contract line, quadrant tables, ILP rationale,
   divergence bullets, operand-order conventions)
4. `workspace/transformers/src/stateful/kvcache.lean` (Lean formalization:
   purpose header, field bullets, `Mirrors Nim's ...` pointers, `CONTRACT:`
   bullets, invariant predicates with one-line statements)
5. `workspace/transformers/src/stateful/kvcache.nim` (Nim side of the same
   module: `DESIGN` rationale, `Usage:` block, `Invariants:` bullets,
   decision tables, lock lifecycle)
6. `workspace/transformers/src/layers/attn.nim` (transformer layer: ASCII
   data-flow diagram, `Args:` / `Returns:` / `Computes:` docs, inline `#`
   rationale for guards and buffer choices)
7. `workspace/transformers/src/layers/rope.nim` (small math/layer module:
   type doc with LIFETIME / DATA FLOW / INVARIANTS / USAGE, numbered
   algorithm steps, explicit contracts)

This skill is self-contained: the seven canonical files above are the
exemplars it is derived from, and the Write Without Hidden Context framing is
integrated in SKILL.md.

## Self-check

Run before committing prose:

- Every public item has a doc comment stating its contract (the name-carrying
  rule only bans comments that restate visible structure)
- Audience check: user-facing docs explain how to use it correctly, not how it
  is done; maintainer comments use industry-standard jargon, never invented
- No wall of text: lifecycles, dataflow, and multidimensional algorithms or
  data structures are diagrams; contracts are bullets
- No banned word from the blocklist appears in consumer-authored prose (this
  skill's own definitions and examples are exempt)
- No history, journey, finding ID, iteration label, or unverifiable claim
- No temporal words: "currently", "as of", "once X lands", "now that"
- `##` comments never point at test files or unrelated subsystems
  (paired-module contract references are fine)
- Module headers are 3-8 tight lines and state the module's contract
- Enumerations are bullets or tables, one concept per bullet
- No `;` or em-dashes in prose. Parens stay whole. Lines break at phrase
  boundaries, cap 128 chars
- Kernel docs carry `Expected input:` / `Output:` and a before/after diagram
  for layout transforms
- Test file headers carry the run command only
- Comments match the code, down to parameter names
