# data_structures

Innovative data structures for Tattletale, with a formal-verification counterpart in Lean 4.

## What it provides

Entry point:
  [`data_structures.nim`](data_structures.nim) imports and re-exports [`src/wavl_tree.nim`](src/wavl_tree.nim).

### WAVL (Weak AVL) tree — `src/wavl_tree.nim`

An intrusive, index-based, `seq`-backed WAVL tree. Self-balancing BST, rank
differences of 1 or 2 between parent and child, `O(log N)` operations,
amortized `O(1)` restructuring per insert/delete.

- **Intrusive design**:
  nodes are not separately allocated. Each entry indexes a parallel
  `WavlLink` `seq` (`p`/`l`/`r`/`rank`) living alongside the caller's
  data. Zero tree-node GC allocations, contiguous, cache-friendly
  (200K nodes ≈ 3.2 MB of links per the header notes).
- **Removal dance**:
  integrates with Nim `seq.del` swap-pop, `fixLinksAfterIndexRemap`
  updates only the ≤3 affected references in `O(1)`.
- **API**:
  `wavlInit`, `wavlInsert`, `wavlFind`, `wavlMin`, `wavlMax`, `wavlDelete`, `fixLinksAfterIndexRemap`, and the `wavlFindBestMatch` template.

### Longest-prefix-match via signed comparator

`wavlFindBestMatch` exploits a comparator returning the *signed
position of first divergence* rather than just `-1/0/+1`.

- The sign drives BST navigation, the magnitude is the shared-prefix length.
- On a miss it returns the neighbor with the longest shared prefix,
  pure `O(log N)`, no linear scan.
- The implementation uses a self-balancing BST as the longest-prefix-match
  index for radix-trie KV caches keyed by 256-token pages.

### Double-array Aho-Corasick — `src/daac.nim`

`buildDaac(patterns, values)` compiles a byte dictionary into two flat
arrays (BASE/CHECK per slot, plus fail links and merged output chains).

- The design follows daachorse BASE+CHECK. The block placement comes
  from arXiv 2207.13870. `nextState` is the goto-plus-fail transition.
  Each state's output chain reports every pattern occurrence ending
  there (standard overlapping semantics).
- Built automata are read-only and deterministic for a fixed input.
- Leftmost and decision semantics live in the scanner layer consuming
  this core (`workspace/toktoktok/src/scan.nim`).

## Formal verification

- [`src/wavl_tree.lean`](src/wavl_tree.lean), a Lean 4 formalization
  of the Nim implementation. References Haeupler/Sen/Tarjan 2015
  "Rank-Balanced Trees" and Gillon 2024 "Verified AVL Trees in Lean 4".
- [`../../formalities/wavl_tree.lean`](../../formalities/wavl_tree.lean),
  a symlink from the `formalities/` directory.

## Tests

- `tests/test_wavl_tree.nim`.
- `tests/test_daac.nim`, DAAC build + overlapping match vs a naive
  exhaustive reference over fuzz dictionaries. Fixed seeds, NUL bytes
  and prefix chains included.

## Status

WAVL tree with LPM support and its Lean formalization are implemented. Additional data structures may be added here over time.

## Related

- Root project README at [`../../README.md`](../../README.md).
