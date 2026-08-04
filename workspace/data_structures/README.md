# data_structures

Innovative data structures for Tattletale, with a formal-verification counterpart in Lean 4.

## What it provides

Entry point: [`data_structures.nim`](data_structures.nim) imports and re-exports [`src/wavl_tree.nim`](src/wavl_tree.nim).

### WAVL (Weak AVL) tree — `src/wavl_tree.nim`

An intrusive, index-based, `seq`-backed WAVL tree. Self-balancing BST with rank differences of 1 or 2 between parent and child, giving `O(log N)` operations with amortized `O(1)` restructuring per insert/delete.

- **Intrusive design**: nodes are not separately allocated; each entry is an index into a parallel `WavlLink` `seq` (`p`/`l`/`r`/`rank`) living alongside the caller's data. Zero tree-node GC allocations, contiguous and cache-friendly (200K nodes ≈ 3.2 MB of links per the header notes).
- **Removal dance**: integrates with Nim `seq.del` swap-pop; `fixLinksAfterIndexRemap` updates only the ≤3 affected references in `O(1)`.
- **API**: `wavlInit`, `wavlInsert`, `wavlFind`, `wavlMin`, `wavlMax`, `wavlDelete`, `fixLinksAfterIndexRemap`, and the `wavlFindBestMatch` template.

### Longest-prefix-match via signed comparator

`wavlFindBestMatch` exploits a comparator returning the *signed position of first divergence* rather than just `-1/0/+1`: the sign drives BST navigation while the magnitude is the shared-prefix length. On a miss it returns the neighbor with the longest shared prefix in pure `O(log N)` — no linear scan. Per the source comment, this is claimed as the first use of a self-balancing BST as a longest-prefix-match index, targeting radix-trie KV caches keyed by 256-token page.

## Formal verification

- [`src/wavl_tree.lean`](src/wavl_tree.lean) — Lean 4 formalization of the Nim implementation (references Haeupler/Sen/Tarjan 2015 "Rank-Balanced Trees" and Gillon 2024 "Verified AVL Trees in Lean 4").
- [`../../formalities/wavl_tree.lean`](../../formalities/wavl_tree.lean) — symlink from the `formalities/` directory.

## Tests

- `tests/test_wavl_tree.nim`.

## Status

WAVL tree with LPM support and its Lean formalization are implemented. Additional data structures may be added here over time.

## Related

- Root project: [`../../README.md`](../../README.md)
