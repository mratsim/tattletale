# data_structures

Innovative data structures for Tattletale, with a formal-verification counterpart in Lean 4.

## What it provides

Public entry points are one root module per data structure, each re-exporting
the implementation under `src/`. A module is imported by path.

| import                                   | provides                                        |
| ---------------------------------------- | ----------------------------------------------- |
| `workspace/data_structures/aho_corasick` | Aho-Corasick automaton over a pattern set       |
| `workspace/data_structures/small_seqs`   | `SmallSeq`, first `N` elements inside the value |
| `workspace/data_structures/wavl_trees`   | WAVL tree used as a longest-prefix-match index  |

### WAVL (Weak AVL) tree — `wavl_trees.nim`

An intrusive, index-based, `seq`-backed WAVL tree. Self-balancing BST with rank differences of 1 or 2 between parent and child, giving `O(log N)` operations with amortized `O(1)` restructuring per insert/delete.

- **Intrusive design**: nodes are not separately allocated; each entry is an index into a parallel `WavlLink` `seq` (`p`/`l`/`r`/`rank`) living alongside the caller's data. Zero tree-node GC allocations, contiguous and cache-friendly (200K nodes ≈ 3.2 MB of links per the header notes).
- **Removal dance**: integrates with Nim `seq.del` swap-pop; `fixLinksAfterIndexRemap` updates only the ≤3 affected references in `O(1)`.
- **API**: `wavlInit`, `wavlInsert`, `wavlFind`, `wavlMin`, `wavlMax`, `wavlDelete`, `fixLinksAfterIndexRemap`, and the `wavlFindBestMatch` template.

### SmallSeq — `small_seqs.nim`

A sequence whose first `N` elements live inside the object. `SmallSeq[N, T]` takes its parameter
order from `array[N, T]`.

- Up to `N` elements are inline, past `N` they live in one heap block at index `i - N`.
- Growth beyond `N` doubles total capacity like a `seq` does, so `N` is inline capacity.
- Crossing `N` is a fast-path exit, not an assertion.

| indices     | storage test | elements live in  | cost at `N = 5`       |
| ----------- | ------------ | ----------------- | --------------------- |
| `0 ..< N`   | `i < N`      | `arr[i]`          | 40 bytes, no heap     |
| `N ..< len` | `i >= N`     | `overflow[i - N]` | 40 bytes plus a block |

- **One comparison on the index picks the storage**, `i < N`, with no discriminant field.
  Growth reallocs the tail only, so no append moves an element out of `arr`.
- **`clear` sets `len` to 0** and retains both parts of the storage. The destructor releases
  the tail when the value goes out of scope.
- **`T` must be a trivial type**, checked by `supportsCopyMem`, which is false for managed
  payloads and for a type defining `=destroy` or `=copy`. `SmallSeq` cannot hold `SmallSeq`,
  `string`, `seq` or `ref`.

Layout carries the design, `int32` lengths and a raw-pointer `overflow` keep the object at 40 bytes
for `N = 5` and 32 for `N = 3` on a 4-byte payload. An element carrying a pointer pads the length
pair and pushes `arr` 8 bytes in, to 64 bytes at `N = 3` and 96 at `N = 5`.

### Longest-prefix-match via signed comparator

`wavlFindBestMatch` exploits a comparator returning the *signed position of first divergence* rather than just `-1/0/+1`: the sign drives BST navigation while the magnitude is the shared-prefix length. On a miss it returns the neighbor with the longest shared prefix in pure `O(log N)` — no linear scan. The implementation uses a self-balancing BST as a longest-prefix-match index for radix-trie KV caches keyed by 256-token pages.

## Formal verification

- [`src/wavl_trees.lean`](src/wavl_trees.lean) formalizes the Nim implementation in Lean 4, following
  Haeupler/Sen/Tarjan 2015 and Gillon 2024.
- [`../../formalities/wavl_trees.lean`](../../formalities/wavl_trees.lean) is the symlink exposed
  from the `formalities/` directory.

## Tests

- `tests/test_wavl_trees.nim`.
- `tests/test_aho_corasick.nim`.
- `tests/test_small_seqs.nim`.

The `SmallSeq` suite runs twice, once plain and once with `-d:nimAllocStats`, the define enabling
the allocator block counts its move and layout claims are checked against.

## Status

WAVL tree with LPM support and its Lean formalization are implemented, alongside the Aho-Corasick
automaton and `SmallSeq`. Additional data structures may be added here over time.

## Related

- Root project: [`../../README.md`](../../README.md)
