# Codegen-size runners for the layout-algebra simplification track.

## Layout algebra

| runner                            | kernels                                                              |
| --------------------------------- | -------------------------------------------------------------------- |
| cgs_layout_complement.nim         | complement, the copyFrom chain, local_tile                           |
| cgs_layout_make_layout_like.nim   | make_layout_like, the shape-preserving stride compaction constructor |
| cgs_layout_make_fragment_like.nim | make_fragment_like, the fragment/rest-path constructor               |
| cgs_layout_compose.nim            | compose over static, nested, rank-1, and runtime layouts             |
| cgs_layout_coalesce.nim           | coalesce                                                             |
| cgs_layout_zip_group.nim          | groupDimensions, zipDimensions                                       |
| cgs_layout_concat.nim             | concat                                                               |
| cgs_layout_pad.nim                | blocked_product, raked_product, tile_to_shape                        |
| cgs_layout_divides.nim            | zipped_divide, logical_divide over all tiler forms                   |
| cgs_layout_products.nim           | tiled_product, flat_product, logical_product                         |
| cgs_layout_local_tile.nim         | local_tile family baselines, selectors, and instruments              |
| cgs_layout_indexing.nim           | crd2idx coord forms, idx2crd divmod, the L(i, j) accessor            |
| cgs_layout_inverses.nim           | right_inverse, left_inverse over dynamic and static layouts          |

All runners live in the cgs_layout_ namespace, layout-algebra subjects.
codegen_size_analysis.nim is the shared analysis module. Every runner ends
with one `cgsReport` call over its measured kernels:

- kernel table, Name, Cost of 1 call (MSL bytes), Marginal, one totals row,
  where Marginal = cost minus the paired baseline, `-` where a kernel has no baseline


- runner-wide stats totals, function LOC buckets at the 5/30/70/150/300/700/
  1400 bounds, overload families with the same bucketing
- per-kernel line attribution with a pareto line, attribution caveats in three lines

Each kernel is a `const xMsl = metal:` block in its runner, mirroring one real
call site. Runner code carries no echo statements, the report is the only output.

Run from the tattletale/ dir through the ceramic_cgs task, the measured
flags live there so the protocol cannot drift per runner:

    nim ceramic_cgs runner=<runner>
    nim ceramic_cgs runner=<runner> dump=true

## Kernel rows

One line per kernel row, row names say what the row measures.

| runner                        | kernel                          | measures                                                                                                |
| ----------------------------- | ------------------------------- | -----------------------------------------------------------------------------------------------------   |
| cgs_layout_complement         | complementDirectKernel          | complement of a runtime-shape rank-2 layout, the direct-call pattern                                    |
| cgs_layout_complement         | complementStaticKernel          | complement of a fully static layout, the compile-time path                                              |
| cgs_layout_complement         | copyChainKernel                 | the copyFrom chain right_inverse + coalesce(compose(dst, R)), the KV-write pattern                      |
| cgs_layout_complement         | localTileCompKernel             | local_tile, complement via zipped_divide -> logical_divide                                              |
| cgs_layout_make_layout_like   | baselineOverheadKernel          | the like input, one runtime layout construction consumed by size, no like call                          |
| cgs_layout_make_layout_like   | makeLayoutLikeCompactKernel     | make_layout_like on a compacting static rank-2 layout (2,3):(2,1)                                       |
| cgs_layout_make_layout_like   | makeLayoutLikeDynStrideKernel   | make_layout_like on a runtime-stride layout, the make_tensor_like site                                  |
| cgs_layout_make_fragment_like | baselineOverheadKernel          | the fragment input, one static V-block layout construction consumed by size, no like call               |
| cgs_layout_make_fragment_like | makeFragmentLikeVKernel         | make_fragment_like with a (16, 2) V block, the tensor-core fragment site                                |
| cgs_layout_make_fragment_like | makeFragmentLikeBroadcastKernel | make_fragment_like with a broadcast V, the epilogue broadcast-bias site                                 |
| cgs_layout_compose            | composeStaticKernel             | compose of two static rank-2 layouts, the composeImpl path                                              |
| cgs_layout_compose            | composeNestedKernel             | compose of a static rank-2 layout with a nested layout, the composeDistribute path                      |
| cgs_layout_compose            | composeRank1Kernel              | compose with a runtime rank-1 LHS, the b-strides scaleBy path                                           |
| cgs_layout_compose            | composeDynKernel                | compose of a runtime layout with a static layout, the thrfrg_A/B/C compose call sites in k_layout_gemm.nim                    |
| cgs_layout_coalesce           | coalesceStaticKernel            | coalesce of a static contiguous rank-3 layout                                                           |
| cgs_layout_coalesce           | coalesceZerosKernel             | coalesce of a static layout with a stride-0 trailing dimension                                          |
| cgs_layout_coalesce           | coalesceDynKernel               | coalesce of a runtime layout, the copy-chain pattern                                                    |
| cgs_layout_zip_group          | groupDimensionsKernel           | groupDimensions wrapping dimensions [0, 2) of a static rank-4 layout                                    |
| cgs_layout_zip_group          | zipDimensionsKernel             | zipDimensions interleaving two static rank-2 layouts                                                    |
| cgs_layout_concat             | concatDirectKernel              | direct concat of two tuple layouts into one layout                                                      |
| cgs_layout_pad                | blockedCurrentKernel            | blocked_product, one rank step of pad, current form                                                     |
| cgs_layout_pad                | blockedRecKernel                | blocked_product over the recursion candidate, Marginal = candidate delta                                |
| cgs_layout_pad                | rakedCurrentKernel              | raked_product, one rank step of pad, current form                                                       |
| cgs_layout_pad                | rakedRecKernel                  | raked_product over the recursion candidate, Marginal = candidate delta                                  |
| cgs_layout_pad                | tileToShapeCurrentKernel        | tile_to_shape, the full consumer chain, current form                                                    |
| cgs_layout_pad                | tileToShapeRecKernel            | tile_to_shape over the recursion candidate, Marginal = candidate delta                                  |
| cgs_layout_divides             | baselineOverheadKernel     | the divide isolation baseline, dynamic rank-2 layout with one direct coord read, no divide chain        |
| cgs_layout_divides             | sizeOverheadKernel              | the size isolation baseline, dynamic rank-2 layout consumed by size, no divide chain                    |
| cgs_layout_divides             | sizeOverheadCompKernel          | the size baseline of the complement + compose input, Marginal over baselineOverheadKernel               |
| cgs_layout_divides             | sizeOverheadRank4Kernel         | the size baseline of the rank-4 input, Marginal over sizeOverheadKernel                                 |
| cgs_layout_divides             | zippedDivideOnlyKernel          | zipped_divide chain in isolation on the dynamic rank-2 input                                            |
| cgs_layout_divides             | logicalDivideOnlyKernel         | logical_divide chain in isolation on the dynamic rank-2 input                                           |
| cgs_layout_divides             | logicalDivideCompKernel         | logical_divide with a Layout tiler, the complement + compose general path                               |
| cgs_layout_divides             | zippedDivideTupleKernel         | zipped_divide with a tuple tiler on a runtime rank-2 layout                                             |
| cgs_layout_divides             | zippedDivideLayoutKernel        | zipped_divide with a Layout tiler                                          |
| cgs_layout_divides             | zippedDivideRank4Kernel         | zipped_divide on a runtime rank-4 layout, the NHWC gmem view pattern                                    |
| cgs_layout_products            | baselineOverheadKernel          | the product input, one runtime block layout plus a static tiler, consumed by size, no product call      |
| cgs_layout_products            | tiledProductKernel              | tiled_product, concat with the block dimension kept grouped                                             |
| cgs_layout_products            | flatProductKernel               | flat_product, concat with both dimensions unpacked                                                      |
| cgs_layout_products            | logicalProductCompKernel        | logical_product, the compose(complement(a, ...), tiler) site                                            |
| cgs_layout_local_tile         | baselineOverheadRank2Kernel     | the tile-read baseline, dynamic rank-2 view with one indexed element read, no tile machinery            |
| cgs_layout_local_tile         | baselineOverheadGlViewKernel    | the tile-read baseline on the rank-4 global-data view, one 4-coord element read, no tile machinery      |
| cgs_layout_local_tile         | localTileDynKernel              | local_tile_dyn on the rank-4 global-data view, the working entry, Marginal over baselineOverheadGlViewKernel  |
| cgs_layout_local_tile         | innerPartitionKernel            | inner_partition on the dynamic rank-2 fallback                                                          |
| cgs_layout_local_tile         | outerPartitionKernel            | outer_partition on the dynamic rank-2 fallback                                                          |
| cgs_layout_local_tile         | localTile2argKernel             | 2-arg local_tile on the dynamic rank-2 fallback, byte-identity with innerPartition is the alias check   |
| cgs_layout_local_tile         | localTile4argKernel             | 4-arg local_tile with projection (1, 1), dice overhead over the 2-arg form                              |
| cgs_layout_local_tile         | localPartitionKernel            | local_partition on the dynamic rank-2 fallback, the idx2crd route                                       |
| cgs_layout_local_tile         | localTileStaticKernel           | fully static rank-2 layout, runtime coords                                                              |
| cgs_layout_local_tile         | twoTileKernel                   | 2 real local_tile calls, Marginal = 2nd-call cost over the localTile2arg row                            |
| cgs_layout_local_tile         | dynFormulaKernel                | the rank-2 inline of the local_tile_dyn body, the formula baseline                                      |
| cgs_layout_indexing           | baselineOverheadKernel          | the indexing baseline, dynamic rank-2 layout with one direct coord read, no indexing machinery          |
| cgs_layout_indexing           | crd2idxRank4Kernel              | crd2idx on a runtime rank-4 layout with a runtime multi-dim coord, the NHWC gmem view read              |
| cgs_layout_indexing           | crd2idxStaticKernel             | crd2idx on a fully static layout, the compile-time folding path                                         |
| cgs_layout_indexing           | idx2crdCpuKernel                | idx2crd_cpu divmod on a runtime rank-2 layout         |
| cgs_layout_indexing           | callOperatorKernel              | the L(i, j) call-operator accessor over the baseline input, an in-bounds check plus crd2idx |
| cgs_layout_inverses           | baselineOverheadKernel          | the inverse baseline, dynamic shape rank-2 layout with static strides consumed by size                  |
| cgs_layout_inverses           | rightInverseKernel              | right_inverse over that layout, the copyFrom quasi-inverse call-site shape                              |
| cgs_layout_inverses           | rightInverseDynStrideKernel     | right_inverse over a runtime-stride rank-2 layout, the chain keeps the static-stride run                |
| cgs_layout_inverses           | rightInverseStaticKernel        | right_inverse over a fully static layout, the compile-time fold path                                    |
| cgs_layout_inverses           | leftInverseKernel               | left_inverse over the dynamic shape rank-2 layout, all strides static per precondition                  |

## Inspecting generated code

Dump the emitted Metal by adding `dump=true` to the task above:

- every analyzed kernel's MSL lands in workspace/ceramic/benchmark/codegen_size/dumps/,
  named `<runner>_<kernel>.msl`, the kernel table gains a Dump column with the file name

- `-d:TTT_CgsDumpDir="/some/dir"` moves the dumps out of the repo tree
- dumps are transient build output and gitignored, the directory survives
  via its tracked .gitkeep

Reading a dump:

- a mangled name splits at the `___` boundary, boundary text spells the Nim origin name
  and a base62 tail spells the mangling suffix

- function spans are brace-balanced, definition header through closing brace,
  that span is the LOC the attribution charges to the owner

- `inline`-qualified headers spell inlined helpers, unqualified headers spell
  kernel entry points, divide-family materialization shows as brace depth,
  each compose-fold instantiation nests its own braces inside the kernel body


## Update protocol

- every simplification commit on these units re-runs the runners,
  the current dumps refresh, the JSON carries the branch and commit
- a regression over 200 B on an untouched call site is a stop-and-report
- always measure through the ceramic_cgs task, `-d:release` folds debug
  runtime checks out of the MSL and shifts the byte counts

Flag hygiene:

- line-info flags `--lineTrace:on`, `--lineDir:on`, `--debugger:native` touch
  host code, not the crucible-generated code, keep them off
- never mix run protocols across one comparison, the byte counts shift
  with the flags and the backend, C and C++ differ by 8-20 B internally

The runners are not part of the suite scan (no `test_`/`t_` prefix), the ceramic_cgs nim task runs them.

## Dumps and reporting

A dump=true run drops:

- the standardized JSON dump lands at `dumps/cur_reports/<runner>.json`,
  the raw kernel dumps under `dumps/cur_kernels/`
- the runner's previous generation moves wholesale to `dumps/old_reports/`
  and `dumps/old_kernels/`, perf-style old and current, per runner

    nim ceramic_cgs runner=cgs_layout_divides dump=true

Each dump carries, schema 1:

- the run metadata: git branch, commit, UTC datetime
- per-kernel cost, marginal, stats, and per-owner attribution
- the runner-wide stats totals and the function LOC and overload bucket ladders

The raw `.msl` dumps stay alongside for attribution work.
Dump artifacts are gitignored.

The complete report and the old-to-current delta, reads both dump
generations:

    python3 workspace/ceramic/benchmark/codegen_size/report_cgs.py --runner cgs_layout_divides

Drop --runner for every runner.

The workflow mirrors perf: record the old dumps, land the change, re-run,
and `report_cgs.py` reports the delta per kernel, per bucket,
and per attribution origin.
