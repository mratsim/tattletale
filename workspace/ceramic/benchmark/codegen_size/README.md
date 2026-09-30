# Codesize ledger runners for the layout-algebra simplification track.

| runner | kernels |
| ------ | ------- |
| cgs_complement.nim | complement, logical_divide, logical_product, copyFrom, local_tile |
| cgs_like.nim | make_layout_like, make_fragment_like |
| cgs_compose.nim | compose over static, nested, rank-1, and runtime layouts |
| cgs_coalesce.nim | coalesce, filter_inactive |
| cgs_zip_group.nim | zipped_divide, groupDimensions, zipDimensions |
| cgs_concat.nim | concat, tiled_product, flat_product |
| cgs_pad.nim | pad pair: current hand-emitted macros vs the measured candidates |
| cgs_local_tile.nim | local_tile family floors, selectors, and the v2 instruments |

codegen_size_analysis.nim is the shared analysis module, bencher reports.nim table style.
Every runner ends with one `cgsReport` call after the per-kernel lines:

- a per-kernel table of bytes, LOC, types, vars, funcs, calls, inline/non-inline
- function-LOC buckets at the 5/30/70/150/300/700/1400 bounds, overload families likewise
- per-kernel line attribution with a pareto cut, size per call site, overload
  variant counts per Nim origin

Each kernel is a `const xMsl = metal:` block in its runner, mirroring one real
call site. The runner prints one `<kernel>: <bytes>` line per kernel, the MSL
source size under the crucible metal backend.

Run from the tattletale/ dir, plain release protocol:

    nim c -r -d:release --outdir:build/wip --nimcache:nimcache/wip workspace/ceramic/benchmark/codegen_size/<runner>.nim

## Baselines and the update protocol

The baseline tables and the measurement protocol live in `.scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md`.

- every simplification commit on these units re-runs the runners,
  updates the table with the new branch-tip hash and sizes
- a regression over 200 B on an untouched call site is a stop-and-report
- always measure with the plain release flags above, `-d:release` folds debug
  runtime checks out of the MSL and shifts the byte counts
- line-info flags `--lineTrace:on`, `--lineDir:on`, `--debugger:native` change
  mangled-name suffixes in the emitted MSL
- never mix run protocols across one comparison, the byte counts shift with the flags

The runners are not part of the suite scan (no `test_`/`t_` prefix), run them manually.
