# Codesize ledger runners for the layout-algebra simplification track.

| runner | kernels |
| ------ | ------- |
| run_complement.nim | complement, logical_divide, logical_product, copyFrom, local_tile |
| run_like.nim | make_layout_like, make_fragment_like |
| run_compose.nim | compose over static, nested, rank-1, and runtime layouts |
| run_coalesce.nim | coalesce, filter_inactive |
| run_zip_group.nim | zipped_divide, groupDimensions, zipDimensions |
| run_concat.nim | concat, tiled_product, flat_product |

Each kernel is a `const xMsl = metal:` block in its runner, mirroring one real
call site. The runner prints one `<kernel>: <bytes>` line per kernel, the MSL
source size under the crucible metal backend.

Run from the tattletale/ dir, suite flags (the `testerCmd` flags of `config.nims`):

    nim c -r -d:release --stackTrace:on --lineTrace:on --lineDir:on --debugger:native \
      --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests workspace/ceramic/experiments/codesize/<runner>.nim

## Baselines and the update protocol

The baseline tables and the measurement protocol live in `.scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md`.

- every simplification commit on these units re-runs the seven runners,
  updates the table with the new branch-tip hash and sizes
- a regression over 200 B on an untouched call site is a stop-and-report
- always measure with the suite flags above, `-d:release` folds debug runtime
  checks out of the MSL and shifts the byte counts

The runners are not part of the suite scan (no `test_`/`t_` prefix), run them manually.
