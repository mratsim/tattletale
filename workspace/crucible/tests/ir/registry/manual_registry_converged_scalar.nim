## Registry divergence ledger: converging msl's and cuda's 8-bit signed name fails the compile
##
## Test that naming cuda's (ctInt, 8) cell with msl's `char` is a compile error
## from the divergence check, not a silent convergence. msl names 8-bit signed
## `char` and 64-bit signed `long`, cuda names the same widths `signed char`
## and `long long`. The ledger keeps the two rows apart.
##
## Run (must fail to compile):
##   nim check --hints:off --warnings:off \
##     --path:./ workspace/crucible/tests/ir/registry/manual_registry_converged_scalar.nim

import std/sequtils
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/targets_rows

static:
  # cuda's real row with the (ctInt, 8) name converged onto msl's `char`
  var converged = cudaScalarRow()
  for c in converged.cells.mitems:
    if c.kind == ctInt and c.width == 8:
      c.spelling = "char"
  checkScalarDivergences(
    [("msl", mslScalarRow()), ("cuda", converged)], ScalarDivergences)
