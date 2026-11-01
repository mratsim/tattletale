## Registry cross-check: a second row for an existing scalar cell fails the compile
##
## Test that a second (ctInt, 8) row beside msl's `char` row is a compile error
## from the genTargets cell-coverage check, not a silent last-one-wins overwrite
##
## Run (must fail to compile):
##   nim check --hints:off --warnings:off \
##     --path:./ workspace/crucible/tests/ir/registry/manual_registry_drifted_row.nim

import std/tables
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/targets_rows
import workspace/crucible/src/codegen/backends/targets_macros

genTargets:
  target mslDrift:
    # msl's real scalar row plus a second row for the (ctInt, 8) cell with the drifted CUDA name
    scalar: ScalarRow(cells: mslScalarRow().cells & @[
      ScalarCell(kind: ctInt, width: 8, spelling: "signed char", supported: true)])
    lit: mslLitRow()
    kind: mslKindRow()
    space: mslSpaceRow()
    conv: mslConvRow()
    coord: mslCoordRow()
    sig: mslSigRow()
    vector: mslVectorRule()
