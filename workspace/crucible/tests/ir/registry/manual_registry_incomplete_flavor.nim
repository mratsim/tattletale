## Registry flavor-key completeness: a flavored target key missing a family fails the compile
##
## Test that a `target fake(fa):` section with only a signature row is a compile
## error from the genTargets family-resolution check, not a silently incomplete flavor
##
## Run (must fail to compile):
##   nim check --hints:off --warnings:off \
##     --path:./ workspace/crucible/tests/ir/registry/manual_registry_incomplete_flavor.nim

import std/tables
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/targets_macros

genTargets:
  target fake(fa):
    sig: SigRow(entryLinkage: "fake ")
