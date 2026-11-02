## Shared C declarator: pointer-to-array declarations.
##
## Test that `cDecl` over the CUDA tables renders the identifier of a pointer
## to array nested into the declarator, one bracket pair per array level.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests workspace/crucible/tests/codegen/printers/t_cdecl_ptr_array.nim

import std/unittest
import workspace/crucible/src/codegen/backends/print_common
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build

suite "shared C declarator - pointer to array":

  test "a pointer to array nests the identifier into the declarator":
    let t = tPtr(caDevice, tArray(4, tFloat(32)))
    check printCDecl(cudaScalarNames, cudaVectorRule, "cuda", t, "v", false) ==
      "float (*v)[4]"

  test "the cast form renders the star with no identifier":
    let t = tPtr(caDevice, tArray(4, tFloat(32)))
    check printCDecl(cudaScalarNames, cudaVectorRule, "cuda", t, "", true) ==
      "float (*)[4]"

  test "a thread-space pointer to array (the var array param shape) names the same":
    let t = tPtr(caThread, tArray(4, tFloat(32)))
    check printCDecl(cudaScalarNames, cudaVectorRule, "cuda", t, "a", false) ==
      "float (*a)[4]"

  test "a pointer to array of array extends the bracket list in declaration order":
    let t = tPtr(caDevice, tArray(2, tArray(3, tFloat(32))))
    check printCDecl(cudaScalarNames, cudaVectorRule, "cuda", t, "m", false) ==
      "float (*m)[2][3]"
