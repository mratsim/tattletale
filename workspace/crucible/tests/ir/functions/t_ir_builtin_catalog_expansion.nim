## Builtin catalog alias expansion
##
## Test that coordinate aliases expand to canonical builtins in a typed GPU
## macro body and a runtime-dim use errors at Nim sem
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/functions/t_ir_builtin_catalog_expansion.nim

import std/strutils
import workspace/crucible

# ═══════════════════════════════════════════════════════════════════════
# 1. In-macro template expansion (verified inside `opencl:`)
# ═══════════════════════════════════════════════════════════════════════
const expansionKernel = opencl:
  proc expansionKernel(C: ptr UncheckedArray[int]) {.global.} =
    C[0] = blockIdx.x              # dot context
    C[1] = blockIdx[1]             # positional tuple-bracket access
    C[2] = get_global_id(0)        # static-dim: folds to component 0
    C[3] = get_global_id(1)        # static-dim: folds to component 1
    C[4] = get_global_id(2)        # static-dim: folds to component 2
    C[5] = thread_index_in_threadgroup
    syncthreads()                  # CUDA-idiom barrier alias
    barrier(CLK_LOCAL_MEM_FENCE)   # OpenCL-idiom barrier alias

block:
  # Aliases must expand to canonical names, and the OpenCL printer must map
  # each canonical back to its native `get_*(d)` call. A canonical name
  # leaking into the output would mean the alias never expanded
  # (or the printer case missed it).
  doAssert "thread_position_in_grid" notin expansionKernel
  doAssert "threadgroup_position_in_grid" notin expansionKernel
  doAssert "thread_position_in_threadgroup" notin expansionKernel
  doAssert "thread_index_in_threadgroup" notin expansionKernel
  # dot context: blockIdx.x -> threadgroup_position_in_grid.x -> get_group_id(0)
  doAssert "get_group_id(0)" in expansionKernel
  # positional: blockIdx[1] -> threadgroup_position_in_grid.y -> get_group_id(1)
  doAssert "get_group_id(1)" in expansionKernel
  # static-dim get_global_id(d) -> thread_position_in_grid.d -> get_global_id(d)
  doAssert "get_global_id(0)" in expansionKernel
  doAssert "get_global_id(1)" in expansionKernel
  doAssert "get_global_id(2)" in expansionKernel
  # flat local index: x-major linearization of get_local_id / get_local_size
  doAssert "(get_local_id(2)*get_local_size(0)*get_local_size(1) + get_local_id(1)*get_local_size(0) + get_local_id(0))" in expansionKernel
  # both barrier aliases fold into the canonical call -> native spelling
  doAssert expansionKernel.count("barrier(CLK_LOCAL_MEM_FENCE)") == 2
  echo "  OK — in-macro alias expansion (dot / positional / static-dim / barrier aliases)"

block:
  # Runtime dim: a non-static argument cannot bind the `static int` template
  # param, so `get_global_id(d)` with a runtime `d` is a loud Nim error.
  # The rejection is verified here inside the `opencl:` macro body.
  static:
    doAssert not compiles(block:
      const bad = opencl:
        proc bad(C: ptr UncheckedArray[int]) {.global.} =
          let d = 0
          C[0] = get_global_id(d)
    )
  echo "  OK — runtime-dim get_global_id(d) rejected loudly inside the macro"

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All builtin catalog expansion tests passed."
