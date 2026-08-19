# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## IR builtin kinds: resolution, clone survival, and real kernel emission.
##
## The coordinate and synchronization kinds live on the Symbol. Cloned idents
## keep them through the shared Symbol ref. The printers consume cloned trees:
## farmTopLevel puts `ast.clone()` in `fnTab`, and codegen re-clones for forward declarations.
## A kind lost on the clone path would compile cleanly and mis-emit. This test asserts:
## - the resolution of all six coordinates and the barrier
## - kind survival through a cloned tree
## - the emitted CUDA text of a kernel referencing every coordinate, plus the barrier, printf, and a shadowing local
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_builtin_kinds.nim

import std/[strutils, tables]
import workspace/crucible
import workspace/crucible/src/codegen/ir/gpu_types

# ═══════════════════════════════════════════════════════════════════════
# 1. Resolution tables
# ═══════════════════════════════════════════════════════════════════════
block:
  doAssert coordBuiltinKind("thread_position_in_grid") == gbkThreadPositionInGrid
  doAssert coordBuiltinKind("threadgroup_position_in_grid") == gbkThreadgroupPositionInGrid
  doAssert coordBuiltinKind("thread_position_in_threadgroup") == gbkThreadPositionInThreadgroup
  doAssert coordBuiltinKind("threads_per_threadgroup") == gbkThreadsPerThreadgroup
  doAssert coordBuiltinKind("threadgroups_per_grid") == gbkThreadgroupsPerGrid
  doAssert coordBuiltinKind("thread_index_in_threadgroup") == gbkThreadIndexInThreadgroup
  doAssert synchroBuiltinKind("threadgroup_barrier") == gbkThreadgroupBarrier
  # Non-coordinate builtins stay unkinded and emit verbatim.
  doAssert coordBuiltinKind("printf") == gbkNone
  doAssert coordBuiltinKind("cvtaGenericToShared") == gbkNone
  # The barrier is a call, not a coordinate. printf is not a barrier.
  doAssert coordBuiltinKind("threadgroup_barrier") == gbkNone
  doAssert synchroBuiltinKind("printf") == gbkNone
  doAssert synchroBuiltinKind("thread_position_in_grid") == gbkNone
  echo "  OK — resolution tables (6 coordinates, 1 barrier, gbkNone else)"

static:
  # The tables hold exactly the catalog's canonical declarations
  # (the static assertion in gpu_types.nim enforces this at compile time).
  doAssert GpuCoordBuiltinKindByName.len == 6
  doAssert GpuSynchroBuiltinKindByName.len == 1

# ═══════════════════════════════════════════════════════════════════════
# 2. Clone survival (the kind lives on the shared Symbol ref)
# ═══════════════════════════════════════════════════════════════════════
block:
  let coordSym = newSymbol("thread_position_in_grid")
  coordSym.coordBuiltin = gbkThreadPositionInGrid
  let coordIdent = GpuAst(kind: gpuIdent, symbol: coordSym)
  let coordClone = coordIdent.clone()
  doAssert coordClone.symbol == coordSym, "clone must share the Symbol ref"
  doAssert coordClone.symbol.coordBuiltin == gbkThreadPositionInGrid,
    "the coordinate kind must survive cloning"

  let barrierSym = newSymbol("threadgroup_barrier")
  barrierSym.synchroBuiltin = gbkThreadgroupBarrier
  let barrierCall = GpuAst(
    kind: gpuCall,
    cName: GpuAst(kind: gpuIdent, symbol: barrierSym),
    cArgs: @[])
  let barrierClone = barrierCall.clone()
  doAssert barrierClone.cName.symbol.synchroBuiltin == gbkThreadgroupBarrier,
    "the synchronization kind must survive cloning"
  echo "  OK — kinds survive clone via the shared Symbol ref"

# ═══════════════════════════════════════════════════════════════════════
# 3. Real CUDA emission (through the fnTab clones)
# ═══════════════════════════════════════════════════════════════════════
const kindCuda = cuda:
  proc kindKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    C[0] = thread_position_in_grid.x
    C[1] = threadgroup_position_in_grid.x
    C[2] = thread_position_in_threadgroup.x
    C[3] = threads_per_threadgroup.x
    C[4] = threadgroups_per_grid.x
    C[5] = thread_index_in_threadgroup
    threadgroup_barrier()
    printf("i = %u", C[0])
    let thread_position_in_grid = 5'u32
    C[6] = thread_position_in_grid

block:
  # Component access of each coordinate kind lowers to its CUDA spelling.
  doAssert "(blockIdx.x*blockDim.x+threadIdx.x)" in kindCuda
  doAssert "C[1] = blockIdx.x" in kindCuda
  doAssert "C[2] = threadIdx.x" in kindCuda
  doAssert "C[3] = blockDim.x" in kindCuda
  doAssert "C[4] = gridDim.x" in kindCuda
  # The flat thread index is the parenthesized x-major linearization.
  doAssert "threadIdx.z*blockDim.x*blockDim.y" in kindCuda
  # The barrier kind lowers to the native CUDA call even though getFnName
  # clobbers the callee symbol to gsProc: the gate is the kind, not symKind.
  doAssert "__syncthreads()" in kindCuda
  # printf has no kind: the call keeps its verbatim name.
  doAssert "printf(\"i = %u\", C[0])" in kindCuda
  # The local shadowing a canonical name has no kind: declaration and use
  # emit verbatim, never a coordinate spelling.
  doAssert "unsigned int thread_position_in_grid = 5U" in kindCuda
  doAssert "C[6] = thread_position_in_grid" in kindCuda
  doAssert "C[6] = (blockIdx" notin kindCuda
  echo "  OK — real CUDA emission (6 coordinates, barrier, printf, shadowed local)"

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All IR builtin-kind tests passed."
