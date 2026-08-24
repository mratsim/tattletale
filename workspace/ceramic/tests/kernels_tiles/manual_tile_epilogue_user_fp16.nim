## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## On-device user-defined epilogue (manual, Metal): the test-local
## `EpiScale` (type + `apply` proc) passed as the object to
## `gemm_with_epilogue`, vs the triple-loop GEMM reference scaled by hand.
##
## The 8×8×8 mma's cross-lane reduction needs subgroup shuffles, which
## Apple's OpenCL-to-Metal translation rejects, so this gate runs on
## Metal on this machine; on OpenCL 2.0+ platforms it runs on OpenCL.
##
## Run: nim c -r --hints:off --warnings:off \
##   --outdir:build/tests/manual_tile_epilogue_user_fp16 \
##   --nimcache:nimcache/tests/manual_tile_epilogue_user_fp16 \
##   workspace/ceramic/tests/kernels_tiles/manual_tile_epilogue_user_fp16.nim

import workspace/crucible
import ../libtest_epilogues
import ../../src/atoms
import ../../src/kernels/k_tile_gemm_epilogues

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  User epilogue: a type plus an `apply` proc
# ═════════════════════════════════════════════════════════════════════════

type EpiScale[T] = object
  ## Scale epilogue: D = s·AB.
  s*: T

func apply[T, Sh, StAB, StR](
    op: EpiScale[T];
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## Per-thread epilogue math: tmp = s·AB.
  const S = toIntVal(size(tmp))
  for i in 0 ..< S:
    tmp(i) = op.s * AB(i)

proc apply[T; R, C: static int; AT, ABT: static MmaAtom](
    op: EpiScale[T];
    tmp: var RtLeft[T, R, C, AT];
    AB: RtLeft[T, R, C, ABT]) {.inline.} =
  ## Per-slot epilogue math (the fragment-resident accumulator form):
  ## tmp = s·AB. The two tile params carry separate atom params. The static
  ## assert enforces the shared subtile grid and per-lane count (see the shipped
  ## applies).
  static:
    doAssert AT.mnk.m == ABT.mnk.m and AT.mnk.n == ABT.mnk.n and
      toIntVal(AT.valuesPerThread(opC)) == toIntVal(ABT.valuesPerThread(opC)),
      "apply: the accumulator and operand tiles must share the atom's subtile grid and per-lane count"
  const rowTiles = R div AT.mnk.m
  const colTiles = C div AT.mnk.n
  const vpt = toIntVal(AT.valuesPerThread(opC))
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        tmp.frags[n][m].frag[v] = op.s * AB.frags[n][m].frag[v]

static:
  doAssert EpiScale[float32] is Epilogue, "EpiScale must satisfy the Epilogue concept"

const msl = metal:
  proc fusedScaleUser(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                      Scale: float32, N, K, M: int32) {.global.} =
    gemm_with_epilogue(D, A, B, N, K, M, EpiScale[float32](s: Scale))

proc runTest() =
  let M = 64
  let N = 32
  let K = 32
  let (Ah, Bh) = fillAB(M, N, K, M, N, K)
  var D = newSeq[float32](M * N)
  var engine = bkMetal.init()
  engine.ingest(msl)
  engine.run << (grid: (N div 32, M div 32), blk: (32, 1)) >> (
    "fusedScaleUser", D, (Ah, Bh, 2.0'f32, int32(M), int32(K), int32(N)))
  assertAllClose(D, scale(gemmRef(M, N, K, M, N, K, Ah, Bh), 2.0'f32))
  echo "epilogue_user ", M, "x", N, "x", K, " (EpiScale s=2): PASS on Metal"

when isMainModule:
  runTest()
