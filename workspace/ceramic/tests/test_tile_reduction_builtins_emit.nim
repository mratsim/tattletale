## Tile-layer reductions: MSL emission + the on-device thread-layout check.
## Asserts the attn kernel's `simd_shuffle_down`/`simd_shuffle` MSL spellings.
## Checks the (1, 1, 1) congruence on-device: (tm, tn) = (0, 0) for every
## lane; the 8×8×8 atom's thread layout keeps one atom per threadgroup.
##
## Run from the tattletale root (also picked up by `nim test_ceramic`):
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/ceramic/tests/test_tile_reduction_builtins_emit.nim

import std/strutils
import workspace/crucible
import workspace/ceramic/src/tile_algebra/tile_fma_partition
import workspace/ceramic/src/kernels/k_tile_attn

const probeMsl = metal:
  proc fmaSliceProbe(outBuf: ptr UncheckedArray[int32]) {.global.} =
    let tid = thread_index_in_threadgroup
    let thr = fmaSlice[getTileConfig(float32), FmaThreadLayout]()
    outBuf[tid * 2] = int32(thr.tm)
    outBuf[tid * 2 + 1] = int32(thr.tn)

const attnMsl = metal:
  proc attnD64(o, q, k, v: ptr UncheckedArray[float16], H, N: int32) {.global.} =
    attn_fwd(q, k, v, o, H, N, 64)

proc runTest() =
  # The attn kernel's online softmax runs the full tree: row_max uses the
  # shuffleDown steps and row_sum adds the leader broadcast (simdShuffle);
  # the mma's cross-lane reduction adds its own gathers.
  doAssert "simd_shuffle_down(" in attnMsl,
    "attn MSL missing simd_shuffle_down from the reduction builtin:\n" & attnMsl
  doAssert "simd_shuffle(" in attnMsl,
    "attn MSL missing simd_shuffle from the reduction builtin:\n" & attnMsl
  let nDown = attnMsl.count("simd_shuffle_down(")
  let nShuffle = attnMsl.count("simd_shuffle(")
  doAssert nDown >= 2 and nShuffle >= 2,
    "attn MSL shuffle count too low: " & $nDown & " down, " & $nShuffle & " shuffle:\n" & attnMsl
  echo "  OK: attn MSL emits ", nDown, " simd_shuffle_down + ", nShuffle,
       " simd_shuffle from the reduction builtin kinds"
  # The config folds at the instantiation site: no config or backend names
  # reach the emitted kernel.
  doAssert "getTileConfig" notin attnMsl and "ccGetBackend" notin attnMsl,
    "attn MSL leaks the config path:\n" & attnMsl
  # The (1, 1, 1) congruence, on-device: fmaSlice's (tm, tn) must be
  # (0, 0) for every one of the 32 lanes. The loads, the mma and the
  # reductions all partition by this mapping.
  var engine = bkMetal.init()
  engine.ingest(probeMsl)
  var res: array[64, int32]
  engine.run << (blk: (32, 1)) >> ("fmaSliceProbe", res, ())
  for tid in 0 ..< 32:
    doAssert res[tid * 2] == int32(0),
      "tid " & $tid & ": tm " & $res[tid * 2] & " != 0"
    doAssert res[tid * 2 + 1] == int32(0),
      "tid " & $tid & ": tn " & $res[tid * 2 + 1] & " != 0"
  echo "  OK: on-device (tm, tn) = (0, 0) for all 32 threads"

when isMainModule:
  runTest()
