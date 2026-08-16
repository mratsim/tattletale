# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## The int8 copy partition: the cp.async 16-byte chunk is 16 int8
## elements (the atom NumPacked 16), so the copy unit is the (16, 1)
## column chunk and the chunk positions are c = thrIdx + i·blockSize
## at the element offsets 16·c. The (16, 8) tile with 4 threads
## gives the thread slice shape (1, 2) and the chunk offsets
## 16·(t + i·4), each unit the atom's 16-byte column chunk.
##
## Runs on the CPU, no GPU needed:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/test_gemm_copy_int8.nim --nimcache:nimcache/tests/test_gemm_copy_int8.nim \
##     workspace/ceramic/tests/gemm/test_gemm_copy_int8.nim

{.experimental: "callOperator".}

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/atoms_copy
import workspace/ceramic/src/kernel_copy_gpu
import workspace/ceramic/tests/layouts_testutils

proc runInt8PartitionTests =
  block:
    check numPacked(CpAsyncAtom[int8]), 16, int
    doAssert tilerMN(CpAsyncAtom[int8]) === (16, 1)
    doAssert tilerMN(CpAsyncAtom[int64]) === (2, 1)
  block:
    ## Compact tile: expected offset m + k·TILE_M, decoded from the
    ## element offset 16·c through the (16, 8) tile shape
    const blockSize = 4
    const TILE_M = 16
    const TILE_K = 8
    const copyUnits = 2   # (16·8) div (16·4)
    const pL = thrfrg_copy(make_layout((TILE_M, TILE_K), (1, TILE_M)),
                           CpAsyncAtom[int8], blockSize)
    for tid in 0 ..< blockSize:
      let origin = crd2idx(pL, tid)
      let sub = slice(pL, (tid, _, _))
      doAssert sub.shape === (1, copyUnits),
        "int8 partition: thread " & $tid & ": slice shape " & $sub.shape &
        ", expected (1, " & $copyUnits & ")"
      for i in 0 ..< copyUnits:
        let off = toIntVal(origin) + toIntVal(crd2idx(sub, (0, i)))
        let c = tid + i * blockSize
        let (m0, k0) = idx2crd((TILE_M, TILE_K), 16 * c)
        let expected = toIntVal(m0) + toIntVal(k0) * TILE_M
        doAssert off == expected,
          "int8 partition: thread " & $tid & " unit " & $i & ": tile offset " &
          $off & ", reference " & $expected
  block:
    ## Padded leading stride: expected offset m + k·80
    const blockSize = 4
    const TILE_M = 16
    const TILE_K = 8
    const copyUnits = 2
    const ldA = 80
    const pL = thrfrg_copy(make_layout((TILE_M, TILE_K), (1, ldA)),
                           CpAsyncAtom[int8], blockSize)
    for tid in 0 ..< blockSize:
      let origin = crd2idx(pL, tid)
      let sub = slice(pL, (tid, _, _))
      for i in 0 ..< copyUnits:
        let off = toIntVal(origin) + toIntVal(crd2idx(sub, (0, i)))
        let c = tid + i * blockSize
        let (m0, k0) = idx2crd((TILE_M, TILE_K), 16 * c)
        let expected = toIntVal(m0) + toIntVal(k0) * ldA
        doAssert off == expected,
          "int8 padded partition: thread " & $tid & " unit " & $i &
          ": tile offset " & $off & ", reference " & $expected
  echo "    int8 copy partition: 3 cases OK"

proc runTests =
  echo "\n── int8 copy partition (16-byte chunk = 16 int8 elements) ──"
  runInt8PartitionTests()
  echo "  All tests passed."

when isMainModule:
  runTests()
