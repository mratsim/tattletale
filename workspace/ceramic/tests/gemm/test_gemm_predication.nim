## CPU unit test: the store-predication mask (cStoreMask) and the copy
## partition (partition_S / partition_D) against independent references.
##
## The store-mask reference derives each C-fragment element's tile
## coordinate from the partition view, without using the helper's own
## atom-offset math: crd2idx(view.layout, i) is the flat tile offset of
## fragment element i. The view and the fragment share the flat
## alignment. Element i of the view is element i of the fragment.
## The tile is col-major, so the row/col is offset mod tileM /
## offset div tileM.
##
## The copy-partition proof checks the partition views against the
## pinned chunk sequence for every thread: the thread's 16-byte chunks
## sit at the flat chunk positions c = tid + i·blockSize of the
## (TILE_M, TILE_K) k-tile. The reference derives each chunk's tile
## offset from the SHAPE coordinates of its flat position 4·c (the
## gemm_cta tApA construction), never from the partition's own layout.
## The gmem side runs a padded leading stride, the smem side the
## compact tile.
##
## Runs on the CPU, no GPU needed:
##   nim c -r workspace/ceramic/tests/gemm/test_gemm_predication.nim

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm/atoms_nvidia
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/ceramic/src/kernel_copy_gpu
import workspace/ceramic/src/atoms_copy
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic

{.experimental: "callOperator".}

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

const
  TILE_M = 32
  TILE_N = 16
  TILE_K = 32                 # the copy-partition k-tile depth
  thrM = tiled.thrM
  thrN = tiled.thrN
  thrK = tiled.thrK
  blockSize = tiled.threadCount()
  copyUnits = (TILE_M * TILE_K) div (4 * blockSize)   # the 16-byte chunks per thread

# The tile view only feeds layout algebra: the data pointer is never
# dereferenced, so one dummy buffer serves the C tile.
var dummy = newSeq[uint32](TILE_M * TILE_N)
let dummyPtr = cast[ptr UncheckedArray[uint32]](addr dummy[0])

proc main() =
  # every thread: the C store mask must match the partition-view reference
  # for the full and ragged extents
  for tid in 0 ..< blockSize:
    let thr = tiled.get_slice(tid)
    let tC = make_view(dummyPtr, (TILE_M, TILE_N), (1, TILE_M))
    let tCv = tiled.partition_C(thr, tC)
    # the partition view's data pointer is already advanced by the thread's
    # partition origin: the tile offset of element i is the view-relative
    # layout offset plus that origin (element units)
    let originC = (cast[int](tCv.data) - cast[int](tC.data)) div int(sizeof(uint32))

    # the store mask over a sweep of valid extents (full, ragged, tiny)
    for (validM, validN) in [(TILE_M, TILE_N), (16, 16), (16, 8), (8, 16),
                             (8, 8), (1, 1)]:
      var expected = 0
      for i in 0 ..< size(tCv.layout):
        let off = originC + toIntVal(crd2idx(tCv.layout, i))
        let m = off mod TILE_M
        let n = off div TILE_M
        if m < validM and n < validN:
          expected = expected or (1 shl i)
      let got = cStoreMask(tiled, tid, TILE_M, TILE_N, validM, validN)
      doAssert got == expected,
        "cStoreMask: thread " & $tid & " valid (" & $validM & ", " & $validN &
        "): got " & $got & ", reference " & $expected

  echo "  [OK] cStoreMask: the C store mask matches the partition-view reference" &
    " for every thread and valid extent"

  # ── the copy partition 1:1 proof (partition_S / partition_D) ──
  # for every thread, the partition view holds exactly the thread's
  # 16-byte chunks of the (TILE_M, TILE_K) k-tile at the flat chunk
  # positions c = tid + i·blockSize. The reference derives each chunk's
  # tile offset from the SHAPE coordinates of the flat position 4·c
  # (the gemm_cta tApA construction), never from the partition's own
  # layout. The gmem side runs a padded leading stride, the smem side
  # the compact tile.
  for tid in 0 ..< blockSize:
    let ldA = 80     # a padded leading stride, runtime in the kernel
    let tA = make_view(dummyPtr, (TILE_M, TILE_K), (1, ldA))
    let tAgA = partition_S(tA, CpAsyncAtom[uint32], blockSize, tid)
    let originA = (cast[int](tAgA.data) - cast[int](tA.data)) div int(sizeof(uint32))
    doAssert tAgA.layout.shape === (1, copyUnits),
      "partition_S: thread " & $tid & ": unit-view shape " & $tAgA.layout.shape &
      ", expected (1, " & $copyUnits & ")"
    for i in 0 ..< copyUnits:
      let c = tid + i * blockSize
      let (m0, k0) = idx2crd((TILE_M, TILE_K), 4 * c)
      let expected = toIntVal(m0) + toIntVal(k0) * ldA
      let got = originA + toIntVal(crd2idx(tAgA.layout, (0, i)))
      doAssert got == expected,
        "partition_S: thread " & $tid & " unit " & $i & ": tile offset " & $got &
        ", reference " & $expected
    let tD = make_view(dummyPtr, (TILE_M, TILE_K), (1, TILE_M))
    let tDsD = partition_D(tD, CpAsyncAtom[uint32], blockSize, tid)
    let originD = (cast[int](tDsD.data) - cast[int](tD.data)) div int(sizeof(uint32))
    doAssert tDsD.layout.shape === (1, copyUnits),
      "partition_D: thread " & $tid & ": unit-view shape " & $tDsD.layout.shape &
      ", expected (1, " & $copyUnits & ")"
    for i in 0 ..< copyUnits:
      let c = tid + i * blockSize
      let (m0, k0) = idx2crd((TILE_M, TILE_K), 4 * c)
      let expected = toIntVal(m0) + toIntVal(k0) * TILE_M
      let got = originD + toIntVal(crd2idx(tDsD.layout, (0, i)))
      doAssert got == expected,
        "partition_D: thread " & $tid & " unit " & $i & ": tile offset " & $got &
        ", reference " & $expected

  echo "  [OK] partition_S / partition_D: the partition views match the pinned" &
    " chunk sequence for every thread, padded gmem stride and compact smem tile"

main()
