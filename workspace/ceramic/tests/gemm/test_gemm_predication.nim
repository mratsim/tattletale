## CPU unit test: the ragged-tile predication helpers (aFragmentTileRow,
## bFragmentTileCol, cStoreMask, aFragmentTileCoord, bFragmentTileCoord)
## against an independent reference.
##
## The reference derives each fragment element's tile coordinate from the
## partition views, without using the helpers' own atom-offset math:
## crd2idx(view.layout, i) is the flat tile offset of fragment element i
## (the same flat alignment the gather uses, element i of the view ←
## element i of the fragment), and the tile is col-major, so the row/col
## is offset mod tileM / mod tileN and the K coordinate is offset div
## tileM / div tileN.
##
## The ragged-K helpers (aFragmentTileCoord / bFragmentTileCoord) must
## agree with the row/col helpers on the row/col component and return the
## full tile K (the offset div tileM / div tileN), the predicate used to
## zero-fill the ragged-K residue k-tile. This is the one place the
## helpers' rest-coordinate decode (restK = i div V through the
## partition's rest modes) is checked against an independent view-based
## reference.
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
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic

{.experimental: "callOperator".}

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

const
  TILE_M = 32
  TILE_N = 16
  TILE_K = 32
  thrM = toIntVal(tiled.threadLayout.shape[0])
  thrN = toIntVal(tiled.threadLayout.shape[1])
  thrK = toIntVal(tiled.threadLayout.shape[2])
  blockSize = toIntVal(tiled.atom.threadCount(opA)) * thrM * thrN * thrK

# The tile views only feed layout algebra: the data pointer is never
# dereferenced, so one dummy buffer serves all three operands.
var dummy = newSeq[uint32](TILE_M * TILE_N * TILE_K)
let dummyPtr = cast[ptr UncheckedArray[uint32]](addr dummy[0])

proc main() =
  # every thread: the A/B fragment rows/cols, the K coordinates and the C
  # store mask must match the partition-view reference for the full and
  # ragged extents
  for tid in 0 ..< blockSize:
    let thr = tiled.get_slice(tid)
    let tA = make_view(dummyPtr, (TILE_M, TILE_K), (1, TILE_M))
    let tB = make_view(dummyPtr, (TILE_N, TILE_K), (1, TILE_N))
    let tC = make_view(dummyPtr, (TILE_M, TILE_N), (1, TILE_M))
    let tAv = tiled.partition_A(thr, tA)
    let tBv = tiled.partition_B(thr, tB)
    let tCv = tiled.partition_C(thr, tC)
    # the partition view's data pointer is already advanced by the thread's
    # partition origin: the tile offset of element i is the view-relative
    # layout offset plus that origin (element units)
    let originA = (cast[int](tAv.data) - cast[int](tA.data)) div int(sizeof(uint32))
    let originB = (cast[int](tBv.data) - cast[int](tB.data)) div int(sizeof(uint32))
    let originC = (cast[int](tCv.data) - cast[int](tC.data)) div int(sizeof(uint32))

    for i in 0 ..< size(tAv.layout):
      let offA = originA + toIntVal(crd2idx(tAv.layout, i))
      let refRow = offA mod TILE_M
      let refK = offA div TILE_M
      let gotRow = aFragmentTileRow(tiled, thr, TILE_M, i)
      doAssert gotRow == refRow,
        "aFragmentTileRow: thread " & $tid & " element " & $i & ": got " & $gotRow &
        ", reference " & $refRow
      let gotA = aFragmentTileCoord(tiled, thr, TILE_M, TILE_K, i)
      doAssert gotA[0] == refRow,
        "aFragmentTileCoord row: thread " & $tid & " element " & $i & ": got " & $gotA[0] &
        ", reference " & $refRow
      doAssert gotA[1] == refK,
        "aFragmentTileCoord k: thread " & $tid & " element " & $i & ": got " & $gotA[1] &
        ", reference " & $refK

    for i in 0 ..< size(tBv.layout):
      let offB = originB + toIntVal(crd2idx(tBv.layout, i))
      let refCol = offB mod TILE_N
      let refK = offB div TILE_N
      let gotCol = bFragmentTileCol(tiled, thr, TILE_N, i)
      doAssert gotCol == refCol,
        "bFragmentTileCol: thread " & $tid & " element " & $i & ": got " & $gotCol &
        ", reference " & $refCol
      let gotB = bFragmentTileCoord(tiled, thr, TILE_N, TILE_K, i)
      doAssert gotB[0] == refCol,
        "bFragmentTileCoord col: thread " & $tid & " element " & $i & ": got " & $gotB[0] &
        ", reference " & $refCol
      doAssert gotB[1] == refK,
        "bFragmentTileCoord k: thread " & $tid & " element " & $i & ": got " & $gotB[1] &
        ", reference " & $refK

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
      let got = cStoreMask(tiled, thr, TILE_M, TILE_N, validM, validN)
      doAssert got == expected,
        "cStoreMask: thread " & $tid & " valid (" & $validM & ", " & $validN &
        "): got " & $got & ", reference " & $expected

  echo "  [OK] predication helpers: A/B fragment rows/cols, the K coordinates and the C store" &
    " mask match the partition-view reference for every thread and valid extent"

main()
