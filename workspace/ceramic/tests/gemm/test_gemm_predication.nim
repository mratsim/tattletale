## CPU unit test: the store-predication mask (cStoreMask) against an
## independent reference.
##
## The reference derives each C-fragment element's tile coordinate from
## the partition view, without using the helper's own atom-offset math:
## crd2idx(view.layout, i) is the flat tile offset of fragment element i.
## The view and the fragment share the flat alignment.
## Element i of the view is element i of the fragment.
## The tile is col-major, so the row/col is offset mod tileM / offset div tileM.
##
## The load-side predication needs no unit test: the chunk coordinates
## come from the tile shape via the shape-based idx2crd
## (stride-independent by construction), covered by the GPU NaN
## verification in the ragged manual tests.
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
  thrM = tiled.thrM
  thrN = tiled.thrN
  thrK = tiled.thrK
  blockSize = tiled.threadCount()

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
      let got = cStoreMask(tiled, thr, TILE_M, TILE_N, validM, validN)
      doAssert got == expected,
        "cStoreMask: thread " & $tid & " valid (" & $validM & ", " & $validN &
        "): got " & $got & ", reference " & $expected

  echo "  [OK] cStoreMask: the C store mask matches the partition-view reference" &
    " for every thread and valid extent"

main()
