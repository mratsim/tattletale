## Codesize ledger, local_tile family.
##
## Every kernel compiles one call site to Metal Shading Language and prints
## its MSL byte size, one call per family member:
## - static tiler (16, 16), runtime coords, one element read per kernel, the body mirrors the tile_io loadTile call shape
## - dynamic shapes, local_tile_dyn alone runs on the rank-4 GlView
## - every selector kernel falls back to the dynamic rank-2 view, the ledger carries its GlView failure texts
##
## Run from the tattletale/ dir, suite flags of config.nims testerCmd, usage in experiments/codesize/README.md.
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra/tiles

# ── floors, no selector machinery ──

# bare kernel, no ceramic machinery
const floorBareMsl = metal:
  proc floorBareKernel(C: ptr UncheckedArray[float32]) {.global.} =
    C[0] = C[1]

# dynamic rank-2 view only
const floorViewMsl = metal:
  proc floorViewKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    C[0] = float32 p.data[0]

# dynamic rank-2 view, one element read
const floorViewIndexMsl = metal:
  proc floorViewIndexKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    C[0] = float32 p[int R0, int C0]

# rank-4 GlView construction only
const floorGlMsl = metal:
  proc floorGlKernel(C: ptr UncheckedArray[float32]; B, D, M, N: int32) {.global.} =
    let gl = gd(C, B, D, M, N)
    C[0] = float32 gl.data[0]

# rank-4 GlView, one 4-coord element read, measures the no-tiler baseline
const floorGlIndexMsl = metal:
  proc floorGlIndexKernel(C: ptr UncheckedArray[float32]; B, D, M, N, R0, C0: int32) {.global.} =
    let gl = gd(C, B, D, M, N)
    C[0] = float32 gl[0, 0, int R0, int C0]

# ── local_tile_dyn, the working GlView entry (tile_algebra/tiles.nim:131) ──

const localTileDynMsl = metal:
  proc localTileDynKernel(C: ptr UncheckedArray[float32]; B, D, M, N, R0, C0: int32) {.global.} =
    let gl = gd(C, B, D, M, N)
    let src = local_tile_dyn(gl, 16, 16, (0, 0, int R0, int C0))
    C[0] = float32 src[int R0, int C0]

# ── partition selectors, dynamic rank-2 fallback ──
# on the GlView the selectors do not compose the loadTile call shape,
# failure texts and full-arity numbers live in the ledger table:
# - inner_partition and local_tile with tiler (16, 16) die inside slice
#   (filterZipWith rank mismatch 2 vs 4)
# - rank-4 tiler forms compile but the within-tile element read dies, extra
#   flat shape leaves trip crd2idxRecur (invalid index)

# inner_partition call site (tensor_selectors.nim:137)
const innerPartitionMsl = metal:
  proc innerPartitionKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let src = inner_partition(p, (16, 16), (int R0, int C0))
    C[0] = float32 src[int R0, int C0]

# outer_partition call site (tensor_selectors.nim:157)
const outerPartitionMsl = metal:
  proc outerPartitionKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let src = outer_partition(p, (16, 16), (int R0, int C0))
    C[0] = float32 src[int R0, int C0]

# 2-arg local_tile, the inner_partition alias (tensor_selectors.nim:177),
# byte-identity with innerPartitionKernel is the alias check
const localTile2argMsl = metal:
  proc localTile2argKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let src = local_tile(p, (16, 16), (int R0, int C0))
    C[0] = float32 src[int R0, int C0]

# 4-arg local_tile with projection (1, 1), dice overhead appears as the delta
# over the 2-arg form (tensor_selectors.nim:182)
const localTile4argMsl = metal:
  proc localTile4argKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let src = local_tile(p, (16, 16), (int R0, int C0), (1, 1))
    C[0] = float32 src[int R0, int C0]

# local_partition call site (tensor_selectors.nim:188), the idx2crd route feeds outer_partition
const localPartitionMsl = metal:
  proc localPartitionKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0, I: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let tile = make_layout((16, 16), (1, 16))
    let src = local_partition(p, tile, int I)
    C[0] = float32 src[int R0, int C0]

# ── fully static layout row ──

# fully static rank-2 layout, runtime coords, mirrors the static-layout
# baseline row of the ledger
const localTileStaticMsl = metal:
  proc localTileStaticKernel(C: ptr UncheckedArray[float32]; R0, C0: int32) {.global.} =
    let p = make_view(C, make_layout((64, 64), (64, 1)))
    let src = local_tile(p, (16, 16), (int R0, int C0))
    C[0] = float32 src[int R0, int C0]

echo "floorBareKernel: ", cstring(floorBareMsl).len
echo "floorViewKernel: ", cstring(floorViewMsl).len
echo "floorViewIndexKernel: ", cstring(floorViewIndexMsl).len
echo "floorGlKernel: ", cstring(floorGlMsl).len
echo "floorGlIndexKernel: ", cstring(floorGlIndexMsl).len
echo "localTileDynKernel: ", cstring(localTileDynMsl).len
echo "innerPartitionKernel: ", cstring(innerPartitionMsl).len
echo "outerPartitionKernel: ", cstring(outerPartitionMsl).len
echo "localTile2argKernel: ", cstring(localTile2argMsl).len
echo "localTile4argKernel: ", cstring(localTile4argMsl).len
echo "localPartitionKernel: ", cstring(localPartitionMsl).len
echo "localTileStaticKernel: ", cstring(localTileStaticMsl).len
