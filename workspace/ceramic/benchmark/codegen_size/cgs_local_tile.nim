## Codesize ledger, local_tile family.
##
## Every kernel compiles one call site to Metal Shading Language and prints
## its MSL byte size, one call per family member:
## - static tiler (16, 16), runtime coords, one element read per kernel, the body mirrors the tile_io loadTile call shape
## - dynamic shapes, local_tile_dyn alone runs on the rank-4 GlView
## - every selector kernel falls back to the dynamic rank-2 view, the ledger carries its GlView failure texts
##
## v2 adds instrument rows on the same selector input.
## - frozenFloor/frozenFloor2 hand-expand the divide chain, the emission a byte-identical rewrite should produce
## - twoTile measures the 2nd-call marginal, zippedDivideOnly/logicalDivideOnly isolate the divide chain
## - dynPair inlines the local_tile_dyn body, marginals and the headroom split print at the end
## MSL dumps of the v2 rows land in /tmp under a zzz_ prefix.
##
## Run from the worktree root, suite flags of config.nims testerCmd, usage in benchmark/codegen_size/README.md.
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

# ── frozen-layer mimic, what a byte-identical rewrite should emit ──
# hand-expanded zipped_divide + slice + concat + make_view chain on the same
# dynamic rank-2 input as the selector kernels
# structure mirrors the inner_partition per-call chain
# (zd, c, keptRest, offset, subLayout), each node rebuilt
# from make_layout, slice, concat, crd2idx, make_view

# frozen mimic, one tile call, stride (1, S) and the runtime within-tile read
const frozenFloorMsl = metal:
  proc frozenFloorKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let zd = make_layout(
      ((Int[16](), Int[16]()),
       ((p.layout.shape[0] + 15) div 16, (p.layout.shape[1] + 15) div 16)),
      ((p.layout.stride[0], p.layout.stride[1]),
       (p.layout.stride[0] * 16, p.layout.stride[1] * 16)))
    let c = (int R0, int C0)
    let keptRest = make_layout(slice(zd.shape[1], c), slice(zd.stride[1], c))
    let offset = crd2idx(c, zd.shape[1], zd.stride[1])
    let subLayout = make_layout(
      concat(zd.shape[0], keptRest.shape),
      concat(zd.stride[0], keptRest.stride))
    let t = make_view(p.data[0].addr +% toIntVal(offset), subLayout)
    C[0] = float32 t[int R0, int C0]

# frozen mimic, 2 tile calls, the per-call frozen cost is the delta over one call
const frozenFloor2Msl = metal:
  proc frozenFloor2Kernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let zd = make_layout(
      ((Int[16](), Int[16]()),
       ((p.layout.shape[0] + 15) div 16, (p.layout.shape[1] + 15) div 16)),
      ((p.layout.stride[0], p.layout.stride[1]),
       (p.layout.stride[0] * 16, p.layout.stride[1] * 16)))
    let c = (int R0, int C0)
    let keptRest = make_layout(slice(zd.shape[1], c), slice(zd.stride[1], c))
    let offset = crd2idx(c, zd.shape[1], zd.stride[1])
    let subLayout = make_layout(
      concat(zd.shape[0], keptRest.shape),
      concat(zd.stride[0], keptRest.stride))
    let t = make_view(p.data[0].addr +% toIntVal(offset), subLayout)
    let c2 = (int C0, int R0)
    let keptRest2 = make_layout(slice(zd.shape[1], c2), slice(zd.stride[1], c2))
    let offset2 = crd2idx(c2, zd.shape[1], zd.stride[1])
    let subLayout2 = make_layout(
      concat(zd.shape[0], keptRest2.shape),
      concat(zd.stride[0], keptRest2.stride))
    let u = make_view(p.data[0].addr +% toIntVal(offset2), subLayout2)
    C[0] = float32 t[int R0, int C0]
    C[1] = float32 u[int R0, int C0]

# 2 real local_tile calls on the same view
# 2nd-call marginal = delta over the 1-call selector row
const twoTileMsl = metal:
  proc twoTileKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let t = local_tile(p, (16, 16), (int R0, int C0))
    let u = local_tile(p, (16, 16), (int C0, int R0))
    C[0] = float32 t[int R0, int C0]
    C[1] = float32 u[int R0, int C0]

# ── divide chains in isolation, same runtime input as the selectors ──

# zipped_divide on a rank-2 dynamic layout, no view, no read
const zippedDivideOnlyMsl = metal:
  proc zippedDivideOnlyKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    let zd = zipped_divide(L, (16, 16))
    C[0] = float32(crd2idx(zd, ((3, 2), (1, 0))))

# logical_divide on a rank-2 dynamic layout, no view, no read
const logicalDivideOnlyMsl = metal:
  proc logicalDivideOnlyKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    let ld = logical_divide(L, (16, 16))
    C[0] = float32(crd2idx(ld, ((3, 2), (1, 0))))

# ── rank-2 inline of the local_tile_dyn body ──
# static 16x16 tile, runtime origin, tile stride carried from the source layout
# same input and read as the selector kernels

const dynPairMsl = metal:
  proc dynPairKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let base = int R0 * 1 * 16 + int C0 * int S * 16
    let t = make_view(p.data[0].addr +% base, (Int[16](), Int[16]()), (1, int S))
    C[0] = float32 t[int R0, int C0]

# ── fully static layout row ──

# fully static rank-2 layout, runtime coords, mirrors the static-layout
# baseline row of the ledger
const localTileStaticMsl = metal:
  proc localTileStaticKernel(C: ptr UncheckedArray[float32]; R0, C0: int32) {.global.} =
    let p = make_view(C, make_layout((64, 64), (64, 1)))
    let src = local_tile(p, (16, 16), (int R0, int C0))
    C[0] = float32 src[int R0, int C0]

# ── kernel sizes ──

echo "floorBareKernel: ", floorBareMsl.len
echo "floorViewKernel: ", floorViewMsl.len
echo "floorViewIndexKernel: ", floorViewIndexMsl.len
echo "floorGlKernel: ", floorGlMsl.len
echo "floorGlIndexKernel: ", floorGlIndexMsl.len
echo "localTileDynKernel: ", localTileDynMsl.len
echo "innerPartitionKernel: ", innerPartitionMsl.len
echo "outerPartitionKernel: ", outerPartitionMsl.len
echo "localTile2argKernel: ", localTile2argMsl.len
echo "localTile4argKernel: ", localTile4argMsl.len
echo "localPartitionKernel: ", localPartitionMsl.len
echo "frozenFloorKernel: ", frozenFloorMsl.len
echo "frozenFloor2Kernel: ", frozenFloor2Msl.len
echo "twoTileKernel: ", twoTileMsl.len
echo "zippedDivideOnlyKernel: ", zippedDivideOnlyMsl.len
echo "logicalDivideOnlyKernel: ", logicalDivideOnlyMsl.len
echo "dynPairKernel: ", dynPairMsl.len
echo "localTileStaticKernel: ", localTileStaticMsl.len

# ── derived metrics ──
# marginal per family member = kernel bytes − floorViewIndex bytes
# local_tile_dyn reads the GlView floor floorGlIndex instead
# headroom split = frozen-layer share vs emission residue over the mimic

echo "marginal innerPartition: ",
  innerPartitionMsl.len - floorViewIndexMsl.len
echo "marginal outerPartition: ",
  outerPartitionMsl.len - floorViewIndexMsl.len
echo "marginal localTile2arg: ",
  localTile2argMsl.len - floorViewIndexMsl.len
echo "marginal localTile4arg: ",
  localTile4argMsl.len - floorViewIndexMsl.len
echo "marginal localPartition: ",
  localPartitionMsl.len - floorViewIndexMsl.len
echo "marginal localTileStatic: ",
  localTileStaticMsl.len - floorViewIndexMsl.len
echo "marginal dynPair: ",
  dynPairMsl.len - floorViewIndexMsl.len
echo "marginal localTileDyn: ",
  localTileDynMsl.len - floorGlIndexMsl.len
echo "frozen-layer share of 1-call marginal: ",
  frozenFloorMsl.len - floorViewIndexMsl.len
echo "emission residue over mimic: ",
  localTile2argMsl.len - frozenFloorMsl.len
echo "2nd-call marginal chain: ",
  twoTileMsl.len - localTile2argMsl.len
echo "frozen-layer per-call cost: ",
  frozenFloor2Msl.len - frozenFloorMsl.len

# MSL dumps of the v2 rows, zzz_ prefix, transient
writeFile("/tmp/zzz_frozenFloor.msl", $frozenFloorMsl)
writeFile("/tmp/zzz_frozenFloor2.msl", $frozenFloor2Msl)
writeFile("/tmp/zzz_twoTile.msl", $twoTileMsl)
writeFile("/tmp/zzz_zippedDivideOnly.msl", $zippedDivideOnlyMsl)
writeFile("/tmp/zzz_logicalDivideOnly.msl", $logicalDivideOnlyMsl)
writeFile("/tmp/zzz_dynPair.msl", $dynPairMsl)
