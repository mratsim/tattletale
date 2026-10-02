## Run:
##   nim ceramic_cgs runner=cgs_layout_local_tile dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra/tiles
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# ── baselines, no tile machinery ──

# dynamic rank-2 view, one element read
const baselineOverheadRank2Msl = metal:
  proc baselineOverheadRank2Kernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    C[0] = float32 p[int R0, int C0]

# rank-4 global-data view, one 4-coord element read, measures the no-tiler baseline
const baselineOverheadGlViewMsl = metal:
  proc baselineOverheadGlViewKernel(C: ptr UncheckedArray[float32]; B, D, M, N, R0, C0: int32) {.global.} =
    let gl = gd(C, B, D, M, N)
    C[0] = float32 gl[0, 0, int R0, int C0]

# ── local_tile_dyn, works on the global-data view (tile_algebra/tiles.nim:131) ──

const localTileDynMsl = metal:
  proc localTileDynKernel(C: ptr UncheckedArray[float32]; B, D, M, N, R0, C0: int32) {.global.} =
    let gl = gd(C, B, D, M, N)
    let src = local_tile_dyn(gl, 16, 16, (0, 0, int R0, int C0))
    C[0] = float32 src[int R0, int C0]

# ── partition selectors, dynamic rank-2 fallback ──
# the selector rows run on the dynamic rank-2 view, the global-data view row is localTileDynKernel

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

# 2-arg local_tile, aliases inner_partition (tensor_selectors.nim:177),
# byte-identity with innerPartitionKernel is the alias check
const localTile2argMsl = metal:
  proc localTile2argKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let src = local_tile(p, (16, 16), (int R0, int C0))
    C[0] = float32 src[int R0, int C0]

# 4-arg local_tile with projection (1, 1), dice overhead appears as the delta over the 2-arg form (tensor_selectors.nim:182)
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

# fully static rank-2 layout, runtime coords, mirrors the indexing runner's static row
const localTileStaticMsl = metal:
  proc localTileStaticKernel(C: ptr UncheckedArray[float32]; R0, C0: int32) {.global.} =
    let p = make_view(C, make_layout((64, 64), (64, 1)))
    let src = local_tile(p, (16, 16), (int R0, int C0))
    C[0] = float32 src[int R0, int C0]

# 2 real local_tile calls on the same view
# 2nd-call marginal = delta over the 1-call selector row
const twoTileMsl = metal:
  proc twoTileKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let t = local_tile(p, (16, 16), (int R0, int C0))
    let u = local_tile(p, (16, 16), (int C0, int R0))
    C[0] = float32 t[int R0, int C0]
    C[1] = float32 u[int R0, int C0]

# ── rank-2 inline of the local_tile_dyn body ──
# static 16x16 tile, runtime origin, tile stride carried from the source layout
# same input and read as the selector kernels

const dynFormulaMsl = metal:
# measured kernel, the formula baseline reads the tile dim and the .data[0] base directly
  proc dynFormulaKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let base = int R0 * 1 * 16 + int C0 * int S * 16
    let t = make_view(p.data[0].addr +% base, (Int[16](), Int[16]()), (1, int S))
    C[0] = float32 t[int R0, int C0]

# ── kernel rows ──
# Marginal = cost minus baseline, computed inside cgsReport, one baseline per row:
# - selector rows, dynFormula and the static row sit on baselineOverheadRank2
# - localTileDyn sits on baselineOverheadGlView
# - twoTile pairs the localTile2arg row, its Marginal column is the 2nd-call cost
cgsReport("cgs_layout_local_tile", [
  cgsReceipt("baselineOverheadRank2Kernel", baselineOverheadRank2Msl),
  cgsReceipt("baselineOverheadGlViewKernel", baselineOverheadGlViewMsl),
  cgsReceipt("localTileDynKernel", localTileDynMsl, baselineOverheadGlViewMsl.len),
  cgsReceipt("innerPartitionKernel", innerPartitionMsl, baselineOverheadRank2Msl.len),
  cgsReceipt("outerPartitionKernel", outerPartitionMsl, baselineOverheadRank2Msl.len),
  cgsReceipt("localTile2argKernel", localTile2argMsl, baselineOverheadRank2Msl.len),
  cgsReceipt("localTile4argKernel", localTile4argMsl, baselineOverheadRank2Msl.len),
  cgsReceipt("localPartitionKernel", localPartitionMsl, baselineOverheadRank2Msl.len),
  cgsReceipt("localTileStaticKernel", localTileStaticMsl, baselineOverheadRank2Msl.len),
  cgsReceipt("twoTileKernel", twoTileMsl, localTile2argMsl.len),
  cgsReceipt("dynFormulaKernel", dynFormulaMsl, baselineOverheadRank2Msl.len)])
