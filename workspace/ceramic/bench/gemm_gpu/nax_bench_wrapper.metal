// Bench wrapper over the vendored mlx steel NAX GEMM. It compiles with
// mlx_steel_gemm.metal, the engine concatenates the two files at
// ingest.
//
// Hardcoded config: BM=64, BN=64, BK=256, WM=WN=2, no batch, no swizzle,
// no out-source. It covers exactly the shapes whose M and N are multiples
// of 64 and whose K is a multiple of 256; the launcher rejects others.

///////////////////////////////////////////////////////////////////////////////
// Bench wrapper over the vendored NAX GEMM, pure gemm (no beta/C),
// the (BM=64, BN=64, BK=256, WM=2, WN=2) gemm_loop instantiation, the
// GPU loops `iters` times per launch
///////////////////////////////////////////////////////////////////////////////

kernel void mlx_gemm_nax_64x64 [[kernel, max_total_threads_per_threadgroup(128)]] (
    device float* D [[buffer(0)]],
    const device half* A [[buffer(1)]],
    const device half* B [[buffer(2)]],
    const constant mlx::steel::GEMMParams& params [[buffer(3)]],
    const constant int& iters [[buffer(4)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using namespace mlx::steel;

  constexpr int BM = 64, BN = 64, BK = 256, WM = 2, WN = 2;
  constexpr short SM = BM / WM;  // rows of the per-simdgroup output tile
  constexpr short SN = BN / WN;  // cols of the per-simdgroup output tile
  constexpr short SK = 32;       // K step per mma round
  constexpr short TM = SM / 16;  // 2x2 fragments of 16x16 per tile
  constexpr short TN = SN / 16;

  if (params.tiles_n <= int(tid.x) || params.tiles_m <= int(tid.y)) {
    return;
  }

  const int c_row = tid.y * BM;
  const int c_col = tid.x * BN;
  const device half* At = A + c_row * params.lda;
  const device half* Bt = B + c_col;
  device float* Dt = D + c_row * params.ldd + c_col;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  At += tm * params.lda;
  Bt += tn;
  Dt += tm * params.ldd + tn;

  for (int i = 0; i < iters; i++) {
    NAXTile<float, TM, TN, BaseNAXFrag<half, half, float, false, false,
        NAXFragDest>> Dtile;
    Dtile = gemm_loop<
        half, SM, SN, SK, BK, false, false, true, true, true, float>(
        At, Bt, params.lda, params.ldb, params.K,
        params.gemm_k_iterations_aligned, SM, SN);
    Dtile.store(Dt, params.ldd);
  }
}
