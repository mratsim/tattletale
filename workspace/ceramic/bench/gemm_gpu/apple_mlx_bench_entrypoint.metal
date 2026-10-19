// Bench wrapper over the vendored mlx steel GEMM. It compiles with
// mlx_steel_gemm.metal, the engine concatenates the two files at
// ingest.

///////////////////////////////////////////////////////////////////////////////
// Bench wrapper over the vendored steel GEMM, pure gemm (no
// beta/C), GEMMKernel<half, float, 64, 64, 16, 2, 2>, the GPU loops
// `iters` times per launch
///////////////////////////////////////////////////////////////////////////////

kernel void mlx_gemm_64x64x16 [[kernel, max_total_threads_per_threadgroup(128)]] (
    device float* D [[buffer(0)]],
    const device half* A [[buffer(1)]],
    const device half* B [[buffer(2)]],
    const constant mlx::steel::GEMMParams& params [[buffer(3)]],
    const constant int& iters [[buffer(4)]],
    threadgroup half* tgp [[threadgroup(0)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  // the staging blocks share one threadgroup region:
  // A = BM * (BK + pad) halfs, B = BK * (BN + pad) halfs
  threadgroup half* As = tgp;
  threadgroup half* Bs = tgp + 64 * (16 + 8);
  for (int i = 0; i < iters; i++) {
    mlx::steel::GEMMKernel<half, float, 64, 64, 16, 2, 2, false, false, true, true>::run(
        A, B, D, &params, As, Bs, simd_lane_id, simd_group_id, tid, lid);
  }
}
