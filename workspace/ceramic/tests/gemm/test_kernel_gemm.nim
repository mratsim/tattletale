## Test: gemm_ref — the naive whole-tile reference oracle (a STUB).
##
## Validates gemm_ref(C, A, B) = C[m,n] += A[m,k] * B[n,k] on hand-checkable
## cases and against the shared flat-array oracle gemm_tf32_ref.
##
## TODO: use production implementation — gemm_ref is the naive stub oracle
## (moved to tests/gemm/gemm_test_lib.nim with the gemm_fragment→gemm_atom
## rename). Every test below pins the stub's semantics only; the real gemm
## path (gemm_ukernel → gemm_tiled) is covered bit-exact by the
## manual_*_cuda GPU tests. Revisit this file to test the production
## implementation once it exists. (The old local ref_gemm is gone —
## comparisons now use the shared gemm_tf32_ref oracle.)

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/tests/gemm/gemm_test_lib

{.experimental: "callOperator".}

template test(label: string; body: untyped) =
  block:
    body
  echo "  [OK] ", label

# ═══════════════════════════════════════════════════════════════
#  Tests — all against the gemm_ref STUB (see TODO in the header)
# ═══════════════════════════════════════════════════════════════

proc runGemmTests =
  test "gemm small uniform (2,3)x(4,3)":
    ## TODO: use production implementation — gemm_ref is the naive stub oracle.
    const M = 2; const N = 4; const K = 3
    var bufA = newSeq[float32](M * K)
    var bufB = newSeq[float32](N * K)
    var bufC = newSeq[float32](M * N)
    for i in 0 ..< M*K: bufA[i] = 1.0'f32
    for i in 0 ..< N*K: bufB[i] = 1.0'f32
    for i in 0 ..< M*N: bufC[i] = 0.0'f32

    let A = make_view(bufA +% 0, make_layout((M, K), (1, M)))
    let B = make_view(bufB +% 0, make_layout((N, K), (1, N)))
    var C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    gemm_ref(C, A, B)

    doAssert C[0, 0] == K.float32
    doAssert C[1, 0] == K.float32
    doAssert C[0, 1] == K.float32

  test "gemm identity (1,1)x(1,1)":
    ## TODO: use production implementation — gemm_ref is the naive stub oracle.
    const M = 1; const N = 1; const K = 1
    var bufA = newSeq[float32](1); bufA[0] = 5.0'f32
    var bufB = newSeq[float32](1); bufB[0] = 7.0'f32
    var bufC = newSeq[float32](1); bufC[0] = 0.0'f32

    let A = make_view(bufA +% 0, make_layout((1, 1)))
    let B = make_view(bufB +% 0, make_layout((1, 1)))
    var C = make_view(bufC +% 0, make_layout((1, 1)))
    gemm_ref(C, A, B)

    doAssert C[0, 0] == 35.0'f32

  test "gemm K=1 (2,1)x(4,1)":
    ## TODO: use production implementation — gemm_ref is the naive stub oracle.
    const M = 2; const N = 4; const K = 1
    var bufA = newSeq[float32](M * K)
    var bufB = newSeq[float32](N * K)
    var bufC = newSeq[float32](M * N)
    for i in 0 ..< M*K: bufA[i] = 2.0'f32
    for i in 0 ..< N*K: bufB[i] = 3.0'f32
    for i in 0 ..< M*N: bufC[i] = 1.0'f32  # pre-filled

    let A = make_view(bufA +% 0, make_layout((M, K), (1, M)))
    let B = make_view(bufB +% 0, make_layout((N, K), (1, N)))
    var C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    gemm_ref(C, A, B)

    # C was pre-filled with 1, so result = 1 + 2*3 = 7
    for m in 0 ..< M:
      for n in 0 ..< N:
        doAssert C[m, n] == 7.0'f32

  test "gemm non-uniform (3,4)x(5,4) against reference":
    ## TODO: use production implementation — gemm_ref is the naive stub oracle.
    ## Compared against the shared gemm_tf32_ref (the old local ref_gemm is gone).
    ## Values are ≤ 16 → tf32-exact (low 13 mantissa bits zero), so the
    ## bit-cast twin buffers feed gemm_tf32_ref identically.
    const M = 3; const N = 5; const K = 4
    var bufA = newSeq[float32](M * K)
    var bufB = newSeq[float32](N * K)
    var tA = newSeq[uint32](M * K)
    var tB = newSeq[uint32](N * K)
    for i in 0 ..< M*K:
      bufA[i] = float32((i * 7) mod 11)
      tA[i] = cast[uint32](bufA[i])
    for i in 0 ..< N*K:
      bufB[i] = float32((i * 13) mod 17)
      tB[i] = cast[uint32](bufB[i])
    var bufC = newSeq[float32](M * N)
    var refC = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufC[i] = 0.0'f32

    let A = make_view(bufA +% 0, make_layout((M, K), (1, M)))
    let B = make_view(bufB +% 0, make_layout((N, K), (1, N)))
    var C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    gemm_ref(C, A, B)

    gemm_tf32_ref(refC, tA, tB, M, N, K, 0.0'f32)
    for m in 0 ..< M:
      for n in 0 ..< N:
        doAssert C[m, n] == refC[m + n * M]

  test "gemm accumulator add (pre-filled C)":
    ## TODO: use production implementation — gemm_ref is the naive stub oracle.
    const M = 2; const N = 3; const K = 2
    var bufA = newSeq[float32](M * K)
    var bufB = newSeq[float32](N * K)
    var bufC = newSeq[float32](M * N)
    for i in 0 ..< M*K: bufA[i] = 2.0'f32
    for i in 0 ..< N*K: bufB[i] = 3.0'f32
    for i in 0 ..< M*N: bufC[i] = 10.0'f32  # pre-filled

    let A = make_view(bufA +% 0, make_layout((M, K), (1, M)))
    let B = make_view(bufB +% 0, make_layout((N, K), (1, N)))
    var C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    gemm_ref(C, A, B)

    # C was pre-filled with 10, K=2, A=2, B=3 → C = 10 + 2*2*3 = 22
    for m in 0 ..< M:
      for n in 0 ..< N:
        doAssert C[m, n] == 22.0'f32

  test "gemm row-major stride (8,8)x(16,8) against reference":
    ## TODO: use production implementation — gemm_ref is the naive stub oracle.
    ## Compared against the shared gemm_tf32_ref; all-ones → tf32-exact.
    const M = 8; const N = 16; const K = 8
    var bufA = newSeq[float32](M * K)
    var bufB = newSeq[float32](N * K)
    var tA = newSeq[uint32](M * K)
    var tB = newSeq[uint32](N * K)
    for i in 0 ..< M*K:
      bufA[i] = 1.0'f32
      tA[i] = cast[uint32](1.0'f32)
    for i in 0 ..< N*K:
      bufB[i] = 1.0'f32
      tB[i] = cast[uint32](1.0'f32)
    var bufC = newSeq[float32](M * N)
    var refC = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufC[i] = 0.0'f32

    # Row-major: (M,K):(K,1), (N,K):(K,1), (M,N):(N,1)
    let A = make_view(bufA +% 0, make_layout((M, K), (K, 1)))
    let B = make_view(bufB +% 0, make_layout((N, K), (K, 1)))
    var C = make_view(bufC +% 0, make_layout((M, N), (N, 1)))
    gemm_ref(C, A, B)

    gemm_tf32_ref(refC, tA, tB, M, N, K, 0.0'f32)
    for m in 0 ..< M:
      for n in 0 ..< N:
        doAssert C[m, n] == refC[m + n * M]

when isMainModule:
  runGemmTests()
  echo "OK: all gemm tests passed"
