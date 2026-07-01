## Test: kernel_gemm — gemm outer product (GPU path)
##
## Tests gemm(A, B, C) = C[m,n] += A[m,k] * B[n,k]
## against a naive reference implementation.

import ../src/int_tuples
import ../src/layouts
import ../src/tensors
import ../src/ptr_arithmetic
import ../src/kernel_gemm_gpu

{.experimental: "callOperator".}

template test(label: string; body: untyped) =
  block:
    body
  echo "  [OK] ", label

# ═══════════════════════════════════════════════════════════════
#  Reference: naive matmul
# ═══════════════════════════════════════════════════════════════

proc ref_gemm[T](C: var seq[T]; A, B: seq[T]; M, N, K: int) =
  ## C(M,N) += A(M,K) × B(N,K)  — column-major flat layout
  for k in 0 ..< K:
    for m in 0 ..< M:
      let a = A[m + k * M]
      for n in 0 ..< N:
        C[m + n * M] += a * B[n + k * N]

# ═══════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════

proc runGemmTests =
  test "gemm small uniform (2,3)x(4,3)":
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
    gemm(C, A, B)

    doAssert C[0, 0] == K.float32
    doAssert C[1, 0] == K.float32
    doAssert C[0, 1] == K.float32

  test "gemm identity (1,1)x(1,1)":
    const M = 1; const N = 1; const K = 1
    var bufA = newSeq[float32](1); bufA[0] = 5.0'f32
    var bufB = newSeq[float32](1); bufB[0] = 7.0'f32
    var bufC = newSeq[float32](1); bufC[0] = 0.0'f32

    let A = make_view(bufA +% 0, make_layout((1, 1)))
    let B = make_view(bufB +% 0, make_layout((1, 1)))
    var C = make_view(bufC +% 0, make_layout((1, 1)))
    gemm(C, A, B)

    doAssert C[0, 0] == 35.0'f32

  test "gemm K=1 (2,1)x(4,1)":
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
    gemm(C, A, B)

    # C was pre-filled with 1, so result = 1 + 2*3 = 7
    for m in 0 ..< M:
      for n in 0 ..< N:
        doAssert C[m, n] == 7.0'f32

  test "gemm non-uniform (3,4)x(5,4) against reference":
    const M = 3; const N = 5; const K = 4
    var bufA = newSeq[float32](M * K)
    var bufB = newSeq[float32](N * K)
    var bufC = newSeq[float32](M * N)
    var refC = newSeq[float32](M * N)
    for i in 0 ..< M*K: bufA[i] = float32((i * 7) mod 11)
    for i in 0 ..< N*K: bufB[i] = float32((i * 13) mod 17)
    for i in 0 ..< M*N: bufC[i] = 0.0'f32

    let A = make_view(bufA +% 0, make_layout((M, K), (1, M)))
    let B = make_view(bufB +% 0, make_layout((N, K), (1, N)))
    var C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    gemm(C, A, B)

    ref_gemm(refC, bufA, bufB, M, N, K)
    for m in 0 ..< M:
      for n in 0 ..< N:
        doAssert C[m, n] == refC[m + n * M]

  test "gemm accumulator add (pre-filled C)":
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
    gemm(C, A, B)

    # C was pre-filled with 10, K=2, A=2, B=3 → C = 10 + 2*2*3 = 22
    for m in 0 ..< M:
      for n in 0 ..< N:
        doAssert C[m, n] == 22.0'f32

  test "gemm row-major stride (8,8)x(16,8) against reference":
    const M = 8; const N = 16; const K = 8
    var bufA = newSeq[float32](M * K)
    var bufB = newSeq[float32](N * K)
    var bufC = newSeq[float32](M * N)
    var refC = newSeq[float32](M * N)
    for i in 0 ..< M*K: bufA[i] = 1.0'f32
    for i in 0 ..< N*K: bufB[i] = 1.0'f32
    for i in 0 ..< M*N: bufC[i] = 0.0'f32

    # Row-major: (M,K):(K,1), (N,K):(K,1), (M,N):(N,1)
    let A = make_view(bufA +% 0, make_layout((M, K), (K, 1)))
    let B = make_view(bufB +% 0, make_layout((N, K), (K, 1)))
    var C = make_view(bufC +% 0, make_layout((M, N), (N, 1)))
    gemm(C, A, B)

    for m in 0 ..< M:
      for n in 0 ..< N:
        doAssert C[m, n] == K.float32

when isMainModule:
  runGemmTests()
  echo "OK: all gemm tests passed"
