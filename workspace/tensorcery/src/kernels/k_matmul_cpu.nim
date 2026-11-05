# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU matmul placeholder: scalar i-k-j GEMM over strided views.
##
## Placeholder until h_mma_registry gains a CPU atom.
##
## - the ex02a SME2 GEMM (workspace/ceramic/examples/ex02a_matmul_handtuned_arm64_sme2.nim) does not compile on current master
## - its local_tile instantiation fails inside the ceramic layout algebra
## - the matmul op runs on this serial reference until that regression is fixed

proc matmulStrided*[T: SomeNumber](
    M, N, K: int, alpha: T,
    A: ptr UncheckedArray[T], rowStrideA, colStrideA: int,
    B: ptr UncheckedArray[T], rowStrideB, colStrideB: int,
    beta: T,
    C: ptr UncheckedArray[T], rowStrideC, colStrideC: int) =
  ## Computes `C <- beta*C + alpha*A*B` over the strided views.
  ##
  ## Contract:
  ## - A is MxK, B is KxN, C is MxN
  ## - `rowStride`/`colStride` are element steps, a stride of 1 selects contiguous storage along that axis
  ## - C accumulates in the element type T, the caller owns C's initial contents
  for j in 0 ..< N:
    for i in 0 ..< M:
      let ci = i * rowStrideC + j * colStrideC
      if beta == T(0):
        C[ci] = T(0)
      elif beta != T(1):
        C[ci] *= beta
  for k in 0 ..< K:
    for j in 0 ..< N:
      let bVal = B[k * rowStrideB + j * colStrideB]
      if bVal != T(0):
        for i in 0 ..< M:
          C[i * rowStrideC + j * colStrideC] += alpha * A[i * rowStrideA + k * colStrideA] * bVal
