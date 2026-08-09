## Phase 3 — wip_mma_gemm: single-tile tensor-core GEMM, bit-exact.
##
## C(16×8) = A(16×8) · B(8×8) — one m16n8k8 tf32 atom, 32 threads, no
## smem, no copies: direct gmem→register fragment fills, one mma.sync,
## direct register→gmem stores. The fragment coords come from the atom's
## (T,V) layouts via crd2idx — the same math partition_A/B/C computes
## host-side (validated in test_partition_host + test_fragment_dump_gpu).
##
## tf32 path: input buffers are uint32 holding tf32 bit patterns
## (host-side f32 → tf32 truncation of the low 13 mantissa bits). The
## DSL's cast emission is a value conversion, not a bitcast.
## entry 14), so bit patterns travel as u32, not via cast.
##
## Reference: exact for small-integer inputs (products and partial sums
## fit f32's 24-bit mantissa), so the test is bit-exact.
## Run with: nim cpp -r workspace/ceramic/experiments/wip_mma_gemm.nim

import std/[strformat, random]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/atoms
import workspace/ceramic/src/atoms_nvidia
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/crucible/src/codegen/nvrtc

# ═════════════════════════════════════════════════════════════════════════
#  Kernel
# ═════════════════════════════════════════════════════════════════════════

const kernelCode = cuda:
  proc mmaGemm16x8x8(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    ## C(16×8) = A(16×8) · B(8×8), tf32, one m16n8k8 atom, 32 threads.
    ## gmem is col-major (CuTe convention): A[m + k·16], B[n + k·8].
    let t = int(threadIdx.x)

    # ── fragment coords from the atom's (T,V) layouts ──
    #   A layout ((4,8),(2,2)):((16,1),(8,64)): off → (lm = off mod 16, lk = off div 16)
    #   B layout ((4,8),2):((8,1),32):          off → (ln = off mod 8,  lk = off div 8)
    #   C layout ((4,8),(2,2)):((32,1),(16,8)): off → (lm = off mod 16, ln = off div 16)
    let aLayout = make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    let bLayout = make_layout(((4, 8), 2), ((8, 1), 32))
    let cLayout = make_layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))

    # ── fragments: direct gmem → registers ──
    var aFrag: array[4, uint32]
    var bFrag: array[2, uint32]
    var cFrag: array[4, float32]
    for v in 0 .. 3:
      let off = crd2idx(aLayout, t + 32 * v)
      aFrag[v] = A[(off mod 16) + (off div 16) * 16]
    for v in 0 .. 1:
      let off = crd2idx(bLayout, t + 32 * v)
      bFrag[v] = B[(off mod 8) + (off div 8) * 8]
    for v in 0 .. 3:
      cFrag[v] = 0.0'f32

    # ── one mma.sync ──
    asm "\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\" : \"+f\"(cFrag[0]), \"+f\"(cFrag[1]), \"+f\"(cFrag[2]), \"+f\"(cFrag[3]) : \"r\"(aFrag[0]), \"r\"(aFrag[1]), \"r\"(aFrag[2]), \"r\"(aFrag[3]), \"r\"(bFrag[0]), \"r\"(bFrag[1])"

    # ── epilogue: registers → gmem ──
    for v in 0 .. 3:
      let off = crd2idx(cLayout, t + 32 * v)
      C[(off mod 16) + (off div 16) * 16] = cFrag[v]

# ═════════════════════════════════════════════════════════════════════════
#  Host side
# ═════════════════════════════════════════════════════════════════════════

func tf32ify(x: float32): uint32 =
  ## f32 → tf32 bit pattern: truncate the low 13 mantissa bits.
  ## For small-integer inputs the mantissa is zero → exact.
  (cast[uint32](x)) and 0xFFFFE000'u32

proc tf32Reference(C: var openArray[float32],
                   A: openArray[uint32], B: openArray[uint32]) =
  ## C[m,n] = Σ_k tf32(A[m,k]) · tf32(B[n,k]) — exact for small ints.
  for m in 0 ..< 16:
    for n in 0 ..< 8:
      var sum = 0.0'f32
      for k in 0 ..< 8:
        let av = cast[float32](A[m + k * 16])
        let bv = cast[float32](B[n + k * 8])
        sum = sum + av * bv
      C[m + n * 16] = sum

when isMainModule:
  var rng = initRand(0xC0FFEE)
  for trial in 0 ..< 16:
    var A = newSeq[uint32](16 * 8)
    var B = newSeq[uint32](8 * 8)
    for i in 0 ..< A.len:
      A[i] = tf32ify(float32(rng.rand(0 .. 15)))   # small ints → tf32-exact
    for i in 0 ..< B.len:
      B[i] = tf32ify(float32(rng.rand(0 .. 15)))

    var refC = newSeq[float32](16 * 8)
    refC.tf32Reference(A, B)

    var gpuC = newSeq[float32](16 * 8)
    var nv = initNvrtc(kernelCode)
    nv.compile()
    nv.getPtx()
    nv.execute("mmaGemm16x8x8", dim3(1), dim3(32), gpuC, (A, B))

    for j in 0 ..< 16 * 8:
      doAssert gpuC[j] == refC[j],
        &"trial {trial} [{j mod 16},{j div 16}]: gpu {gpuC[j]} != ref {refC[j]}"
  echo "  OK — single-tile mma GEMM bit-exact vs reference (16 trials)"
