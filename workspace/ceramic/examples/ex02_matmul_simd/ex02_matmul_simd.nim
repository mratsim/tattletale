## ex02_matmul_simd — SIMD-accelerated serial GEMM
##
## Extends ex01 with:
##   - Per-architecture SIMD micro-kernels (AVX+FMA, AVX-512)
##   - Runtime CPU feature dispatch
##   - Activation function as enum with `genEpilogue` macro
##     (inlines the activation body, zero dispatch overhead)
##
## BLIS 5-loop structure (same as ex01):
##
##   Loop 5 (jc): column panels of C/B
##   Loop 4 (pc): rank-k updates over K
##   Loop 3 (ic): row blocks of A
##   Loop 2 (jr): micro-panels of B
##   Loop 1 (ir): micro-tiles of A
##
##   C[M, N] += α · f(A[M, K] × B[K, N]) + β · C[M, N]

{.experimental: "callOperator".}

import std/math
import ../../src/int_tuples
import ../../src/layouts
import ../../src/layout_algebra
import ../../src/tensors
import ../../src/kernel_copy_cpu
import ../../src/kernel_fillwith_cpu
export int_tuples, layouts, layout_algebra, tensors

# ═══════════════════════════════════════════════════════════════════════════
#  Activation enum + epilogue_body template
# ═══════════════════════════════════════════════════════════════════════════

type Activation* = enum
  akIdentity
  akReLU

template genEpilogue*(epilogueName: untyped; activationBody: untyped): untyped =
  ## Generate an epilogue proc with the activation inlined.
  ## Inside `activationBody`, use `x` for the AB accumulator value.
  proc `epilogueName`[T; MR, NR: static int; Sh, St](
      C: TensorView[T, Sh, St];
      AB: array[MR, array[NR, T]];
      mr, nr: int;
      alpha, beta: T) =
    if beta == T(0):
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          C[i, j] = T(0)
    elif beta != T(1):
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          C[i, j] *= beta
    if alpha == T(1):
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          let x {.inject.} = AB[i][j]
          C[i, j] += activationBody
    else:
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          let x {.inject.} = AB[i][j]
          C[i, j] += alpha * activationBody

# Generate concrete epilogue procs (zero dispatch overhead)
genEpilogue(epilogue_identity):
  x

genEpilogue(epilogue_relu):
  if x > T(0): x else: T(0)

# ═══════════════════════════════════════════════════════════════════════════
#  pack_layout — derive pack-buffer layout from zipped_divide
# ═══════════════════════════════════════════════════════════════════════════

template pack_layout*(zd: Layout; transposed: static bool): auto =
  let tileCompact = make_layout(zd.shape[0], LayoutLeft)
  let tile_size = product(zd.shape[0])
  let restCompact = make_layout(zd.shape[1],
    when transposed: LayoutRight else: LayoutLeft)
  let restScaled = mapLeavesWith(restCompact):
    (it_sh, it_st * tile_size)
  nested_product(tileCompact, restScaled)

# ═══════════════════════════════════════════════════════════════════════════
#  MmaAtom — micro-kernel parameters
# ═══════════════════════════════════════════════════════════════════════════

type MmaAtom = object
  mr*, nr*, kc*: int

const
  L1_CACHE_SIZE = 32 * 1024
  L2_CACHE_SIZE = 256 * 1024

# ═══════════════════════════════════════════════════════════════════════════
#  Cache & tile sizing (Laser heuristics)
# ═══════════════════════════════════════════════════════════════════════════

proc autoTileParams(atom: static MmaAtom; T: typedesc; M, K: int): tuple[mc, kc: int] =
  const mr = atom.mr
  const nr = atom.nr
  const kc_atom = atom.kc
  result.kc = min(512, K)     # Laser: 2048 / sizeof(T) for float32
  result.kc = (result.kc div kc_atom) * kc_atom
  if result.kc < kc_atom:
    result.kc = min(K, kc_atom)
  result.mc = min(192, M)     # Laser: 768 / sizeof(T) for float32
  result.mc = (result.mc div mr) * mr
  if result.mc < mr:
    result.mc = min(M, mr)

# ═══════════════════════════════════════════════════════════════════════════
#  Micro-kernel dispatch
# ═══════════════════════════════════════════════════════════════════════════

import ./gemm_ukernel_generic

const simdArch {.strdefine.} = "auto"

when simdArch == "auto":
  when defined(amd64):
    when defined(avx512f):
      import ./gemm_ukernel_avx512
      const resolvedArch = "avx512"
    elif defined(avx):
      import ./gemm_ukernel_avx_fma
      const resolvedArch = "avx_fma"
    else:
      const resolvedArch = "generic"
  else:
    const resolvedArch = "generic"
elif simdArch == "avx512":
  import ./gemm_ukernel_avx512
  const resolvedArch = "avx512"
elif simdArch == "avx_fma":
  import ./gemm_ukernel_avx_fma
  const resolvedArch = "avx_fma"
else:
  const resolvedArch = "generic"

when simdArch != "auto":
  # Manual SIMD arch override — ensure C++ compiler gets the right flags
  when resolvedArch == "avx_fma":
    {.passC: "-mavx -mfma".}
  elif resolvedArch == "avx512":
    {.passC: "-mavx512f -mfma".}

proc simdArchString*(): string = resolvedArch



template gemm_ukernel(packA, packB, AB, kc: untyped): untyped =
  when resolvedArch == "avx512":
    gemm_ukernel_avx512(packA, packB, AB, kc)
  elif resolvedArch == "avx_fma":
    gemm_ukernel_avx_fma(packA, packB, AB, kc)
  else:
    gemm_ukernel_generic(packA, packB, AB, kc)

# ═══════════════════════════════════════════════════════════════════════════
#  Epilogue dispatch
# ═══════════════════════════════════════════════════════════════════════════

template gemm_epilogue(activation: Activation; C, AB, mr, nr, alpha, beta: untyped): untyped =
  if activation == akReLU:
    epilogue_relu(C, AB, mr, nr, alpha, beta)
  else:
    epilogue_identity(C, AB, mr, nr, alpha, beta)

# ═══════════════════════════════════════════════════════════════════════════
#  gemm_strided — BLIS 5-loop GEMM
# ═══════════════════════════════════════════════════════════════════════════

proc gemm_strided*[T: SomeNumber](
    M, N, K: int;
    alpha: T;
    A: ptr UncheckedArray[T]; rowStrideA, colStrideA: int;
    B: ptr UncheckedArray[T]; rowStrideB, colStrideB: int;
    beta: T;
    C: ptr UncheckedArray[T]; rowStrideC, colStrideC: int;
    activation: Activation = akIdentity) =

  # ── Micro-kernel tile dimensions ──
  const
    mr = 6
    nr = 16
    kc_atom = 8

  # ── Cache-block dimensions ──
  const atom = MmaAtom(mr: mr, nr: nr, kc: kc_atom)
  let (mc, kc) = autoTileParams(atom, T, M, K)
  if mc < mr or kc < kc_atom:
    for i in 0 ..< M:
      for j in 0 ..< N:
        var acc = 0.T
        for k in 0 ..< K:
          acc += A[i * rowStrideA + k * colStrideA] * B[k * rowStrideB + j * colStrideB]
        let ci = i * rowStrideC + j * colStrideC
        C[ci] = beta * C[ci] + alpha * acc
    return

  # ── Derived quantities ──
  let num_jr = ceil_div(N, nr)
  let nc     = num_jr * nr
  let num_ir = mc div mr
  let num_ic = ceil_div(M, mc)
  let num_pc = ceil_div(K, kc)

  # ── Matrix views ──
  let vA = make_view(A, make_layout((M, K), (rowStrideA, colStrideA)))
  let vB = make_view(B, make_layout((K, N), (rowStrideB, colStrideB)))
  var vC = make_view(C, make_layout((M, N), (rowStrideC, colStrideC)))

  # ── Panel layouts ──
  let panelA_lay = make_layout((mc, kc), (rowStrideA, colStrideA))
  let panelB_lay = make_layout((kc, nc), (rowStrideB, colStrideB))

  # ── Pack buffer layouts ──
  let packALay = make_layout((num_ir, kc, mr), LayoutRight)
  let packBLay = make_layout((num_jr, kc, nr), LayoutRight)
  var packDataA = newSeq[T](int(cosize(packALay)))
  var packDataB = newSeq[T](int(cosize(packBLay)))
  var packA = make_view(packDataA, packALay)
  var packB = make_view(packDataB, packBLay)

  # ── zipped_divide + pack_layout ──
  let srcA_zd = zipped_divide(panelA_lay, (mr, 1))
  let dstA_zd = pack_layout(srcA_zd, transposed = true)
  let srcB_zd = zipped_divide(panelB_lay, (1, nr))
  let dstB_zd = pack_layout(srcB_zd, transposed = false)

  let pA = tiled_divide(vA.layout, (mc, kc))
  let pB = tiled_divide(vB.layout, (kc, nc))

  # ── Loop 4 (pc): rank-k updates ──
  for pc in 0 ..< num_pc:
    let current_kc = min(K - pc * kc, kc)
    if current_kc <= 0: continue
    let last_k = (pc == num_pc - 1) and (current_kc < kc)

    for jc in 0 ..< 1:
      let panelB = local_tile(vB, pB, pc, jc)

      if last_k:
        packB.fillWith_cpu(0.T)
        let srcB_edge = make_view(panelB,
          make_layout(((1, nr), (current_kc, num_jr)), srcB_zd.stride))
        var dstB_edge = make_view(packB,
          make_layout(((1, nr), (current_kc, num_jr)), dstB_zd.stride))
        copySameShape_cpu(dstB_edge, srcB_edge)
      else:
        let src4B = make_view(panelB, srcB_zd)
        var dst4B = make_view(packB, dstB_zd)
        copySameShape_cpu(dst4B, src4B)

      # ── Loop 3 (ic): row blocks of A ──
      for ic in 0 ..< num_ic:
        let current_mc = min(M - ic * mc, mc)
        if current_mc <= 0: continue
        let last_m = (current_mc < mc)
        let num_ir_eff = if last_m: ceil_div(current_mc, mr) else: num_ir

        let panelA = local_tile(vA, pA, ic, pc)

        if last_m or last_k:
          let mr_eff = min(mr, current_mc)
          packA.fillWith_cpu(0.T)
          let srcA_edge = make_view(panelA,
            make_layout(((mr_eff, 1), (num_ir_eff, current_kc)), srcA_zd.stride))
          var dstA_edge = make_view(packA,
            make_layout(((mr_eff, 1), (num_ir_eff, current_kc)), dstA_zd.stride))
          copySameShape_cpu(dstA_edge, srcA_edge)
        else:
          let src4A = make_view(panelA, srcA_zd)
          var dst4A = make_view(packA, dstA_zd)
          copySameShape_cpu(dst4A, src4A)

        # ── Loop 2 (jr): micro-panels of B ──
        for jr in 0 ..< num_jr:
          let cCol = jr * nr
          if cCol >= N: break
          let bTile = packB.slice((jr, _, _))

          # ── Loop 1 (ir): micro-tiles of A (innermost) ──
          for ir in 0 ..< num_ir_eff:
            let cRow = ic * mc + ir * mr
            if cRow >= M: break
            let aTile = packA.slice((ir, _, _))
            var AB: array[mr, array[nr, T]]

            gemm_ukernel(aTile.data, bTile.data, AB, current_kc)

            let cTile = displace(vC, (cRow, cCol))
            gemm_epilogue(activation, cTile, AB, mr, nr, alpha, beta)

# ═══════════════════════════════════════════════════════════════════════════
#  Convenience overload — openArray[T]
# ═══════════════════════════════════════════════════════════════════════════

proc gemm_strided*[T: SomeNumber](
    M, N, K: int;
    alpha: T;
    A: openArray[T]; rowStrideA, colStrideA: int;
    B: openArray[T]; rowStrideB, colStrideB: int;
    beta: T;
    C: var openArray[T]; rowStrideC, colStrideC: int;
    activation: Activation = akIdentity) =
  gemm_strided[T](
    M, N, K, alpha,
    cast[ptr UncheckedArray[T]](addr A[0]), rowStrideA, colStrideA,
    cast[ptr UncheckedArray[T]](addr B[0]), rowStrideB, colStrideB,
    beta,
    cast[ptr UncheckedArray[T]](addr C[0]), rowStrideC, colStrideC,
    activation)

# ═══════════════════════════════════════════════════════════════════════════
#  Test driver
# ═══════════════════════════════════════════════════════════════════════════

when isMainModule:
  import std/[random, strutils]

  proc gemm_reference(M, N, K: int; alpha: float32;
      A: openArray[float32]; rsA, csA: int;
      B: openArray[float32]; rsB, csB: int;
      beta: float32; C: var openArray[float32]; rsC, csC: int) =
    for j in 0 ..< N:
      for i in 0 ..< M:
        let ci = i * rsC + j * csC
        C[ci] = if beta == 0.0'f32: 0.0'f32
                elif beta != 1.0'f32: C[ci] * beta
                else: C[ci]
    for j in 0 ..< N:
      for k in 0 ..< K:
        let bv = B[k * rsB + j * csB]
        if bv != 0.0'f32:
          for i in 0 ..< M:
            C[i * rsC + j * csC] += alpha * A[i * rsA + k * csA] * bv

  proc test(M, N, K: int; rsA, csA, rsB, csB, rsC, csC: int; label: string) =
    echo "\n### ", label
    randomize(42)
    let aLen = max((M-1)*rsA + (K-1)*csA + 1, 0)
    let bLen = max((K-1)*rsB + (N-1)*csB + 1, 0)
    let cLen = max((M-1)*rsC + (N-1)*csC + 1, 0)
    var A = newSeq[float32](aLen)
    var B = newSeq[float32](bLen)
    var C_ref = newSeq[float32](cLen)
    var C_tst = newSeq[float32](cLen)
    for i in 0 ..< A.len:   A[i] = rand(1.0'f32)
    for i in 0 ..< B.len:   B[i] = rand(1.0'f32)
    for i in 0 ..< C_ref.len: C_ref[i] = rand(1.0'f32)
    for i in 0 ..< C_tst.len: C_tst[i] = C_ref[i]

    let alpha = 1.0'f32; let beta = 1.0'f32
    gemm_reference(M, N, K, alpha, A, rsA, csA, B, rsB, csB, beta, C_ref, rsC, csC)
    gemm_strided(M, N, K, alpha, A, rsA, csA, B, rsB, csB, beta, C_tst, rsC, csC, akIdentity)

    var err: float32 = 0
    for i in 0 ..< cLen:
      err = max(err, abs(C_ref[i] - C_tst[i]))
    echo "  simd arch: ", resolvedArch
    echo "  max error: ", err.formatFloat(ffScientific, 2),
         if err < 1e-4: "  v" else: "  x"

  test(16, 16, 16, 1, 16, 1, 16, 1, 16, "Square 16x16 col-major")
  test(16, 16, 16, 16, 1, 16, 1, 16, 1, "Square 16x16 row-major")
  test(8, 8, 4, 1, 10, 1, 10, 1, 10, "Non-square (8x4)")
  test(10, 10, 5, 1, 12, 1, 12, 1, 12, "Non-power-of-2 (10x5)")
  test(128, 128, 128, 1, 128, 1, 128, 1, 128, "Large 128x128 col-major")
  test(6, 16, 64,  1, 16, 1, 16, 1, 16, "UKernel 6x16")
  test(6, 32, 64,  1, 32, 1, 32, 1, 32, "UKernel 6x32")
  test(14, 32, 64, 1, 32, 1, 32, 1, 32, "UKernel 14x32")
  test(2, 2, 2, 1, 2, 1, 2, 1, 2, "Tiny 2x2 (triple-loop path)")
  echo "\nDone."
