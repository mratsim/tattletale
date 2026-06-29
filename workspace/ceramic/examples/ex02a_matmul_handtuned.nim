## ex02a_matmul_handtuned — Hand-tuned SIMD-accelerated serial GEMM
##
## The "low-level" contrast point: manual pack loops, raw pointer arithmetic,
## aligned loads, k-loop unrolling, prefetch, effective_beta.
##
## BLIS 5-loop structure:
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
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
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

template genEpilogue_raw*(epilogueName: untyped; activationBody: untyped): untyped =
  ## Generate an epilogue proc taking raw pointer + strides.
  ## Inside `activationBody`, use `x` for the AB accumulator value.
  proc `epilogueName`[T; MR, NR: static int](
      C: ptr UncheckedArray[T];
      AB: array[MR, array[NR, T]];
      mr, nr: int;
      alpha, beta: T;
      rsC, csC: int) {.inline.} =
    if beta == T(0):
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          C[i * rsC + j * csC] = T(0)
    elif beta != T(1):
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          C[i * rsC + j * csC] *= beta
    if alpha == T(1):
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          let x {.inject.} = AB[i][j]
          C[i * rsC + j * csC] += activationBody
    else:
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          let x {.inject.} = AB[i][j]
          C[i * rsC + j * csC] += alpha * activationBody

# Generate concrete epilogue procs (zero dispatch overhead)
genEpilogue(epilogue_identity):
  x
genEpilogue(epilogue_relu):
  if x > T(0): x else: T(0)

# Raw-pointer epilogue procs
genEpilogue_raw(epilogue_identity_raw):
  x
genEpilogue_raw(epilogue_relu_raw):
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

import workspace/ceramic/examples/ex02_matmul_microkernels/gemm_ukernel_generic
import workspace/cpuplatforms/x86/simd_x86

const simdArch {.strdefine.} = "auto"

when simdArch == "auto":
  when defined(amd64):
    when defined(avx512f):
      import workspace/ceramic/examples/ex02_matmul_microkernels/gemm_ukernel_avx512
      const resolvedArch = "avx512"
    elif defined(avx):
      import workspace/ceramic/examples/ex02_matmul_microkernels/gemm_ukernel_avx_fma_ex02a
      const resolvedArch = "avx_fma"
    else:
      const resolvedArch = "generic"
  else:
    const resolvedArch = "generic"
elif simdArch == "avx512":
  import workspace/ceramic/examples/ex02_matmul_microkernels/gemm_ukernel_avx512
  const resolvedArch = "avx512"
elif simdArch == "avx_fma":
  import workspace/ceramic/examples/ex02_matmul_microkernels/gemm_ukernel_avx_fma_ex02a
  const resolvedArch = "avx_fma"
else:
  const resolvedArch = "generic"

when resolvedArch == "generic":
  {.warning: "SIMD arch is 'generic'. For SIMD acceleration compile with -d:simdArch=avx_fma or -d:simdArch=avx512.".}

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

template gemm_epilogue(activation: Activation; C, AB, mr, nr, alpha, beta, rsC, csC: untyped): untyped =
  ## Raw-pointer epilogue dispatch. `C` is ptr UncheckedArray[T] (from displace).
  if activation == akReLU:
    epilogue_relu_raw(C, AB, mr, nr, alpha, beta, rsC, csC)
  else:
    epilogue_identity_raw(C, AB, mr, nr, alpha, beta, rsC, csC)

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
  let packSizeA = int(cosize(packALay))
  let packSizeB = int(cosize(packBLay))
  var packMemA = newSeq[T](packSizeA + (32 div sizeof(T)))
  var packMemB = newSeq[T](packSizeB + (32 div sizeof(T)))
  # Align to 32 bytes (AVX alignment requirement)
  let alignA = (cast[int](addr packMemA[0]) + 31) and not 31
  let alignB = (cast[int](addr packMemB[0]) + 31) and not 31
  var packA = make_view(cast[ptr UncheckedArray[T]](alignA), packALay)
  var packB = make_view(cast[ptr UncheckedArray[T]](alignB), packBLay)

  # ── zipped_divide + pack_layout ──
  let srcA_zd = zipped_divide(panelA_lay, (mr, 1))
  let dstA_zd = pack_layout(srcA_zd, transposed = true)
  let srcB_zd = zipped_divide(panelB_lay, (1, nr))
  let dstB_zd = pack_layout(srcB_zd, transposed = false)

  let pA = tiled_divide(vA.layout, (mc, kc))
  let pB = tiled_divide(vB.layout, (kc, nc))
  var packA_ptr = cast[ptr UncheckedArray[T]](alignA)
  var packB_ptr = cast[ptr UncheckedArray[T]](alignB)

  # ── Loop 4 (pc): rank-k updates ──
  for pc in 0 ..< num_pc:
    let current_kc = min(K - pc * kc, kc)
    if current_kc <= 0: continue
    let last_k = (pc == num_pc - 1) and (current_kc < kc)
    let effective_beta = if pc == 0: beta else: T(1)

    for jc in 0 ..< 1:
      let panelB = local_tile(vB, (kc, nc), (pc, jc))
      let pB_ptr = cast[ptr UncheckedArray[T]](panelB.data)
      let pB_rs = panelB.layout.stride[0]  # row stride of panel
      let pB_cs = panelB.layout.stride[1]  # col stride of panel

      # Pack B — explicit triple loop (copyMem-compatible stride, SIMD-friendly)
      let packB_aligned = cast[ptr UncheckedArray[T]](builtin_assume_aligned(cast[pointer](packB_ptr), 32))
      if pB_cs == 1:
        # B rows are contiguous — use copyMem
        for jr in 0 ..< num_jr:
          for k in 0 ..< current_kc:
            let dstOff = jr * kc * nr + k * nr
            let srcOff = k * pB_rs + jr * nr
            copyMem(addr packB_aligned[dstOff], addr pB_ptr[srcOff], nr * sizeof(T).int)
      else:
        for jr in 0 ..< num_jr:
          for k in 0 ..< current_kc:
            let dstOff = jr * kc * nr + k * nr
            let srcOff = k * pB_rs + jr * nr * pB_cs
            for jj in 0 ..< nr:
              packB_aligned[dstOff + jj] = pB_ptr[srcOff + jj * pB_cs]

      # ── Loop 3 (ic): row blocks of A ──
      for ic in 0 ..< num_ic:
        let current_mc = min(M - ic * mc, mc)
        if current_mc <= 0:
          continue
        let last_m = (current_mc < mc)
        let num_ir_eff = if last_m: ceil_div(current_mc, mr)
                         else: num_ir

        let panelA = local_tile(vA, (mc, kc), (ic, pc))

        let pA_ptr = cast[ptr UncheckedArray[T]](panelA.data)
        let pA_rs = panelA.layout.stride[0]
        let pA_cs = panelA.layout.stride[1]

        # Pack A — explicit triple loop (with copyMem for contiguous rows)
        let packA_aligned = cast[ptr UncheckedArray[T]](builtin_assume_aligned(cast[pointer](packA_ptr), 32))
        if pA_rs == 1:
          # Rows are contiguous in memory (column-major) — use copyMem
          for ir in 0 ..< num_ir_eff:
            let srcRow = ir * mr
            let lastTile = (srcRow + mr) > current_mc
            for k in 0 ..< current_kc:
              let dstOff = ir * kc * mr + k * mr
              let srcOff = srcRow + k * pA_cs
              if lastTile:
                let valid = current_mc - srcRow
                copyMem(addr packA_aligned[dstOff], addr pA_ptr[srcOff], valid * sizeof(T).int)
                for ii in valid ..< mr:
                  packA_aligned[dstOff + ii] = T(0)
              else:
                copyMem(addr packA_aligned[dstOff], addr pA_ptr[srcOff], mr * sizeof(T).int)
        else:
          for ir in 0 ..< num_ir_eff:
            let srcRow = ir * mr
            for k in 0 ..< current_kc:
              let dstOff = ir * kc * mr + k * mr
              let srcOff = srcRow * pA_rs + k * pA_cs
              for ii in 0 ..< mr:
                if (srcRow + ii) < current_mc:
                  packA_aligned[dstOff + ii] = pA_ptr[srcOff + ii * pA_rs]
                else:
                  packA_aligned[dstOff + ii] = T(0)

        # ── Loop 2 (jr): micro-panels of B ──
        let packB_ptr = cast[ptr UncheckedArray[T]](alignB)
        let packA_ptr = cast[ptr UncheckedArray[T]](alignA)
        let packA_jump = kc * mr   # elements per ir slice
        for jr in 0 ..< num_jr:
          let cCol = jr * nr
          if cCol >= N:
            break
          
          let bTilePtr = cast[ptr UncheckedArray[T]](packB_ptr)
          let bOffset = jr * kc * nr   # jr * kc * nr

          # ── Loop 1 (ir): micro-tiles of A (innermost) ──
          for ir in 0 ..< num_ir_eff:
            let cRow = ic * mc + ir * mr
            let aOffset = ir * packA_jump
            var AB {.noInit.}: array[mr, array[nr, T]]
            # Prefetch next B and A panels (matching Laser's gebp_mkernel pattern)
            builtin_prefetch(cast[pointer](cast[int](bTilePtr) +% bOffset *% sizeof(T).int), 0, 1)
            builtin_prefetch(cast[pointer](cast[int](packA_ptr) +% aOffset *% sizeof(T).int), 0, 1)
            gemm_ukernel(
              cast[ptr UncheckedArray[T]](cast[int](packA_ptr) +% aOffset *% sizeof(T).int),
              cast[ptr UncheckedArray[T]](cast[int](bTilePtr) +% bOffset *% sizeof(T).int),
              AB, current_kc)

            # Epilogue: displace (layout algebra) for view, raw-pointer dispatch
            let eff_mr = min(mr, M - cRow)
            let eff_nr = min(nr, N - cCol)
            let cTile {.noInit.} = displace(vC, (cRow, cCol))
            gemm_epilogue(activation, cast[ptr UncheckedArray[T]](cTile.data),
                          AB, eff_mr, eff_nr, alpha, effective_beta, rowStrideC, colStrideC)

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

  proc test(M, N, K: int; rsA, csA, rsB, csB, rsC, csC: int; label: string; tol: float32 = 1e-4'f32;
      alpha: float32 = 1.0'f32; beta: float32 = 1.0'f32) =
    ## Run a GEMM test: compare example's gemm_strided against a naive reference.
    ## Tests default to alpha=1, beta=1. Pass different values to exercise
    ## K==0 / alpha==0 fast-path and general alpha/beta correctness.
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

    gemm_reference(M, N, K, alpha, A, rsA, csA, B, rsB, csB, beta, C_ref, rsC, csC)
    gemm_strided(M, N, K, alpha, A, rsA, csA, B, rsB, csB, beta, C_tst, rsC, csC, akIdentity)

    var err: float32 = 0
    for i in 0 ..< cLen:
      err = max(err, abs(C_ref[i] - C_tst[i]))
    echo "  max error: ", err.formatFloat(ffScientific, 2),
         if err < tol: " ✅" else: " ❌"
    doAssert err < tol, "Error tolerance exceeded (got " & $err & ")"

  test(16, 16, 16, 1, 16, 1, 16, 1, 16, "Square 16x16 col-major")
  test(16, 16, 16, 16, 1, 16, 1, 16, 1, "Square 16x16 row-major")
  test(8, 8, 4, 1, 10, 1, 10, 1, 10, "Non-square (8x4)")
  test(10, 10, 5, 1, 12, 1, 12, 1, 12, "Non-power-of-2 (10x5)")
  test(128, 128, 128, 1, 128, 1, 128, 1, 128, "Large 128x128 col-major")
  test(1024, 1024, 1024, 1024, 1, 1024, 1, 1024, 1, "Large 1024x1024 row-major", tol = 1e-3'f32)
  test(6, 16, 64,  1, 16, 1, 16, 1, 16, "UKernel 6x16")
  test(6, 32, 64,  1, 32, 1, 32, 1, 32, "UKernel 6x32")
  test(14, 32, 64, 1, 32, 1, 32, 1, 32, "UKernel 14x32")
  test(2, 2, 2, 1, 2, 1, 2, 1, 2, "Tiny 2x2 (triple-loop path)")

  # Edge cases: K==0 / alpha==0 should still apply beta to C
  test(8, 8, 0, 1, 8, 1, 8, 1, 8, "K=0, beta=1")
  test(8, 8, 0, 1, 8, 1, 8, 1, 8, "K=0, beta=2", beta = 2.0'f32)
  test(8, 8, 0, 1, 8, 1, 8, 1, 8, "K=0, beta=0", beta = 0.0'f32)
  test(8, 8, 16, 1, 8, 1, 8, 1, 8, "alpha=0, beta=1", alpha = 0.0'f32)
  test(8, 8, 16, 1, 8, 1, 8, 1, 8, "alpha=0, beta=2", alpha = 0.0'f32, beta = 2.0'f32)
  test(8, 8, 16, 1, 8, 1, 8, 1, 8, "alpha=0, beta=0", alpha = 0.0'f32, beta = 0.0'f32)
  echo "\nDone."
