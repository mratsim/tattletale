## ex02a_matmul_handtuned_arm64_sme2: hand-tuned SME2-accelerated serial GEMM
##
## mr = nr = 32, kc_atom = 16: the 32×32 output tile maps to four 16×16 ZA
## tiles (4-way ILP). SME micro-kernel dispatch on arm64, generic fallback
## elsewhere.
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
  ## Activation function applied to the A·B accumulator before the alpha/beta
  ## combination in the epilogue.
  akIdentity
  akReLU

template genEpilogue*(epilogueName: untyped, relu: static bool, activationBody: untyped): untyped =
  ## Generate an epilogue proc with the activation inlined.
  ## Inside `activationBody`, use `x` for the AB accumulator value.
  ## `relu` selects the NEON fast path's activation (same fmax semantics).
  proc `epilogueName`[T; MR, NR: static int; Sh, St](
      C: TensorView[T, Sh, St],
      AB: array[MR, array[NR, T]],
      mr, nr: int,
      alpha, beta: T) =
    when defined(arm64) and T is float32:
      # Fast path: full 32×32 tile with contiguous C rows (col stride 1, row
      # stride ≥ tile width). NEON 4-lane ops. Alpha/beta/activation flags use
      # the scalar path's comparisons, so both paths agree element-for-element
      # for all finite inputs.
      if mr == 32 and nr == 32 and C.layout.stride[1] == 1 and C.layout.stride[0] >= 32:
        neonEpilogueF3232x32(
          cast[ptr float32](C.data), cint(C.layout.stride[0]),
          cast[ptr float32](addr AB[0][0]),
          alpha, beta,
          cint(relu), cint(alpha == T(1)), cint(beta == T(0)), cint(beta == T(1)))
        return
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
genEpilogue(epilogue_identity, relu = false):
  x
genEpilogue(epilogue_relu, relu = true):
  if x > T(0): x else: T(0)

# ═══════════════════════════════════════════════════════════════════════════
#  pack_layout: derive pack-buffer layout from zipped_divide
# ═══════════════════════════════════════════════════════════════════════════

template pack_layout*(zd: Layout, transposed: static bool): auto =
  ## Derive the pack-buffer layout from a `zipped_divide` layout: tile elements
  ## compact, then the remaining dimension scaled by the tile size.
  let tileCompact = make_layout(zd.shape[0], LayoutLeft)
  let tile_size = product(zd.shape[0])
  let restCompact = make_layout(zd.shape[1],
    when transposed: LayoutRight else: LayoutLeft)
  let restScaled = mapLeavesWith(restCompact):
    (it_sh, it_st * tile_size)
  nested_product(tileCompact, restScaled)

# ═══════════════════════════════════════════════════════════════════════════
#  MmaAtom: micro-kernel parameters
# ═══════════════════════════════════════════════════════════════════════════

type MmaAtom = object
  mr*, nr*, kc*: int

# ═══════════════════════════════════════════════════════════════════════════
#  Cache & tile sizing
# ═══════════════════════════════════════════════════════════════════════════

# Cache tile size caps for float32: kc = 2048, mc = 192.
proc autoTileParams(atom: static MmaAtom, M, K: int): tuple[mc, kc: int] =
  const mr = atom.mr
  const nr = atom.nr
  const kc_atom = atom.kc
  result.kc = min(2048, K)    # kc = 2048 keeps a single pc pass at K = 2048.
                              # smaller kc re-reads C per pass and measures worse
  result.kc = (result.kc div kc_atom) * kc_atom
  if result.kc < kc_atom:
    result.kc = min(K, kc_atom)
  result.mc = min(192, M)     # Laser: 768 B / sizeof(float32)
  result.mc = (result.mc div mr) * mr
  if result.mc < mr:
    result.mc = min(M, mr)

# ═══════════════════════════════════════════════════════════════════════════
#  Micro-kernel dispatch
# ═══════════════════════════════════════════════════════════════════════════

import workspace/ceramic/examples/ex02_matmul_microkernels/gemm_ukernel_generic
import workspace/ceramic/examples/ex02_matmul_microkernels/gemm_ukernel_arm64_sme2
import workspace/ceramic/examples/ex02_matmul_microkernels/gemm_packing_arm64_sme2
export gemm_ukernel_arm64_sme2  # epilogue templates instantiate in the caller's scope

## Clang prefetch builtin used by the GEMM loops. Available on every target,
## so the off-arm64 fallback compiles.
proc builtin_prefetch*(p: pointer, rw: cint, locality: cint) {.importc: "__builtin_prefetch", nodecl.}
## Clang alignment-assert builtin used by the pack paths. Available on every
## target, so the off-arm64 fallback compiles.
proc builtin_assume_aligned*(p: pointer, alignment: csize_t): pointer {.importc: "__builtin_assume_aligned", nodecl.}

const simdArch {.strdefine.} = "auto"

when simdArch == "auto":
  when defined(arm64):
    const resolvedArch = "sme"
  else:
    const resolvedArch = "generic"
elif simdArch == "sme":
  when defined(arm64):
    const resolvedArch = "sme"
  else:
    {.error: "simdArch='sme' requires an arm64 target with SME (Apple M4 or newer).".}
else:
  const resolvedArch = "generic"

when resolvedArch == "generic":
  {.warning: "SIMD arch is 'generic'. For SME acceleration compile with -d:simdArch=sme on arm64.".}

proc simdArchString*(): string = resolvedArch


template gemm_ukernel(packA, packB, AB, kc: untyped): untyped =
  when resolvedArch == "sme":
    gemmUkernelSme(packA, packB, AB, kc)
  else:
    gemm_ukernel_generic(packA, packB, AB, kc)

# ═══════════════════════════════════════════════════════════════════════════
#  Epilogue dispatch
# ═══════════════════════════════════════════════════════════════════════════

template gemm_epilogue(activation: Activation, C, AB, mr, nr, alpha, beta: untyped): untyped =
  ## TensorView epilogue dispatch.
  if activation == akReLU:
    epilogue_relu(C, AB, mr, nr, alpha, beta)
  else:
    epilogue_identity(C, AB, mr, nr, alpha, beta)

# ═══════════════════════════════════════════════════════════════════════════
#  gemm_strided: BLIS 5-loop GEMM
# ═══════════════════════════════════════════════════════════════════════════

## BLIS 5-loop GEMM: `C ← beta·C + alpha·f(A·B)` with a micro-kernel dispatch
## (SME on arm64, scalar fallback otherwise).
##
## `rowStrideA/colStrideA`: element strides of the M×K row-major/column-major A view
## (rowStrideA = 1 ⇒ column-major). Same convention for B (K×N) and C (M×N).
## `activation`: elementwise function applied to the A·B accumulator before the alpha/beta combination (akIdentity or akReLU).
##
## Postcondition: all M·N lanes of C are written exactly once, in the order
## `C[i*rowStrideC + j*colStrideC]`, regardless of tile overhang.
proc gemm_strided*[T: SomeNumber](
    M, N, K: int,
    alpha: T,
    A: ptr UncheckedArray[T], rowStrideA, colStrideA: int,
    B: ptr UncheckedArray[T], rowStrideB, colStrideB: int,
    beta: T,
    C: ptr UncheckedArray[T], rowStrideC, colStrideC: int,
    activation: Activation = akIdentity) =

  # ── Micro-kernel tile dimensions ──
  # mr = nr = 32: four 16×16 ZA quadrants (4-way ILP, ~2 TFLOP/s SME ceiling).
  const
    mr = 32
    nr = 32
    kc_atom = 16

  # ── Cache-block dimensions ──
  const atom = MmaAtom(mr: mr, nr: nr, kc: kc_atom)
  let (mc, kc) = autoTileParams(atom, M, K)
  if mc < mr or kc < kc_atom:
    for i in 0 ..< M:
      for j in 0 ..< N:
        var acc = 0.T
        for k in 0 ..< K:
          acc += A[i * rowStrideA + k * colStrideA] * B[k * rowStrideB + j * colStrideB]
        let ci = i * rowStrideC + j * colStrideC
        if activation == akReLU:
          acc = max(acc, T(0))
        C[ci] = if beta == T(0): alpha * acc
                else: beta * C[ci] + alpha * acc
    return

  # ── Derived quantities ──
  let num_jr = ceil_div(N, nr)
  let nc     = num_jr * nr
  let num_ir = mc div mr
  let num_ic = ceil_div(M, mc)
  let num_pc = ceil_div(K, kc)
  doAssert activation == akIdentity or num_pc == 1,
    "gemm_strided: activation is applied per k-block; K must fit one kc block (K=" &
    $K & ", kc=" & $kc & ")"

  # ── Matrix views ──
  let vA = make_view(A, (M, K), (rowStrideA, colStrideA))
  let vB = make_view(B, (K, N), (rowStrideB, colStrideB))
  var vC = make_view(C, (M, N), (rowStrideC, colStrideC))

  # ── Pack buffer layouts ──
  let packALay = make_layout((num_ir, kc, mr), LayoutRight)
  let packBLay = make_layout((num_jr, kc, nr), LayoutRight)
  let packSizeA = int(cosize(packALay))
  let packSizeB = int(cosize(packBLay))
  var packMemA = newSeq[T](packSizeA + (32 div sizeof(T)))
  var packMemB = newSeq[T](packSizeB + (32 div sizeof(T)))
  # Align to 32 bytes. ld1w/st1w need no alignment (any alignment works).
  # Kept for a uniform pack layout across ukernels.
  let alignA = (cast[int](addr packMemA[0]) + 31) and not 31
  let alignB = (cast[int](addr packMemB[0]) + 31) and not 31

  # ── Pack buffer base pointers ──
  var packA_ptr = cast[ptr UncheckedArray[T]](alignA)
  var packB_ptr = cast[ptr UncheckedArray[T]](alignB)

  # ── Loop 4 (pc): rank-k updates ──
  for pc in 0 ..< num_pc:
    let current_kc = min(K - pc * kc, kc)
    if current_kc <= 0: continue
    let effective_beta = if pc == 0: beta else: T(1)

    # B is packed across the full N dimension: nc = num_jr*nr covers all of N.
    let panelB = local_tile(vB, (kc, nc), (pc, 0))
    let pB_ptr = cast[ptr UncheckedArray[T]](panelB.data)
    let pB_rs = panelB.layout.stride[0]  # row stride of panel
    let pB_cs = panelB.layout.stride[1]  # col stride of panel

    # Pack B: explicit triple loop (copyMem-compatible stride, SIMD-friendly)
    let packB_aligned = cast[ptr UncheckedArray[T]](builtin_assume_aligned(cast[pointer](packB_ptr), 32))
    for jr in 0 ..< num_jr:
      let eff = min(nr, N - jr * nr)   # lanes valid in the last micro-panel
      if pB_cs == 1:
        when defined(arm64) and T is float32:
          if N mod nr == 0 and num_jr mod 4 == 0:
            # Four full panels per pass: one B-row visit reads 512 contiguous bytes
            # (DRAM row locality) instead of a 128-B slice. The group's remaining jr iterations are no-ops.
            if jr mod 4 == 0:
              smePackBCopy32f32x4(
                cast[ptr float32](cast[int](packB_aligned) +% ((jr + 0) * kc * nr) *% sizeof(T).int),
                cast[ptr float32](cast[int](packB_aligned) +% ((jr + 1) * kc * nr) *% sizeof(T).int),
                cast[ptr float32](cast[int](packB_aligned) +% ((jr + 2) * kc * nr) *% sizeof(T).int),
                cast[ptr float32](cast[int](packB_aligned) +% ((jr + 3) * kc * nr) *% sizeof(T).int),
                cast[ptr float32](cast[int](pB_ptr) +% (jr * nr) *% sizeof(T).int),
                cint(pB_rs), cint(current_kc))
          elif eff == nr:
            # 32 valid lanes per row: 128-B NEON copies (2× ld1/st1 per row).
            # Partial-column last panels (eff < nr) keep the copyMem loop
            # below, so B is never read past its valid lanes.
            neonPackBCopy32f32(
              cast[ptr float32](cast[int](packB_aligned) +% (jr * kc * nr) *% sizeof(T).int),
              cast[ptr float32](cast[int](pB_ptr) +% (jr * nr) *% sizeof(T).int),
              cint(pB_rs), cint(current_kc))
          else:
            # B rows are contiguous: copyMem
            for k in 0 ..< current_kc:
              let dstOff = jr * kc * nr + k * nr
              let srcOff = k * pB_rs + jr * nr
              copyMem(addr packB_aligned[dstOff], addr pB_ptr[srcOff], eff * sizeof(T).int)
              for jj in eff ..< nr:
                packB_aligned[dstOff + jj] = T(0)
        else:
          # B rows are contiguous: copyMem
          for k in 0 ..< current_kc:
            let dstOff = jr * kc * nr + k * nr
            let srcOff = k * pB_rs + jr * nr
            copyMem(addr packB_aligned[dstOff], addr pB_ptr[srcOff], eff * sizeof(T).int)
            for jj in eff ..< nr:
              packB_aligned[dstOff + jj] = T(0)
      else:
        for k in 0 ..< current_kc:
          let dstOff = jr * kc * nr + k * nr
          let srcOff = k * pB_rs + jr * nr * pB_cs
          for jj in 0 ..< eff:
            packB_aligned[dstOff + jj] = pB_ptr[srcOff + jj * pB_cs]
          for jj in eff ..< nr:
            packB_aligned[dstOff + jj] = T(0)

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

      # Pack A: explicit triple loop (with copyMem for contiguous rows)
      let packA_aligned = cast[ptr UncheckedArray[T]](builtin_assume_aligned(cast[pointer](packA_ptr), 32))
      if pA_rs == 1:
        # Rows are contiguous in memory (column-major): use copyMem
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
      elif pA_cs == 1:
        when defined(arm64) and T is float32:
          # Row-major A: NEON 8×8 trn pack, or the streaming mova 16×16 pack
          # when kc % 16 == 0 (16-column ZA tile). validRows zero-fills rows
          # past current_mc, and leftover k steps use the scalar gather below.
          static: doAssert mr == 32  # helpers' 8-row/16-row store geometry
          let kBlocks = current_kc div 8
          let kBlocks16 = current_kc div 16
          let kcEven16 = current_kc mod 16 == 0
          let kDone = if kcEven16: kBlocks16 * 16 else: kBlocks * 8
          for ir in 0 ..< num_ir_eff:
            let srcRow = ir * mr
            let dstBase = ir * kc * mr
            if kBlocks > 0:
              if kcEven16:
                # Two 16-row groups per 32-row tile.
                for g in 0 ..< 2:
                  let groupValid = max(0, min(16, current_mc - srcRow - g * 16))
                  smePackATranspose16rows(
                    cast[ptr float32](cast[int](packA_aligned) +% (dstBase + g * 16) *% sizeof(T).int),
                    cast[ptr float32](cast[int](pA_ptr) +% ((srcRow + g * 16) * pA_rs) *% sizeof(T).int),
                    cint(pA_rs), cint(kBlocks16), cint(groupValid))
              else:
                # Four 8-row groups per 32-row tile.
                for g in 0 ..< 4:
                  let groupValid = max(0, min(8, current_mc - srcRow - g * 8))
                  neonPackATranspose8rows(
                    cast[ptr float32](cast[int](packA_aligned) +% (dstBase + g * 8) *% sizeof(T).int),
                    cast[ptr float32](cast[int](pA_ptr) +% ((srcRow + g * 8) * pA_rs) *% sizeof(T).int),
                    cint(pA_rs), cint(kBlocks), cint(groupValid))
            for k in kDone ..< current_kc:
              let dstOff = dstBase + k * mr
              let srcOff = srcRow * pA_rs + k * pA_cs
              for ii in 0 ..< mr:
                if (srcRow + ii) < current_mc:
                  packA_aligned[dstOff + ii] = pA_ptr[srcOff + ii * pA_rs]
                else:
                  packA_aligned[dstOff + ii] = T(0)
        else:
          # Scalar gather: same pack layout as the NEON path, without the 8×8 block transpose.
          # Rows past current_mc store zeros.
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

      # ── Loop 1 (ir): micro-tiles of A, then Loop 2 (jr): micro-panels of B ──
      # ir outer, jr inner: the packed A tile stays resident across the jr
      # sweep, while the inner loop streams the packed B panels. Tile visit order
      # changes, so no element's accumulation order is affected.
      let packB_ptr = cast[ptr UncheckedArray[T]](alignB)
      let packA_ptr = cast[ptr UncheckedArray[T]](alignA)
      let packA_jump = kc * mr   # elements per ir slice
      for ir in 0 ..< num_ir_eff:
        let cRow = ic * mc + ir * mr
        let aOffset = ir * packA_jump
        let eff_mr = min(mr, M - cRow)
        for jr in 0 ..< num_jr:
          let cCol = jr * nr
          if cCol >= N:
            break

          let bTilePtr = cast[ptr UncheckedArray[T]](packB_ptr)
          let bOffset = jr * kc * nr   # elements into the packed B buffer for micro-panel jr
          var AB {.noInit.}: array[mr, array[nr, T]]
          # Prefetch the next B and A panels.
          builtin_prefetch(cast[pointer](cast[int](bTilePtr) +% bOffset *% sizeof(T).int), 0, 1)
          builtin_prefetch(cast[pointer](cast[int](packA_ptr) +% aOffset *% sizeof(T).int), 0, 1)
          # Epilogue: displace (layout algebra) for view, raw-pointer dispatch
          let eff_nr = min(nr, N - cCol)
          let cTile {.noInit.} = displace(vC, (cRow, cCol))
          when defined(arm64) and T is float32:
            if eff_mr == mr and eff_nr == nr and
               vC.layout.stride[1] == 1 and vC.layout.stride[0] >= nr:
              # Full 32×32 tile with contiguous C rows: kernel extracts ZA straight to C,
              # fusing alpha/beta/ReLU in streaming mode (no AB scratch, no separate epilogue).
              smeGemmUkernel32x32EpiDv(
                cast[ptr float32](cast[int](packA_ptr) +% aOffset *% sizeof(T).int),
                cast[ptr float32](cast[int](bTilePtr) +% bOffset *% sizeof(T).int),
                cast[ptr float32](cTile.data),
                cint(vC.layout.stride[0]), cint(current_kc),
                alpha, effective_beta,
                cint(activation == akReLU),
                cint(alpha == T(1)), cint(effective_beta == T(0)),
                cint(effective_beta == T(1)))
            else:
              gemm_ukernel(
                cast[ptr UncheckedArray[T]](cast[int](packA_ptr) +% aOffset *% sizeof(T).int),
                cast[ptr UncheckedArray[T]](cast[int](bTilePtr) +% bOffset *% sizeof(T).int),
                AB, current_kc)
              gemm_epilogue(activation, cTile, AB, eff_mr, eff_nr, alpha, effective_beta)
          else:
            gemm_ukernel(
              cast[ptr UncheckedArray[T]](cast[int](packA_ptr) +% aOffset *% sizeof(T).int),
              cast[ptr UncheckedArray[T]](cast[int](bTilePtr) +% bOffset *% sizeof(T).int),
              AB, current_kc)
            gemm_epilogue(activation, cTile, AB, eff_mr, eff_nr, alpha, effective_beta)

# ═══════════════════════════════════════════════════════════════════════════
#  Convenience overload: openArray[T]
# ═══════════════════════════════════════════════════════════════════════════

proc gemm_strided*[T: SomeNumber](
    M, N, K: int,
    alpha: T,
    A: openArray[T], rowStrideA, colStrideA: int,
    B: openArray[T], rowStrideB, colStrideB: int,
    beta: T,
    C: var openArray[T], rowStrideC, colStrideC: int,
    activation: Activation = akIdentity) =
  ## openArray convenience overload of the pointer `gemm_strided`: same contract.
  ## A, B and C must be non-empty. A K == 0 call still needs non-empty A and B
  ## buffers (the natural M×0 and 0×N shapes trip the doAssert below).
  doAssert A.len > 0 and B.len > 0 and C.len > 0, "gemm_strided: empty matrices not supported"
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
  import workspace/ceramic/benchmark/bench_utils

  when defined(arm64):
    doAssert simdArchString() == "sme",
      "SME path not active on arm64 (resolvedArch = " & simdArchString() & ")"

  proc test(M, N, K: int, rsA, csA, rsB, csB, rsC, csC: int, label: string, tol: float32 = 1e-4'f32,
      alpha: float32 = 1.0'f32, beta: float32 = 1.0'f32) =
    # Run a GEMM test: compare this example's gemm_strided against a naive reference.
    # Tests default to alpha=1, beta=1. Pass different values to exercise
    # the K==0 / alpha==0 edge cases and general alpha/beta correctness. K==0
    # cases run through the scalar fallback (kc < kc_atom), never the SME
    # kernel's kc == 0 branch, which gemm_strided cannot reach.
    echo "\n### ", label
    randomize(42)
    let aLen = max((M-1)*rsA + (K-1)*csA + 1, 0)
    let bLen = max((K-1)*rsB + (N-1)*csB + 1, 0)
    let cLen = max((M-1)*rsC + (N-1)*csC + 1, 0)
    let bPad = 32
    # Pack loop clamps its copy width to N - jr*nr, so B is never over-read.
    # Pad keeps B.len > 0 for the K == 0 shape (the footprint formula yields
    # 0 elements for a 0×N matrix), satisfying the openArray overload's doAssert.
    var A = newSeq[float32](aLen)
    var B = newSeq[float32](bLen + bPad)
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

  # Direct kernel-level kc == 0 test: the SME asm's zeroed-tile branch
  # (`cbz w14, 2f`) is unreachable through gemm_strided, since K=0 routes
  # to the scalar fallback.
  when defined(arm64):
    block:
      var packA: array[16, float32]
      var packB: array[16, float32]
      var AB: array[16, array[16, float32]]
      for i in 0 ..< 16:
        for j in 0 ..< 16:
          AB[i][j] = 123.0'f32
      gemmUkernelSme(
        cast[ptr UncheckedArray[float32]](addr packA[0]),
        cast[ptr UncheckedArray[float32]](addr packB[0]),
        AB, 0)
      for i in 0 ..< 16:
        for j in 0 ..< 16:
          doAssert AB[i][j] == 0.0'f32, "kernel kc=0 must zero the tile"
      echo "  Kernel kc=0 (zeroed tile): ✅"
  when defined(arm64):
    block:
      # 32×32 kernel kc == 0: the zeroed-tile branch.
      var packA: array[32, float32]
      var packB: array[32, float32]
      var AB: array[32, array[32, float32]]
      for i in 0 ..< 32:
        for j in 0 ..< 32:
          AB[i][j] = 123.0'f32
      gemmUkernelSme(
        cast[ptr UncheckedArray[float32]](addr packA[0]),
        cast[ptr UncheckedArray[float32]](addr packB[0]),
        AB, 0)
      for i in 0 ..< 32:
        for j in 0 ..< 32:
          doAssert AB[i][j] == 0.0'f32, "32x32 kernel kc=0 must zero the tile"
      echo "  32x32 Kernel kc=0 (zeroed tile): ✅"
  when defined(arm64):
    block:
      # Fused kernel kc == 0: extracts the zeroed ZA tiles with a beta read.
      # Expected result: C = beta*C + alpha*0.
      var packA: array[32, float32]
      var packB: array[32, float32]
      var C: array[32 * 32, float32]
      var Cref: array[32 * 32, float32]
      randomize(42)
      for i in 0 ..< C.len:
        C[i] = rand(1.0'f32)
        Cref[i] = C[i]
      smeGemmUkernel32x32EpiDv(addr packA[0], addr packB[0], addr C[0],
        cint(32), cint(0), 1.0'f32, 2.0'f32, 0, 1, 0, 0)  # kc=0, beta=2
      for i in 0 ..< C.len:
        doAssert abs(C[i] - 2.0'f32 * Cref[i]) < 1e-6, "fused kc=0 must yield beta*C"
      echo "  Fused kernel kc=0 (beta*C): ✅"
  test(16, 16, 16, 16, 1, 16, 1, 16, 1, "Square 16x16 row-major")
  test(20, 32, 18, 18, 1, 32, 1, 32, 1, "Edge tiles: M=20, K=18 (non-multiples of 16)")
  test(32, 20, 32, 32, 1, 20, 1, 20, 1, "Edge tiles: N=20 (non-multiple of 16)")
  test(17, 16, 17, 17, 1, 16, 1, 16, 1, "Edge tiles: M=17, K=17 (minimal overhang)")
  test(48, 48, 48, 48, 1, 48, 1, 48, 1, "Non-square 48x48")
  test(128, 128, 128, 1, 128, 1, 128, 1, 128, "Large 128x128 col-major")
  test(512, 512, 512, 512, 1, 512, 1, 512, 1, "Large 512x512 row-major", tol = 1e-3'f32)
  # A-pack zero-group regression: K = 1..7 (mod 16) leaves a final pc block
  # with current_kc < 8, so the NEON transpose pack must handle zero 8-column
  # groups without walking out of bounds.
  test(32, 32, 17, 32, 1, 32, 1, 32, 1, "K=17: A-pack nColGroups=0 (regression)")
  test(32, 32, 513, 513, 1, 513, 1, 513, 1, "K=513: A-pack nColGroups=0 at kc=512")
  # K=18: pc=1 leaves current_kc=2, so the 32×32 kernel runs its kc % 4
  # oddments tail and the A-pack scalar gather covers all k-steps.
  # K=36/M=36: pc=1 leaves current_kc=4 (kernel single-block path). M=36
  # makes ic=1 a 4-row block, exercising the A-transpose partial 8-row group
  # and the pack's overhang zero-fill.
  test(32, 32, 18, 32, 1, 32, 1, 32, 1, "K=18: kernel oddments tail (kc=2) + A-pack remainder")
  test(36, 36, 36, 36, 1, 36, 1, 36, 1, "K=36/M=36: kernel single-block kc=4, A-pack 4-step remainder, partial 8-row group")
  # K=19: pc=1 leaves current_kc=3: the kernel's oddments-only tail runs
  # w17 = 3 iterations of the `6:` loop (kc % 4 == 3, no full blocks).
  # K=47: pc=1 leaves current_kc=15, w16 = 3 blocks plus a 3-oddment tail
  # exercising the mixed prologue+oddments path in the fused kernel.
  test(32, 32, 19, 32, 1, 32, 1, 32, 1, "K=19: kc=3 oddments-only tail (kc % 4 == 3)")
  test(32, 32, 47, 47, 1, 47, 1, 47, 1, "K=47: blocks + 3-oddment tail (kc=32, tail kc=15)")
  test(8, 8, 4, 1, 10, 1, 10, 1, 10, "Non-square (8x4) scalar path")
  test(2, 2, 2, 1, 2, 1, 2, 1, 2, "Tiny 2x2 (triple-loop path)")
  # Edge cases: alpha==0 (K>0) should still apply beta to C
  test(32, 32, 32, 32, 1, 32, 1, 32, 1, "alpha=0, beta=1", alpha = 0.0'f32)
  test(32, 32, 32, 32, 1, 32, 1, 32, 1, "alpha=0.5, beta=2", alpha = 0.5'f32, beta = 2.0'f32)
  test(32, 32, 32, 32, 1, 32, 1, 32, 1, "alpha=0, beta=0", alpha = 0.0'f32, beta = 0.0'f32)
  # K==0: scalar fallback with an empty k-loop, C = beta*C + alpha*0
  test(32, 32, 0, 32, 1, 32, 1, 32, 1, "K=0, beta=1 (no accumulation)")
  test(32, 32, 0, 32, 1, 32, 1, 32, 1, "K=0, beta=2", beta = 2.0'f32)
  block:
    # ReLU via the fused SME extract (fclamp): full 32×32 tile, row-major C.
    # A rows alternate +1/-1 by column and B rows alternate -1/+1 by row:
    # every A·B accumulator is -32, so the fclamp is observable. An identity
    # clamp would leave -32 in C.
    let (M, N, K) = (32, 32, 32)
    let rs = N
    var A = newSeq[float32](M * K)
    var B = newSeq[float32](K * N)
    for i in 0 ..< M:
      for k in 0 ..< K:
        A[i * K + k] = if (k mod 2) == 0: 1.0'f32 else: -1.0'f32
    for k in 0 ..< K:
      for j in 0 ..< N:
        B[k * N + j] = if (k mod 2) == 0: -1.0'f32 else: 1.0'f32
    var C_ref = newSeq[float32](M * N)
    var C_tst = newSeq[float32](M * N)
    randomize(42)
    for i in 0 ..< M * N: C_ref[i] = rand(1.0'f32)
    for i in 0 ..< M * N: C_tst[i] = C_ref[i]
    # Reference: C = beta*C + alpha*max(A·B, 0) with alpha=1, beta=1
    for i in 0 ..< M:
      for j in 0 ..< N:
        var acc = 0.0'f32
        for k in 0 ..< K:
          acc += A[i * K + k] * B[k * N + j]
        C_ref[i * N + j] += max(acc, 0.0'f32)
    gemm_strided(M, N, K, 1.0'f32, A, rs, 1, B, rs, 1, 1.0'f32, C_tst, rs, 1, akReLU)
    var err: float32 = 0
    for i in 0 ..< M * N:
      err = max(err, abs(C_ref[i] - C_tst[i]))
    doAssert err < 1e-4'f32, "fused ReLU tolerance exceeded (got " & $err & ")"
    echo "  ReLU via the fused SME extract (fclamp): max error ", err.formatFloat(ffScientific, 2), " ✅"
  block:
    # ReLU on a partial tile (N=20): the fused dispatch keeps tiles below
    # 32×32 on the AB-store kernel + epilogue path, so this check runs
    # the epilogue's ReLU with clamped and pass-through rows. A rows alternate
    # by (i+k) parity and B rows by k parity, so every accumulator is +32 on
    # even i rows and -32 on odd i rows.
    let (M, N, K) = (32, 20, 32)
    var A = newSeq[float32](M * K)
    var B = newSeq[float32](K * N)
    for i in 0 ..< M:
      for k in 0 ..< K:
        A[i * K + k] = if ((i + k) mod 2) == 0: 1.0'f32 else: -1.0'f32
    for k in 0 ..< K:
      for j in 0 ..< N:
        B[k * N + j] = if (k mod 2) == 0: 1.0'f32 else: -1.0'f32
    var C_ref = newSeq[float32](M * N)
    var C_tst = newSeq[float32](M * N)
    randomize(42)
    for i in 0 ..< M * N: C_ref[i] = rand(1.0'f32)
    for i in 0 ..< M * N: C_tst[i] = C_ref[i]
    # Reference: C = C + max(A·B, 0): +32 on even rows, 0 on odd rows
    for i in 0 ..< M:
      for j in 0 ..< N:
        var acc = 0.0'f32
        for k in 0 ..< K:
          acc += A[i * K + k] * B[k * N + j]
        C_ref[i * N + j] += max(acc, 0.0'f32)
    gemm_strided(M, N, K, 1.0'f32, A, K, 1, B, N, 1, 1.0'f32, C_tst, N, 1, akReLU)
    var err: float32 = 0
    for i in 0 ..< M * N:
      err = max(err, abs(C_ref[i] - C_tst[i]))
    doAssert err < 1e-4'f32, "partial-tile ReLU tolerance exceeded (got " & $err & ")"
    echo "  ReLU on partial tile (old AB+epilogue path): max error ", err.formatFloat(ffScientific, 2), " ✅"
  block:
    # Fused ReLU with alpha=0.5, beta=2: the extract applies fclamp first, then
    # alpha, then beta*C. The same -32 accumulator pattern plus one NaN A lane:
    # fclamp(NaN) = 0 on M4, so row 0 must come out as beta*C,
    # matching the scalar max(NaN, 0) = 0.
    let (M, N, K) = (32, 32, 32)
    let rs = N
    var A = newSeq[float32](M * K)
    var B = newSeq[float32](K * N)
    for i in 0 ..< M:
      for k in 0 ..< K:
        A[i * K + k] = if (k mod 2) == 0: 1.0'f32 else: -1.0'f32
    A[0] = NaN
    for k in 0 ..< K:
      for j in 0 ..< N:
        B[k * N + j] = if (k mod 2) == 0: -1.0'f32 else: 1.0'f32
    var C_orig = newSeq[float32](M * N)
    var C_ref = newSeq[float32](M * N)
    var C_tst = newSeq[float32](M * N)
    randomize(42)
    for i in 0 ..< M * N: C_orig[i] = rand(1.0'f32)
    for i in 0 ..< M * N:
      C_ref[i] = C_orig[i]
      C_tst[i] = C_orig[i]
    # Reference: C = 2*C + 0.5*max(A·B, 0), with max(NaN, 0) = 0.
    for i in 0 ..< M:
      for j in 0 ..< N:
        var acc = 0.0'f32
        for k in 0 ..< K:
          acc += A[i * K + k] * B[k * N + j]
        C_ref[i * N + j] = 2.0'f32 * C_ref[i * N + j] + 0.5'f32 * max(acc, 0.0'f32)
    gemm_strided(M, N, K, 0.5'f32, A, rs, 1, B, rs, 1, 2.0'f32, C_tst, rs, 1, akReLU)
    var err: float32 = 0
    for i in 0 ..< M * N:
      err = max(err, abs(C_ref[i] - C_tst[i]))
    doAssert err < 1e-4'f32, "fused ReLU alpha/beta tolerance exceeded (got " & $err & ")"
    echo "  ReLU via fused SME extract (alpha=0.5, beta=2): max error ", err.formatFloat(ffScientific, 2), " ✅"
    # NaN lane (row 0) must equal beta*C: alpha*max(NaN, 0) = 0, asserted
    # against the independent pre-GEMM C values.
    for j in 0 ..< N:
      doAssert abs(C_tst[j] - 2.0'f32 * C_orig[j]) < 1e-6,
        "NaN lane must stay beta*C (got " & $C_tst[j] & ")"
    echo "  ReLU NaN lane (fclamp NaN->0, C = beta*C): ✅"
  when defined(arm64):
    block:
      # NEON epilogue's ReLU (fmax) branch is unreachable through
      # gemm_strided: the fused dispatch owns every full 32×32 contiguous
      # tile, which is the NEON fast path's only precondition.
      var AB: array[32, array[32, float32]]
      var C: array[32 * 32, float32]
      var Cref: array[32 * 32, float32]
      randomize(42)
      for i in 0 ..< 32:
        for j in 0 ..< 32:
          AB[i][j] = if ((i + j) mod 2) == 0: 1.0'f32 else: -1.0'f32
      for i in 0 ..< C.len:
        C[i] = rand(1.0'f32)
        Cref[i] = C[i] + (if ((i div 32 + i mod 32) mod 2) == 0: 1.0'f32 else: 0.0'f32)
      neonEpilogueF3232x32(addr C[0], cint(32), addr AB[0][0],
        1.0'f32, 1.0'f32, 1, 1, 0, 1)
      for i in 0 ..< C.len:
        doAssert abs(C[i] - Cref[i]) < 1e-6, "NEON epilogue ReLU mismatch (got " & $C[i] & ")"
      echo "  NEON epilogue ReLU (fmax, direct call): ✅"
  echo "\nDone."
