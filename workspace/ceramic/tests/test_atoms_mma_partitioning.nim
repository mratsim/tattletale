## Tests for MMA fragment partitioning: partition_A/B/C.
##
## The per-thread fragment: which elements of A/B/C a thread holds, in
## register order (v inner), with rest positions (value tiling beyond the
## tiled-mma unit) outer — the (MMA, M, K) fragment shape.
##
## References:
##   [CUTE-PAR] CuTe C++: atom/mma_atom.hpp — partition_A/B/C via the
##              thrfrg chain (logical_divide → zipped_divide → compose)
##   [CUTE-EX]  Expected coordinates in sections 2-4 were extracted by
##              running the CuTe partition implementation itself (host-side
##              build, no GPU: make_tiled_mma / get_slice / partition_* on
##              identity tensors), for the SM80_16x8x8_F32TF32TF32F32_TN
##              atom under the listed tilings. They pin the exact
##              coordinate-level contract of the partition — the level
##              neither CuTe's own tests (whole-GEMM only) nor the other
##              references cover.
##   [TL-GRID]  tensor-layouts: layout_utils.tile_mma_grid (the analysis
##              cross-check used by the experiment tests)
##
## Coverage: tile-size derivation, fragment size = V × rest, exact
## per-thread coordinates (single atom, thread tiling boundaries, rest
## positions, K-tiled), rejection of non-multiple tile shapes, coverage
## with the expected cross-atom multiplicity.

import std/[strformat, sequtils]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/atoms
import workspace/ceramic/src/atoms_nvidia
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/ceramic/tests/layouts_testutils

{.experimental: "callOperator".}

const atom = SM80_16x8x8_F32TF32TF32F32_TN
  ## m16n8k8 tf32: V_A = 4, V_B = 2, V_C = 4; T = 32 threads per atom.

# ═════════════════════════════════════════════════════════════════════════
#  verifyFragments — test-only coverage check (kept out of src: it is a
#  verification helper, not part of the partition API)
# ═════════════════════════════════════════════════════════════════════════

proc checkMultiplicity(counts: seq[int]; expected: int; label: string) =
  ## Every tile element must appear in exactly `expected` fragments.
  for i in 0 ..< counts.len:
    doAssert counts[i] == expected,
      &"element {i}: multiplicity {counts[i]}, expected {expected}"

template verifyFragments*(mma: untyped; tileM, tileK: static int;
                          operand: static MmaOperand) =
  ## Every tile element appears in EXACTLY the expected number of threads'
  ## fragments, with no duplicates within one thread group:
  ##   A: ThrN copies (A is shared across N-atoms)
  ##   B: ThrM copies (B is shared across M-atoms)
  ##   C: ThrK copies (C is k-independent: each tk group holds a full copy)
  ## Checks: per-thread count == vCount × rest; per-element multiplicity ==
  ## expected.
  const
    T = static(mma.atom.threadCount(operand).toIntVal())
    vCount = static(mma.atom.valuesPerThread(operand).toIntVal())
    thrM = static(mma.threadLayout.shape[0].toIntVal())
    thrN = static(mma.threadLayout.shape[1].toIntVal())
    thrK = static(mma.threadLayout.shape[2].toIntVal())
    atomLayout = when operand == opA: mma.atom.aLayout
                 elif operand == opB: mma.atom.bLayout
                 else: mma.atom.cLayout
    tileRows = tileM
    restM = when operand == opA: tileM div (thrM * static(mma.atom.mnk.m))
            elif operand == opB: 1
            else: tileM div (thrM * static(mma.atom.mnk.m))
    restN = when operand == opB: tileM div (thrN * static(mma.atom.mnk.n))
            else: tileK div (thrN * static(mma.atom.mnk.n))
    restK = tileK div (thrK * static(mma.atom.mnk.k))
    expected = case operand
               of opA: thrN
               of opB: thrM
               of opC: thrK
  block:
    let p = when operand == opA: mma.partition_A(tileM, tileK)
            elif operand == opB: mma.partition_B(tileM, tileK)
            else: mma.partition_C(tileM, tileK)
    var counts = newSeq[int](tileM * tileK)
    for flatThread in 0 ..< T * thrM * thrN * thrK:
      let tv = flatThread mod T
      let (tm, tn, tk) = idx2crd(mma.threadLayout.shape, flatThread div T)
      let tc = idx2crd(mode(atomLayout, 0).shape, tv)
      when operand == opA:
        for rk in 0 ..< restK:
          for rm in 0 ..< restM:
            for v in 0 ..< vCount:
              let vc = idx2crd(mode(atomLayout, 1).shape, v)
              let off = crd2idx(p, ((tc, vc), (tm, tk), (rm, rk))).toIntVal()
              counts[(off mod tileRows) + (off div tileRows) * tileRows].inc
      elif operand == opB:
        for rk in 0 ..< restK:
          for rn in 0 ..< restN:
            for v in 0 ..< vCount:
              let vc = idx2crd(mode(atomLayout, 1).shape, v)
              let off = crd2idx(p, ((tc, vc), (tn, tk), (rn, rk))).toIntVal()
              counts[(off mod tileRows) + (off div tileRows) * tileRows].inc
      else:
        for rn in 0 ..< restN:
          for rm in 0 ..< restM:
            for v in 0 ..< vCount:
              let vc = idx2crd(mode(atomLayout, 1).shape, v)
              let off = crd2idx(p, ((tc, vc), (tm, tn), (rm, rn))).toIntVal()
              counts[(off mod tileRows) + (off div tileRows) * tileRows].inc
    checkMultiplicity(counts, expected, $operand)

template fragCoords*(mma: untyped; operand: static MmaOperand;
                     tileM, tileK: static int; t: int): seq[(int, int)] =
  ## The (row, col) coordinates of thread t's fragment, in fragment order
  ## (v inner, rest outer, restM fastest) — decoded from the partition
  ## layout's nested coords; the atom T/V modes and the thread position
  ## are decomposed via the shape-based idx2crd(shape, flat).
  const
    T = static(mma.atom.threadCount(operand).toIntVal())
    vCount = static(mma.atom.valuesPerThread(operand).toIntVal())
    thrM = static(mma.threadLayout.shape[0].toIntVal())
    thrN = static(mma.threadLayout.shape[1].toIntVal())
    thrK = static(mma.threadLayout.shape[2].toIntVal())
    atomLayout = when operand == opA: mma.atom.aLayout
                 elif operand == opB: mma.atom.bLayout
                 else: mma.atom.cLayout
    tileRows = tileM
    restM = when operand == opA: tileM div (thrM * static(mma.atom.mnk.m))
            elif operand == opB: 1
            else: tileM div (thrM * static(mma.atom.mnk.m))
    restN = when operand == opB: tileM div (thrN * static(mma.atom.mnk.n))
            else: tileK div (thrN * static(mma.atom.mnk.n))
    restK = tileK div (thrK * static(mma.atom.mnk.k))
  block:
    let p = when operand == opA: mma.partition_A(tileM, tileK)
            elif operand == opB: mma.partition_B(tileM, tileK)
            else: mma.partition_C(tileM, tileK)
    let tv = t mod T
    let (tm, tn, tk) = idx2crd(mma.threadLayout.shape, t div T)
    let tc = idx2crd(mode(atomLayout, 0).shape, tv)
    var r: seq[(int, int)]
    when operand == opA:
      for rk in 0 ..< restK:
        for rm in 0 ..< restM:
          for v in 0 ..< vCount:
            let vc = idx2crd(mode(atomLayout, 1).shape, v)
            let off = crd2idx(p, ((tc, vc), (tm, tk), (rm, rk))).toIntVal()
            r.add (off mod tileRows, off div tileRows)
    elif operand == opB:
      for rk in 0 ..< restK:
        for rn in 0 ..< restN:
          for v in 0 ..< vCount:
            let vc = idx2crd(mode(atomLayout, 1).shape, v)
            let off = crd2idx(p, ((tc, vc), (tn, tk), (rn, rk))).toIntVal()
            r.add (off mod tileRows, off div tileRows)
    else:
      for rn in 0 ..< restN:
        for rm in 0 ..< restM:
          for v in 0 ..< vCount:
            let vc = idx2crd(mode(atomLayout, 1).shape, v)
            let off = crd2idx(p, ((tc, vc), (tm, tn), (rm, rn))).toIntVal()
            r.add (off mod tileRows, off div tileRows)
    r

template tiled(thrM, thrN, thrK: static int): untyped =
  ## Build a TiledMma with the tf32 atom and a (thrM, thrN, thrK) tiling.
  TiledMma[typeof(atom), typeof(make_layout((thrM, thrN, thrK)))](
    atom: atom, threadLayout: make_layout((thrM, thrN, thrK)))

# ═════════════════════════════════════════════════════════════════════════
#  1. Derived quantities — tile sizes and fragment sizes
#     CuTe: make_tiled_mma(atom, thr_layout) — tile_size_mnk
#     Fragment size per thread = V registers × rest positions
# ═════════════════════════════════════════════════════════════════════════

proc runDerivedQuantityTests =
  block:  # tile sizes derive from atom shape × thread tiling (3×5 tiled)
    #   tile M = ThrM·m = 3·16 = 48, N = ThrN·n = 5·8 = 40, K = ThrK·k = 8
    const mma = tiled(3, 5, 1)
    doAssert mma.threadLayout.shape[0].toIntVal() * atom.mnk.m == 48, "tile M"
    doAssert mma.threadLayout.shape[1].toIntVal() * atom.mnk.n == 40, "tile N"
    doAssert mma.threadLayout.shape[2].toIntVal() * atom.mnk.k == 8,  "tile K"
    # compile-time derived: thread count (T-mode of the layouts) and
    # values per thread (V-mode) are Int[N], not runtime int
    check atom.threadCount(opA), 32, Int
    check atom.valuesPerThread(opA), 4, Int
    check atom.valuesPerThread(opB), 2, Int
    check atom.valuesPerThread(opC), 4, Int

  block:  # K-tiled: tile K = ThrK · kAtom = 2·8 = 16
    const mma = tiled(2, 2, 2)
    doAssert mma.threadLayout.shape[2].toIntVal() * atom.mnk.k == 16, "tile K"

  block:  # fragment size = V × rest per thread (3×5 tiled, rest (7,9,1))
    #   A tile (336, 8):  rest M = 336/48 = 7, rest K = 8/8 = 1 → 4·7 = 28
    #   B tile (360, 8):  rest N = 360/40 = 9, rest K = 1        → 2·9 = 18
    #   C tile (336, 360): rest M = 7, rest N = 9                → 4·7·9 = 252
    const mma = tiled(3, 5, 1)
    doAssert fragCoords(mma, opA, 336, 8, 1).len == 28, "A fragment size"
    doAssert fragCoords(mma, opB, 360, 8, 1).len == 18, "B fragment size"
    doAssert fragCoords(mma, opC, 336, 360, 1).len == 252, "C fragment size"

  echo "  1. Derived quantities: 3 blocks OK"

# ═════════════════════════════════════════════════════════════════════════
#  2. Exact coordinates — single atom and thread-tiling boundaries
#     What: the exact (row, col) sets a thread's fragment holds, for a
#     single atom and a 2×2 tiling. Why these threads: the thread-per-atom
#     boundary (31/32), the atom-position boundary (63/64), and one thread
#     per corner — together they pin the get_slice decomposition (tm, tn),
#     the local-thread mapping, and the operand-sharing pattern (A shared
#     across N-atoms, B shifting with the N-atom, C per (tm, tn)).
#     Expected values: [CUTE-EX]
# ═════════════════════════════════════════════════════════════════════════

proc runThreadTilingBoundaryTests =
  block:  # single atom (1×1), rest (1,1,1), thread 0
    const mma = tiled(1, 1, 1)
    doAssert fragCoords(mma, opA, 16, 8, 0) == @[(0,0),(8,0),(0,4),(8,4)]
    doAssert fragCoords(mma, opB, 8, 8, 0) == @[(0,0),(0,4)]
    doAssert fragCoords(mma, opC, 16, 8, 0) == @[(0,0),(0,1),(8,0),(8,1)]

  block:  # 2×2 tiled, rest (1,1,1): threads 0 / 31 / 32 / 63 / 64 / 127
    const mma = tiled(2, 2, 1)
    let expA = @[
      @[(0,0),(8,0),(0,4),(8,4)],      # t0:   atom (0,0)
      @[(7,3),(15,3),(7,7),(15,7)],    # t31:  atom (0,0), last local thread
      @[(16,0),(24,0),(16,4),(24,4)],  # t32:  atom (1,0), first local thread
      @[(23,3),(31,3),(23,7),(31,7)],  # t63:  atom (1,0)
      @[(0,0),(8,0),(0,4),(8,4)],      # t64:  atom (0,1) — A shared with t0
      @[(23,3),(31,3),(23,7),(31,7)]]  # t127: atom (1,1)
    let expB = @[
      @[(0,0),(0,4)], @[(7,3),(7,7)],
      @[(0,0),(0,4)], @[(7,3),(7,7)],
      @[(8,0),(8,4)], @[(15,3),(15,7)]]  # B shifts with the N-atom
    let expC = @[
      @[(0,0),(0,1),(8,0),(8,1)],        # t0
      @[(7,6),(7,7),(15,6),(15,7)],      # t31
      @[(16,0),(16,1),(24,0),(24,1)],    # t32
      @[(23,6),(23,7),(31,6),(31,7)],    # t63
      @[(0,8),(0,9),(8,8),(8,9)],        # t64
      @[(23,14),(23,15),(31,14),(31,15)]]  # t127
    for i, t in [0, 31, 32, 63, 64, 127]:
      doAssert fragCoords(mma, opA, 32, 8, t) == expA[i], &"A t{t}"
      doAssert fragCoords(mma, opB, 16, 8, t) == expB[i], &"B t{t}"
      doAssert fragCoords(mma, opC, 32, 16, t) == expC[i], &"C t{t}"

  echo "  2. Thread-tiling boundaries: 2 blocks OK"

# ═════════════════════════════════════════════════════════════════════════
#  3. Exact coordinates — rest positions (value tiling)
#     What: the fragment when the tile exceeds the tiled-mma unit
#     (rest > 1): pure value tiling (1×1 tiled) and tiled + value tiling
#     (3×5 tiled), first and last threads. Why: pins the rest-position
#     ordering (mma-position outer, col-major (RestM, RestK), v inner) —
#     the (MMA, M, K) fragment shape.
#     Expected values: [CUTE-EX]
# ═════════════════════════════════════════════════════════════════════════

proc runRestPositionTests =
  block:  # 1×1 tiled, rest (7,1)/(9,1)/(7,9) — pure value tiling, thread 0
    const mma = tiled(1, 1, 1)
    let expA = @[
      (0,0),(8,0),(0,4),(8,4), (16,0),(24,0),(16,4),(24,4),
      (32,0),(40,0),(32,4),(40,4), (48,0),(56,0),(48,4),(56,4),
      (64,0),(72,0),(64,4),(72,4), (80,0),(88,0),(80,4),(88,4),
      (96,0),(104,0),(96,4),(104,4)]
    doAssert fragCoords(mma, opA, 112, 8, 0) == expA,
      &"1×1 rest(7,1) A: {fragCoords(mma, opA, 112, 8, 0)}"
    let expB = @[
      (0,0),(0,4), (8,0),(8,4), (16,0),(16,4), (24,0),(24,4),
      (32,0),(32,4), (40,0),(40,4), (48,0),(48,4), (56,0),(56,4),
      (64,0),(64,4)]
    doAssert fragCoords(mma, opB, 72, 8, 0) == expB, "1×1 rest(9,1) B"
    # C rest (7,9): size 252; corner samples at the (rn, rm) corners
    let fC = fragCoords(mma, opC, 112, 72, 0)
    doAssert fC.len == 252
    doAssert fC[0..3] == @[(0,0),(0,1),(8,0),(8,1)]
    doAssert fC[4..7] == @[(16,0),(16,1),(24,0),(24,1)]
    doAssert fC[28..31] == @[(0,8),(0,9),(8,8),(8,9)]
    doAssert fC[248..251] == @[(96,64),(96,65),(104,64),(104,65)]

  block:  # 3×5 tiled, rest (7,9,1): first (t1) and last (t479) thread
    const mma = tiled(3, 5, 1)
    # thread 1: atom (0,0), local thread 1 → A k=1, C n=2
    let expA1 = @[
      (0,1),(8,1),(0,5),(8,5), (48,1),(56,1),(48,5),(56,5),
      (96,1),(104,1),(96,5),(104,5), (144,1),(152,1),(144,5),(152,5),
      (192,1),(200,1),(192,5),(200,5), (240,1),(248,1),(240,5),(248,5),
      (288,1),(296,1),(288,5),(296,5)]
    doAssert fragCoords(mma, opA, 336, 8, 1) == expA1, "3×5 A t1"
    let expB1 = @[
      (0,1),(0,5), (40,1),(40,5), (80,1),(80,5), (120,1),(120,5),
      (160,1),(160,5), (200,1),(200,5), (240,1),(240,5), (280,1),(280,5),
      (320,1),(320,5)]
    doAssert fragCoords(mma, opB, 360, 8, 1) == expB1, "3×5 B t1"
    let fC1 = fragCoords(mma, opC, 336, 360, 1)
    doAssert fC1.len == 252
    doAssert fC1[0..3] == @[(0,2),(0,3),(8,2),(8,3)]
    doAssert fC1[248..251] == @[(288,322),(288,323),(296,322),(296,323)]
    # thread 479 (last of 480): atom (2,4), local thread 31
    let expA479 = @[
      (39,3),(47,3),(39,7),(47,7), (87,3),(95,3),(87,7),(95,7),
      (135,3),(143,3),(135,7),(143,7), (183,3),(191,3),(183,7),(191,7),
      (231,3),(239,3),(231,7),(239,7), (279,3),(287,3),(279,7),(287,7),
      (327,3),(335,3),(327,7),(335,7)]
    doAssert fragCoords(mma, opA, 336, 8, 479) == expA479, "3×5 A t479"
    doAssert fragCoords(mma, opB, 360, 8, 479)[0..3] == @[(39,3),(39,7),(79,3),(79,7)]
    doAssert fragCoords(mma, opC, 336, 360, 479)[0..3] == @[(39,38),(39,39),(47,38),(47,39)]

  echo "  3. Rest positions (value tiling): 2 blocks OK"

# ═════════════════════════════════════════════════════════════════════════
#  4. Exact coordinates — K-tiled mma (thrK > 1)
#     What: coordinates when the atom tiling includes K (2×2×2): the
#     k-coordinates shift by the atom's k-step for tk = 1 threads, and C is
#     unaffected by tk (each tk group holds a full copy of the same C
#     fragment). Why: pins the tk decomposition of get_slice and the
#     k-independent C partition.
#     Expected values: [CUTE-EX]
# ═════════════════════════════════════════════════════════════════════════

proc runKTiledTests =
  block:  # 2×2×2 tiled, rest (1,1,1): threads 0 / 128 (tk=1) / 255
    const mma = tiled(2, 2, 2)
    doAssert fragCoords(mma, opA, 32, 16, 0) == @[(0,0),(8,0),(0,4),(8,4)]
    doAssert fragCoords(mma, opB, 16, 16, 0) == @[(0,0),(0,4)]
    doAssert fragCoords(mma, opC, 32, 16, 0) == @[(0,0),(0,1),(8,0),(8,1)]
    # thread 128: tk = 1 → k coords 8, 12; C unchanged (C has no K)
    doAssert fragCoords(mma, opA, 32, 16, 128) == @[(0,8),(8,8),(0,12),(8,12)]
    doAssert fragCoords(mma, opB, 16, 16, 128) == @[(0,8),(0,12)]
    doAssert fragCoords(mma, opC, 32, 16, 128) == @[(0,0),(0,1),(8,0),(8,1)]
    # thread 255: atom (1,1,1), local thread 31
    doAssert fragCoords(mma, opA, 32, 16, 255) == @[(23,11),(31,11),(23,15),(31,15)]
    doAssert fragCoords(mma, opB, 16, 16, 255) == @[(15,11),(15,15)]
    doAssert fragCoords(mma, opC, 32, 16, 255) == @[(23,14),(23,15),(31,14),(31,15)]

  echo "  4. K-tiled mma: 1 block OK"

# ═════════════════════════════════════════════════════════════════════════
#  5. Rejections and structural invariants
# ═════════════════════════════════════════════════════════════════════════

proc runRejectionAndInvariantTests =
  block:  # non-multiple tile shape is rejected (doAssert fires)
    const mma = tiled(2, 2, 1)
    try:
      discard mma.partition_A(33, 8)   # 33 not a multiple of 32
      doAssert false, "expected AssertionDefect for non-multiple tile shape"
    except AssertionDefect:
      discard

  block:  # coverage with rest modes — every element exactly the expected
    #        multiplicity (A ×ThrN, B ×ThrM, C ×ThrK), including rest
    const mma = tiled(3, 5, 1)
    mma.verifyFragments(336, 8, opA)
    mma.verifyFragments(360, 8, opB)
    mma.verifyFragments(336, 360, opC)

  block:  # K-tiled coverage (2×2×2): the thrK-thread loop + C × thrK
    const mma = tiled(2, 2, 2)
    mma.verifyFragments(32, 16, opA)
    mma.verifyFragments(16, 16, opB)
    mma.verifyFragments(32, 16, opC)

  block:  # single atom with value tiling only (1×1, rest (7,9,1)) — the
    #        tensor-layouts tile_mnk expansion shape
    const mma = tiled(1, 1, 1)
    mma.verifyFragments(112, 8, opA)
    mma.verifyFragments(72, 8, opB)
    mma.verifyFragments(112, 72, opC)

  block:  # every thread of a 3×5 tiling has the same fragment SIZE
    const mma = tiled(3, 5, 1)
    for t in 0 ..< 480:
      doAssert fragCoords(mma, opA, 336, 8, t).len == 28, &"A size t{t}"
      doAssert fragCoords(mma, opB, 360, 8, t).len == 18, &"B size t{t}"
      doAssert fragCoords(mma, opC, 336, 360, t).len == 252, &"C size t{t}"

  echo "  5. Rejections and invariants: 5 blocks OK"

# ═════════════════════════════════════════════════════════════════════════
#  Test runner
# ═════════════════════════════════════════════════════════════════════════

proc runTests =
  runDerivedQuantityTests()
  runThreadTilingBoundaryTests()
  runRestPositionTests()
  runKTiledTests()
  runRejectionAndInvariantTests()
  echo "\nALL TESTS PASSED"

when isMainModule:
  runTests()
