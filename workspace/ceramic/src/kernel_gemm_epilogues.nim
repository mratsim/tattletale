## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Epilogue fusion operations for the tiled GEMM.
##
## One concept, one walk: `apply(D, AB)` computes the output tile
##   D = f(AB)
## where
##   D   is the destination (var; the output tile in registers or gmem)
##   AB  is the accumulated GEMM result (the accumulator fragment)
##
## Every other input is OP STATE: the op object is the complete epilogue
## configuration, carrying its operands as fields (EpiAXPBY's C,
## EpiAddBias's bias). The concept's `preflight()` stages those fields
## into smem (async memcpy / TMA in the full design; a stub here). It
## subsumes the getLoads declaration: the fields and the staging body
## ARE the load spec. See GEMM-ARCHITECTURE §4 Level 3.
##
## The math is zipped by SIZE, not by dimension. The loop iterates
## `size(D)` and indexes D(i), AB(i) each through its own layout.
##
## No rank or stride assumption on any operand. Rank-2 gmem tiles,
## rank-3 accumulator fragments (V, RestM, RestN), nested layouts and
## broadcast (stride-0) layouts all resolve through the layout algebra.
## That includes an AMD MFMA fragment with irregular rows, the nested
## AMX layout (1,(16,16)):(0,(1,16)), and a stride-0 bias broadcast view.
##
## A 2D (i, j) loop over the tile's (M, N) would break on every
## non-col-major fragment.
##
## The op is a compile-time functor: static dispatch, runtime fields
## (alpha/beta as data), `func` bodies. Each op's `apply` hoists its
## dispatch out of the element loop (the ex02a genEpilogue pattern).
## No per-element branches on op state.
##
## The concept is Nim concepts V2 (bare `concept`, `Self` = the matched
## type). Two matching gotchas, both probe-verified:
##   1. the concept signature must mirror the ops' type-class structure
##      exactly. An op taking `var (TensorView or Tensor)` needs the
##      same union in the concept, or the match fails;
##   2. the ops' apply signatures must use the EXPLICIT union, not the
##      `AnyTensor` alias. V2 matching cannot infer the shape params
##      (ShAB, StAB, ...) through the alias (probe 9, "cannot
##      instantiate"). The alias is fine for plain procs and templates
##      (tensors.nim, axpby, partition_A/B/C), just not here.
##
## Loads (getLoads, LoadKind, ikBroadcast) and capability flags are the
## NEXT step per the design. This file ships the math-only PoC.

import ./int_tuples
import ./layouts
import ./tensors

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  Epilogue concept
# ═════════════════════════════════════════════════════════════════════════

type Epilogue* = concept
  proc preflight(op: var Self)
  proc apply(op: Self; D: var (TensorView or Tensor);
             AB: TensorView or Tensor)

# ═════════════════════════════════════════════════════════════════════════
#  EpiAXPBY: D = α·AB + β·C
# ═════════════════════════════════════════════════════════════════════════

type EpiAXPBY*[T, Sh, StC] = object
  ## Linear combination of the accumulator and the C operand. C is op
  ## state, carried in the object as part of the epilogue configuration
  ## (the same mechanism as EpiAddBias's bias). preflight stages it into
  ## smem in the full design.
  alpha*, beta*: T
  C*: TensorView[T, Sh, StC]

func initEpiAXPBY*[T, Sh, StC](
    alpha: T; beta: T; C: TensorView[T, Sh, StC]): EpiAXPBY[T, Sh, StC] =
  ## Build the op with its C operand. The C view is the same shape as
  ## the output tile (shared Sh, enforced by the compiler).
  result = EpiAXPBY[T, Sh, StC](alpha: alpha, beta: beta, C: C)

proc preflight*[T, Sh, StC](op: var EpiAXPBY[T, Sh, StC]) {.inline.} =
  ## Stub. In the full design this stages op.C into smem with an async
  ## copy (TMA / cp.async), and skips it entirely when beta == 0 (the
  ## load-level bandwidth win of the C-skip guarantee, §8o).
  discard

func apply*[T, Sh, StD, StAB, StC](
    op: EpiAXPBY[T, Sh, StC];
    D: var (TensorView[T, Sh, StD] or Tensor[T, Sh, StD]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = α·AB + β·C, element-wise over the D tile's (M, N) shape.
  ## Dispatch hoisted out of the loops (four flat paths):
  ##   β == 0 → C is never read, this saves memory bandwidth
  ##   α == 1 → the α multiply is skipped
  ## On GPU the branches are uniform
  ##   α/β are identical for every thread
  ##   no warp divergence, the cost is on instruction cache / code size
  let n = toIntVal(size(D.layout))
  if op.beta == T(0):
    if op.alpha == T(1):
      for i in 0 ..< n:
        D(i) = AB(i)
    else:
      for i in 0 ..< n:
        D(i) = op.alpha * AB(i)
  elif op.alpha == T(1):
    for i in 0 ..< n:
      # This is a single FMA instruction,
      # specializing for β == doesn't gain performance
      D(i) = AB(i) + op.beta * op.C(i)
  else:
    for i in 0 ..< n:
      D(i) = op.alpha * AB(i) + op.beta * op.C(i)

# ═════════════════════════════════════════════════════════════════════════
#  EpiIdentity: D = AB (α=1, β=0, compile-time constants)
# ═════════════════════════════════════════════════════════════════════════

type EpiIdentity* = object
  ## Identity epilogue: D = AB, C ignored. Used when the GEMM has no C
  ## operand. C stays in the signature (the concept is uniform) but is
  ## never read.

proc preflight*(op: var EpiIdentity) {.inline.} =
  ## Stub. Nothing to stage: the identity epilogue has no inputs.

func apply*[T, Sh, StD, StAB](
    op: EpiIdentity;
    D: var (TensorView[T, Sh, StD] or Tensor[T, Sh, StD]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  let n = toIntVal(size(D.layout))
  for i in 0 ..< n:
    D(i) = AB(i)

# ═════════════════════════════════════════════════════════════════════════
#  EpiAddBias: D = AB + bias (column broadcast)
# ═════════════════════════════════════════════════════════════════════════

type EpiAddBias*[T, Sh, St] = object
  ## D = AB + bias. The bias is op state, carried in the object as part
  ## of the epilogue configuration, not a per-call operand. The load
  ## side produces a same-size broadcast view of the bias column
  ## (stride-0 row mode, LoadKind ikBroadcast in the full design). The
  ## math is a plain same-size add.
  bias*: TensorView[T, Sh, St]

func initEpiAddBias*[T, Sh, St](bias: TensorView[T, Sh, St]): EpiAddBias[T, Sh, St] =
  ## Build the op from the broadcast bias view (the load side produces
  ## it: a stride-0 row-mode view of the bias column, same shape as the
  ## output tile).
  result.bias = bias

proc preflight*[T, Sh, St](op: var EpiAddBias[T, Sh, St]) {.inline.} =
  ## Stub. In the full design this stages op.bias into smem (the
  ## broadcast column, stride-0 view) with an async copy.
  discard

func apply*[T, Sh, StD, StAB, StB](
    op: EpiAddBias[T, Sh, StB];
    D: var (TensorView[T, Sh, StD] or Tensor[T, Sh, StD]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  let n = toIntVal(size(D.layout))
  for i in 0 ..< n:
    D(i) = AB(i) + op.bias(i)

# ═════════════════════════════════════════════════════════════════════════
#  EpiReLU: D = max(0, AB)
# ═════════════════════════════════════════════════════════════════════════

type EpiReLU* = object
  ## Rectified linear unit: D = max(0, AB)

proc preflight*(op: var EpiReLU) {.inline.} =
  ## Stub. Nothing to stage: ReLU has no inputs.

func apply*[T, Sh, StD, StAB](
    op: EpiReLU;
    D: var (TensorView[T, Sh, StD] or Tensor[T, Sh, StD]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  let n = toIntVal(size(D.layout))
  for i in 0 ..< n:
    D(i) = if AB(i) > T(0): AB(i) else: T(0)
