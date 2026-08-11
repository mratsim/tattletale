## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Epilogue fusion operations for the tiled GEMM — the `Epilogue` concept.
##
## One concept, one walk: `apply(D, AB, C)` computes the output tile
##   D = f(AB, C)
## where
##   D  — destination (var; the output tile in registers or gmem)
##   AB — the accumulated GEMM result (the accumulator fragment)
##   C  — the epilogue operand (initial C, bias, or anything declared via
##        getLoads in the full design — see GEMM-ARCHITECTURE §4 Level 3)
##
## The op is a compile-time functor: static dispatch, runtime fields
## (alpha/beta as data), `func` bodies. Each op's `apply` hoists its
## dispatch out of the element loop (the ex02a genEpilogue pattern) —
## no per-element branches on op state.
##
## The concept is Nim concepts V2 (bare `concept`, `Self` = the matched
## type). Two matching gotchas, both probe-verified:
##   1. the concept signature must mirror the ops' type-class structure
##      exactly — an op taking `var (TensorView or Tensor)` needs the
##      same union in the concept, or the match fails;
##   2. the ops' apply signatures must use the EXPLICIT union, not the
##      `AnyTensor` alias: V2 matching cannot infer the shape params
##      (ShAB, StAB, ...) through the alias (probe 9 — "cannot
##      instantiate"). The alias is fine for plain procs/templates
##      (tensors.nim, axpby, partition_A/B/C), just not here.
##
## Loads (getLoads, LoadKind, ikBroadcast) and capability flags are the
## NEXT step per the design — this file ships the math-only PoC.

import ./int_tuples
import ./layouts
import ./tensors

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  Epilogue concept
# ═════════════════════════════════════════════════════════════════════════

type Epilogue* = concept
  proc apply(op: Self; D: var (TensorView or Tensor);
             AB: TensorView or Tensor; C: TensorView or Tensor)

# ═════════════════════════════════════════════════════════════════════════
#  EpiAXPBY — D = α·AB + β·C
# ═════════════════════════════════════════════════════════════════════════

type EpiAXPBY*[T] = object
  ## Linear combination of the accumulator and the C operand.
  alpha*, beta*: T

func apply*[T, ShD, StD, ShAB, StAB, ShC, StC](
    op: EpiAXPBY[T];
    D: var (TensorView[T, ShD, StD] or Tensor[T, ShD, StD]);
    AB: TensorView[T, ShAB, StAB] or Tensor[T, ShAB, StAB];
    C: TensorView[T, ShC, StC] or Tensor[T, ShC, StC]) {.inline.} =
  ## D = α·AB + β·C, element-wise over the D tile's (M, N) shape.
  ## Dispatch hoisted out of the loops (four flat paths):
  ##   β == 0 → C is never read, this saves memory bandwidth
  ##   α == 1 → the α multiply is skipped
  ## On GPU the branches are uniform
  ##   α/β are identical for every thread
  ##   no warp divergence, the cost is on instruction cache / code size
  let M = toIntVal(mode(D.layout, 0).shape)
  let N = toIntVal(mode(D.layout, 1).shape)
  if op.beta == T(0):
    if op.alpha == T(1):
      for i in 0 ..< M:
        for j in 0 ..< N:
          D[i, j] = AB[i, j]
    else:
      for i in 0 ..< M:
        for j in 0 ..< N:
          D[i, j] = op.alpha * AB[i, j]
  elif op.alpha == T(1):
    for i in 0 ..< M:
      for j in 0 ..< N:
        # This is a single FMA instruction,
        # specializing for β == doesn't gain performance
        D[i, j] = AB[i, j] + op.beta * C[i, j]
  else:
    for i in 0 ..< M:
      for j in 0 ..< N:
        D[i, j] = op.alpha * AB[i, j] + op.beta * C[i, j]

# ═════════════════════════════════════════════════════════════════════════
#  EpiIdentity — D = AB  (α=1, β=0, compile-time constants)
# ═════════════════════════════════════════════════════════════════════════

type EpiIdentity* = object
  ## Identity epilogue: D = AB, C ignored. Used when the GEMM has no C
  ## operand. C stays in the signature (the concept is uniform) but is
  ## never read.

func apply*[T, ShD, StD, ShAB, StAB, ShC, StC](
    op: EpiIdentity;
    D: var (TensorView[T, ShD, StD] or Tensor[T, ShD, StD]);
    AB: TensorView[T, ShAB, StAB] or Tensor[T, ShAB, StAB];
    C: TensorView[T, ShC, StC] or Tensor[T, ShC, StC]) {.inline.} =
  let M = toIntVal(mode(D.layout, 0).shape)
  let N = toIntVal(mode(D.layout, 1).shape)
  for i in 0 ..< M:
    for j in 0 ..< N:
      D[i, j] = AB[i, j]

# ═════════════════════════════════════════════════════════════════════════
#  EpiAddBias — D = AB + bias  (column broadcast)
# ═════════════════════════════════════════════════════════════════════════

type EpiAddBias* = object
  ## D = AB + C where C is a column vector of length N broadcast across
  ## the M rows (a "broadcasted column add"). The broadcast lives on the
  ## LOAD side in the full design (LoadKind ikBroadcast) — here the math
  ## is a plain add and C arrives as a rank-1 (N,) view.

func apply*[T, ShD, StD, ShAB, StAB, ShC, StC](
    op: EpiAddBias;
    D: var (TensorView[T, ShD, StD] or Tensor[T, ShD, StD]);
    AB: TensorView[T, ShAB, StAB] or Tensor[T, ShAB, StAB];
    C: TensorView[T, ShC, StC] or Tensor[T, ShC, StC]) {.inline.} =
  let M = toIntVal(mode(D.layout, 0).shape)
  let N = toIntVal(mode(D.layout, 1).shape)
  doAssert toIntVal(size(C.layout)) == N,
    "EpiAddBias: C must be a column vector of length N (the D tile's N)"
  for i in 0 ..< M:
    for j in 0 ..< N:
      D[i, j] = AB[i, j] + C[j]

# ═════════════════════════════════════════════════════════════════════════
#  EpiReLU — D = max(0, AB)
# ═════════════════════════════════════════════════════════════════════════

type EpiReLU* = object
  ## Rectified linear unit: D = max(0, AB)

func apply*[T, ShD, StD, ShAB, StAB, ShC, StC](
    op: EpiReLU;
    D: var (TensorView[T, ShD, StD] or Tensor[T, ShD, StD]);
    AB: TensorView[T, ShAB, StAB] or Tensor[T, ShAB, StAB];
    C: TensorView[T, ShC, StC] or Tensor[T, ShC, StC]) {.inline.} =
  let M = toIntVal(mode(D.layout, 0).shape)
  let N = toIntVal(mode(D.layout, 1).shape)
  for i in 0 ..< M:
    for j in 0 ..< N:
      D[i, j] = if AB[i, j] > T(0): AB[i, j] else: T(0)
