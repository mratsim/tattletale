## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#      Tile epilogues: the Epilogue concept and the shipped set
#
# ############################################################
#
# An epilogue transforms the accumulated GEMM result into the output tile.
# The user contract is the `Epilogue` concept below.
# The shipped set: identity, ReLU, the bias column broadcast, bias+ReLU, and α·AB + β·C.

import workspace/crucible
import ../int_tuples
import ../layouts
import ../layout_constructors
import ../tensors
import ../ptr_arithmetic
import ../atoms
import ./tiles
import ./tile_config
import ./tile_fma_partition
import ./tile_ops
import ./tile_mma

export tiles, tile_ops, tile_mma
# Tile surface re-exported so the epilogue flow resolves at the call site.

# ═════════════════════════════════════════════════════════════════════════
#  Tile Epilogue concept
# ═════════════════════════════════════════════════════════════════════════

type Epilogue* = concept
  ## A tile epilogue computes the output tile D = f(AB) from the
  ## accumulated GEMM result. The user writes the epilogue type, one
  ## TensorView field per captured operand, plus `apply` in the
  ## view-backed and the fragment-resident accumulator forms.
  proc apply(op: Self, tmp: var (TensorView or Tensor), AB: TensorView or Tensor)
  proc apply(op: Self, tmp: var RtLeft, AB: RtLeft)

# ═════════════════════════════════════════════════════════════════════════
#  Operand captures: the tile-shaped gmem views
# ═════════════════════════════════════════════════════════════════════════

func biasView*[T; R, C: static int](buf: ptr UncheckedArray[T]): TensorView[T, (Int[R], Int[C]), (Int[0], Int[1])] =
  ## The bias as a stride-0-row view over the tile's columns (one value
  ## per column broadcast across the rows). R×C are the output tile dims.
  make_view(buf, (R, C), (0, 1))

func cView*[T; R, C: static int](buf: ptr UncheckedArray[T]): TensorView[T, (Int[R], Int[C]), (Int[C], Int[1])] =
  ## The full C operand as a tile-shaped view, its row stride the tile's
  ## column count. R×C are the output tile dims.
  make_view(buf, (R, C), (C, 1))

# ═════════════════════════════════════════════════════════════════════════
#  EpiIdentity
# ═════════════════════════════════════════════════════════════════════════

type EpiIdentity* = object
  ## Identity: D = AB.

func apply*[T, Sh, StAB, StR](
    op: EpiIdentity;
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = AB, per element.
  const S = toIntVal(size(tmp))
  for i in 0 ..< S:
    tmp(i) = AB(i)

proc apply*[T; R, C: static int; AT, ABT: static MmaAtom; TL: static ThreadLayout](
    op: EpiIdentity;
    tmp: var RtLeft[T, R, C, AT, TL];
    AB: RtLeft[T, R, C, ABT, TL]) {.inline.} =
  ## D = AB, per owned slot.
  static:
    doAssert AT.mnk.m == ABT.mnk.m and AT.mnk.n == ABT.mnk.n and
      toIntVal(AT.valuesPerThread(opC)) == toIntVal(ABT.valuesPerThread(opC)),
      "apply: the accumulator and operand tiles must share the atom's subtile grid and per-lane count"
  const rowTiles = R div AT.mnk.m
  const colTiles = C div AT.mnk.n
  const vpt = toIntVal(AT.valuesPerThread(opC))
  let thr = fmaSlice[AT, TL]()
  for n in countup(thr.tm, rowTiles - 1, TL.thrM):
    for m in countup(thr.tn, colTiles - 1, TL.thrN):
      for v in 0 ..< vpt:
        tmp.frags[n][m].frag[v] = AB.frags[n][m].frag[v]

# ═════════════════════════════════════════════════════════════════════════
#  EpiReLU
# ═════════════════════════════════════════════════════════════════════════

type EpiReLU* = object
  ## Rectified linear unit: D = max(0, AB).

func apply*[T, Sh, StAB, StR](
    op: EpiReLU;
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = max(0, AB), per element.
  const S = toIntVal(size(tmp))
  for i in 0 ..< S:
    tmp(i) = max(AB(i), T(0))

proc apply*[T; R, C: static int; AT, ABT: static MmaAtom; TL: static ThreadLayout](
    op: EpiReLU;
    tmp: var RtLeft[T, R, C, AT, TL];
    AB: RtLeft[T, R, C, ABT, TL]) {.inline.} =
  ## D = max(0, AB), per owned slot.
  static:
    doAssert AT.mnk.m == ABT.mnk.m and AT.mnk.n == ABT.mnk.n and
      toIntVal(AT.valuesPerThread(opC)) == toIntVal(ABT.valuesPerThread(opC)),
      "apply: the accumulator and operand tiles must share the atom's subtile grid and per-lane count"
  const rowTiles = R div AT.mnk.m
  const colTiles = C div AT.mnk.n
  const vpt = toIntVal(AT.valuesPerThread(opC))
  let thr = fmaSlice[AT, TL]()
  for n in countup(thr.tm, rowTiles - 1, TL.thrM):
    for m in countup(thr.tn, colTiles - 1, TL.thrN):
      for v in 0 ..< vpt:
        tmp.frags[n][m].frag[v] =
          max(AB.frags[n][m].frag[v], T(0))

# ═════════════════════════════════════════════════════════════════════════
#  EpiAXPBY
# ═════════════════════════════════════════════════════════════════════════

type EpiAXPBY*[T, Sh, StC] = object
  ## D = α·AB + β·C, per element.
  alpha*, beta*: T
  C_gmem*: TensorView[T, Sh, StC]

func initEpiAXPBY*[T, Sh, StC](alpha: T; beta: T;
                               C: TensorView[T, Sh, StC]): EpiAXPBY[T, Sh, StC] =
  ## Gmem capture of C plus the scalars.
  EpiAXPBY[T, Sh, StC](alpha: alpha, beta: beta, C_gmem: C)

func apply*[T, Sh, StAB, StC, StR](
    op: EpiAXPBY[T, Sh, StC];
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = α·AB + β·C, per element. β = 0 skips reading C, α = 1 skips
  ## the multiply. The branches are uniform per thread.
  const S = toIntVal(size(tmp))
  if op.beta == T(0):
    if op.alpha == T(1):
      for i in 0 ..< S:
        tmp(i) = AB(i)
    else:
      for i in 0 ..< S:
        tmp(i) = op.alpha * AB(i)
  elif op.alpha == T(1):
    for i in 0 ..< S:
      tmp(i) = AB(i) + op.beta * op.C_gmem(i)
  else:
    for i in 0 ..< S:
      tmp(i) = op.alpha * AB(i) + op.beta * op.C_gmem(i)

proc apply*[T; R, C: static int; AT, ABT: static MmaAtom; TL: static ThreadLayout; Sh, StC](
    op: EpiAXPBY[T, Sh, StC];
    tmp: var RtLeft[T, R, C, AT, TL];
    AB: RtLeft[T, R, C, ABT, TL]) {.inline.} =
  ## D = α·AB + β·C, per owned slot, the C view the per-lane shard.
  static:
    doAssert AT.mnk.m == ABT.mnk.m and AT.mnk.n == ABT.mnk.n and
      toIntVal(AT.valuesPerThread(opC)) == toIntVal(ABT.valuesPerThread(opC)),
      "apply: the accumulator and operand tiles must share the atom's subtile grid and per-lane count"
  const rowTiles = R div AT.mnk.m
  const colTiles = C div AT.mnk.n
  const vpt = toIntVal(AT.valuesPerThread(opC))
  let thr = fmaSlice[AT, TL]()
  if op.beta == T(0):
    if op.alpha == T(1):
      for n in countup(thr.tm, rowTiles - 1, TL.thrM):
        for m in countup(thr.tn, colTiles - 1, TL.thrN):
          for v in 0 ..< vpt:
            tmp.frags[n][m].frag[v] = AB.frags[n][m].frag[v]
    else:
      for n in countup(thr.tm, rowTiles - 1, TL.thrM):
        for m in countup(thr.tn, colTiles - 1, TL.thrN):
          for v in 0 ..< vpt:
            tmp.frags[n][m].frag[v] =
              op.alpha * AB.frags[n][m].frag[v]
  elif op.alpha == T(1):
    for n in countup(thr.tm, rowTiles - 1, TL.thrM):
      for m in countup(thr.tn, colTiles - 1, TL.thrN):
        for v in 0 ..< vpt:
          tmp.frags[n][m].frag[v] =
            AB.frags[n][m].frag[v] + op.beta * op.C_gmem[n, m, v]
  else:
    for n in countup(thr.tm, rowTiles - 1, TL.thrM):
      for m in countup(thr.tn, colTiles - 1, TL.thrN):
        for v in 0 ..< vpt:
          tmp.frags[n][m].frag[v] =
            op.alpha * AB.frags[n][m].frag[v] +
            op.beta * op.C_gmem[n, m, v]

# ═════════════════════════════════════════════════════════════════════════
#  EpiAddBias
# ═════════════════════════════════════════════════════════════════════════

type EpiAddBias*[T, Sh, St] = object
  ## D = AB + bias, the bias a column vector broadcast over the tile rows via its stride-0 row view.
  bias_gmem*: TensorView[T, Sh, St]

func initEpiAddBias*[T, Sh, St](bias: TensorView[T, Sh, St]): EpiAddBias[T, Sh, St] {.inline.} =
  ## Gmem capture of the bias.
  EpiAddBias[T, Sh, St](bias_gmem: bias)

func apply*[T, Sh, StAB, StB, StR](
    op: EpiAddBias[T, Sh, StB];
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = AB + bias, the bias a column vector broadcast over the tile rows.
  const S = toIntVal(size(tmp))
  for i in 0 ..< S:
    tmp(i) = AB(i) + op.bias_gmem(i)

proc apply*[T; R, C: static int; AT, ABT: static MmaAtom; TL: static ThreadLayout; Sh, StB](
    op: EpiAddBias[T, Sh, StB];
    tmp: var RtLeft[T, R, C, AT, TL];
    AB: RtLeft[T, R, C, ABT, TL]) {.inline.} =
  ## D = AB + bias, per owned slot, the bias view the per-lane shard.
  static:
    doAssert AT.mnk.m == ABT.mnk.m and AT.mnk.n == ABT.mnk.n and
      toIntVal(AT.valuesPerThread(opC)) == toIntVal(ABT.valuesPerThread(opC)),
      "apply: the accumulator and operand tiles must share the atom's subtile grid and per-lane count"
  const rowTiles = R div AT.mnk.m
  const colTiles = C div AT.mnk.n
  const vpt = toIntVal(AT.valuesPerThread(opC))
  let thr = fmaSlice[AT, TL]()
  for n in countup(thr.tm, rowTiles - 1, TL.thrM):
    for m in countup(thr.tn, colTiles - 1, TL.thrN):
      for v in 0 ..< vpt:
        tmp.frags[n][m].frag[v] =
          AB.frags[n][m].frag[v] + op.bias_gmem[n, m, v]

# ═════════════════════════════════════════════════════════════════════════
#  EpiLinearBiasReLU
# ═════════════════════════════════════════════════════════════════════════

type EpiLinearBiasReLU*[T, Sh, St] = object
  ## D = max(0, AB + bias), the bias a column vector broadcast over the
  ## tile rows.
  bias_gmem*: TensorView[T, Sh, St]

func initEpiLinearBiasReLU*[T, Sh, St](bias: TensorView[T, Sh, St]): EpiLinearBiasReLU[T, Sh, St] {.inline.} =
  ## Gmem capture of the bias.
  EpiLinearBiasReLU[T, Sh, St](bias_gmem: bias)

func apply*[T, Sh, StAB, StB, StR](
    op: EpiLinearBiasReLU[T, Sh, StB];
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = max(0, AB + bias), the bias a column vector broadcast over the tile rows.
  const S = toIntVal(size(tmp))
  for i in 0 ..< S:
    tmp(i) = max(AB(i) + op.bias_gmem(i), T(0))

proc apply*[T; R, C: static int; AT, ABT: static MmaAtom; TL: static ThreadLayout; Sh, StB](
    op: EpiLinearBiasReLU[T, Sh, StB];
    tmp: var RtLeft[T, R, C, AT, TL];
    AB: RtLeft[T, R, C, ABT, TL]) {.inline.} =
  ## D = max(0, AB + bias), per owned slot, the bias view the per-lane
  ## shard.
  static:
    doAssert AT.mnk.m == ABT.mnk.m and AT.mnk.n == ABT.mnk.n and
      toIntVal(AT.valuesPerThread(opC)) == toIntVal(ABT.valuesPerThread(opC)),
      "apply: the accumulator and operand tiles must share the atom's subtile grid and per-lane count"
  const rowTiles = R div AT.mnk.m
  const colTiles = C div AT.mnk.n
  const vpt = toIntVal(AT.valuesPerThread(opC))
  let thr = fmaSlice[AT, TL]()
  for n in countup(thr.tm, rowTiles - 1, TL.thrM):
    for m in countup(thr.tn, colTiles - 1, TL.thrN):
      for v in 0 ..< vpt:
        tmp.frags[n][m].frag[v] =
          max(AB.frags[n][m].frag[v] + op.bias_gmem[n, m, v], T(0))
