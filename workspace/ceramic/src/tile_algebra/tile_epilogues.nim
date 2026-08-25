## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import ../int_tuples
import ../layouts
import ../layout_constructors
import ../tensors
import ../ptr_arithmetic
import ../atoms_mma_partitioning
import ./tiles
import ./tile_config
import ./tile_ops_unary
import ./tile_ops_binary
import ./tile_ops_reductions
import ./tile_mma

export tiles, tile_ops_unary, tile_ops_binary,
       tile_ops_reductions, tile_mma
# Tile surface re-exported so the epilogue flow resolves at the call site.

# ═════════════════════════════════════════════════════════════════════════
#  Tile Epilogue concept
# ═════════════════════════════════════════════════════════════════════════

type Epilogue* = concept
  ## A tile epilogue computes the output tile D = f(AB)
  ## from the accumulated GEMM result.
  proc apply(op: Self, tmp: var (TensorView or Tensor), AB: TensorView or Tensor)
  proc apply(op: Self, tmp: var RtLeft, AB: RtLeft)

# ═════════════════════════════════════════════════════════════════════════
#  Operand captures: the tile-shaped gmem views
# ═════════════════════════════════════════════════════════════════════════

func biasView*(T: typedesc, R, C: static int,
               buf: ptr UncheckedArray[T]): TensorView[T, (Int[R], Int[C]), (Int[0], Int[1])] =
  make_view(buf, (R, C), (0, 1))

func cView*(T: typedesc, R, C: static int,
            buf: ptr UncheckedArray[T]): TensorView[T, (Int[R], Int[C]), (Int[C], Int[1])] =
  make_view(buf, (R, C), (C, 1))

# ═════════════════════════════════════════════════════════════════════════
#  EpiIdentity
# ═════════════════════════════════════════════════════════════════════════

type EpiIdentity* = object
  ## Identity: D = AB.

func apply*[T, Sh, StAB, StR](
    op: EpiIdentity,
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]),
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = AB, per element.
  const S = toIntVal(size(tmp))
  for i in 0 ..< S:
    tmp(i) = AB(i)

func apply*[T; R, C: static int; A: static MmaAtom](
    op: EpiIdentity,
    tmp: var RtLeft[T, R, C, A],
    AB: RtLeft[T, R, C, A]) {.inline.} =
  ## D = AB, per owned slot.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = toIntVal(A.valuesPerThread(opC))
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        tmp.frags[n][m].frag[v] = AB.frags[n][m].frag[v]

# ═════════════════════════════════════════════════════════════════════════
#  EpiReLU
# ═════════════════════════════════════════════════════════════════════════

type EpiReLU* = object
  ## Rectified linear unit: D = max(0, AB).

func apply*[T, Sh, StAB, StR](
    op: EpiReLU,
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]),
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = max(0, AB), per element.
  const S = toIntVal(size(tmp))
  for i in 0 ..< S:
    tmp(i) = max(AB(i), T(0))

func apply*[T; R, C: static int; A: static MmaAtom](
    op: EpiReLU,
    tmp: var RtLeft[T, R, C, A],
    AB: RtLeft[T, R, C, A]) {.inline.} =
  ## D = max(0, AB), per owned slot.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = toIntVal(A.valuesPerThread(opC))
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
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

func initEpiAXPBY*[T, Sh, StC](alpha: T, beta: T,
                               C: TensorView[T, Sh, StC]): EpiAXPBY[T, Sh, StC] =
  ## Gmem capture of C plus the scalars.
  EpiAXPBY[T, Sh, StC](alpha: alpha, beta: beta, C_gmem: C)

func apply*[T, Sh, StAB, StC, StR](
    op: EpiAXPBY[T, Sh, StC],
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]),
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = α·AB + β·C, per element.
  ## β = 0 skips reading C, saving memory bandwidth
  ## α = 1 skips the multiply.
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

func apply*[T; R, C: static int; A: static MmaAtom; Sh, StC](
    op: EpiAXPBY[T, Sh, StC],
    tmp: var RtLeft[T, R, C, A],
    AB: RtLeft[T, R, C, A]) {.inline.} =
  ## D = α·AB + β·C, per owned slot, the C view the per-lane shard.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = toIntVal(A.valuesPerThread(opC))
  if op.beta == T(0):
    if op.alpha == T(1):
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for v in 0 ..< vpt:
            tmp.frags[n][m].frag[v] = AB.frags[n][m].frag[v]
    else:
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for v in 0 ..< vpt:
            tmp.frags[n][m].frag[v] =
              op.alpha * AB.frags[n][m].frag[v]
  elif op.alpha == T(1):
    for n in 0 ..< rowTiles:
      for m in 0 ..< colTiles:
        for v in 0 ..< vpt:
          tmp.frags[n][m].frag[v] =
            AB.frags[n][m].frag[v] + op.beta * op.C_gmem[n, m, v]
  else:
    for n in 0 ..< rowTiles:
      for m in 0 ..< colTiles:
        for v in 0 ..< vpt:
          tmp.frags[n][m].frag[v] =
            op.alpha * AB.frags[n][m].frag[v] +
            op.beta * op.C_gmem[n, m, v]

# ═════════════════════════════════════════════════════════════════════════
#  Strided AXPBY: D = α·AB + β·C, C with runtime strides
# ═════════════════════════════════════════════════════════════════════════

type StridedOperand*[T] = object
  ## A gmem operand with runtime (BLIS) row/col strides. The shard
  ## fills `base` with the tile-origin + lane-cell offset.
  data*: ptr UncheckedArray[T]
  rsc*, csc*: int32
  base*: int32

type EpiAXPBYStrided*[T] = object
  ## D = α·AB + β·C, C addressed with runtime strides.
  alpha*, beta*: T
  C*: StridedOperand[T]

func initEpiAXPBY*[T](alpha, beta: T, C: ptr UncheckedArray[T],
                      rsc, csc: int32): EpiAXPBYStrided[T] =
  ## The runtime-strided form: C with explicit row/col strides (BLIS).
  EpiAXPBYStrided[T](alpha: alpha, beta: beta,
                     C: StridedOperand[T](data: C, rsc: rsc, csc: csc, base: 0))

func apply*[T; R, C: static int; A: static MmaAtom](
    op: EpiAXPBYStrided[T],
    tmp: var RtLeft[T, R, C, A],
    AB: RtLeft[T, R, C, A]) {.inline.} =
  ## D = α·AB + β·C, per owned slot. C is read at (row, col) with the
  ## runtime strides (rsc, csc); β = 0 skips the read.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = toIntVal(A.valuesPerThread(opC))
  if op.beta == T(0):
    for n in 0 ..< rowTiles:
      for m in 0 ..< colTiles:
        for v in 0 ..< vpt:
          tmp.frags[n][m].frag[v] = op.alpha * AB.frags[n][m].frag[v]
  else:
    for n in 0 ..< rowTiles:
      for m in 0 ..< colTiles:
        let cOff = op.C.base + int32(n * A.getM()) * op.C.rsc +
                                int32(m * A.getN()) * op.C.csc
        for v in 0 ..< vpt:
          tmp.frags[n][m].frag[v] =
            op.alpha * AB.frags[n][m].frag[v] +
            op.beta * op.C.data[int(cOff) + int(v) * int(op.C.csc)]

# ═════════════════════════════════════════════════════════════════════════
#  EpiAddBias
# ═════════════════════════════════════════════════════════════════════════

type EpiAddBias*[T, Sh, St] = object
  ## D = AB + bias, the bias a column vector broadcast over the tile rows via its stride-0 row view.
  bias_gmem*: TensorView[T, Sh, St]

func initEpiAddBias*[T, Sh, St](bias: TensorView[T, Sh, St]): EpiAddBias[T, Sh, St] {.inline.} =
  EpiAddBias[T, Sh, St](bias_gmem: bias)

func apply*[T, Sh, StAB, StB, StR](
    op: EpiAddBias[T, Sh, StB],
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]),
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = AB + bias, the bias a column vector broadcast over the tile rows.
  const S = toIntVal(size(tmp))
  for i in 0 ..< S:
    tmp(i) = AB(i) + op.bias_gmem(i)

func apply*[T; R, C: static int; A: static MmaAtom; Sh, StB](
    op: EpiAddBias[T, Sh, StB],
    tmp: var RtLeft[T, R, C, A],
    AB: RtLeft[T, R, C, A]) {.inline.} =
  ## D = AB + bias, per owned slot, the bias view the per-lane shard.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = toIntVal(A.valuesPerThread(opC))
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        tmp.frags[n][m].frag[v] =
          AB.frags[n][m].frag[v] + op.bias_gmem[n, m, v]

# ═════════════════════════════════════════════════════════════════════════
#  EpiLinearBiasReLU
# ═════════════════════════════════════════════════════════════════════════

type EpiLinearBiasReLU*[T, Sh, St] = object
  ## D = max(0, AB + bias), the bias a column vector broadcast over the tile rows.
  bias_gmem*: TensorView[T, Sh, St]

func initEpiLinearBiasReLU*[T, Sh, St](bias: TensorView[T, Sh, St]): EpiLinearBiasReLU[T, Sh, St] {.inline.} =
  ## Gmem capture of the bias.
  EpiLinearBiasReLU[T, Sh, St](bias_gmem: bias)

func apply*[T, Sh, StAB, StB, StR](
    op: EpiLinearBiasReLU[T, Sh, StB],
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]),
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = max(0, AB + bias), the bias a column vector broadcast over the tile rows.
  const S = toIntVal(size(tmp))
  for i in 0 ..< S:
    tmp(i) = max(AB(i) + op.bias_gmem(i), T(0))

func apply*[T; R, C: static int; A: static MmaAtom; Sh, StB](
    op: EpiLinearBiasReLU[T, Sh, StB],
    tmp: var RtLeft[T, R, C, A],
    AB: RtLeft[T, R, C, A]) {.inline.} =
  ## D = max(0, AB + bias), per owned slot, the bias view the per-lane shard.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = toIntVal(A.valuesPerThread(opC))
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        tmp.frags[n][m].frag[v] =
          max(AB.frags[n][m].frag[v] + op.bias_gmem[n, m, v], T(0))
