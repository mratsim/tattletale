## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            GEMM epilogues: user contract + shipped set
#
# ############################################################
#
# An epilogue computes D = f(AB): a user type capturing constants or
# TensorView operands, an `apply` proc (the per-epilogue math), the
# shared `finalStore` (the store) and a `storeMask` (the valid tile
# range, predicated per element).
#
# Lifecycle:
#   1. capture: the kernel builds the epilogue value with gmem operand
#      views (initEpiAddBias(bias_gmem), ...)
#   2. shard: per captured operand, gmem → per-thread storage
#      (the legacy `shard`/`preflight`)
#   3. apply: the per-thread f(AB) over the accumulator
#   4. store: `finalStore` writes the tile, masked by `storeMask`

import ./int_tuples
import ./layouts
import ./tensors
import ./atoms
import ./atoms_mma_partitioning

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  Epilogues for GEneralized Matrix Multiplication
# ═════════════════════════════════════════════════════════════════════════

type Epilogue* = concept
  ## An epilogue computes the output tile
  ##   D = f(AB)
  ## where
  ##   D   is the destination (the output tile in gmem, written per thread)
  ##   AB  is the accumulated GEMM result (the accumulator fragment)
  ##
  ## Function `f` is user-defined and can capture constants, vectors
  ## or tensors and compose operations between the accumulator AB and the captured input(s).
  ## Example operations:
  ## - scaling by a factor           D = α·AB
  ## - adding a broadcasted vector   D = AB + b
  ## - adding another tensor         D = AB + C
  ##
  ## Each epilogue implements `apply` (the per-epilogue math) and shares
  ## the contract's `finalStore` (the store). Each epilogue also carries
  ## a `storeMask` field controlling what is zeroed in D or copied into D.
  ##
  ## `shard` partitions the captured gmem operands onto threads via `partition_C`.
  ## `preflight` stages gmem → smem, a no-op here: the operands are read per-thread from gmem in `apply`.
  ##
  ## Concepts v2 match procs, not templates: `apply` is a `func {.inline.}` proc on every epilogue.
  proc apply(op: Self, tmp: var (TensorView or Tensor), AB: TensorView or Tensor)
  proc finalStore(op: Self, D: var (TensorView or Tensor), tmp: (TensorView or Tensor))

# ═════════════════════════════════════════════════════════════════════════
#  finalStore, shared by every op
# ═════════════════════════════════════════════════════════════════════════

func finalStore*[T, Sh, StR, StD](
    op: Epilogue;
    D: var (TensorView[T, Sh, StD] or Tensor[T, Sh, StD]);
    tmp: (TensorView[T, Sh, StR] or Tensor[T, Sh, StR])) {.inline.} =
  ## Copy f(AB) to the global memory destination tensor D. Shared: one implementation for every epilogue.
  ## storeMask describes the valid (M, N) range of the tile:
  ## full tile when all bits are set, predicated per element otherwise.

  const S = toIntVal(size(D.layout))

  if op.storeMask == (1 shl S) - 1:
    # Full tile
    for i in 0 ..< S:
      D(i) = tmp(i)
  else:
    # Ragged tile, only copy the valid coordinates
    for i in 0 ..< S:
      if ((op.storeMask shr i) and 1) != 0:
        D(i) = tmp(i)

# ═════════════════════════════════════════════════════════════════════════
#  EpiAXPBY: D = α·AB + β·C
# ═════════════════════════════════════════════════════════════════════════

type EpiAXPBY*[T, Sh, StC] = object
  ## Linear combination of the accumulator and the C operand.
  ## D = α·AB + β·C, element-wise over the tile's per-thread elements.
  alpha*, beta*: T
  C_gmem*: TensorView[T, Sh, StC]
  storeMask* = -1 # Store predication: describes the valid (M, N) range of the tile

func initEpiAXPBY*[T, Sh, StC](
    alpha: T;
    beta: T;
    C: TensorView[T, Sh, StC]): EpiAXPBY[T, Sh, StC] =
  ## Constructor: the scalars and the gmem capture of C.
  result = EpiAXPBY[T, Sh, StC](alpha: alpha, beta: beta, C_gmem: C)

func shard*[T, ShC, StC](
    op: EpiAXPBY[T, ShC, StC];
    tma: static TiledMma; thr: ThrSlice; mCTA, nCTA: int): auto =
  ## Partition the epilogue `C` operand onto threads (the legacy path
  ## gemm_cta drives).
  const tileM = tma.thrM * tma.atom.mnk.m
  const tileN = tma.thrN * tma.atom.mnk.n
  initEpiAXPBY(op.alpha, op.beta, tma.partition_C(thr, local_tile(op.C_gmem, (tileM, tileN), (mCTA, nCTA))))

template preflight*[T, Sh, StC](op: var EpiAXPBY[T, Sh, StC]): untyped =
  ## No-op. The legacy gemm_cta path reads C per-thread from gmem in `apply`.
  discard

func apply*[T, Sh, StAB, StC, StR](
    op: EpiAXPBY[T, Sh, StC];
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = α·AB + β·C, element-wise over the tile's (M, N) shape.

  # Dispatch hoisted out of the loop:
  #   β == 0 → C is never read, this saves memory bandwidth
  #   α == 1 → the α multiply is skipped
  # On GPU the branches are uniform
  #   α/β are identical for every thread
  #   no warp divergence, the cost is on instruction cache / code size
  let S = size(tmp.layout)
  if op.beta == T(0):
    if op.alpha == T(1):
      for i in 0 ..< S:
        tmp(i) = AB(i)
    else:
      for i in 0 ..< S:
        tmp(i) = op.alpha * AB(i)
  elif op.alpha == T(1):
    for i in 0 ..< S:
      # A single FMA per element
      tmp(i) = AB(i) + op.beta * op.C_gmem(i)
  else:
    for i in 0 ..< S:
      tmp(i) = op.alpha * AB(i) + op.beta * op.C_gmem(i)

# ═════════════════════════════════════════════════════════════════════════
#  EpiIdentity: D = AB (α=1, β=0, compile-time constants)
# ═════════════════════════════════════════════════════════════════════════

type EpiIdentity* = object
  ## Identity epilogue: D = AB.
  storeMask* = -1  # Store predication: describes the valid (M, N) range of the tile

func shard*(op: EpiIdentity; tma: static TiledMma; thr: ThrSlice; mCTA, nCTA: int): EpiIdentity {.inline.} =
  ## No-op (EpiIdentity has no operands): the legacy gemm_cta path.
  op

template preflight*(op: var EpiIdentity): untyped =
  ## No-op. The legacy gemm_cta path. EpiIdentity has no operands.
  discard

func apply*[T, Sh, StAB, StR](
    op: EpiIdentity;
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = AB, element-wise copy over the tile's (M, N) shape.
  for i in 0 ..< size(tmp.layout):
    tmp(i) = AB(i)

# ═════════════════════════════════════════════════════════════════════════
#  EpiAddBias: D = AB + bias (column broadcast)
# ═════════════════════════════════════════════════════════════════════════

type EpiAddBias*[T, Sh, St] = object
  ## D = AB + bias.
  ## Bias is a column vector broadcasted onto AB.
  bias_gmem*: TensorView[T, Sh, St]
  storeMask* = -1 # Store predication: describes the valid (M, N) range of the tile

func initEpiAddBias*[T, Sh, St](bias: TensorView[T, Sh, St]): EpiAddBias[T, Sh, St] {.inline.} =
  ## Constructor: the gmem capture of the bias.
  EpiAddBias[T, Sh, St](bias_gmem: bias)

template shard*[T, Sh, St](
    op: EpiAddBias[T, Sh, St];
    tma: static TiledMma; thr: ThrSlice; mCTA, nCTA: int): auto =
  ## Partition the epilogue `bias` operand onto threads (the legacy path gemm_cta drives).
  const tileM = tma.thrM * tma.atom.mnk.m
  const tileN = tma.thrN * tma.atom.mnk.n
  initEpiAddBias(tma.partition_C(thr, local_tile(op.bias_gmem, (tileM, tileN), (mCTA, nCTA))))

template preflight*[T, Sh, St](op: var EpiAddBias[T, Sh, St]): untyped =
  ## No-op. The legacy gemm_cta path reads bias per-thread from gmem in `apply`.
  discard

func apply*[T, Sh, StAB, StB, StR](
    op: EpiAddBias[T, Sh, StB];
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = AB + bias, with bias a column vector broadcasted onto AB
  for i in 0 ..< size(tmp.layout):
    tmp(i) = AB(i) + op.bias_gmem(i)

# ═════════════════════════════════════════════════════════════════════════
#  EpiLinearBiasReLU: D = max(0, AB + bias) (column broadcast)
# ═════════════════════════════════════════════════════════════════════════

type EpiLinearBiasReLU*[T, Sh, St] = object
  ## The fused epilogue D = max(0, AB + bias), bias a column vector
  ## broadcasted onto AB (the same stride-0-row view as EpiAddBias,
  ## with the clamp added).
  bias_gmem*: TensorView[T, Sh, St]
  storeMask* = -1 # Store predication: describes the valid (M, N) range of the tile

func initEpiLinearBiasReLU*[T, Sh, St](bias: TensorView[T, Sh, St]): EpiLinearBiasReLU[T, Sh, St] {.inline.} =
  ## Constructor: the gmem capture of the bias.
  EpiLinearBiasReLU[T, Sh, St](bias_gmem: bias)

template shard*[T, Sh, St](
    op: EpiLinearBiasReLU[T, Sh, St];
    tma: static TiledMma; thr: ThrSlice; mCTA, nCTA: int): auto =
  ## Partition the epilogue `bias` operand onto threads (the legacy path gemm_cta drives).
  const tileM = tma.thrM * tma.atom.mnk.m
  const tileN = tma.thrN * tma.atom.mnk.n
  initEpiLinearBiasReLU(tma.partition_C(thr, local_tile(op.bias_gmem, (tileM, tileN), (mCTA, nCTA))))

template preflight*[T, Sh, St](op: var EpiLinearBiasReLU[T, Sh, St]): untyped =
  ## No-op. The legacy gemm_cta path reads bias per-thread from gmem in `apply`.
  discard

func apply*[T, Sh, StAB, StB, StR](
    op: EpiLinearBiasReLU[T, Sh, StB];
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = max(0, AB + bias), with bias a column vector broadcasted onto AB.
  for i in 0 ..< size(tmp.layout):
    tmp(i) = max(AB(i) + op.bias_gmem(i), T(0))

# ═════════════════════════════════════════════════════════════════════════
#  EpiReLU: D = max(0, AB)
# ═════════════════════════════════════════════════════════════════════════

type EpiReLU* = object
  ## Rectified linear unit: D = max(0, AB)
  storeMask* = -1 # Store predication: describes the valid (M, N) range of the tile


template shard*(op: EpiReLU; tma: static TiledMma; thr: ThrSlice; mCTA, nCTA: int): auto =
  ## No-op (EpiReLU has no operands): the legacy gemm_cta path.
  op

template preflight*(op: var EpiReLU): untyped =
  ## No-op. The legacy gemm_cta path. EpiReLU has no operands to stage.
  discard

func apply*[T, Sh, StAB, StR](
    op: EpiReLU;
    tmp: var (TensorView[T, Sh, StR] or Tensor[T, Sh, StR]);
    AB: TensorView[T, Sh, StAB] or Tensor[T, Sh, StAB]) {.inline.} =
  ## D = max(0, AB), element-wise over the tile's (M, N) shape.
  for i in 0 ..< size(tmp.layout):
    tmp(i) = max(AB(i), T(0))
