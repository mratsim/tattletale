# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#    Tile storage: the register tile types
#
# ############################################################
#
# The two register tile types, element-parametrized and backend-agnostic:
# `rt_l` (LayoutLeft, col-major) and `rt_r` (LayoutRight, row-major). The tile layer never names a backend type.
# Per-subtile storage is the tile_config `FragmentOf` type. Every hardware detail
# (the atom, the fragment, the register ops, the lane forms) comes from tile_config.

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../layout_indexing
import ../atoms
import ../tensors
import ../kernel_gemm/atoms_apple
import ../kernel_gemm/atoms_universal
import ./tile_config

export laneFm, laneFn, fragLaneCoeffs
## The lane forms are defined in tile_config (the mma reads them too);
## re-exported here so tile consumers keep one import surface.

# ═════════════════════════════════════════════════════════════════════════
#  The register tiles: rt_l (LayoutLeft) / rt_r (LayoutRight)
# ═════════════════════════════════════════════════════════════════════════
#
# An R×C register tile holds one fragment per atom subtile. The frags
# nesting is the tile's outer index:
#
#   rt_l (LayoutLeft, col-major): frags[n][m]   rt_r (LayoutRight, row-major): frags[m][n]
#   n over R div A.mnk.m                          m over C div A.mnk.n
#   m over C div A.mnk.n                          n over R div A.mnk.m
#
# fp16/bf16 tiles are the operand tiles, fp32 tiles the accumulators.
# Zero dims are a compile error. Kernel tiles are sized as multiples of
# the atom's M·N.

type
  RtLeft*[T; R, C: static int; A: static MmaAtom; TL: static ThreadLayout = FmaThreadLayout] = object
    ## LayoutLeft (col-major) register tile of an R×C matrix: `frags[n][m]` is the
    ## fragment of the subtile at row n, col m.
    ## fp32 tiles are the accumulators (the gemm D, the attention S and O)
    ## and the fp32 epilogue operands.
    frags*: array[R div A.mnk.m, array[C div A.mnk.n, FragmentOf[A, T]]]

  RtRight*[T; R, C: static int; A: static MmaAtom; TL: static ThreadLayout = FmaThreadLayout] = object
    ## LayoutRight (row-major) register tile of an R×C matrix: `frags[m][n]` is the
    ## fragment of the subtile at row n, col m.
    ## fp32 tiles are the accumulators (the gemm D, the attention S and O)
    ## and the fp32 epilogue operands.
    frags*: array[C div A.mnk.n, array[R div A.mnk.m, FragmentOf[A, T]]]

template rt_l*(T: typedesc; R, C: static int; A: untyped = getTileConfig(T);
               TL: static ThreadLayout = FmaThreadLayout): typedesc =
  ## LayoutLeft (col-major) register tile for an R×C matrix.
  RtLeft[T, R, C, A, TL]

template rt_r*(T: typedesc; R, C: static int; A: untyped = getTileConfig(T);
               TL: static ThreadLayout = FmaThreadLayout): typedesc =
  ## LayoutRight (row-major) register tile for an R×C matrix.
  RtRight[T, R, C, A, TL]

# ═════════════════════════════════════════════════════════════════════════
#  ColVecOf: the row-reduction col-vec
# ═════════════════════════════════════════════════════════════════════════

template ColVecOf*(T: typedesc; R, C: static int; A: untyped = getTileConfig(T)): typedesc =
  ## The col-vec of an R×C tile: the reduced value of each row,
  ## replicated across the row's slots.
  Tensor[T,
         (Int[R div A.mnk.m], Int[toIntVal(A.valuesPerThread(opA))]),
         (Int[toIntVal(A.valuesPerThread(opA))], Int[1])]

static:
  # The derivation must reproduce the atom's documented lane forms on every lane:
  # fm = (qid and 4) + ((lane div 2) and 3),  fn = (qid and 2)·2 + (lane and 1)·2,
  # with qid = lane div 4. The universal 8×8×8 atoms share the Apple AC
  # layout, so the same forms fix both.
  for lane in 0 ..< 32:
    doAssert laneFm[APPLE_8x8x8_F16](lane) == ((lane div 4) and 4) + ((lane div 2) and 3),
      "fm derivation drifted from the atom's lane form"
    doAssert laneFn[APPLE_8x8x8_F16](lane) == ((lane div 4) and 2) * 2 + (lane and 1) * 2,
      "fn derivation drifted from the atom's lane form"
    doAssert laneFm[UNIVERSAL_FMA_F16](lane) == ((lane div 4) and 4) + ((lane div 2) and 3),
      "universal fm drifted from the Apple lane form"
    doAssert laneFn[UNIVERSAL_FMA_F16](lane) == ((lane div 4) and 2) * 2 + (lane and 1) * 2,
      "universal fn drifted from the Apple lane form"
