# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile-shaped MMA atoms and the per-operand thread->value maps
## addressing register tiles.
##
## - `tiledMma`: the atom resolved for the backend, repeated over
##   (MT, NT, KT) tiles
## - `getMmaLayout`: one thread's register tile shape for one operand
## - `getTVCrdLayout`: the thread->value map over one tile, the tile
##   shape composed with the atom's thread map
##
## Must run inside a cuda:/metal:/opencl: block, the backend picks the atom.

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/ism_coord_strides
import workspace/ceramic/src/layout_algebra/layouts
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch

# ═══════════════════════════════════════════════════════════════
#  MMA atom resolution, tile descriptor
# ═══════════════════════════════════════════════════════════════

proc getMmaAtom*(TAcc, TA, TB: typedesc): static MmaAtom {.inline.} =
  ## Backend-appropriate MMA atom for the requested element types.
  ## Must run inside a cuda:/metal:/opencl: block, the backend picks the atom.
  static:
    doAssert TAcc is float32, "the picked atoms accumulate in fp32"
  when ccGetBackend() == ctMetal:
    static:
      doAssert TA is TB, "the Metal simdgroup atoms require equal operand types"
    when TA is float16: APPLE_8x8x8_F16
    elif TA is float32: APPLE_8x8x8_F32
    elif TA is bfloat16: APPLE_8x8x8_BF16
    else: {.error: "no Metal MMA atom for the requested element types".}
  else:
    when TAcc is float32 and TA is float16 and TB is float16:
      UNIVERSAL_1x1x1_F32F16F16F32
    elif TAcc is float32 and TA is float32 and TB is float32:
      UNIVERSAL_1x1x1_F32F32F32F32
    elif TAcc is float32 and TA is bfloat16 and TB is bfloat16:
      UNIVERSAL_1x1x1_F32BF16BF16F32
    else: {.error: "no MMA atom for the requested element types".}

type TiledMma*[TAcc, TA, TB; MT, NT, KT: static int] = object
  ## A tile-shaped MMA schedule: (MT, NT, KT) = repetitions of the atom's cell.
  atom*: MmaAtom

proc tiledMma*[TAcc, TA, TB](
      _: typedesc[TAcc], _: typedesc[TA], _: typedesc[TB],
      MT, NT, KT: static int
    ): TiledMma[TAcc, TA, TB, MT, NT, KT] {.inline.} =
  const atom = getMmaAtom(TAcc, TA, TB)
  when MT mod atom.getM() == 0 and NT mod atom.getN() == 0 and KT mod atom.getK() == 0:
    result = TiledMma[TAcc, TA, TB, MT, NT, KT](atom: atom)
  else:
    {.error: "the tile dimensions are not multiples of the resolved atom's cell dimensions".}

# ═══════════════════════════════════════════════════════════════
#  Tile schedule queries
# ═══════════════════════════════════════════════════════════════

template getThreadCount*(mma: static TiledMma): int =
  mma.atom.getThreadCount()

# ═══════════════════════════════════════════════════════════════
#  Per-operand register tiles and thread->value maps
# ═══════════════════════════════════════════════════════════════

template getMmaLayout*(mma: static TiledMma, op: static MmaOperand): untyped =
  ## Register tile shape of one thread for one operand:
  ## (values-per-thread, (outer repetitions, last-dimension repetitions))
  const
    vpt = mma.atom.valuesPerThread(op)
    cellOuter = (when op == opB: mma.atom.getN() else: mma.atom.getM())
    cellLast = (when op == opC: mma.atom.getN() else: mma.atom.getK())
    repOuter = (when op == opB: mma.NT else: mma.MT) div cellOuter
    repLast = (when op == opC: mma.NT else: mma.KT) div cellLast
  make_layout((vpt, (repOuter, repLast)))

template getMatmulThreadmap*(mma: static TiledMma, op: static MmaOperand): untyped =
  ## Thread map for one operand: each thread's register slot within the GEMM tile,
  ## (atom part, cell-repetition part).
  ##
  ## Contract:
  ## - the atom layout is pre-converted to tile-flat integer strides
  ##   (flat_tile o shard_id o atom_map, one static composition per atom)
  ##   before the cell strides are attached
  ## - the pre-conversion keeps the map injective: composing the raw
  ##   atom map lets two (thread, value) slots claim the same tile element
  const
    atomMap = mma.atom.getLayout(op)
    cellOuter = (when op == opB: mma.atom.getN() else: mma.atom.getM())
    cellLast = (when op == opC: mma.atom.getN() else: mma.atom.getK())
    repOuter = (when op == opB: mma.NT else: mma.MT) div cellOuter
    repLast = (when op == opC: mma.NT else: mma.KT) div cellLast
    tileOuter = cellOuter * repOuter
    atomTileFlat = compose(make_layout((cellOuter, cellLast), (1, tileOuter)),
                           atomMap)
  make_layout((atomTileFlat.shape, (repOuter, repLast)),
              (atomTileFlat.stride, (cellOuter, tileOuter * cellLast)))

# ═══════════════════════════════════════════════════════════════
#  The thread->value map over one tile
# ═══════════════════════════════════════════════════════════════

type ThreadValueMap*[Sh, St, TSh] = object
  ## Thread->value layout of one operand over one tile shape
  layout*: Layout[Sh, St]
  tileShape*: TSh

proc toThreadValueMap*[Sh, St, TSh](l: Layout[Sh, St], tileShape: TSh):
    ThreadValueMap[Sh, St, TSh] =
  ThreadValueMap[Sh, St, TSh](layout: l, tileShape: tileShape)

template getThreadPartitioning*(tv: ThreadValueMap): untyped =
  tv.layout.selectDimensions(0).selectDimensions(0).shape

template getTVCrdLayout*(mma: static TiledMma, op: static MmaOperand): untyped =
  const threadMap = mma.getMatmulThreadmap(op)
  const tileShape = ((when op == opB: mma.NT else: mma.MT),
                     (when op == opC: mma.NT else: mma.KT))
  toThreadValueMap(compose(make_identity_layout(tileShape), threadMap),
                   tileShape)

# Compile-time spec lock: the thread->value layouts for the APPLE_8x8x8_F32
# atom on a 32x32x16 GEMM tile must equal the pycute reference
# (experiments/e05_load_tile_oneshot_pycute.py, get_tv_crd_layout) byte-exact
# on shape and coordinate stride for all three operands.
#
# Contract: the atom part of the threadmap is pre-converted to tile-flat
# integer strides before the identity composition, which keeps the composed
# layout injective (two (thread, value) slots never claim the same tensor element).
const specMma = TiledMma[float32, float32, float32, 32, 32, 16](
  atom: APPLE_8x8x8_F32)
static:
  doAssert specMma.getTVCrdLayout(opA).layout ===
    ((((2, 2, 2, 2, 2), 2), (4, 2)),
     (((E(2, 1), E(1, 0), E(2, 0), E(4, 1), E(4, 0)), E(1, 1)),
      (E(8, 0), E(8, 1))))
  doAssert specMma.getTVCrdLayout(opB).layout ===
    ((((2, 2, 2, 2, 2), 2), (4, 2)),
     (((E(2, 0), E(1, 1), E(2, 1), E(4, 0), E(4, 1)), E(1, 0)),
      (E(8, 0), E(8, 1))))
  doAssert specMma.getTVCrdLayout(opC).layout ===
    ((((2, 2, 2, 2, 2), 2), (4, 4)),
     (((E(2, 1), E(1, 0), E(2, 0), E(4, 1), E(4, 0)), E(1, 1)),
      (E(8, 0), E(8, 1))))
