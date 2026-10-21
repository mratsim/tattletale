# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties

# ############################################################
#
#            Matrix Multiply Accumulate (MMA)
#
# ############################################################

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
    atomTileFlat = compose(make_layout((cellOuter, cellLast), (1, tileOuter)), atomMap)
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
  tv.layout.shape[0][0]

template getThreadValueMap*(mma: static TiledMma, op: static MmaOperand): untyped =
  const threadMap = mma.getMatmulThreadmap(op)
  const tileShape = ((when op == opB: mma.NT else: mma.MT),
                     (when op == opC: mma.NT else: mma.KT))
  toThreadValueMap(compose(make_identity_layout(tileShape), threadMap),
                   tileShape)
