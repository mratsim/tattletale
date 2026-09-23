## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run command, from the repo root:
## - nim test_ceramic, the tests/gemm scan compiles this suite
##
## CPU tests for gemm_kernel's ragged M/N derivation and the epilogues'
## store-mask-guarded gmem operand reads.
##
## The device path (gemm_kernel/gemm_cta) does not compile at HEAD,
## the atom partition chain loses static shapes and `compiles` guards
## cannot run here. That blocker is pre-existing and tracked separately.
##
## Runtime ragged validation for the tile GEMM family is the Metal suite tests/kernels_tiles/manual_tile_gemm_ragged_fp16.nim.
##
## Host coverage:
##   - the thread-layout derivation from M/N padded to atom multiples,
##     host-computed exactly as gemm_kernel's static block derives it
##   - EpiAXPBY, EpiAddBias and EpiLinearBiasReLU apply, an operand element
##     read only where the store mask's bit is set, masked-off lanes
##     on the zero fill, and the all-bits mask reading unguarded

import std/math
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/hardware/h_configgen
import workspace/ceramic/src/hardware/h_registry
import workspace/ceramic/src/hardware/h_properties
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/src/kernel_gemm_epilogues

{.experimental: "callOperator".}

template test(label: string; body: untyped) =
  block:
    body
  echo "  [OK] ", label

# ═════════════════════════════════════════════════════════════════════════
#  The padded thread-layout derivation
# ═════════════════════════════════════════════════════════════════════════

template paddedTo(extent, atom: static int): int =
  ## Smallest multiple of `atom` that covers `extent`.
  ((extent + atom - 1) div atom) * atom

const atom = atom_selector(uint32, uint32, float32)
const atomM = atom.getM()
const atomN = atom.getN()

static:
  doAssert atomM == 16 and atomN == 8,
    "the padded-atom derivation is built on the SM80 16x8x8 TF32 atom (" & $atomM & "x" & $atomN & ")"

# Worked examples locked independently of the formula, the padded extent
# is the next atom multiple at or above the problem extent and stays put
# for aligned extents. Each case proves the derivation at compile time, exactly as
# gemm_kernel's static block derives it, padded extents → thread
# layout → tile, the CTA grid ceil(problem/tile).
template paddedCase(probM, probN, expectM, expectN: static int) =
  const
    Mp = paddedTo(probM, atomM)
    Np = paddedTo(probN, atomN)
    tma = make_tiled_mma(atom, threadLayoutOf(atom, Mp, Np))
    (tileM, tileN, tileK) = tile_shape(tma, 32)
  static:
    doAssert Mp == expectM and Np == expectN,
      "padding (" & $probM & ", " & $probN & ") → (" & $Mp & ", " & $Np & "), expected (" &
      $expectM & ", " & $expectN & ")"
    # The padded extent covers the problem and stays minimal, one atom's
    # overhang at most, and the atom divides it exactly.
    doAssert Mp >= probM and Mp < probM + atomM and Mp mod atomM == 0
    doAssert Np >= probN and Np < probN + atomN and Np mod atomN == 0
    doAssert tileM == expectM and tileN == expectN and tileK == 32,
      "tile (" & $tileM & ", " & $tileN & ", " & $tileK & ")"
    doAssert (probM + tileM - 1) div tileM == 1 and (probN + tileN - 1) div tileN == 1,
      "one CTA per tile covers the padded extent"

static:
  paddedCase(37, 52, 48, 56)
  paddedCase(64, 64, 64, 64)
  paddedCase(17, 9, 32, 16)
  paddedCase(33, 25, 48, 32)

proc runDerivationTests =
  test "padded-to-atom-multiple derivation (static, host)":
    discard

# ═════════════════════════════════════════════════════════════════════════
#  Store-mask-guarded epilogue operand reads
# ═════════════════════════════════════════════════════════════════════════

const S = 4
const Shp = (Int[1](),)

func fragView(buf: var seq[float32]): auto =
  ## A 4-element fragment view, the per-thread C-fragment shape gemm_cta drives.
  make_view(buf, make_layout((S,), (1,)))

proc runEpilogueGuardTests =

  # EpiAXPBY, α=2, β=3, mask 0b0101, lanes 0 and 2 read C and lanes 1
  # and 3 take the zero fill. C is NaN-prefilled, any read past a set
  # bit would poison tmp.
  test "EpiAXPBY ragged mask guards the C read per bit (α=2, β=3)":
    var bufAB = newSeq[float32](S)
    var bufC = newSeq[float32](S)
    for i in 0 ..< S: bufAB[i] = float32(i + 1)
    for i in 0 ..< S: bufC[i] = NaN
    bufC[0] = 10.0'f32
    bufC[2] = 30.0'f32
    let AB = fragView(bufAB)
    let C = fragView(bufC)
    var op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    op.storeMask = 0b0101
    var tmp = make_tensor(float32, AB.layout.shape)
    op.apply(tmp, AB)
    for i in 0 ..< S:
      doAssert not tmp(i).isNaN, "lane " & $i & " read C against its mask bit"
    doAssert tmp(0) == 2.0'f32 * 1.0'f32 + 3.0'f32 * 10.0'f32
    doAssert tmp(1) == 2.0'f32 * 2.0'f32           # zero fill, C never read
    doAssert tmp(2) == 2.0'f32 * 3.0'f32 + 3.0'f32 * 30.0'f32
    doAssert tmp(3) == 2.0'f32 * 4.0'f32

  # Same op, α=1:
  #   the fused-multiply-add branch must honor the mask too.
  test "EpiAXPBY ragged mask guards the C read per bit (α=1 branch)":
    var bufAB = newSeq[float32](S)
    var bufC = newSeq[float32](S)
    for i in 0 ..< S: bufAB[i] = float32(i + 1)
    for i in 0 ..< S: bufC[i] = NaN
    bufC[1] = 100.0'f32
    let AB = fragView(bufAB)
    let C = fragView(bufC)
    var op = initEpiAXPBY(1.0'f32, 0.5'f32, C)
    op.storeMask = 0b0010
    var tmp = make_tensor(float32, AB.layout.shape)
    op.apply(tmp, AB)
    doAssert not tmp(0).isNaN and not tmp(2).isNaN and not tmp(3).isNaN
    doAssert tmp(0) == 1.0'f32
    doAssert tmp(1) == 2.0'f32 + 0.5'f32 * 100.0'f32
    doAssert tmp(2) == 3.0'f32
    doAssert tmp(3) == 4.0'f32

  # β=0 hoists the C read out entirely, ragged mask or not.
  test "EpiAXPBY β=0 never reads C, ragged mask included":
    var bufAB = newSeq[float32](S)
    var bufC = newSeq[float32](S)
    for i in 0 ..< S: bufAB[i] = float32(i + 1)
    for i in 0 ..< S: bufC[i] = NaN
    let AB = fragView(bufAB)
    let C = fragView(bufC)
    var op = initEpiAXPBY(2.0'f32, 0.0'f32, C)
    op.storeMask = 0b0101
    var tmp = make_tensor(float32, AB.layout.shape)
    op.apply(tmp, AB)
    for i in 0 ..< S:
      doAssert tmp(i) == 2.0'f32 * float32(i + 1)
      doAssert not tmp(i).isNaN

  # The all-bits mask is the full-tile fast path with unguarded reads,
  # and the result equals the ragged-path result with the same mask.
  test "EpiAXPBY full-tile fast path (all bits set) equals the ragged path":
    var bufAB = newSeq[float32](S)
    var bufC = newSeq[float32](S)
    for i in 0 ..< S: bufAB[i] = float32(i + 1)
    for i in 0 ..< S: bufC[i] = float32(10 * (i + 1))
    let AB = fragView(bufAB)
    let C = fragView(bufC)
    var tmpRagged = make_tensor(float32, AB.layout.shape)
    block:
      var op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
      op.storeMask = (1 shl S) - 1
      op.apply(tmpRagged, AB)
    var tmpFast = make_tensor(float32, AB.layout.shape)
    block:
      var op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
      op.storeMask = -1            # the constructor default, all bits
      op.apply(tmpFast, AB)
    for i in 0 ..< S:
      doAssert tmpRagged(i) == tmpFast(i)
      doAssert tmpFast(i) == 2.0'f32 * float32(i + 1) + 3.0'f32 * float32(10 * (i + 1))

  # EpiAddBias:
  #   the bias element is read only on set bits. The bias is a column
  # broadcast so lane i's source element is i's column, and the guard
  # decision is per flat element.
  test "EpiAddBias ragged mask guards the bias read per bit":
    var bufAB = newSeq[float32](S)
    var bufB = newSeq[float32](S)
    for i in 0 ..< S: bufAB[i] = float32(i + 1)
    for i in 0 ..< S: bufB[i] = NaN
    bufB[1] = 7.0'f32
    let AB = fragView(bufAB)
    let bias = fragView(bufB)
    var op = initEpiAddBias(bias)
    op.storeMask = 0b0010
    var tmp = make_tensor(float32, AB.layout.shape)
    op.apply(tmp, AB)
    doAssert not tmp(0).isNaN and not tmp(2).isNaN and not tmp(3).isNaN
    doAssert tmp(0) == 1.0'f32
    doAssert tmp(1) == 2.0'f32 + 7.0'f32
    doAssert tmp(2) == 3.0'f32
    doAssert tmp(3) == 4.0'f32

  test "EpiAddBias full-tile fast path (all bits set) reads unguarded":
    var bufAB = newSeq[float32](S)
    var bufB = newSeq[float32](S)
    for i in 0 ..< S: bufAB[i] = float32(i + 1)
    for i in 0 ..< S: bufB[i] = float32(i)
    let AB = fragView(bufAB)
    let bias = fragView(bufB)
    var op = initEpiAddBias(bias)
    op.storeMask = -1
    var tmp = make_tensor(float32, AB.layout.shape)
    op.apply(tmp, AB)
    for i in 0 ..< S:
      doAssert tmp(i) == float32(i + 1) + float32(i)

  # EpiLinearBiasReLU:
  #   ReLU clamps negative sums to 0, and a masked-off lane's NaN bias
  # must not reach the comparison.
  test "EpiLinearBiasReLU ragged mask guards the bias read per bit":
    var bufAB = newSeq[float32](S)
    var bufB = newSeq[float32](S)
    for i in 0 ..< S: bufAB[i] = float32(i + 1)
    for i in 0 ..< S: bufB[i] = NaN
    bufB[0] = -100.0'f32            # set bit, clamped by ReLU
    bufB[3] = 50.0'f32
    let AB = fragView(bufAB)
    let bias = fragView(bufB)
    var op = initEpiLinearBiasReLU(bias)
    op.storeMask = 0b1001
    var tmp = make_tensor(float32, AB.layout.shape)
    op.apply(tmp, AB)
    for i in 0 ..< S:
      doAssert not tmp(i).isNaN, "lane " & $i & " read the bias against its mask bit"
    doAssert tmp(0) == 0.0'f32      # max(1 - 100, 0)
    doAssert tmp(1) == 2.0'f32      # zero fill, C is never read
    doAssert tmp(2) == 3.0'f32
    doAssert tmp(3) == 4.0'f32 + 50.0'f32

  test "EpiLinearBiasReLU full-tile fast path (all bits set) reads unguarded":
    var bufAB = newSeq[float32](S)
    var bufB = newSeq[float32](S)
    for i in 0 ..< S: bufAB[i] = float32(i + 1)
    for i in 0 ..< S: bufB[i] = float32(-(i + 1))   # every sum negative → 0
    let AB = fragView(bufAB)
    let bias = fragView(bufB)
    var op = initEpiLinearBiasReLU(bias)
    op.storeMask = -1
    var tmp = make_tensor(float32, AB.layout.shape)
    op.apply(tmp, AB)
    for i in 0 ..< S:
      doAssert tmp(i) == 0.0'f32

proc runTest() =
  runDerivationTests()
  runEpilogueGuardTests()

when isMainModule:
  runTest()
