# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Structural contracts of the register tile types:
##   - the per-element-type atom mapping (TileConfigFor)
##   - the per-backend fragment dispatch (FragmentOf)
##   - the subtile grid derived from the atom's mnk
##   - the LayoutLeft/LayoutRight storage nesting of rt_l/rt_r
##   - the divisibility guard
##   - the fm/fn lane forms
##   - the FMA register-op signatures
## Host-only, no GPU.
##
## Run: nim c -r --hints:off --warnings:off \
##   --outdir:build/tests/test_tiles --nimcache:nimcache/tests/test_tiles \
##   workspace/ceramic/tests/test_tiles.nim

import std/typetraits
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm/atoms_apple
import workspace/ceramic/src/kernel_gemm/atoms_universal
import workspace/ceramic/src/kernel_gemm/atoms_nvidia
import workspace/ceramic/src/tile_algebra/tile_config
import workspace/ceramic/src/tile_algebra/tiles
import workspace/ceramic/src/tile_algebra/tile_fma_partition
import workspace/crucible

const T1TL = ThreadLayout(thrM: 1, thrN: 1, thrK: 1)
  ## The one-thread layout: equal to the FMA config's default, spelled
  ## out so the explicit-layout assertions are readable.

# ── FragmentOf: the per-lane register fragment dispatch ─────────────────

static:
  # The FMA fragment is the per-lane value array: two slots per lane on
  # the 8×8×8 universal atoms (the lane's (fm, fn) and (fm, fn+1) cells),
  # like the Apple 8×8×8 atoms.
  doAssert FragmentOf[UNIVERSAL_FMA_F16, float16].frag is array[2, float16],
    "the FMA fragment holds valuesPerThread=2 values per lane"
  doAssert FragmentOf[UNIVERSAL_FMA_F32, float32].frag is array[2, float32],
    "the fp32 FMA fragment holds two values per lane"
  doAssert FragmentOf[UNIVERSAL_FMA_BF16, bfloat16].frag is array[2, bfloat16],
    "the bf16 FMA fragment holds two values per lane"
  doAssert FragmentOf[APPLE_8x8x8_F16, float16].frag is array[2, float16],
    "the Apple atom's fragment holds its 2 values per lane"

# ── the subtile grid and the storage nesting ─────────────────────────────

static:
  # rt_l: LayoutLeft (col-major). A 32×32 tile on the 8×8 FMA atom holds
  # a 4×4 grid of 8×8 subtiles.
  var d32: rt_l(float16, 32, 32, UNIVERSAL_FMA_F16)
  doAssert d32.frags.len == 4 and d32.frags[0].len == 4,
    "the 8×8 atom gives a 4×4 subtile grid on a 32×32 tile"
  var q16: rt_l(float16, 8, 16, UNIVERSAL_FMA_F16)
  doAssert q16.frags.len == 1 and q16.frags[0].len == 2,
    "the 8×16 tile holds a 1×2 subtile grid"
  # rt_r: LayoutRight (row-major). A 32×8 transposed-B tile holds a 1×4 grid, col subtiles outer.
  var kt: rt_r(float16, 32, 8, UNIVERSAL_FMA_F16)
  doAssert kt.frags.len == 1 and kt.frags[0].len == 4,
    "rt_r nests col subtiles outer"
  # The grid derives from the atom's mnk, not a hardcoded count.
  # The m16n8k16 NVIDIA atom gives a 2×2 grid on a 32×16 tile.
  var nv: rt_l(float16, 32, 16, SM80_16x8x16_F32F16F16F32_TN)
  doAssert nv.frags.len == 2 and nv.frags[0].len == 2,
    "the 16×8 NVIDIA atom must give a 2×2 grid on 32×16"
  # The thread layout is a defaulted type param: the default (1, 1, 1)
  # resolves at instantiation, an explicit layout overrides.
  var dflt: rt_l(float32, 32, 32, UNIVERSAL_FMA_F32)
  doAssert dflt.frags.len == 4,
    "the default thread layout must not change the subtile grid"
  var t1: rt_l(float32, 32, 32, UNIVERSAL_FMA_F32, T1TL)
  doAssert t1.frags.len == 4,
    "an explicit thread layout must not change the subtile grid"

# ── the dims guard ───────────────────────────────────────────────────────

static:
  # No dims guard: zero dims build an empty frags array (legal Nim) and
  # negative dims are rejected by the compiler's own array-size check.
  # Kernel tiles are sized as multiples of the atom.
  doAssert compiles(rt_l(float16, 32, 32, UNIVERSAL_FMA_F16)),
    "valid dims must compile"
  # The default atom resolves per backend inside a DSL block only: outside one,
  # getTileConfig trips on the missing backend (a designed failure).
  doAssert not compiles(rt_l(float16, 32, 32)),
    "the default atom must resolve only inside a DSL block"

# ── the FMA register-op signatures ───────────────────────────────────────

static:
  # zero fills the per-lane array. mma takes the D, A and B fragments
  # (device-only: it reads the lane id and gathers other lanes).
  var frag: FragmentOf[UNIVERSAL_FMA_F32, float32]
  doAssert compiles(zero(frag.frag)),
    "zero must take the per-lane value array"
  doAssert compiles(mma[UNIVERSAL_FMA_F32, float32, float32, float32](
    frag.frag, frag.frag, frag.frag)),
    "mma must take the D, A and B fragments"

# ── the element-type guard ───────────────────────────────────────────────

# The probe runs inside a DSL block, where the missing-backend tripwire is off.
# The error branch is the only failure source it can verify.
const typeGuardProbe = metal:
  proc guardProbe() {.global.} =
    static:
      doAssert compiles(getTileConfig(float32)),
        "the fp32 atom must resolve inside the DSL block"
      doAssert not compiles(getTileConfig(float64)),
        "getTileConfig must reject float64 with its error branch"

# ── the cross-lane reduction tree ────────────────────────────────────────

static:
  # The row-sum shuffle tree walks the lane's fragment-column bits, which
  # for the 8×8×8 atom are lanes 1 and 8 apart: deltas [1, 8], 2 steps,
  # and the leader mask clears bits 0 and 3 (the row group
  # {base, base+1, base+8, base+9}).
  let (deltas, steps, mask) = fmaTree[UNIVERSAL_FMA_F32, FmaThreadLayout]()
  doAssert steps == 2 and mask == 0b01001'u32 and deltas[0..1] == [1, 8],
    "the 8×8×8 tree walks the fragment-column bits (lanes 1 and 8 apart)"
  let (d2, s2, m2) = fmaTree[UNIVERSAL_FMA_F32, T1TL]()
  doAssert s2 == 2 and m2 == 0b01001'u32 and d2[0..1] == [1, 8],
    "the explicit (1, 1, 1) layout must give the same atom tree"

proc runRegisterOps() =
  # Host smoke of the register-op signatures: zero is host arithmetic,
  # mma is device-only (its shuffle gathers return garbage on the host),
  # so only its signature is compile-checked above.
  var acc: array[2, float32]
  zero(acc)
  doAssert acc[0] == 0.0'f32 and acc[1] == 0.0'f32,
    "zero must fill the per-lane elements with 0"

# ── fm/fn: the lane forms ────────────────────────────────────────────────

func laneBits(lane: int): (int, int, int, int, int) =
  ## lane = b0 + 2·b1 + 4·b2 + 8·b3 + 16·b4 (the atom's lane-bit decomposition).
  (lane and 1, (lane shr 1) and 1, (lane shr 2) and 1,
   (lane shr 3) and 1, (lane shr 4) and 1)

static:
  # fm/fn reproduce the documented lane forms on every lane:
  # fm = (qid and 4) + ((lane div 2) and 3), fn = (qid and 2)·2 + (lane and 1)·2
  # with qid = lane div 4. These are the load's per-lane offsets.
  # The universal atoms share the Apple AC layout, so the forms match.
  for lane in 0 ..< 32:
    let (b0, b1, b2, b3, b4) = laneBits(lane)
    let qid = b2 + 2 * b3 + 4 * b4
    doAssert laneFm[APPLE_8x8x8_F16](lane) == (qid and 4) + (b1 + 2 * b2),
      "fm drifted from the lane form"
    doAssert laneFn[APPLE_8x8x8_F16](lane) == (qid and 2) * 2 + b0 * 2,
      "fn drifted from the lane form"
    doAssert laneFm[UNIVERSAL_FMA_F16](lane) == (qid and 4) + (b1 + 2 * b2),
      "universal fm drifted from the Apple lane form"
    doAssert laneFn[UNIVERSAL_FMA_F16](lane) == (qid and 2) * 2 + b0 * 2,
      "universal fn drifted from the Apple lane form"

runRegisterOps()

echo "TILE FRAGMENT HOST PASS"
