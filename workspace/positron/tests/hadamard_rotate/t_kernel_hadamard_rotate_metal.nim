# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run from the repository root:
##   nim c -r --threads:off -o:build/tests/t_hadamard_metal \
##     workspace/positron/tests/hadamard_rotate/t_kernel_hadamard_rotate_metal.nim

import std/[random, strformat]
import workspace/crucible
import ./hadamard_rotate_template

proc checkShape(engine: HwEngine, rng: var Rand, M, C: int, label: string, gridLines = 0, gridBlocks = 0) =
  ## One shape x all variants, bit-exact against the reference.
  ## gridLines / gridBlocks override the natural launch grid, 0 keeps it.
  let nElems = M * C
  var srcBits = newSeq[uint16](nElems)
  var preBits = newSeq[uint16](C)
  var postBits = newSeq[uint16](C)
  for i in 0 ..< nElems:
    srcBits[i] = fp32ToFp16(rng.rand(-2.0'f32 .. 2.0'f32))
  for c in 0 ..< C:
    preBits[c] = fp32ToFp16(rng.rand(0.25'f32 .. 1.75'f32))
    postBits[c] = fp32ToFp16(rng.rand(0.25'f32 .. 1.75'f32))
  for v in ScaleVariant:
    let
      lines = if gridLines == 0: M else: gridLines
      blocks = if gridBlocks == 0: C div 128 else: gridBlocks
      got = engine.launchHadamardGrid(v, srcBits, preBits, postBits, M, C, InvSqrt128,
        lines, blocks)
    let want = referenceVariant(srcBits, preBits, postBits, M, C, InvSqrt128, v)
    var bad = 0
    for i in 0 ..< nElems:
      if got[i] != want[i]:
        inc bad
    doAssert bad == 0,
      &"{label} {v}: {bad}/{nElems} elements differ from the reference"

proc main =
  var engine = newEngine(runMetal)
  var rng = initRand(0xC17D19'i64)
  checkShape(engine, rng, 3, 256, "two blocks")
  checkShape(engine, rng, 1, 128, "single block")
  checkShape(engine, rng, 7, 640, "five blocks")
  checkShape(engine, rng, 4, 384, "three blocks")
  checkShape(engine, rng, 33, 1024, "eight blocks tail")
  checkShape(engine, rng, 7, 640, "grid-stride covers the work from (1, 1)",
    gridLines = 1, gridBlocks = 1)
  echo "T_HADAMARD_ROTATE_METAL VERDICT: bit-exact vs reference on all shapes and variants"

main()
