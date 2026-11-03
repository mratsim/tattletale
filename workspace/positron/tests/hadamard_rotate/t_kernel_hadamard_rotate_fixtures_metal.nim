# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run from the repository root:
##   nim c -r --threads:off -o:build/tests/t_hadamard_fixtures \
##     workspace/positron/tests/hadamard_rotate/t_kernel_hadamard_rotate_fixtures_metal.nim
##
## The recorded output_both frame is a recording artifact (it duplicates output_pre), the both variant checks against the host reference.

import std/[os, strformat, tables]
import workspace/crucible
import ./hadamard_rotate_template

type Fixture = object
  file: string
  M, C: int

const Fixtures = [
  Fixture(file: "hadamard_single_block.safetensor", M: 3, C: 128),
  Fixture(file: "hadamard_two_blocks.safetensor", M: 2, C: 256),
  Fixture(file: "hadamard_odd_blocks.safetensor", M: 4, C: 384),
  Fixture(file: "hadamard_eight_blocks.safetensor", M: 1, C: 1024),
  Fixture(file: "hadamard_batch2_eight_blocks.safetensor", M: 2, C: 1024),
]

proc checkFixture(engine: HwEngine, fx: Fixture) =
  let path = FixtureDir / fx.file
  let input = loadFixtureF16(path, "input", fx.M * fx.C)
  let suh = loadFixtureF16(path, "suh", fx.C)
  let svh = loadFixtureF16(path, "svh", fx.C)
  let frame = loadUniformStatsFrame(path & ".stats.json.zst")
  for v in ScaleVariant:
    let got = engine.launchHadamard(v, input, suh, svh, fx.M, fx.C, InvSqrt128)
    if v == varBoth:
      let want = referenceVariant(input, suh, svh, fx.M, fx.C, InvSqrt128, v)
      doAssert got == want, &"{fx.file} {v}: kernel differs from the host reference"
    else:
      let fp = statsFingerprint(got)
      fp.assertFingerprintMatches(frame[v.variantFrameName()],
        &"{fx.file} {v}")
  echo &"  {fx.file}: frames matched (none/pre/post) or host-reference exact (both)"

proc main =
  var engine = newEngine(runMetal)
  for fx in Fixtures:
    checkFixture(engine, fx)
  echo "T_HADAMARD_ROTATE_FIXTURES_METAL VERDICT: all fixture checks green"

main()
