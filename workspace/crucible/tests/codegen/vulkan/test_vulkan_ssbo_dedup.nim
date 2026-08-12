## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan: SSBO dedup by position (not by param name).
##
## When two kernels in the same module have pointer params with different
## names at the same position, the Nth parameter should always get
## binding N (matching `execVulkan`'s position-based descriptor binding).
## The SSBO variable name should be consistent so both kernels can
## reference the same buffer.

import std/strformat
import std/strutils
import workspace/crucible/src/codegen/gpu_compiler
const code = vulkan:
  proc kernel1(a: ptr UncheckedArray[uint32];
               output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + 1'u32
  proc kernel2(x: ptr UncheckedArray[uint32];
               y: ptr UncheckedArray[uint32]) {.global.} =
    y[0] = x[0] * 2'u32

# Check that SSBO bindings use position (0, 1) not param names (0,1,2,3)
# Both kernels should share binding 0 for the first param and binding 1 for the second.

# kernel1's `a` should be at binding 0
doAssert code.contains("binding = 0"), &"Expected binding = 0 in:\n{code}"
doAssert code.contains("binding = 1"), &"Expected binding = 1 in:\n{code}"

# There should be exactly 2 SSBO declarations, not 4
var ssboCount = 0
for line in code.split('\n'):
  if line.contains("buffer Buf") and line.contains("layout"):
    ssboCount += 1
doAssert ssboCount == 2, &"Expected exactly 2 SSBOs, got {ssboCount}\n{code}"

# kernel1 uses position-0 buffer (currently named `a`)
# kernel2 should have its `x` renamed to match the canonical name at position 0
doAssert code.contains("void kernel1()"), &"Missing kernel1 in:\n{code}"
doAssert code.contains("void kernel2()"), &"Missing kernel2 in:\n{code}"

echo code
echo "  OK — SSBO dedup by position"
