# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Fixture-frame reader for the transformer suites.
##
## Consumers import the check layer from harness/harness.nim, the device
## selection from harness/select_device.nim, and take only
## `zstdReadFixture` from this module.

import
  std/strutils,
  workspace/zstd/zstd_highlevel

proc zstdReadFixture*(fixturePath: string): string =
  ## JSON text of one fixture json sidecar, the single payload
  ## inside the `.json.zst` frame.
  ##
  ## Args:
  ## the frame path itself, or the sidecar stem without any container suffix
  ## - a missing, empty or corrupt frame raises,
  ## never a silent empty result
  let framePath =
    if fixturePath.endsWith(".json.zst"): fixturePath
    else: fixturePath & ".json.zst"
  let frame = readFile(framePath)
  if frame.len == 0:
    raise newException(IOError, "fixture frame holds no bytes: " & framePath)
  result = frame.zstdDecompress(string)
