# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test harness for the transformer suites: tolerance budgets, fixture
## reading, analytic invariants, provenance stamps, device selection,
## and selftest.
##
## SPEC.md states the check semantics, PLAYBOOK.md the new-suite checklist.

import
  std/strutils,
  harness/tolerance,
  workspace/zstd/zstd_highlevel,
  harness/invariants,
  harness/provenance,
  harness/device,
  harness/selftest

export zstd_highlevel
export harness.tolerance, harness.invariants,
  harness.provenance, harness.device, harness.selftest

proc zstdReadFixture*(fixturePath: string): string =
  ## JSON text of one fixture json sidecar, the single payload of
  ## the `.json.zst` frame. The argument resolves in two forms:
  ## the frame path itself, or the sidecar stem without any
  ## container suffix. A missing, empty or corrupt frame raises
  ## an error, never a silent empty result.
  let framePath =
    if fixturePath.endsWith(".json.zst"): fixturePath
    else: fixturePath & ".json.zst"
  let frame = readFile(framePath)
  if frame.len == 0:
    raise newException(IOError, "fixture frame holds no bytes: " & framePath)
  result = frame.zstdDecompress(string)
