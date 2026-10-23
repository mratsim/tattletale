# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Locked Lean project checks, as tests
##
## Test that the three README checks hold, so a future edit to the locked
## project cannot silently break them:
##
## - `lake build` stays green (ToolchainSmoke is a lake root)
## - ToolchainSmoke.lean elaborates under warningAsError
## - Negative.lean stays rejected by a failed proof, no sorry in the output
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[os, osproc, strutils]
import lean_testutils

proc runTest() =
  block: # the locked project's checks: build green, toolchain smoke green, negative control rejected
    let (buildOut, buildCode) = execCmdEx("lake build", workingDir = leanProjectDir)
    doAssert buildCode == 0, "lake build must stay green:\n" & buildOut

    let (smokeOut, smokeCode) = elaborate("ToolchainSmoke.lean")
    doAssert smokeCode == 0, "ToolchainSmoke.lean must stay green:\n" & smokeOut

    let (negOut, negCode) = elaborate("Negative.lean")
    doAssert negCode != 0, "Negative.lean must stay rejected (a false claim " &
      "proved!):\n" & negOut
    doAssert "declaration uses 'sorry'" notin negOut,
      "the negative control must be rejected by a failed proof, not a sorry"
    echo "  locked project gates (lake build, ToolchainSmoke, Negative.lean): ok"

when isMainModule:
  runTest()
  echo "t_lean_project: all checks passed"
