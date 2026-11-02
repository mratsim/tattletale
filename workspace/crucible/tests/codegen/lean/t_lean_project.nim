## Locked Lean project checks, as tests
##
## Test that lake build stays green, ToolchainSmoke elaborates and Negative.lean fails its rfl proof
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[os, osproc, strutils]
import lean_testutils

proc runTest() =
  block: # locked project checks: build green, toolchain smoke green, negative control rejected
    # a warm lake cache replays silently, the clean build is what proves the roots compile
    removeDir(leanProjectDir / ".lake")
    let (buildOut, buildCode) = execCmdEx("lake build", workingDir = leanProjectDir)
    doAssert buildCode == 0, "lake build must stay green:\n" & buildOut
    doAssert fileExists(leanProjectDir / ".lake/build/lib/lean/AddKernel.olean"),
      "the clean build must produce the AddKernel olean, without it the " &
      "negative control proves nothing"

    let (smokeOut, smokeCode) = elaborate("ToolchainSmoke.lean")
    doAssert smokeCode == 0, "ToolchainSmoke.lean must stay green:\n" & smokeOut

    let (negOut, negCode) = elaborate("Negative.lean")
    doAssert negCode != 0, "Negative.lean must stay rejected (a false claim " &
      "proved!):\n" & negOut
    # rejection identity: an unbuilt AddKernel fails as unknown module prefix,
    # only the built module can fail the rfl proof
    let rejectedByRfl = "Tactic `rfl` failed" in negOut and
      "is not definitionally equal" in negOut
    doAssert rejectedByRfl,
      "the rejection must be the rfl failure on the false claim, not a " &
      "module-resolution or build error:\n" & negOut
    let noSorry = "declaration uses 'sorry'" notin negOut
    doAssert noSorry,
      "the negative control must be rejected by a failed proof, not a sorry"
    echo "  locked project gates (lake build, ToolchainSmoke, Negative.lean): ok"

when isMainModule:
  runTest()
  echo "t_lean_project: all checks passed"
