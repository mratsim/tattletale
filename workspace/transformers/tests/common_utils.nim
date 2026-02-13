# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/strformat,
  std/macros,
  workspace/libtorch as F

macro traceExec*(body: untyped): untyped =
  result = nnkStmtList.newTree()
  for statement in body:
    let stmtRepr = statement.repr
    let echoNode = nnkCall.newTree(
      ident"debugEcho",
      newLit("Will execute '" & stmtRepr & "'")
    )
    result.add(echoNode)
    result.add(statement)

template catchTorchExceptions*(body: bool): bool =
  ## Use this for debugging
  ## Error: unhandled exception: no exception to reraise [ReraiseDefect]

  when not defined(cpp) and defined(nimCheck):
    {.error: "You are running 'nim check' in C mode. It will misreport that C++ exceptions can't be caught because they aren't ref objects.".}

  try:
    body
  except TorchError as e: # nim check will misreport this. Ignore and test with `nim cpp` compilation
    echo "❌ C++ torch::Error caught:\n---------------------------\n", $e.what()
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    false
  except CatchableError as e:
    echo "❌ Exception caught:\n---------------------------\n", e.msg
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    false

proc runTest*(name: string, body: proc(): bool) =
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Section: " & name
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  let passed = catchTorchExceptions(body())
  if passed:
    echo "✅ PASS | ", name
  else:
    echo "❌ FAIL | ", name
  echo ""

proc assertAllClose*(
  actual, expected: TorchTensor,
  rtol = 1e-2, abstol = 1e-3,
  msg = ""
) =
  let allClose = F.allClose(actual, expected, rtol, abstol)
  if not allClose:
    echo "Assertion failed: allClose"
    if msg.len > 0:
      echo msg
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Actual[0, 0..<5, 0..<5]:\n", actual[0, 0..<5, 0..<5]
    echo "Expected[0, 0..<5, 0..<5]:\n", expected[0, 0..<5, 0..<5]
    raise newException(AssertionDefect, "allClose assertion failed")

template assertDefined*(tensor: untyped, name: string = "") =
  if not tensor.isDefined():
    raise newException(
      AssertionDefect,
      "Tensor '" & (if name.len > 0: name else: astToStr(tensor)) & "' is not defined"
    )
