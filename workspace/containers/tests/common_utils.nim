#
#
#            Vec Container Test Utilities
#        (c) Copyright 2025 Tattletale contributors
#
#    See the file "copying.txt", included in this
#    distribution, for details about the copyright.
#

## Common utilities for Vec container tests.
## Provides C++ exception handling for tests involving TorchTensor and other FFI types.

import
  std/strformat,
  workspace/libtorch as F

template catchCppExceptions*(body: bool): bool =
  ## Catch C++ exceptions from TorchTensor operations.
  ## Returns true if body executed successfully, false if an exception was caught.
  ##
  ## Use this for debugging tests that involve C++ FFI types.
  
  when not defined(cpp) and defined(nimCheck):
    {.error: "You are running 'nim check' in C mode. It will misreport that C++ exceptions can't be caught because they aren't ref objects.".}

  try:
    body
  except TorchError as e:
    echo "❌ C++ torch::Error caught:"
    echo "---------------------------"
    echo $e.what()
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    false
  except CatchableError as e:
    echo "❌ Exception caught:"
    echo "---------------------------"
    echo e.msg
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    false

proc runTest*(name: string, body: proc(): bool) =
  ## Run a test with C++ exception handling.
  ## Prints PASS/FAIL status.
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Section: " & name
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  let passed = catchCppExceptions(body())
  if passed:
    echo "✅ PASS | ", name
  else:
    echo "❌ FAIL | ", name
  echo ""

proc assertAllClose*(
  actual, expected: F.TorchTensor,
  rtol = 2e-2'f64, abstol = 2e-2'f64,
  msg = ""
) =
  ## Assert that two tensors are close within tolerance.
  ## Raises AssertionDefect if they differ.
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
  ## Assert that a tensor is defined (initialized).
  if not tensor.isDefined():
    raise newException(
      AssertionDefect,
      "Tensor '" & (if name.len > 0: name else: astToStr(tensor)) & "' is not defined"
    )