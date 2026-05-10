#
#
#            LibTorch Test Utilities
#        (c) Copyright 2025 Tattletale contributors
#
#    See the file "copying.txt", included in this
#    distribution, for details about the copyright.
#

## Common utilities for tests involving TorchTensor and other libtorch FFI types.
##
## Provides:
## - C++ exception handling for tests
## - Test runner with formatted output
## - Tensor assertion helpers

import
  std/strformat,
  std/macros,
  workspace/libtorch as F

# =============================================================================
# C++ Exception Handling
# =============================================================================

template catchCppExceptions*(body: bool): bool =
  ## Catch C++ exceptions from TorchTensor operations.
  ## Returns true if body executed successfully, false if an exception was caught.
  ##
  ## Use this for tests that involve C++ FFI types like TorchTensor.
  
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

# =============================================================================
# Test Runner
# =============================================================================

proc runTest*(name: string, body: proc(): bool) =
  ## Run a test with C++ exception handling.
  ## Prints PASS/FAIL status with formatted output.
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Section: " & name
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  let passed = catchCppExceptions(body())
  if passed:
    echo "✅ PASS | ", name
  else:
    echo "❌ FAIL | ", name
  echo ""

# =============================================================================
# Tensor Assertions
# =============================================================================

proc assertAllClose*(
  actual, expected: F.TorchTensor,
  rtol = 2e-2'f64, abstol = 2e-2'f64,
  msg = ""
) =
  ## Assert that two tensors are close within tolerance.
  ## Raises AssertionDefect if they differ.
  ##
  ## Args:
  ##   actual: The tensor produced by the test
  ##   expected: The expected tensor values
  ##   rtol: Relative tolerance (default: 2e-2)
  ##   abstol: Absolute tolerance (default: 2e-2)
  ##   msg: Optional error message
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
  ## Raises AssertionDefect if tensor is not defined.
  ##
  ## Args:
  ##   tensor: The tensor to check
  ##   name: Optional name for error message (defaults to variable name)
  if not tensor.isDefined():
    raise newException(
      AssertionDefect,
      "Tensor '" & (if name.len > 0: name else: astToStr(tensor)) & "' is not defined"
    )

template assertShape*(tensor: untyped, expectedShape: varargs[int], name: string = "") =
  ## Assert that a tensor has the expected shape.
  ## Raises AssertionDefect if shape doesn't match.
  ##
  ## Args:
  ##   tensor: The tensor to check
  ##   expectedShape: Expected dimensions
  ##   name: Optional name for error message
  let expected = @[expectedShape]
  let actual = @(tensor.shape)
  if actual != expected:
    let tensorName = if name.len > 0: name else: astToStr(tensor)
    raise newException(
      AssertionDefect,
      "Tensor '" & tensorName & "' shape mismatch. Expected: " & $expected & ", Got: " & $actual
    )

template assertDtype*(tensor: untyped, expectedDtype: F.ScalarKind, name: string = "") =
  ## Assert that a tensor has the expected dtype.
  ## Raises AssertionDefect if dtype doesn't match.
  let actual = tensor.scalarType()
  if actual != expectedDtype:
    let tensorName = if name.len > 0: name else: astToStr(tensor)
    raise newException(
      AssertionDefect,
      "Tensor '" & tensorName & "' dtype mismatch. Expected: " & $expectedDtype & ", Got: " & $actual
    )

template assertClose*(
  actual, expected: F.TorchTensor,
  rtol = 2e-2'f64, abstol = 2e-2'f64,
  msg = ""
) =
  ## Assert that two tensors are close within tolerance.
  ## Raises AssertionDefect if they differ.
  ## Alias for assertAllClose for consistency with other assert* templates.
  assertAllClose(actual, expected, rtol, abstol, msg)

# =============================================================================
# Debug Helpers
# =============================================================================

macro traceExec*(body: untyped): untyped =
  ## Debug macro to trace statement execution.
  ## Prints each statement before executing it.
  result = nnkStmtList.newTree()
  for statement in body:
    let stmtRepr = statement.repr
    let echoNode = nnkCall.newTree(
      ident"debugEcho",
      newLit("Will execute '" & stmtRepr & "'")
    )
    result.add(echoNode)
    result.add(statement)

proc printTensor*(t: F.TorchTensor, label: string = "") =
  ## Print a tensor with an optional label.
  ## Useful for debugging test failures.
  if label.len > 0:
    echo label, ":"
  t.print()
  echo ""

proc printTensorShape*(t: F.TorchTensor, label: string = "") =
  ## Print tensor shape and dtype with an optional label.
  if label.len > 0:
    echo label, ":"
  echo "  Shape: ", t.shape, ", Dtype: ", t.scalarType()
  echo ""