# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Common utilities for tests involving Tensor and other libtorch FFI types.
##
## Provides:
## - C++ exception handling for tests
## - Test runner with formatted output
## - Tensor assertion helpers

import
  std/strformat,
  std/strutils,
  std/macros,
  workspace/libtorch/src/tensors

from workspace/libtorch/src/raw_libtorch import TorchError, CppStdException, what

# =============================================================================
# C++ Exception Handling
# =============================================================================

template catchCppExceptions*(body: bool): bool =
  ## Catch C++ exceptions from Tensor operations.
  ## Returns true if body executed successfully, false if an exception was caught.
  ##
  ## Use this for tests that involve C++ FFI types like Tensor.

  when not defined(cpp) and defined(nimCheck):
    {.error: "You are running 'nim check' in C mode. It will misreport that C++ exceptions can't be caught because they aren't ref objects.".}

  try:
    body
  except TorchError as e:
    echo "❌ C++ torch::Error caught (this shouldn't happen, they should be wrapped in Nim exceptions)"
    echo "---------------------------"
    echo $e.what()
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    false
  except CppStdException as e:
    echo "❌ Raw C++ exception caught (this shouldn't happen, they should be wrapped in Nim exceptions)"
    echo "---------------------------"
    echo $e.what()
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    false
  except LibTorchDefect as e:
    # LibTorchDefect already has pretty printing by default
    echo e.msg
    false
  except CatchableError as e:
    echo "❌ Exception caught:"
    echo "---------------------------"
    echo e.msg
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    false
  except Defect as e:
    echo "❌ Defect caught:"
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
  ## Exits with code 1 on first failure.
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Section: " & name
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  let passed = catchCppExceptions(body())
  if passed:
    echo "✅ PASS | ", name
  else:
    echo "❌ FAIL | ", name
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ FAILED: ", name
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    quit(1)
  echo ""


# =============================================================================
# Tensor Assertions
# =============================================================================

proc assertAllClose*(
  actual, expected: Tensor,
  rtol = 2e-2'f64, abstol = 2e-2'f64,
  msg = ""): bool =
  ## Assert that two tensors are close within tolerance.
  ## Returns false if they differ (for use in runTest).
  ##
  ## Args:
  ##   actual: The tensor produced by the test
  ##   expected: The expected tensor values
  ##   rtol: Relative tolerance (default: 2e-2)
  ##   abstol: Absolute tolerance (default: 2e-2)
  ##   msg: Optional error message
  let allClose = actual.allClose(expected, rtol, abstol)
  if not allClose:
    echo "Assertion failed: allClose"
    if msg.len > 0:
      echo msg
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Actual:   "; actual.print()
    echo "Expected: "; expected.print()
    return false
  true

template assertDefined*(tensor: untyped, name: string = ""): bool =
  ## Assert that a tensor is defined (initialized).
  ## Returns false if tensor is not defined (for use in runTest).
  ##
  ## Args:
  ##   tensor: The tensor to check
  ##   name: Optional name for error message (defaults to variable name)
  if not tensor.isDefined():
    echo "Tensor '" & (if name.len > 0: name else: astToStr(tensor)) & "' is not defined"
    false
  else:
    true

template assertShape*(tensor: untyped, expectedShape: openArray[int], name: string = ""): bool =
  ## Assert that a tensor has the expected shape.
  ## Returns false if shape doesn't match (for use in runTest).
  ##
  ## Args:
  ##   tensor: The tensor to check
  ##   expectedShape: Expected dimensions
  ##   name: Optional name for error message
  let actual = @tensor.shape.asNimView()
  let expected = @expectedShape
  if actual != expected:
    let tensorName = if name.len > 0: name else: astToStr(tensor)
    echo "Tensor '" & tensorName & "' shape mismatch. Expected: " & $expected & ", Got: " & $actual
    false
  else:
    true

template assertDtype*(tensor: untyped, expectedDtype: ScalarKind, name: string = ""): bool =
  ## Assert that a tensor has the expected dtype.
  ## Returns false if dtype doesn't match (for use in runTest).
  let actual = tensor.scalarType()
  if actual != expectedDtype:
    let tensorName = if name.len > 0: name else: astToStr(tensor)
    echo "Tensor '" & tensorName & "' dtype mismatch. Expected: " & $expectedDtype & ", Got: " & $actual
    false
  else:
    true

template assertClose*(
  actual, expected: Tensor,
  rtol = 2e-2'f64, abstol = 2e-2'f64,
  msg = ""): bool =
  ## Assert that two tensors are close within tolerance.
  ## Returns false if they differ (for use in runTest).
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

proc printTensor*(t: Tensor, label: string = "") =
  ## Print a tensor with an optional label.
  ## Useful for debugging test failures.
  if label.len > 0:
    echo label, ":"
  t.print()
  echo ""

proc printTensorShape*(t: Tensor, label: string = "") =
  ## Print tensor shape and dtype with an optional label.
  if label.len > 0:
    echo label, ":"
  echo "  Shape: ", t.shape, ", Dtype: ", t.scalarType()
  echo ""

# =============================================================================
# Pointer Debug Helpers
# =============================================================================

proc ptrHex*(p: pointer): string =
  ## Convert a pointer address to hex string for debugging.
  ## Useful for detecting memory aliasing issues.
  ##
  ## Example:
  ##   echo "tensor.data_ptr() = 0x", tensor.data_ptr().ptrHex()
  toHex(cast[uint](p))

proc dataPtrHex*(tensor: Tensor): string =
  ## Get tensor data pointer as hex string.
  tensor.data_ptr().ptrHex()

proc shapePtrHex*(tensor: Tensor): string =
  ## Get tensor shape pointer as hex string.
  tensor.shape[0].unsafeAddr.ptrHex()