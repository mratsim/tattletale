# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Common utilities for tests involving TorchTensor and other libtorch FFI types.
##
## Provides:
## - C++ exception handling for tests
## - Test runner with formatted output
## - Tensor assertion helpers

import
  std/strformat,
  std/strutils,
  std/macros,
  workspace/libtorch/src/raw_libtorch as F

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
    echo "----------------------------"
    echo $e.what()
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    false
  except CppStdException as e:
    echo "❌ Raw C++ exception caught:"
    echo "----------------------------"
    echo $e.what()
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    false
  except CatchableError as e:
    echo "❌ Exception caught:"
    echo "----------------------------"
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
    echo "Actual:   "; actual.print()
    echo "Expected: "; expected.print()
    raise newException(AssertionDefect, "allClose assertion failed")

template assertShape*(tensor: untyped, expectedShape: openArray[int], name: string = "") =
  ## Assert that a tensor has the expected shape.
  ## Raises AssertionDefect if shape doesn't match.
  ##
  ## Args:
  ##   tensor: The tensor to check
  ##   expectedShape: Expected dimensions
  ##   name: Optional name for error message
  let actual = @tensor.shape.asNimView()
  let expected = @expectedShape
  if actual != expected:
    let tensorName = if name.len > 0: name else: astToStr(tensor)
    raise newException(
      AssertionDefect,
      "Tensor '" & tensorName & "' shape mismatch. Expected: " & $expected & ", Got: " & $actual
    )

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
  echo "  Shape: ", t.shape.asNimView(), ", Dtype: ", t.scalarType()
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

proc dataPtrHex*(tensor: F.TorchTensor): string =
  ## Get tensor data pointer as hex string.
  F.data_ptr(tensor).ptrHex()

proc shapePtrHex*(tensor: F.TorchTensor): string =
  ## Get tensor shape pointer as hex string.
  cast[pointer](tensor.shape.data()).ptrHex()