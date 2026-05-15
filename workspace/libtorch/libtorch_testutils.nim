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
  std/strutils,
  std/macros,
  workspace/libtorch/src/tensors

from workspace/libtorch/src/raw_libtorch import TorchError, CppStdException, what

# =============================================================================
# C++ Exception Handling
# =============================================================================

template catchExceptions*(body: bool): bool =
  ## Catch exceptions from Tensor operations.
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
  let passed = catchExceptions(body())
  if passed:
    echo "✅ PASS | ", name
  else:
    echo "❌ FAIL | ", name
    quit(1)
  echo ""


# =============================================================================
# Tensor Assertions
# =============================================================================

template assertAllClose*(
          tensor, expectedTensor: Tensor,
          rtol = 2e-2'f64, abstol = 2e-2'f64,
          msg = "") =
  ## Assert that two tensors are close within tolerance.
  ## Returns false if they differ (for use in runTest).
  ##
  ## Args:
  ##   actual: The tensor produced by the test
  ##   expected: The expected tensor values
  ##   rtol: Relative tolerance (default: 2e-2)
  ##   abstol: Absolute tolerance (default: 2e-2)
  ##   msg: Optional error message

  # Ensure computation is done only once and side-effect are done only once:
  let actual = tensor
  let expected = expectedTensor

  let allClose = actual.allClose(expected, rtol, abstol)
  if not allClose:
    echo "Assertion failed: allClose"
    echo "  '", astToStr(tensor), "'"
    if msg.len > 0:
      echo msg
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    # Print only a 5-element snippet along each dimension to avoid flooding the terminal
    let showN = 5
    var actualSnippet = actual
    var expectedSnippet = expected
    for d in 0 ..< actual.dim():
      let sn = min(showN, actual.size(d))
      actualSnippet = actualSnippet.narrow(d, 0, sn)
      expectedSnippet = expectedSnippet.narrow(d, 0, sn)
    echo "Actual (shape: ", actual.shape, "):"
    actualSnippet.print()
    echo ""
    echo "Expected (shape: ", expected.shape, "):"
    expectedSnippet.print()
    raise newException(AssertionDefect, "[ttt] allClose assertion failed")

template assertShape*(tensor: untyped, expectedShape: openArray[int], msg = ""): bool =
  ## Assert that a tensor has the expected shape.
  ## Returns false if shape doesn't match (for use in runTest).
  ##
  ## Args:
  ##   tensor: The tensor to check
  ##   expectedShape: Expected dimensions
  ##   name: Optional name for error message

  # Ensure computation is done only once and side-effect are done only once:
  let actual = @tensor.shape
  let expected = @expectedShape

  if actual != expected:
    echo "Assertion failed: allClose"
    echo "  '", astToStr(tensor), "'"
    if msg.len > 0:
      echo msg
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Actual:   "; actual.print()
    echo "Expected: "; expected.print()
    raise newException(AssertionDefect, "[ttt] Shape assertion failed")

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
  echo "  Shape: ", t.shape, ", dtype: ", t.scalarType()
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