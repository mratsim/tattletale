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
  std/strutils,
  workspace/libtorch/src/raw_libtorch as F

# =============================================================================
# C++ Exception Handling
# =============================================================================

template catchExceptions*(body: bool): bool =
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
  let passed = catchExceptions(body())
  if passed:
    echo "✅ PASS | ", name
  else:
    echo "❌ FAIL | ", name
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