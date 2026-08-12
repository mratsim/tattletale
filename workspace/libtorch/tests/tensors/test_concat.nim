# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests for cat/stack - adapted from bug_test_concat.nim

import
  workspace/libtorch/src/tensors,
  workspace/libtorch/libtorch_testutils

proc main() =
  echo "=== Test Suite: torch::cat ==="
  echo ""

  # =============================================================================
  # Test 1: cat with varargs
  # =============================================================================
  runCppTest "cat with varargs":
    proc(): bool =
      let a = randn(2, 3, kFloat32)
      let b = randn(2, 3, kFloat32)
      let c = cat(a, b, axis = 0)
      doAssert c.dim() == 2
      doAssert c.size(0) == 4
      doAssert c.size(1) == 3
      true

  # =============================================================================
  # Test 2: cat with lvalue array + lvalue ArrayRef[Tensor]
  # =============================================================================
  runCppTest "cat with lvalue array + lvalue ArrayRef":
    proc(): bool =
      let a = randn(2, 3, kFloat32)
      let b = randn(2, 3, kFloat32)
      let ab = [a, b]
      let c = cat(ab, axis = 0)
      doAssert c.dim() == 2
      doAssert c.size(0) == 4
      doAssert c.size(1) == 3
      true

  # =============================================================================
  # Test 3: cat with lvalue array + rvalue ArrayRef[Tensor]
  # =============================================================================
  runCppTest "cat with lvalue array + rvalue ArrayRef":
    proc(): bool =
      let a = randn(2, 3, kFloat32)
      let b = randn(2, 3, kFloat32)
      let ab = [a, b]
      let c = cat(ab, axis = 0)
      doAssert c.dim() == 2
      doAssert c.size(0) == 4
      doAssert c.size(1) == 3
      true

  # =============================================================================
  # Test 4: cat with rvalue array + rvalue ArrayRef[Tensor]
  # =============================================================================
  runCppTest "cat with rvalue array + rvalue ArrayRef":
    proc(): bool =
      let a = randn(2, 3, kFloat32)
      let b = randn(2, 3, kFloat32)
      let c = cat([a, b], axis = 0)
      doAssert c.dim() == 2
      doAssert c.size(0) == 4
      doAssert c.size(1) == 3
      true

  # =============================================================================
  # Test 5: cat with rvalue array + [sugar] implicit conversion ArrayRef[Tensor]
  # =============================================================================
  runCppTest "cat with rvalue array + implicit conversion":
    proc(): bool =
      let a = randn(2, 3, kFloat32)
      let b = randn(2, 3, kFloat32)
      let c = cat([a, b], axis = 0)
      doAssert c.dim() == 2
      doAssert c.size(0) == 4
      doAssert c.size(1) == 3
      true

  # =============================================================================
  # Test 6: cat with lvalue seq + lvalue ArrayRef[Tensor]
  # =============================================================================
  runCppTest "cat with lvalue seq + lvalue ArrayRef":
    proc(): bool =
      let a = randn(2, 3, kFloat32)
      let b = randn(2, 3, kFloat32)
      var tensors = @[a, b]
      let c = cat(tensors, axis = 0)
      doAssert c.dim() == 2
      doAssert c.size(0) == 4
      doAssert c.size(1) == 3
      true

  # =============================================================================
  # Test 7: cat with lvalue seq + rvalue ArrayRef[Tensor]
  # =============================================================================
  runCppTest "cat with lvalue seq + rvalue ArrayRef":
    proc(): bool =
      let a = randn(2, 3, kFloat32)
      let b = randn(2, 3, kFloat32)
      var tensors = @[a, b]
      let c = cat(tensors, axis = 0)
      doAssert c.dim() == 2
      doAssert c.size(0) == 4
      doAssert c.size(1) == 3
      true

  # =============================================================================
  # Test 8: cat with rvalue seq + rvalue ArrayRef[Tensor]
  # =============================================================================
  runCppTest "cat with rvalue seq + rvalue ArrayRef":
    proc(): bool =
      let a = randn(2, 3, kFloat32)
      let b = randn(2, 3, kFloat32)
      let c = cat(@[a, b], axis = 0)
      doAssert c.dim() == 2
      doAssert c.size(0) == 4
      doAssert c.size(1) == 3
      true

  # =============================================================================
  # Test 9: cat with rvalue seq + [sugar] implicit conversion ArrayRef[Tensor]
  # =============================================================================
  runCppTest "cat with @[a, b] syntax (implicit seq)":
    proc(): bool =
      let a = randn(2, 3, kFloat32)
      let b = randn(2, 3, kFloat32)
      let c = cat(@[a, b], 0)
      doAssert c.dim() == 2
      doAssert c.size(0) == 4
      doAssert c.size(1) == 3
      true

  # =============================================================================
  # Test stack with Tensor
  # =============================================================================

  runCppTest "stack two tensors":
    proc(): bool =
      let a = randn(2, 3, kFloat32)
      let b = randn(2, 3, kFloat32)
      let c = stack(a, b, dim = 0)
      doAssert c.dim() == 3
      doAssert c.size(0) == 2
      doAssert c.size(1) == 2
      doAssert c.size(2) == 3
      true

  runCppTest "stack many tensors":
    proc(): bool =
      let a = randn(2, 3, kFloat32)
      let b = randn(2, 3, kFloat32)
      let c = randn(2, 3, kFloat32)
      let result = stack(a, b, c, dim = 0)
      doAssert result.size(0) == 3
      doAssert result.size(1) == 2
      doAssert result.size(2) == 3
      true

  # =============================================================================
  # Test chunk with Tensor
  # =============================================================================

  runCppTest "chunk tensor":
    proc(): bool =
      let a = ones(4, 3, kFloat32)
      let chunks = chunk(a, 2, dim = 0)
      doAssert chunks.len == 2
      doAssert chunks[0].size(0) == 2
      doAssert chunks[1].size(0) == 2
      true

  runCppTest "chunk single element":
    proc(): bool =
      let a = ones(3, 4, kFloat32)
      let chunks = chunk(a, 3, dim = 0)
      doAssert chunks.len == 3
      doAssert chunks[0].size(0) == 1
      doAssert chunks[1].size(0) == 1
      doAssert chunks[2].size(0) == 1
      true

  # =============================================================================
  # Test unbind with Tensor
  # =============================================================================

  runCppTest "unbind along dim 0":
    proc(): bool =
      let a = ones(3, 4, 5, kFloat32)
      let parts = unbind(a, dim = 0)
      doAssert parts.len == 3
      doAssert parts[0].dim() == 2
      doAssert parts[0].size(0) == 4
      doAssert parts[0].size(1) == 5
      true

  runCppTest "unbind along dim 1":
    proc(): bool =
      let a = ones(2, 3, 4, kFloat32)
      let parts = unbind(a, dim = 1)
      doAssert parts.len == 3
      doAssert parts[0].size(0) == 2
      doAssert parts[0].size(1) == 4
      true

  runCppTest "unbind default dim":
    proc(): bool =
      let a = ones(3, 4, 5, kFloat32)
      let parts = unbind(a)
      doAssert parts.len == 3
      doAssert parts[0].dim() == 2
      true

  echo "=== All tests cat, stack, chunks, unbind completed ==="

when isMainModule:
  main()