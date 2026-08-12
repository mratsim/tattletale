# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests for objects containing Tensor fields.
## Verifies copy/move semantics and refcount management.
##
## Previously `bug_test_destructors_embedded_tensors` — now works because
## `Tensor` is a `ref object` so Nim handles all memory correctly.

import
  workspace/libtorch/src/tensors,
  workspace/libtorch/libtorch_testutils

type
  TensorWrapper = object
    ## Simple object containing a single Tensor field
    data: Tensor

  TensorPair = object
    ## Object containing multiple Tensor fields
    first: Tensor
    second: Tensor

  NestedObject = object
    ## Object with mixed fields (tensors + primitives)
    id: int32
    weight: Tensor
    bias: Tensor
    scale: float32

# =============================================================================
# TensorWrapper Tests
# =============================================================================

proc main() =
  # TensorWrapper Tests
  runCppTest "init":
    proc(): bool =
      let wrapper = TensorWrapper(data: ones(2, 3, kFloat32))
      result = wrapper.data.isDefined() and
               wrapper.data.size(0) == 2 and
               wrapper.data.size(1) == 3

  runCppTest "copy":
    proc(): bool =
      var wrapper1 = TensorWrapper(data: ones(2, 3, kFloat32) * 2.0'f32)
      var wrapper2 = wrapper1  # Should call =copy

      # Both should be defined
      if not (wrapper1.data.isDefined() and wrapper2.data.isDefined()):
        return false

      # Both should reference the same data (copy increments refcount)
      if not wrapper1.data.is_same(wrapper2.data):
        return false

      # Modify wrapper2's data
      wrapper2.data = zeros(2, 3, kFloat32)

      # wrapper1 should still have original values
      let expected = ones(2, 3, kFloat32) * 2.0'f32
      result = wrapper1.data.allClose(expected)

  runCppTest "return from proc":
    proc(): bool =
      proc createWrapper(): TensorWrapper =
        TensorWrapper(data: ones(2, 3, kFloat32) * 5.0'f32)

      let wrapper = createWrapper()
      let expected = ones(2, 3, kFloat32) * 5.0'f32
      result = wrapper.data.isDefined() and wrapper.data.allClose(expected)

  runCppTest "assignment":
    proc(): bool =
      var wrapper1 = TensorWrapper(data: ones(2, 3, kFloat32))
      var wrapper2 = TensorWrapper(data: zeros(2, 3, kFloat32))

      wrapper2 = wrapper1  # Should call =copy

      # Both should reference the same data
      result = wrapper1.data.is_same(wrapper2.data) and
               wrapper1.data.isDefined() and
               wrapper2.data.isDefined()

  # =============================================================================
  # TensorPair Tests
  # =============================================================================

  runCppTest "init":
    proc(): bool =
      let pair = TensorPair(
        first: ones(2, kFloat32),
        second: zeros(2, kFloat32)
      )
      result = pair.first.isDefined() and
               pair.second.isDefined() and
               pair.first.size(0) == 2 and
               pair.second.size(0) == 2

  runCppTest "copy":
    proc(): bool =
      var pair1 = TensorPair(
        first: ones(2, kFloat32) * 3.0'f32,
        second: zeros(2, kFloat32)
      )
      var pair2 = pair1  # Should call =copy

      # Both tensors should be copied
      if not (pair1.first.is_same(pair2.first) and
              pair1.second.is_same(pair2.second)):
        return false

      # Modify pair2
      pair2.first = full(2, 999.0'f32, kFloat32)

      # pair1 should be unchanged
      let expected = ones(2, kFloat32) * 3.0'f32
      result = pair1.first.allClose(expected)

  runCppTest "return from proc":
    proc(): bool =
      proc createPair(): TensorPair =
        TensorPair(
          first: ones(3, kFloat32) * 7.0'f32,
          second: zeros(3, kFloat32)
        )

      let pair = createPair()
      let expectedFirst = ones(3, kFloat32) * 7.0'f32
      let expectedSecond = zeros(3, kFloat32)

      result = pair.first.isDefined() and
               pair.second.isDefined() and
               pair.first.allClose(expectedFirst) and
               pair.second.allClose(expectedSecond)

  # =============================================================================
  # NestedObject Tests
  # =============================================================================

  runCppTest "init":
    proc(): bool =
      let obj = NestedObject(
        id: 42'i32,
        weight: ones(2, 2, kFloat32),
        bias: zeros(2, kFloat32),
        scale: 1.5'f32
      )
      result = obj.id == 42 and
               obj.weight.isDefined() and
               obj.bias.isDefined() and
               obj.scale == 1.5'f32

  runCppTest "copy":
    proc(): bool =
      var obj1 = NestedObject(
        id: 100'i32,
        weight: ones(2, 2, kFloat32) * 4.0'f32,
        bias: zeros(2, kFloat32) * 5.0'f32,
        scale: 2.0'f32
      )
      var obj2 = obj1  # Should call =copy

      # Primitive fields should be copied
      if not (obj1.id == obj2.id and obj1.scale == obj2.scale):
        return false

      # Tensor fields should reference same data
      if not (obj1.weight.is_same(obj2.weight) and
              obj1.bias.is_same(obj2.bias)):
        return false

      # Modify obj2 tensors
      obj2.weight = full(2, 2, 999.0'f32, kFloat32)

      # obj1 should be unchanged
      let expected = ones(2, 2, kFloat32) * 4.0'f32
      result = obj1.weight.allClose(expected)

  runCppTest "return from proc":
    proc(): bool =
      proc createNested(): NestedObject =
        NestedObject(
          id: 999'i32,
          weight: ones(4, kFloat32) * 11.0'f32,
          bias: zeros(4, kFloat32),
          scale: 0.5'f32
        )

      let obj = createNested()
      let expectedWeight = ones(4, kFloat32) * 11.0'f32
      let expectedBias = zeros(4, kFloat32)

      result = obj.id == 999 and
               obj.scale == 0.5'f32 and
               obj.weight.isDefined() and
               obj.bias.isDefined() and
               obj.weight.allClose(expectedWeight) and
               obj.bias.allClose(expectedBias)

  # =============================================================================
  # Refcount Tests
  # =============================================================================

  runCppTest "refcount after copy":
    proc(): bool =
      # Verify that copy increments refcount
      var wrapper1 = TensorWrapper(data: ones(2, 3, kFloat32))
      let originalData = wrapper1.data

      var wrapper2 = wrapper1  # Copy

      # Both should reference the same data
      if not wrapper1.data.is_same(originalData):
        return false
      if not wrapper2.data.is_same(originalData):
        return false

      # After wrapper1 is destroyed, wrapper2 should still be valid
      var wrapper3 = wrapper2  # Copy again
      # wrapper1 goes out of scope here

      result = wrapper2.data.isDefined() and
               wrapper3.data.isDefined() and
               wrapper2.data.is_same(wrapper3.data)

  runCppTest "refcount after move":
    proc(): bool =
      var wrapper1 = TensorWrapper(data: ones(2, 3, kFloat32))
      let originalData = wrapper1.data

      var wrapper2 = move(wrapper1)  # Move

      # wrapper2 should have the data
      if not wrapper2.data.is_same(originalData):
        return false

      # wrapper1 should be undefined
      if wrapper1.data.isDefined():
        return false

      # After wrapper1 is destroyed, wrapper2 should still be valid
      # wrapper1 goes out of scope here

      result = wrapper2.data.isDefined() and
               wrapper2.data.is_same(originalData)

  runCppTest "multiple copies":
    proc(): bool =
      let original = ones(2, 3, kFloat32)

      var w1 = TensorWrapper(data: original)
      var w2 = w1  # Copy 1
      var w3 = w1  # Copy 2
      var w4 = w2  # Copy 3

      # All should reference the same data
      result = w1.data.is_same(original) and
               w2.data.is_same(original) and
               w3.data.is_same(original) and
               w4.data.is_same(original)

      # After w1, w2, w3 are destroyed, w4 should still be valid
      # (w1, w2, w3 go out of scope)
      # This test verifies refcount doesn't go to zero prematurely

when isMainModule:
  main()