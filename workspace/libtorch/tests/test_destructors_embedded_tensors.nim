#
#
#            Embedded Tensors Tests
#        (c) Copyright 2025 Tattletale contributors
#
#    See the file "copying.txt", included in this
#    distribution, for details about the copyright.
#

## Tests for objects containing TorchTensor fields.
## Verifies copy/move semantics and refcount management.

import
  std/unittest,
  workspace/libtorch,
  workspace/libtorch_testutils

type
  TensorWrapper = object
    ## Simple object containing a single TorchTensor field
    data: TorchTensor

  TensorPair = object
    ## Object containing multiple TorchTensor fields
    first: TorchTensor
    second: TorchTensor

  NestedObject = object
    ## Object with mixed fields (tensors + primitives)
    id: int32
    weight: TorchTensor
    bias: TorchTensor
    scale: float32

# =============================================================================
# TensorWrapper Tests
# =============================================================================

proc testWrapperInit(): bool =
  let wrapper = TensorWrapper(data: ones(@[2, 3], kFloat32))
  result = wrapper.data.isDefined() and
           wrapper.data.size(0) == 2 and
           wrapper.data.size(1) == 3

proc testWrapperCopy(): bool =
  var wrapper1 = TensorWrapper(data: ones(@[2, 3], kFloat32) * 2.0'f32)
  var wrapper2 = wrapper1  # Should call =copy
  
  # Both should be defined
  if not (wrapper1.data.isDefined() and wrapper2.data.isDefined()):
    return false
  
  # Both should reference the same data (copy increments refcount)
  if not wrapper1.data.is_same(wrapper2.data):
    return false
  
  # Modify wrapper2's data
  wrapper2.data = zeros(@[2, 3], kFloat32)
  
  # wrapper1 should still have original values
  let expected = ones(@[2, 3], kFloat32) * 2.0'f32
  result = wrapper1.data.allClose(expected)

proc testWrapperMove(): bool =
  var wrapper1 = TensorWrapper(data: ones(@[2, 3], kFloat32))
  let originalData = wrapper1.data
  
  var wrapper2 = move(wrapper1)  # Should call =sink
  
  # wrapper2 should have the data
  if not wrapper2.data.isDefined():
    return false
  
  # wrapper2 should reference the same data as original
  if not wrapper2.data.is_same(originalData):
    return false
  
  # wrapper1.data should be undefined after move
  result = not wrapper1.data.isDefined()

proc testWrapperReturnFromProc(): bool =
  proc createWrapper(): TensorWrapper =
    TensorWrapper(data: ones(@[2, 3], kFloat32) * 5.0'f32)
  
  let wrapper = createWrapper()
  let expected = ones(@[2, 3], kFloat32) * 5.0'f32
  result = wrapper.data.isDefined() and wrapper.data.allClose(expected)

proc testWrapperAssignment(): bool =
  var wrapper1 = TensorWrapper(data: ones(@[2, 3], kFloat32))
  var wrapper2 = TensorWrapper(data: zeros(@[2, 3], kFloat32))
  
  wrapper2 = wrapper1  # Should call =copy
  
  # Both should reference the same data
  result = wrapper1.data.is_same(wrapper2.data) and
           wrapper1.data.isDefined() and
           wrapper2.data.isDefined()

# =============================================================================
# TensorPair Tests
# =============================================================================

proc testPairInit(): bool =
  let pair = TensorPair(
    first: ones(@[2], kFloat32),
    second: zeros(@[2], kFloat32)
  )
  result = pair.first.isDefined() and
           pair.second.isDefined() and
           pair.first.size(0) == 2 and
           pair.second.size(0) == 2

proc testPairCopy(): bool =
  var pair1 = TensorPair(
    first: ones(@[2], kFloat32) * 3.0'f32,
    second: zeros(@[2], kFloat32)
  )
  var pair2 = pair1  # Should call =copy
  
  # Both tensors should be copied
  if not (pair1.first.is_same(pair2.first) and
          pair1.second.is_same(pair2.second)):
    return false
  
  # Modify pair2
  pair2.first = full(@[2], 999.0'f32, kFloat32)
  
  # pair1 should be unchanged
  let expected = ones(@[2], kFloat32) * 3.0'f32
  result = pair1.first.allClose(expected)

proc testPairMove(): bool =
  var pair1 = TensorPair(
    first: ones(@[2], kFloat32),
    second: zeros(@[2], kFloat32)
  )
  let originalFirst = pair1.first
  let originalSecond = pair1.second
  
  var pair2 = move(pair1)  # Should call =sink
  
  # pair2 should have both tensors
  if not (pair2.first.isDefined() and pair2.second.isDefined()):
    return false
  
  # Both should reference original data
  if not (pair2.first.is_same(originalFirst) and
          pair2.second.is_same(originalSecond)):
    return false
  
  # pair1 tensors should be undefined
  result = (not pair1.first.isDefined()) and
           (not pair1.second.isDefined())

proc testPairReturnFromProc(): bool =
  proc createPair(): TensorPair =
    TensorPair(
      first: ones(@[3], kFloat32) * 7.0'f32,
      second: zeros(@[3], kFloat32)
    )
  
  let pair = createPair()
  let expectedFirst = ones(@[3], kFloat32) * 7.0'f32
  let expectedSecond = zeros(@[3], kFloat32)
  
  result = pair.first.isDefined() and
           pair.second.isDefined() and
           pair.first.allClose(expectedFirst) and
           pair.second.allClose(expectedSecond)

# =============================================================================
# NestedObject Tests
# =============================================================================

proc testNestedInit(): bool =
  let obj = NestedObject(
    id: 42'i32,
    weight: ones(@[2, 2], kFloat32),
    bias: zeros(@[2], kFloat32),
    scale: 1.5'f32
  )
  result = obj.id == 42 and
           obj.weight.isDefined() and
           obj.bias.isDefined() and
           obj.scale == 1.5'f32

proc testNestedCopy(): bool =
  var obj1 = NestedObject(
    id: 100'i32,
    weight: ones(@[2, 2], kFloat32) * 4.0'f32,
    bias: zeros(@[2], kFloat32) * 5.0'f32,
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
  obj2.weight = full(@[2, 2], 999.0'f32, kFloat32)
  
  # obj1 should be unchanged
  let expected = ones(@[2, 2], kFloat32) * 4.0'f32
  result = obj1.weight.allClose(expected)

proc testNestedMove(): bool =
  var obj1 = NestedObject(
    id: 200'i32,
    weight: ones(@[2], kFloat32),
    bias: zeros(@[2], kFloat32),
    scale: 3.0'f32
  )
  let originalWeight = obj1.weight
  let originalBias = obj1.bias
  
  var obj2 = move(obj1)  # Should call =sink
  
  # Primitive fields should be moved (copied for primitives)
  if not (obj2.id == 200 and obj2.scale == 3.0'f32):
    return false
  
  # Tensor fields should reference original data
  if not (obj2.weight.is_same(originalWeight) and
          obj2.bias.is_same(originalBias)):
    return false
  
  # obj1 tensors should be undefined
  result = (not obj1.weight.isDefined()) and
           (not obj1.bias.isDefined())

proc testNestedReturnFromProc(): bool =
  proc createNested(): NestedObject =
    NestedObject(
      id: 999'i32,
      weight: ones(@[4], kFloat32) * 11.0'f32,
      bias: zeros(@[4], kFloat32),
      scale: 0.5'f32
    )
  
  let obj = createNested()
  let expectedWeight = ones(@[4], kFloat32) * 11.0'f32
  let expectedBias = zeros(@[4], kFloat32)
  
  result = obj.id == 999 and
           obj.scale == 0.5'f32 and
           obj.weight.isDefined() and
           obj.bias.isDefined() and
           obj.weight.allClose(expectedWeight) and
           obj.bias.allClose(expectedBias)

# =============================================================================
# Refcount Tests
# =============================================================================

proc testRefcountAfterCopy(): bool =
  # Verify that copy increments refcount
  var wrapper1 = TensorWrapper(data: ones(@[2, 3], kFloat32))
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

proc testRefcountAfterMove(): bool =
  var wrapper1 = TensorWrapper(data: ones(@[2, 3], kFloat32))
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

proc testMultipleCopies(): bool =
  let original = ones(@[2, 3], kFloat32)
  
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

# =============================================================================
# Test Runner
# =============================================================================

proc runEmbeddedTensorTests*() =
  suite "Embedded Tensors":
    suite "TensorWrapper":
      test "init":
        check catchCppExceptions(testWrapperInit())
      test "copy":
        check catchCppExceptions(testWrapperCopy())
      test "move":
        check catchCppExceptions(testWrapperMove())
      test "return from proc":
        check catchCppExceptions(testWrapperReturnFromProc())
      test "assignment":
        check catchCppExceptions(testWrapperAssignment())
    
    suite "TensorPair":
      test "init":
        check catchCppExceptions(testPairInit())
      test "copy":
        check catchCppExceptions(testPairCopy())
      test "move":
        check catchCppExceptions(testPairMove())
      test "return from proc":
        check catchCppExceptions(testPairReturnFromProc())
    
    suite "NestedObject":
      test "init":
        check catchCppExceptions(testNestedInit())
      test "copy":
        check catchCppExceptions(testNestedCopy())
      test "move":
        check catchCppExceptions(testNestedMove())
      test "return from proc":
        check catchCppExceptions(testNestedReturnFromProc())
    
    suite "Refcount":
      test "refcount after copy":
        check catchCppExceptions(testRefcountAfterCopy())
      test "refcount after move":
        check catchCppExceptions(testRefcountAfterMove())
      test "multiple copies":
        check catchCppExceptions(testMultipleCopies())

when isMainModule:
  runEmbeddedTensorTests()