#
#
#            Vec Container Tests
#        (c) Copyright 2025 Tattletale contributors
#
#    See the file "copying.txt", included in this
#    distribution, for details about the copyright.
#

import
  std/unittest,
  workspace/containers,
  workspace/libtorch,
  workspace/libtorch_testutils

proc testNewWithLength(): bool =
  let v = Vec[TorchTensor].new(5)
  result = v.len == 5

proc testNewEmpty(): bool =
  let v = Vec[TorchTensor].new(0)
  result = v.len == 0

proc testIndexAccess(): bool =
  var v = Vec[TorchTensor].new(3)
  v[0] = zeros(@[2, 3], kFloat32)
  v[1] = ones(@[2, 3], kFloat32)
  v[2] = full(@[2, 3], 2.0'f32, kFloat32)
  result = (v[0].dim() == 2 and v[0].size(0) == 2 and v[0].size(1) == 3 and
            v[1].dim() == 2 and v[1].size(0) == 2 and v[1].size(1) == 3 and
            v[2].dim() == 2 and v[2].size(0) == 2 and v[2].size(1) == 3)

proc testIndexOutOfBounds(): bool =
  var v = Vec[TorchTensor].new(3)
  try:
    discard v[10]  # Out of bounds access
    result = false  # Should have raised
  except IndexError:
    result = true

proc testCopySemantics(): bool =
  var v1 = Vec[TorchTensor].new(2)
  v1[0] = ones(@[2, 3], kFloat32)
  v1[1] = zeros(@[2, 3], kFloat32)
  var v2 = v1  # Calls =copy
  result = (v2.len == 2 and
            v1[0] == v2[0] and
            v1[1] == v2[1])

proc testMoveSemantics(): bool =
  var v1 = Vec[TorchTensor].new(2)
  v1[0] = ones(@[2, 3], kFloat32)
  var v2 = move(v1)  # Calls =sink
  result = (v2.len == 2 and v1.len == 0)

proc testScopeExit(): bool =
  proc inner(): Vec[TorchTensor] =
    var v = Vec[TorchTensor].new(2)
    v[0] = ones(@[2, 3], kFloat32)
    v[1] = zeros(@[2, 3], kFloat32)
    return v

  let v = inner()
  result = (v.len == 2 and
            v[0].dim() == 2 and v[0].size(0) == 2 and v[0].size(1) == 3 and
            v[1].dim() == 2 and v[1].size(0) == 2 and v[1].size(1) == 3)

proc testToOpenArrayIteration(): bool =
  var v = Vec[TorchTensor].new(3)
  v[0] = full(@[2], 1.0'f32, kFloat32)
  v[1] = full(@[2], 2.0'f32, kFloat32)
  v[2] = full(@[2], 3.0'f32, kFloat32)

  var totalElements = 0
  for tensor in v.toOpenArray():
    totalElements += tensor.numel()
  result = totalElements == 6  # 3 tensors × 2 elements each

proc testDupHook(): bool =
  var v1 = Vec[TorchTensor].new(2)
  v1[0] = ones(@[2, 3], kFloat32)
  v1[1] = zeros(@[2, 3], kFloat32)
  let v2 = `=dup`(v1)
  result = (v2.len == 2 and
            v1.len == 2 and
            v1[0] == v2[0])

proc testSelfAssignmentCopy(): bool =
  var v = Vec[TorchTensor].new(2)
  v[0] = ones(@[2, 3], kFloat32)
  v[1] = zeros(@[2, 3], kFloat32)
  v = v  # Self-assignment should work
  result = (v.len == 2 and v[0].dim() == 2 and v[0].size(0) == 2)

proc testRefcountManagement(): bool =
  # Verify that tensors in Vec properly manage refcounts
  var v = Vec[TorchTensor].new(1)
  let original = ones(@[2, 3], kFloat32)
  v[0] = original

  # Both should reference the same data
  if not v[0].is_same(original):
    return false

  # After Vec is destroyed, original should still be valid
  # (because =copy increments refcount)
  var v2 = Vec[TorchTensor].new(1)
  v2[0] = original
  # v goes out of scope here, but original should still work
  result = original.isDefined() and original.numel() == 6

proc testDestructionAfterMove(): bool =
  # Verify that moved-from Vec can be safely destroyed
  var v1 = Vec[TorchTensor].new(2)
  v1[0] = ones(@[2, 3], kFloat32)
  v1[1] = zeros(@[2, 3], kFloat32)
  var v2 = move(v1)
  # v1 is now moved-from (len=0)
  # When v1 goes out of scope, =destroy should handle it safely
  result = v1.len == 0

proc testCopyCreatesIndependentCopy(): bool =
  var v1 = Vec[TorchTensor].new(3)
  v1[0] = full(@[2], 1.0'f32, kFloat32)
  v1[1] = full(@[2], 2.0'f32, kFloat32)
  v1[2] = full(@[2], 3.0'f32, kFloat32)

  var v2 = v1  # Deep copy
  v2[0] = full(@[2], 999.0'f32, kFloat32)  # Modify copy

  # Check that v1[0] is unchanged (still 1.0) and v2[0] is 999.0
  let expected1 = full(@[2], 1.0'f32, kFloat32)
  let expected2 = full(@[2], 999.0'f32, kFloat32)
  result = (v1[0] == expected1 and v2[0] == expected2)

proc testItemsIterator(): bool =
  var v = Vec[TorchTensor].new(3)
  v[0] = full(@[1], 1.0'f32, kFloat32)
  v[1] = full(@[1], 2.0'f32, kFloat32)
  v[2] = full(@[1], 3.0'f32, kFloat32)

  var sum = 0.0'f32
  for tensor in v:
    sum += tensor.item(float32)
  result = sum == 6.0

proc testMitemsIterator(): bool =
  var v = Vec[TorchTensor].new(3)
  v[0] = full(@[1], 1.0'f32, kFloat32)
  v[1] = full(@[1], 2.0'f32, kFloat32)
  v[2] = full(@[1], 3.0'f32, kFloat32)

  # Modify elements through mitems
  for tensor in v.mitems():
    let val = tensor.item(float32)
    tensor = full(@[1], val * 2, kFloat32)

  # Verify modification
  result = (v[0].item(float32) == 2.0 and
            v[1].item(float32) == 4.0 and
            v[2].item(float32) == 6.0)

proc runVecTests*() =
  suite "Vec[TorchTensor]":
    test "new with length":
      check catchCppExceptions(testNewWithLength())
    test "new empty":
      check catchCppExceptions(testNewEmpty())
    test "index access":
      check catchCppExceptions(testIndexAccess())
    test "index out of bounds":
      check testIndexOutOfBounds()
    test "copy semantics":
      check catchCppExceptions(testCopySemantics())
    test "move semantics":
      check catchCppExceptions(testMoveSemantics())
    test "scope exit":
      check catchCppExceptions(testScopeExit())
    test "toOpenArray iteration":
      check catchCppExceptions(testToOpenArrayIteration())
    test "dup hook":
      check catchCppExceptions(testDupHook())
    test "self-assignment copy":
      check catchCppExceptions(testSelfAssignmentCopy())
    test "refcount management":
      check catchCppExceptions(testRefcountManagement())
    test "destruction after move":
      check catchCppExceptions(testDestructionAfterMove())
    test "copy creates independent copy":
      check catchCppExceptions(testCopyCreatesIndependentCopy())
    test "items iterator":
      check catchCppExceptions(testItemsIterator())
    test "mitems iterator":
      check catchCppExceptions(testMitemsIterator())

when isMainModule:
  runVecTests()
