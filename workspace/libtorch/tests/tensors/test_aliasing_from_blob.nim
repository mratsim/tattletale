# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed under MIT or Apache v2
#
# Test tensor memory aliasing - verify pointer access
# Adapted from test_aliasing_from_blob.nim

import
  std/strutils,
  workspace/libtorch/src/tensors,
  workspace/libtorch/libtorch_testutils

proc main() =
  echo "=== Test 1: Tensor.shape returns openArray ==="
  echo ""

  # Create first tensor
  let t1 = zeros(64, 128, 256, kFloat32)
  echo "t1.shape = ", @(t1.shape)
  echo "t1.shape.len = ", t1.shape.len
  echo ""

  # Create second tensor
  let t2 = zeros(1, 2, 3, 4, 5, kFloat32)
  echo "t2.shape = ", @(t2.shape)
  echo "t2.shape.len = ", t2.shape.len
  echo ""

  echo "=== Test 2: Tensor.data_ptr() ==="
  echo ""

  # Create a tensor
  let tensor = randn(2, 3, 4, kFloat32)
  echo "tensor.shape = ", @(tensor.shape)
  echo ""

  # Get data pointer
  let tensorDataPtr = tensor.data_ptr()
  echo "tensor.data_ptr() = 0x", toHex(cast[uint](tensorDataPtr))
  echo ""

  if tensorDataPtr == nil:
    echo "❌ FAIL: data_ptr() returned nil"
  else:
    echo "✓ PASS: data_ptr() returned valid pointer"
  echo ""

  echo "=== Test 3: from_blob preserves addresses ==="
  echo ""

  # Create source data
  var sourceData: array[4, float32] = [1.0, 2.0, 3.0, 4.0]
  let sourceDataPtr = sourceData[0].unsafeAddr
  echo "Source data address = 0x", toHex(cast[uint](sourceDataPtr))

  # Create tensor from blob
  let tensorFromBlob = from_blob(sourceDataPtr, 2, 2, kFloat32)
  echo "Tensor from blob:"
  echo "  tensor.shape = ", @(tensorFromBlob.shape)
  echo "  tensor.data_ptr() = 0x", toHex(cast[uint](tensorFromBlob.data_ptr()))
  echo ""

  # Verify addresses match
  let blobDataPtr = tensorFromBlob.data_ptr()

  if cast[uint](blobDataPtr) == cast[uint](sourceDataPtr):
    echo "✓ PASS: data_ptr matches source data address"
    echo "  → from_blob does NOT copy data (zero-copy view)"
  else:
    echo "❌ FAIL: data_ptr differs from source data address"
    echo "  Expected: 0x", toHex(cast[uint](sourceDataPtr))
    echo "  Got:      0x", toHex(cast[uint](blobDataPtr))
  echo ""

  echo "=== All tests completed ==="

when isMainModule:
  main()