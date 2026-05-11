# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed under MIT or Apache v2
#
# Test tensor memory aliasing - verify pointer access
# Mirrors raw_torch_tensors/test_aliasing_from_blob.nim

import
  std/strutils,
  workspace/libtorch/src/tensors,
  workspace/libtorch/libtorch_testutils

proc main() =
  runTest "Tensor.shape returns openArray view":
    proc(): bool =
      # Create first tensor and get shape view
      let t1 = zeros(64, 128, 256, kFloat32)
      echo "  t1.shape = ", @(t1.shape)
      echo "  t1.shape.len = ", t1.shape.len

      let dataPtr1 = t1.shape[0].unsafeAddr
      echo "  t1.shape[0].unsafeAddr = 0x", toHex(cast[uint](dataPtr1))
      echo ""

      # Create second tensor and get shape view
      let t2 = zeros(1, 2, 3, 4, 5, kFloat32)
      echo "  t2.shape = ", @(t2.shape)
      echo "  t2.shape.len = ", t2.shape.len

      let dataPtr2 = t2.shape[0].unsafeAddr
      echo "  t2.shape[0].unsafeAddr = 0x", toHex(cast[uint](dataPtr2))
      echo ""

      # Compare addresses
      let addr1 = cast[uint](dataPtr1)
      let addr2 = cast[uint](dataPtr2)

      let diff = if addr2 > addr1: addr2 - addr1 else: addr1 - addr2
      echo "  Address difference: ", diff, " bytes"
      echo ""

      if addr1 == addr2:
        echo "  ⚠️  Same address (stack reuse)"
      elif addr2 > addr1:
        echo "  ✓ Addresses growing upward"
      else:
        echo "  ✓ Addresses growing downward"
      echo ""
      true

  runTest "Tensor.data_ptr()":
    proc(): bool =
      # Create a tensor
      let tensor = randn(2, 3, 4, kFloat32)
      echo "  tensor.shape = ", @(tensor.shape)
      echo ""

      # Get data pointer
      let tensorDataPtr = tensor.data_ptr()
      echo "  tensor.data_ptr() = 0x", toHex(cast[uint](tensorDataPtr))
      echo ""

      if tensorDataPtr == nil:
        echo "  ❌ FAIL: data_ptr() returned nil"
        return false
      else:
        echo "  ✓ PASS: data_ptr() returned valid pointer"
      echo ""
      true

  runTest "from_blob preserves addresses":
    proc(): bool =
      # Create source data
      var sourceData: array[4, float32] = [1.0, 2.0, 3.0, 4.0]
      let sourceDataPtr = sourceData[0].unsafeAddr
      echo "  Source data address = 0x", toHex(cast[uint](sourceDataPtr))
      echo ""

      # Create tensor from blob using wrapper API
      let tensorFromBlob = from_blob(sourceDataPtr, 2, 2, kFloat32)
      echo "  Tensor from blob:"
      echo "    tensor.shape = ", @(tensorFromBlob.shape)
      echo "    tensor.data_ptr() = 0x", toHex(cast[uint](tensorFromBlob.data_ptr()))
      echo ""

      # Verify addresses match
      let blobDataPtr = tensorFromBlob.data_ptr()

      if cast[uint](blobDataPtr) == cast[uint](sourceDataPtr):
        echo "  ✓ PASS: data_ptr matches source data address"
        echo "    → from_blob does NOT copy data (zero-copy view)"
      else:
        echo "  ❌ FAIL: data_ptr differs from source data address"
        echo "    Expected: 0x", toHex(cast[uint](sourceDataPtr))
        echo "    Got:      0x", toHex(cast[uint](blobDataPtr))
        return false
      echo ""
      true

  echo "=== All tests completed ==="

when isMainModule:
  main()