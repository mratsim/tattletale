# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed under MIT or Apache v2
#
# Test tensor memory aliasing - verify pointer access

import
  std/strutils,
  workspace/libtorch/src/raw_libtorch as F

proc main() =
  echo "=== Test 1: IntArrayRef.data() ==="
  echo ""

  # Create first IntArrayRef
  let shape1 = F.asTorchView(64, 128, 256)
  echo "shape1.size() = ", shape1.size()

  let dataPtr1 = shape1.data()
  echo "shape1.data() = 0x", toHex(cast[uint](dataPtr1))
  echo ""

  # Create second IntArrayRef
  let shape2 = F.asTorchView(1, 2, 3, 4, 5)
  echo "shape2.size() = ", shape2.size()

  let dataPtr2 = shape2.data()
  echo "shape2.data() = 0x", toHex(cast[uint](dataPtr2))
  echo ""

  # Compare addresses
  let addr1 = cast[uint](dataPtr1)
  let addr2 = cast[uint](dataPtr2)

  let diff = if addr2 > addr1: addr2 - addr1 else: addr1 - addr2
  echo "Address difference: ", diff, " bytes"
  echo ""

  if addr1 == addr2:
    echo "⚠️  Same address (stack reuse)"
  elif addr2 > addr1:
    echo "✓ Addresses growing upward"
  else:
    echo "✓ Addresses growing downward"
  echo ""

  echo "=== Test 2: TorchTensor.data_ptr() ==="
  echo ""

  # Create a tensor
  let tensor = F.randn([2, 3, 4], scalarKind=F.kFloat32)
  echo "tensor.shape = ", tensor.shape

  # Get data pointer
  let tensorDataPtr = F.data_ptr(tensor)
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

  # Create shape
  let shape = F.asTorchView(2, 2)
  let shapePtr = shape.data()
  echo "Shape address = 0x", toHex(cast[uint](shapePtr))
  echo ""

  # Create tensor from blob
  let tensorFromBlob = F.from_blob(sourceDataPtr, shape, scalarKind=F.kFloat32)
  echo "Tensor from blob:"
  echo "  tensor.shape = ", tensorFromBlob.shape
  echo "  tensor.data_ptr() = 0x", toHex(cast[uint](F.data_ptr(tensorFromBlob)))
  echo "  tensor.shape.data() = 0x", toHex(cast[uint](tensorFromBlob.shape.data()))
  echo ""

  # Verify addresses match
  let blobDataPtr = F.data_ptr(tensorFromBlob)
  let blobShapePtr = tensorFromBlob.shape.data()

  if cast[uint](blobDataPtr) == cast[uint](sourceDataPtr):
    echo "✓ PASS: data_ptr matches source data address"
    echo "  → from_blob does NOT copy data (zero-copy view)"
  else:
    echo "❌ FAIL: data_ptr differs from source data address"
    echo "  Expected: 0x", toHex(cast[uint](sourceDataPtr))
    echo "  Got:      0x", toHex(cast[uint](blobDataPtr))
  echo ""

  doAssert cast[uint](blobShapePtr) != cast[uint](shapePtr),
    "shape.data() should differ from input shape address (libtorch should copy shape)"
  echo "✓ shape.data() differs from input shape address"
  echo "  → libtorch copied the shape array internally"
  echo "  → This is SAFE - tensor owns its shape memory"
  echo ""

  echo "=== All tests completed ==="

when isMainModule:
  main()