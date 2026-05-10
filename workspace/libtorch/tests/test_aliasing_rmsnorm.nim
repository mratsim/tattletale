# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed under MIT or Apache v2
#
# Test rms_norm aliasing - does rms_norm modify the weight tensor?

import
  std/strutils,
  workspace/libtorch as F,
  workspace/libtorch/src/abi/neural_nets,
  workspace/libtorch_testutils

proc main() =
  runTest "rms_norm weight aliasing":
    proc(): bool =
      echo "=== Test: Does rms_norm modify weight tensor? ==="
      echo ""

      # Create weight tensor
      let weight = F.ones([64], scalarKind=F.kFloat32)
      echo "Original weight:"
      echo "  shape = ", @(weight.shape.asNimView())
      echo "  data_ptr() = 0x", weight.dataPtrHex()
      echo "  shape.data() = 0x", weight.shapePtrHex()
      weight.print()
      echo ""

      # Store weight in a "layer" object (simulating NormLayer.init without clone)
      type NormLayer = object
        weight*: TorchTensor
        eps*: float
      
      let normLayer = NormLayer(weight: weight, eps: 1e-6)
      
      echo "After storing in normLayer (no clone):"
      echo "  weight.data_ptr() = 0x", weight.dataPtrHex()
      echo "  normLayer.weight.data_ptr() = 0x", normLayer.weight.dataPtrHex()
      if weight.data_ptr() == normLayer.weight.data_ptr():
        echo "  ⚠️  SAME data_ptr - weight and normLayer.weight share memory!"
      echo ""

      # Create input tensor
      let input = F.randn([2, 8, 64], scalarKind=F.kFloat32)
      echo "Input tensor:"
      echo "  shape = ", @(input.shape.asNimView())
      echo ""

      # Call rms_norm
      echo "Calling rms_norm..."
      let normalized_shape = F.asTorchView(64)
      let output = F.rms_norm(input, normalized_shape, normLayer.weight, 1e-6)
      echo ""

      # Check if original weight was modified
      echo "After rms_norm:"
      echo "  weight.shape = ", @(weight.shape.asNimView())
      echo "  weight.data_ptr() = 0x", weight.dataPtrHex()
      echo "  normLayer.weight.shape = ", @(normLayer.weight.shape.asNimView())
      echo "  normLayer.weight.data_ptr() = 0x", normLayer.weight.dataPtrHex()
      weight.print()
      echo ""

      # Check if shape changed
      if @(weight.shape.asNimView()) != @[64]:
        echo "❌ BUG: weight.shape changed from [64] to ", @(weight.shape.asNimView())
        echo ""
        echo "This proves rms_norm (or our wrapper) modifies the weight tensor."
        echo "When weight is shared (no clone), the modification affects all references."
        return false
      else:
        echo "✓ weight.shape unchanged ([64])"
      
      # Check if data values changed by checking first few elements
      let weightData = cast[ptr UncheckedArray[float32]](F.data_ptr(weight))
      var dataChanged = false
      for i in 0..<5:
        if weightData[i] != 1.0'f32:
          dataChanged = true
          echo "  weight[", i, "] = ", weightData[i], " (expected 1.0)"
          break
      
      if dataChanged:
        echo "⚠️  weight data values changed (no longer all 1.0)"
        echo "  This means rms_norm modifies weight data in place"
      else:
        echo "✓ weight data values unchanged (all 1.0)"
      echo ""

      echo "=== Conclusion ==="
      echo ""
      if @(weight.shape.asNimView()) == @[64] and not dataChanged:
        echo "✓ rms_norm does NOT modify the weight tensor"
        echo "  The aliasing bug must be elsewhere (e.g., tensor storage sharing)"
      else:
        echo "⚠️  rms_norm DOES modify the weight tensor"
        echo "  Layer init functions MUST clone weight to prevent aliasing"
      echo ""
      true

when isMainModule:
  main()