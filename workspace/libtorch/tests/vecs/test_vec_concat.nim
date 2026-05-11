# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed under MIT or Apache v2
#
# Test torch::cat with Vec[TorchTensor]

import
  std/strutils,
  workspace/libtorch/src/raw_libtorch as F,
  workspace/libtorch/src/vecs/vecs,
  ../raw_libtorch_testutils,


proc main() =
  echo "=== Test Suite: torch::cat with Vec ==="
  echo ""

  # =============================================================================
  # Test 1: cat with lvalue Vec + lvalue ArrayRef[TorchTensor]
  # =============================================================================
  runTest "cat with lvalue Vec + lvalue ArrayRef[TorchTensor]":
    proc(): bool =
      echo "Test 1: cat with lvalue Vec + lvalue ArrayRef[TorchTensor]"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      var tensors = Vec[TorchTensor].new(2)
      tensors[0] = a
      tensors[1] = b

      let abv = tensors.asTorchView()

      let c = F.cat(abv, axis=0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  # =============================================================================
  # Test 2: cat with lvalue Vec + rvalue ArrayRef[TorchTensor]
  # =============================================================================
  runTest "cat with lvalue Vec + rvalue ArrayRef[TorchTensor]":
    proc(): bool =
      echo "Test 2: cat with lvalue Vec + rvalue ArrayRef[TorchTensor]"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      var tensors = Vec[TorchTensor].new(2)
      tensors[0] = a
      tensors[1] = b

      let c = F.cat(tensors.asTorchView(), axis=0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  # =============================================================================
  # Test 3: cat with rvalue Vec + rvalue ArrayRef[TorchTensor]
  # =============================================================================
  runTest "cat with rvalue Vec + rvalue ArrayRef[TorchTensor]":
    proc(): bool =
      echo "Test 3: cat with rvalue Vec + rvalue ArrayRef[TorchTensor]"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      var tensors = Vec[TorchTensor].new(2)
      tensors[0] = a
      tensors[1] = b

      let c = F.cat(tensors.asTorchView(), axis=0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  echo "=== All tests completed ==="

when isMainModule:
  main()