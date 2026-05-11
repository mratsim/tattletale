# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed under MIT or Apache v2
#
# Systematic test suite for torch::cat
# Tests different input types

import
  std/strutils,
  workspace/libtorch/src/raw_libtorch as F,
  ../raw_libtorch_testutils

proc main() =
  echo "=== Test Suite: torch::cat ==="
  echo ""

  # =============================================================================
  # Test 1: cat with lvalue CppVector
  # =============================================================================
  runTest "cat with lvalue CppVector":
    proc(): bool =
      echo "Test 1: cat with lvalue CppVector"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      var tensors = init(CppVector[TorchTensor])
      tensors.add(a)
      tensors.add(b)

      let c = F.cat(tensors, 0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  # =============================================================================
  # Test 2: cat with lvalue array + lvalue ArrayRef[TorchTensor]
  # =============================================================================
  runTest "cat with lvalue array + lvalue ArrayRef[TorchTensor]":
    proc(): bool =
      echo "Test 2: cat with lvalue array + lvalue ArrayRef[TorchTensor]"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      let ab = [a, b]
      let abv = ab.asTorchView()

      let c = F.cat(abv, axis=0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  # =============================================================================
  # Test 3: cat with lvalue array + rvalue ArrayRef[TorchTensor]
  # =============================================================================
  runTest "cat with lvalue array + rvalue ArrayRef[TorchTensor]":
    proc(): bool =
      echo "Test 3: cat with lvalue array + rvalue ArrayRef[TorchTensor]"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      let ab = [a, b]
      let c = F.cat(ab.asTorchView(), axis=0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  # =============================================================================
  # Test 4: cat with rvalue array + rvalue ArrayRef[TorchTensor]
  # =============================================================================
  runTest "cat with rvalue array + rvalue ArrayRef[TorchTensor]":
    proc(): bool =
      echo "Test 4: cat with rvalue array + rvalue ArrayRef[TorchTensor]"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      let c = F.cat([a, b].asTorchView(), axis=0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  # =============================================================================
  # Test 5: cat with rvalue array + [sugar] implicit conversion ArrayRef[TorchTensor]
  # =============================================================================
  runTest "cat with rvalue array + [sugar] implicit conversion ArrayRef[TorchTensor]":
    proc(): bool =
      echo "Test 5: cat with rvalue array + [sugar] implicit conversion ArrayRef[TorchTensor]"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      let c = F.cat([a, b], axis=0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  # =============================================================================
  # Test 6: cat with lvalue seq + lvalue ArrayRef[TorchTensor]
  # =============================================================================
  runTest "cat with lvalue seq + lvalue ArrayRef[TorchTensor]":
    proc(): bool =
      echo "Test 6: cat with lvalue seq + lvalue ArrayRef[TorchTensor]"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      var tensors = @[a, b]
      let abv = tensors.asTorchView()

      echo "  tensors.data[0].addr = 0x", tensors[0].dataPtrHex()
      echo "  tensors.data[1].addr = 0x", tensors[1].dataPtrHex()
      echo "  abv.data() = 0x", cast[uint](abv.data()).toHex(16)

      let c = F.cat(abv, axis=0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  # =============================================================================
  # Test 7: cat with lvalue seq + rvalue ArrayRef[TorchTensor]
  # =============================================================================
  runTest "cat with lvalue seq + rvalue ArrayRef[TorchTensor]":
    proc(): bool =
      echo "Test 7: cat with lvalue seq + rvalue ArrayRef[TorchTensor]"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      var tensors = @[a, b]

      echo "  tensors.data[0].addr = 0x", tensors[0].dataPtrHex()
      echo "  tensors.data[1].addr = 0x", tensors[1].dataPtrHex()

      let c = F.cat(tensors.asTorchView(), axis=0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  # =============================================================================
  # Test 8: cat with rvalue seq + rvalue ArrayRef[TorchTensor]
  # =============================================================================
  runTest "cat with rvalue seq + rvalue ArrayRef[TorchTensor]":
    proc(): bool =
      echo "Test 8: cat with rvalue seq + rvalue ArrayRef[TorchTensor]"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      let c = F.cat(@[a, b].asTorchView(), axis=0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  # =============================================================================
  # Test 9: cat with rvalue seq + [sugar] implicit conversion ArrayRef[TorchTensor]
  # =============================================================================
  runTest "cat with @[a, b] syntax (implicit seq)":
    proc(): bool =
      echo "Test 9: cat with @[a, b] syntax (implicit seq)"
      let a = F.randn([2, 3], scalarKind=F.kFloat32)
      let b = F.randn([2, 3], scalarKind=F.kFloat32)

      let c = F.cat(@[a, b], 0)

      echo "  c.shape = ", @(c.shape.asNimView())
      echo "  c.data_ptr = 0x", c.dataPtrHex()
      echo ""
      true

  echo "=== All tests completed ==="

when isMainModule:
  main()