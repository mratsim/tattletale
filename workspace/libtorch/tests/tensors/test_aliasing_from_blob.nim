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
  runCppTest "Tensor.shape returns openArray view":
    proc(): bool =
      # This may crash with GCC v15.2.0 in the very first line `let t1 = zeros(64, 128, 256, kFloat32)`
      # but not with GCC v15.2.1 or Clang v22.1.x
      #
      # Thread 1 "test_aliasing_f" received signal SIGSEGV, Segmentation fault.
      # c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reset_not_null_ (target=0x0) at /home/beta/Programming/Perso/workspace-tattletale/tattletale/workspace/libtorch/vendor/libtorch/include/c10/util/intrusive_ptr.h:405
      # 405         if (detail::is_uniquely_owned(
      # (gdb) bt
      # #0  c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reset_not_null_ (target=0x0) at /home/beta/Programming/Perso/workspace-tattletale/tattletale/workspace/libtorch/vendor/libtorch/include/c10/util/intrusive_ptr.h:405
      # #1  0x000055555556a225 in c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reset_ (this=0x7fffffffd5d0) at /home/beta/Programming/Perso/workspace-tattletale/tattletale/workspace/libtorch/vendor/libtorch/include/c10/util/intrusive_ptr.h:398
      # #2  c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::~intrusive_ptr (this=0x7fffffffd5d0) at /home/beta/Programming/Perso/workspace-tattletale/tattletale/workspace/libtorch/vendor/libtorch/include/c10/util/intrusive_ptr.h:532
      # #3  at::TensorBase::~TensorBase (this=0x7fffffffd5d0) at /home/beta/Programming/Perso/workspace-tattletale/tattletale/workspace/libtorch/vendor/libtorch/include/ATen/core/TensorBase.h:119
      # #4  at::Tensor::~Tensor (this=0x7fffffffd5d0) at /home/beta/Programming/Perso/workspace-tattletale/tattletale/workspace/libtorch/vendor/libtorch/include/ATen/core/TensorBody.h:94
      # #5  zeros__OOZOOZsrcZtensors_u120 (size_p0=size_p0@entry=0x555555576410 <TM__pyeSeDytlwzdVj9cD9bOyVGQ_4>, size_p0Len_0=size_p0Len_0@entry=3, scalarKind_p1=c10::ScalarType::Float) at /home/beta/Programming/Perso/workspace-tattletale/tattletale/nimcache/wip/@mtest_aliasing_from_blob.nim.cpp:344
      # #6  0x000055555556a534 in colonanonymous___test95aliasing95from95blob_u5 () at /home/beta/Programming/Perso/workspace-tattletale/tattletale/nimcache/wip/@mtest_aliasing_from_blob.nim.cpp:432
      # #7  0x0000555555569906 in runTest__OOZOOZlibtorch95testutils_u10 (name_p0=..., body_p1=...) at /home/beta/Programming/Perso/workspace-tattletale/tattletale/nimcache/wip/@m..@s..@slibtorch_testutils.nim.cpp:254
      # #8  0x0000555555569d47 in main__test95aliasing95from95blob_u4 () at /home/beta/Programming/Perso/workspace-tattletale/tattletale/nimcache/wip/@mtest_aliasing_from_blob.nim.cpp:858
      # #9  0x0000555555569ecd in NimMainModule () at /home/beta/Programming/Perso/workspace-tattletale/tattletale/nimcache/wip/@mtest_aliasing_from_blob.nim.cpp:916
      # #10 0x0000555555569f48 in NimMainInner () at /home/beta/Programming/Perso/workspace-tattletale/tattletale/nimcache/wip/@mtest_aliasing_from_blob.nim.cpp:890
      # #11 0x0000555555559631 in main (argc=<optimized out>, args=<optimized out>, env=<optimized out>) at /home/beta/Programming/Perso/workspace-tattletale/tattletale/nimcache/wip/@mtest_aliasing_from_blob.nim.cpp:909
      #
      # This is indicative of extra destructors being inserted after C++ std::move / Nim =sink
      # which should be suppressed by {.nodestroy.}

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

  runCppTest "Tensor.data_ptr()":
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

  runCppTest "from_blob preserves addresses":
    proc(): bool =
      # Create source data
      var sourceData: array[4, float32] = [1.0, 2.0, 3.0, 4.0]
      let sourceDataPtr = sourceData[0].unsafeAddr
      echo "  Source data address = 0x", toHex(cast[uint](sourceDataPtr))
      echo ""

      # Create tensor from blob using wrapper API
      let tensorFromBlob = from_blob(sourceDataPtr, [2, 2], kFloat32)
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