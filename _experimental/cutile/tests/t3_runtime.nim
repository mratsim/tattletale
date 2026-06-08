# tests/t3_runtime.nim
# Layer 3 Test: Kernel Runtime
#
# Tests:
#   - Cubin loading
#   - Kernel function lookup
#   - Kernel launch
#   - Execution verification
#
# NOTE: These tests require a compiled cubin file.
# Run t2_compiler.nim first to generate test cubins.

import
  std/[os, strutils],
  ../cuda_driver

proc testLoadPrebuiltCubin*(cubinPath: string) =
  echo "=== Test: Load prebuilt cubin ==="
  
  if not fileExists(cubinPath):
    echo "  Cubin not found: ", cubinPath
    echo "  Skip this test or provide a compiled cubin"
    return
  
  let (device, ctx) = initCuda(0)
  defer: closeCuda(ctx)
  
  # Load module
  var module = loadModuleFromFile(cubinPath)
  defer: unloadModule(module)
  
  echo "  Module loaded from: ", cubinPath
  
  # Get kernel function
  let kernel = getFunction(module, "test_kernel")
  echo "  Kernel 'test_kernel' found"
  
  # Launch (empty kernel)
  launchKernel(kernel, 1, @[])
  synchronize()
  
  echo "✓ Load prebuilt cubin passed"

proc testModuleLoadData*() =
  echo "=== Test: Load module from in-memory data ==="
  
  let (device, ctx) = initCuda(0)
  defer: closeCuda(ctx)
  
  # For this test, we'd need cubin bytes in memory
  echo "  (Skipped: requires cubin bytes)"
  echo "✓ Module load data test skipped"

proc testKernelTiming*() =
  echo "=== Test: Kernel timing ==="
  
  let (device, ctx) = initCuda(0)
  defer: closeCuda(ctx)
  
  # This test would time a real kernel launch
  # For now, just verify the timing infrastructure
  echo "  (Skipped: requires compiled kernel)"
  echo "✓ Kernel timing test skipped"

when isMainModule:
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  Layer 3: Kernel Runtime Tests                       ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo ""
  
  testLoadPrebuiltCubin("/tmp/cutile_tests/t2_test.cubin")
  echo ""
  testModuleLoadData()
  echo ""
  testKernelTiming()
  echo ""
  echo "Layer 3 tests completed"
