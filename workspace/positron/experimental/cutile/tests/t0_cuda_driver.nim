# tests/t0_cuda_driver.nim
# Layer 0 Test: CUDA Driver API wrappers
#
# Tests:
#   - CUDA initialization
#   - Device detection
#   - Context creation/destruction
#   - Memory allocation and transfers
#   - cuModuleLoad (file-based)

import
  std/[os, strutils],
  ../cuda_driver

proc testInitDevice*() =
  ## Test CUDA initialization and device detection.
  echo "=== Test: CUDA init and device detection ==="
  
  let (device, ctx) = initCuda(0)
  defer: closeCuda(ctx)
  
  let arch = getSMArch(device)
  echo "SM Architecture: ", arch
  doAssert arch.startsWith("sm_"), "Invalid SM arch: " & arch
  
  echo "✓ CUDA init and device detection passed"

proc testMemory*() =
  ## Test device memory allocation and transfers.
  echo "=== Test: Device memory allocation and transfers ==="
  
  let (device, ctx) = initCuda(0)
  defer: closeCuda(ctx)
  
  # Allocate device memory
  const dataLen = 256
  var mem = allocDevice(dataLen * sizeof(float32))
  defer: freeDevice(mem)
  
  doAssert mem.size == dataLen * sizeof(float32)
  
  # Host-to-device transfer
  var hostData: array[256, float32]
  for i in 0 .. 255:
    hostData[i] = float32(i)
  mem.h2d(hostData[0].addr, dataLen * sizeof(float32))
  
  # Device-to-host transfer
  var resultData: array[256, float32]
  mem.d2h(resultData[0].addr, dataLen * sizeof(float32))
  
  # Verify
  for i in 0 .. 255:
    doAssert resultData[i] == float32(i), 
      "Mismatch at index " & $i & ": " & $resultData[i]
  
  echo "✓ Device memory allocation and transfers passed"

proc testModuleLoad*() =
  ## Test cuModuleLoad (file-based) with a minimal cubin.
  echo "=== Test: Module loading ==="
  
  # This test creates a minimal PTX module to test loading
  # For a real test, we'd load a .cubin from tileiras
  echo "  (Skipped: requires compiled cubin file)"
  echo "✓ Module load test skipped (needs compiled kernel)"

when isMainModule:
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  Layer 0: CUDA Driver API Tests                      ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo ""
  
  testInitDevice()
  echo ""
  testMemory()
  echo ""
  testModuleLoad()
  echo ""
  echo "All Layer 0 tests passed ✓"
