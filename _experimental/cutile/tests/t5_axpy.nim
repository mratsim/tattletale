# tests/t5_axpy.nim
# Layer 5 Test: End-to-end AXPY kernel
#
# Full pipeline test:
#   1. Build AXPY kernel with DSL
#   2. Compile to bytecode
#   3. Compile bytecode to cubin via tileiras
#   4. Load cubin
#   5. Launch kernel
#   6. Verify results
#
# Requires: CUDA GPU, tileiras (CUDA Toolkit 13.1+)

import
  std/[os, strutils, math],
  ../cuda_driver,
  ../bytecode,
  ../compiler,
  ../dsl

proc testEndToEndAxpY*() =
  echo "=== Test: End-to-end AXPY ==="
  
  # Step 1: Initialize CUDA
  let (device, ctx) = initCuda(0)
  defer: closeCuda(ctx)
  
  let gpuArch = getSMArch(device)
  echo "  Target GPU: ", gpuArch
  
  # Step 2: Check tileiras
  var tileirasPath = ""
  try:
    tileirasPath = findTileiras()
  except:
    echo "  tileiras not available, skipping end-to-end test"
    echo "  Install CUDA Toolkit 13.1+ or set CUTILE_TILEIRAS"
    return
  
  echo "  tileiras at: ", tileirasPath
  
  # Step 3: Build AXPY kernel
  const tileSize = 128
  const N = 1024
  const gridX = 1024 div 128  # 8 blocks
  
  echo "  Building AXPY kernel (tileSize=", tileSize, ", N=", N, ")"
  let m = buildAxpYKernel(tileSize)
  
  let bc = toBytecode(m)
  echo "  Bytecode size: ", bc.len, " bytes"
  
  # Step 4: Compile to cubin
  let testDir = "/tmp/cutile_tests"
  if not dirExists(testDir): createDir(testDir)
  
  let bcPath = testDir / "t5_axpy.bc"
  let cubinPath = testDir / "t5_axpy.cubin"
  
  # Write bytecode
  let f = open(bcPath, fmWrite)
  f.write(bc[0].unsafeAddr, bc.len)
  f.close()
  
  try:
    compileBytecodeToCubin(bcPath, cubinPath, gpuArch)
    echo "  Compiled cubin: ", getFileSize(cubinPath), " bytes"
  except CompileError, OSError as e:
    echo "  Compilation failed: ", e.msg
    return
  
  # Step 5: Load kernel
  let module = loadModuleFromFile(cubinPath)
  defer: unloadModule(module)
  
  let kernel = getFunction(module, "axpy_kernel")
  echo "  Kernel loaded: axpy_kernel"
  
  # Step 6: Allocate device memory
  var xMem = allocDevice(N * sizeof(float32))
  var yMem = allocDevice(N * sizeof(float32))
  defer:
    freeDevice(xMem)
    freeDevice(yMem)
  
  # Step 7: Prepare host data
  var hx: array[1024, float32]
  var hy: array[1024, float32]
  for i in 0 .. 1023:
    hx[i] = float32(i)
    hy[i] = float32(i * 2)
  
  # Step 8: Copy to device
  xMem.h2d(hx[0].addr, N * sizeof(float32))
  yMem.h2d(hy[0].addr, N * sizeof(float32))
  
  # Step 9: Launch kernel
  let alpha = 2.5f32
  
  # TileIR kernel args: x_ptr, y_ptr, alpha
  # For TileIR, we pass the device pointers and scalar alpha
  let args: array[3, pointer] = [
    cast[pointer](xMem.ptr.int64.uint64),
    cast[pointer](yMem.ptr.int64.uint64),
    alpha.unsafeAddr
  ]
  
  echo "  Launching kernel with gridX=", gridX
  launchKernel(kernel, gridX.uint32, args)
  synchronize()
  
  # Step 10: Copy results back
  var resultData: array[1024, float32]
  yMem.d2h(resultData[0].addr, N * sizeof(float32))
  
  # Step 11: Verify
  var errors = 0
  for i in 0 .. 1023:
    let expected = alpha * hx[i] + hy[i]
    let actual = resultData[i]
    if abs(expected - actual) > 0.01f32:
      if errors < 5:
        echo "  Error at index ", i, ": expected ", expected, " got ", actual
      inc errors
  
  if errors == 0:
    echo "✓ AXPY results verified (", N, " elements, 0 errors)"
  else:
    echo "✗ AXPY verification failed (", errors, " errors out of ", N, ")"
  
  # Clean up
  try: removeFile(bcPath)
  except: discard
  try: removeFile(cubinPath)
  except: discard

when isMainModule:
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  Layer 5: End-to-End AXPY Test                       ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo ""
  
  testEndToEndAxpY()
  echo ""
  echo "Layer 5 test completed"
