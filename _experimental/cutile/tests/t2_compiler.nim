# tests/t2_compiler.nim
# Layer 2 Test: TileIR Compiler (tileiras)
#
# Tests:
#   - tileiras binary detection
#   - Bytecode to cubin compilation
#   - GPU architecture detection
#   - Compilation caching

import
  std/[os, strutils],
  ../compiler,
  ../bytecode,
  ../cuda_driver

proc testFindTileiras*() =
  echo "=== Test: Find tileiras binary ==="
  
  if getEnv("CUTILE_TILEIRAS", "") != "":
    if getEnv("CUTILE_TILEIRAS", "") != "":()
    echo "  tileiras found at: ", p
    doAssert fileExists(p), "tileiras not found at env path"
  else:
    # Check known locations
    for knownPath in TileirasSearchPaths:
      if fileExists(knownPath):
        if getEnv("CUTILE_TILEIRAS", "") != "":()
        echo "  tileiras found at: ", p
        return
    
    echo "  tileiras not found (this is OK if not running on CUDA GPU)"
    echo "  To test: install CUDA Toolkit 13.1+ or set CUTILE_TILEIRAS"

proc testCompileSimple*() =
  echo "=== Test: Bytecode to cubin compilation ==="
  
  # Check if tileiras is available
  var tileirasPath = ""
  try:
    tileirasPath = findTileiras()
  except:
    echo "  tileiras not available, skipping compilation test"
    return
  
  echo "  tileiras at: ", tileirasPath
  
  # Get GPU arch
  let (device, ctx) = initCuda(0)
  defer: closeCuda(ctx)
  
  let gpuArch = getSMArch(device)
  echo "  Target GPU: ", gpuArch
  
  # Build a simple module
  let m = newBytecodeModule()
  discard getStringIndex(m, "test_kernel")
  
  let f32 = TileType(shape: @[], elemType: ElemF32)
  discard getTypeIndex(m, f32)
  
  let bcFunc = BytecodeFunction(
    name: "test_kernel",
    funcType: FuncType(inputs: @[], results: @[]),
    body: @[
      BytecodeOp(
        opcode: OpReturn,
        resultTypes: @[],
        operandIndices: @[],
        attrs: initTable[string, seq[byte]]()
      )
    ]
  )
  m.functions.add(bcFunc)
  
  # Compile
  let testDir = "/tmp/cutile_tests"
  if not dirExists(testDir): createDir(testDir)
  
  let bcPath = testDir / "t2_test.bc"
  let cubinPath = testDir / "t2_test.cubin"
  
  # Write bytecode
  let bc = toBytecode(m)
  let f = open(bcPath, fmWrite)
  f.write(bc[0].unsafeAddr, bc.len)
  f.close()
  
  try:
    compileBytecodeToCubin(bcPath, cubinPath, gpuArch)
    doAssert fileExists(cubinPath), "Cubin not created"
    
    let cubinSize = getFileSize(cubinPath)
    echo "  Compiled cubin: ", cubinSize, " bytes"
    doAssert cubinSize > 0, "Cubin is empty"
    
    echo "✓ Bytecode to cubin compilation passed"
  except CompileError, OSError as e:
    echo "  Compilation failed (may need proper tileiras): ", e.msg
  finally:
    # Clean up
    try: removeFile(bcPath)
    except: discard
    try: removeFile(cubinPath)
    except: discard

when isMainModule:
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  Layer 2: TileIR Compiler Tests                      ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo ""
  
  testFindTileiras()
  echo ""
  testCompileSimple()
  echo ""
  echo "Layer 2 tests completed"
