## cutile/examples/e00_minimal.nim
## Minimal TileIR bytecode — hand-crafted to match cutile-rs format exactly.

import
  std/[os, strutils],
  ../bytecode,
  ../dsl

# ------------------------------------------------------------
# Build the simplest valid BytecodeModule
# ------------------------------------------------------------

proc buildMinimalModule(): BytecodeModule =
  let m = newBytecodeModule()

  # Manually register the function name string
  discard getStringIndex(m, "empty_kernel")

  # Register the function signature type: pointer to f32, shape=[]
  let sigTy = TileType(shape: @[], elemType: ElemPointer)
  discard getTypeIndex(m, sigTy)

  # Add the function
  m.functions.add(BytecodeFunction(
    name: "empty_kernel",
    funcType: FuncType(inputs: @[], results: @[]),
    body: @[]
  ))

  return m

proc buildSaxpyModule(): BytecodeModule =
  let m = newBytecodeModule()

  # Pre-register strings and types (avoid by-value copy issues)
  discard getStringIndex(m, "saxpy_kernel")

  let ptrF32 = TileType(shape: @[], elemType: ElemPointer)
  let scalarF32 = TileType(shape: @[], elemType: ElemF32)
  let tile128F32 = TileType(shape: @[128], elemType: ElemF32)
  let tile128Ptr = TileType(shape: @[128], elemType: ElemPointer)
  let tile1Ptr = TileType(shape: @[1], elemType: ElemPointer)
  let tile1F32 = TileType(shape: @[1], elemType: ElemF32)
  let tokenTy = TileType(shape: @[], elemType: ElemI1)  # token uses I1

  discard getTypeIndex(m, ptrF32)
  discard getTypeIndex(m, scalarF32)
  discard getTypeIndex(m, tile128F32)
  discard getTypeIndex(m, tile128Ptr)
  discard getTypeIndex(m, tile1Ptr)
  discard getTypeIndex(m, tile1F32)
  discard getTypeIndex(m, tokenTy)

  # Track op indices. In TileIR bytecode, op result indices are sequential.
  var kb = newKernel("saxpy_kernel", @[ptrF32, ptrF32, scalarF32], @[])

  let iota = kb.iota(@[128'i64], ElemI32)
  let xPtr1 = kb.reshape(0, tile1Ptr)
  let xPtrTile = kb.broadcast(xPtr1, tile128Ptr)
  let xPtrs = kb.offset(xPtrTile, iota, tile128Ptr)
  let (xData, _) = kb.loadPtrTko(xPtrs, tile128F32)

  let yPtr1 = kb.reshape(1, tile1Ptr)
  let yPtrTile = kb.broadcast(yPtr1, tile128Ptr)
  let yPtrs = kb.offset(yPtrTile, iota, tile128Ptr)
  let (yData, _) = kb.loadPtrTko(yPtrs, tile128F32)

  let alphaTile = kb.broadcast(2, tile128F32)
  let result = kb.fma(alphaTile, xData, yData, tile128F32)
  kb.storePtrTko(yPtrs, result)
  kb.ret()

  return kb.build()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

proc run() =
  let tmpDir = "/tmp/cutile_examples"
  if not dirExists(tmpDir): createDir(tmpDir)

  echo "=== e00_minimal: Module → Bytecode ==="
  echo ""

  echo "[minimal]"
  let m = buildMinimalModule()
  let bc = toBytecode(m)
  echo "  Bytecode: ", bc.len, " bytes"
  doAssert bc.len > 0
  doAssert bc[0] == 0x7F'u8
  echo "  ✓ Header magic correct"
  doAssert m.functions.len == 1
  doAssert m.functions[0].name == "empty_kernel"
  echo "  ✓ Function: ", m.functions[0].name
  echo "  m.strings = ", m.strings.repr
  echo "  m.types.len = ", m.types.len

  let path = tmpDir / "e00_minimal.bc"
  var s = newStringOfCap(bc.len)
  for b in bc: s.add(chr(b))
  writeFile(path, s)
  echo "  Wrote: ", path

  echo ""
  echo "[saxpy]"
  let m2 = buildSaxpyModule()
  let bc2 = toBytecode(m2)
  echo "  Bytecode: ", bc2.len, " bytes"
  echo "  m.strings = ", m2.strings.repr
  echo "  m.types.len = ", m2.types.len
  echo "  m.functions.len = ", m2.functions.len
  echo "  m.functions[0].body.len = ", m2.functions[0].body.len

  let path2 = tmpDir / "e00_saxpy.bc"
  var s2 = newStringOfCap(bc2.len)
  for b in bc2: s2.add(chr(b))
  writeFile(path2, s2)
  echo "  Wrote: ", path2

  echo ""
  echo "Done."

when isMainModule:
  run()
