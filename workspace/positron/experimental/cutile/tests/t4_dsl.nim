# tests/t4_dsl.nim
# Layer 4 Test: DSL for TileIR

import
  std/[os, strutils, tables],
  ../dsl,
  ../bytecode

proc testKernelBuilder*() =
  echo "=== Test: Kernel builder ==="
  
  let kb = newKernel(
    "test_kernel",
    @[
      TileType(shape: @[], elemType: ElemPointer),
      TileType(shape: @[], elemType: ElemF32)
    ],
    @[]
  )
  
  doAssert kb.funcName == "test_kernel"
  doAssert kb.funcType.inputs.len == 2
  doAssert kb.numResults == 2
  
  let iota = kb.iota(@[128.int64], ElemI32)
  doAssert iota == 2
  doAssert kb.numResults == 3
  
  let bid = kb.getTileBlockId()
  doAssert bid == 3
  
  let tile128F32 = TileType(shape: @[128.int64], elemType: ElemF32)
  let alphaBroadcast = kb.broadcast(1, tile128F32)
  doAssert alphaBroadcast == 4
  doAssert kb.numResults == 5
  
  kb.ret()
  
  let m = kb.build()
  doAssert m.functions.len == 1
  doAssert m.functions[0].name == "test_kernel"
  doAssert m.functions[0].body.len == 4
  
  echo "✓ Kernel builder passed"

proc testAxpYKernel*() =
  echo "=== Test: AXPY kernel construction ==="
  
  let m = buildAxpYKernel(128)
  
  doAssert m.functions.len == 1
  doAssert m.functions[0].name == "axpy_kernel"
  
  let body = m.functions[0].body
  echo "  Operations in AXPY kernel: ", body.len
  
  doAssert body[0].opcode == OpGetTileBlockId
  doAssert body[1].opcode == OpIota
  
  var foundFma = false
  for op in body:
    if op.opcode == OpFma:
      foundFma = true
      doAssert op.operandIndices.len == 3
  doAssert foundFma, "FMA not found"
  
  let bc = toBytecode(m)
  doAssert bc.len > 0
  doAssert bc[0] == 0x7F'u8
  
  echo "  Bytecode size: ", bc.len, " bytes"
  echo "✓ AXPY kernel construction passed"
  
  let testDir = "/tmp/cutile_tests"
  if not dirExists(testDir): createDir(testDir)
  let path = testDir / "t4_axpy.bc"
  var s = newStringOfCap(bc.len)
  for i in 0 ..< bc.len:
    s.add(chr(bc[i]))
  writeFile(path, s)

proc testPrintKernel*() =
  echo "=== Test: Print kernel construction ==="
  
  let m = buildPrintKernel(128)
  doAssert m.functions.len == 1
  doAssert m.functions[0].name == "print_kernel"
  
  let bc = toBytecode(m)
  doAssert bc.len > 0
  
  echo "  Bytecode size: ", bc.len, " bytes"
  echo "✓ Print kernel construction passed"

proc testOperationTypes*() =
  echo "=== Test: Operation type handling ==="
  
  let kb = newKernel("type_test", @[], @[])
  
  let tile128F32 = TileType(shape: @[128.int64], elemType: ElemF32)
  
  discard kb.addF(0, 1, tile128F32)
  doAssert kb.body[^1].opcode == OpAddF
  
  discard kb.mulF(0, 1, tile128F32)
  doAssert kb.body[^1].opcode == OpMulF
  
  discard kb.subF(0, 1, tile128F32)
  doAssert kb.body[^1].opcode == OpSubF
  
  discard kb.divF(0, 1, tile128F32)
  doAssert kb.body[^1].opcode == OpDivF
  
  discard kb.fma(0, 1, 2, tile128F32)
  doAssert kb.body[^1].opcode == OpFma
  doAssert kb.body[^1].operandIndices.len == 3
  
  kb.ret()
  echo "✓ Operation type handling passed"

when isMainModule:
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  Layer 4: DSL Tests                                  ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo ""
  
  testKernelBuilder()
  echo ""
  testAxpYKernel()
  echo ""
  testPrintKernel()
  echo ""
  testOperationTypes()
  echo ""
  echo "All Layer 4 tests passed ✓"
