# tests/t1_bytecode.nim
# Layer 1 Test: TileIR Bytecode Writer

import
  std/[os, tables],
  ../bytecode

proc testVarint*() =
  echo "=== Test: Varint encoding ==="
  
  var s: seq[byte] = @[]
  writeVarInt(s, 0)
  doAssert s == @[0'u8], "Varint 0 failed"
  
  s = @[]
  writeVarInt(s, 1)
  doAssert s == @[1'u8], "Varint 1 failed"
  
  s = @[]
  writeVarInt(s, 127)
  doAssert s == @[127'u8], "Varint 127 failed"
  
  s = @[]
  writeVarInt(s, 128)
  doAssert s == @[128'u8, 1'u8], "Varint 128 failed"
  
  s = @[]
  writeVarInt(s, 300)
  doAssert s == @[172'u8, 2'u8], "Varint 300 failed: " & $s
  
  echo "✓ Varint encoding passed"

proc testHeader*() =
  echo "=== Test: Header generation ==="
  
  var s: seq[byte] = @[]
  writeHeader(s)
  
  doAssert s.len >= 8, "Header too short"
  for i in 0 .. 7:
    doAssert s[i] == MagicNumber[i], 
      "Magic mismatch at byte " & $i
  doAssert s[8] == BytecodeMajor.uint8
  doAssert s[9] == BytecodeMinor.uint8
  
  echo "✓ Header generation passed (", s.len, " bytes)"

proc testTypeSection*() =
  echo "=== Test: Type table generation ==="
  
  let m = newBytecodeModule()
  let f32 = TileType(shape: @[], elemType: ElemF32)
  let tile128F32 = TileType(shape: @[128], elemType: ElemF32)
  
  let idx0 = getTypeIndex(m, f32)
  let idx1 = getTypeIndex(m, tile128F32)
  
  doAssert idx0 != idx1, "Duplicate type indices"
  doAssert m.types.len == 2, "Expected 2 types"
  
  var s: seq[byte] = @[]
  writeTypeSection(s, m)
  doAssert s.len > 0, "Type section empty"
  
  echo "✓ Type table generation passed (", s.len, " bytes)"

proc testStringSection*() =
  echo "=== Test: String table generation ==="
  
  let m = newBytecodeModule()
  let idx0 = getStringIndex(m, "hello")
  let idx1 = getStringIndex(m, "world")
  let idx2 = getStringIndex(m, "hello")  # reuse
  
  doAssert idx0 == 0
  doAssert idx1 == 1
  doAssert idx2 == 0, "String dedup failed"
  doAssert m.strings.len == 2
  
  var s: seq[byte] = @[]
  writeStringSection(s, m.strings)
  doAssert s.len > 0, "String section empty"
  
  echo "✓ String table generation passed (", s.len, " bytes)"

proc testOperationEncoding*() =
  echo "=== Test: Operation encoding ==="
  
  let m = newBytecodeModule()
  let tileT = TileType(shape: @[128], elemType: ElemF32)
  
  let op = BytecodeOp(
    opcode: OpAddF,
    resultTypes: @[tileT],
    operandIndices: @[0, 1],
    attrs: initTable[string, seq[byte]]()
  )
  
  var s: seq[byte] = @[]
  writeOperation(s, op, m)
  
  doAssert s.len > 0, "Operation encoding empty"
  doAssert s[0] == 0x02'u8, "AddF opcode wrong"
  
  echo "✓ Operation encoding passed (", s.len, " bytes)"

proc testFullModule*() =
  echo "=== Test: Full module serialization ==="
  
  let m = newBytecodeModule()
  discard getStringIndex(m, "axpy_kernel")
  
  let f32 = TileType(shape: @[], elemType: ElemF32)
  let tile128F32 = TileType(shape: @[128], elemType: ElemF32)
  let tileI32 = TileType(shape: @[128], elemType: ElemI32)
  discard getTypeIndex(m, f32)
  discard getTypeIndex(m, tile128F32)
  discard getTypeIndex(m, tileI32)
  
  let bcFunc = BytecodeFunction(
    name: "axpy_kernel",
    funcType: FuncType(
      inputs: @[TileType(shape: @[], elemType: ElemPointer)],
      results: @[]
    ),
    body: @[
      BytecodeOp(
        opcode: OpGetTileBlockId,
        resultTypes: @[TileType(shape: @[], elemType: ElemI32)],
        operandIndices: @[],
        attrs: initTable[string, seq[byte]]()
      ),
      BytecodeOp(
        opcode: OpIota,
        resultTypes: @[tile128F32],
        operandIndices: @[],
        attrs: initTable[string, seq[byte]]()
      ),
      BytecodeOp(
        opcode: OpReturn,
        resultTypes: @[],
        operandIndices: @[],
        attrs: initTable[string, seq[byte]]()
      )
    ]
  )
  m.functions.add(bcFunc)
  
  let bc = toBytecode(m)
  
  doAssert bc.len > 0, "Bytecode empty"
  doAssert bc[0] == 0x7F'u8, "Magic byte 0 wrong"
  doAssert bc[1] == 'T'.uint8, "Magic byte 1 wrong"
  doAssert bc[bc.len - 1] == SectionEnd, "Missing end marker"
  
  echo "✓ Full module serialization passed (", bc.len, " bytes)"
  
  let testDir = "/tmp/cutile_tests"
  if not dirExists(testDir): createDir(testDir)
  let path = testDir / "t1_module.bc"
  var s = newStringOfCap(bc.len)
  for i in 0 ..< bc.len:
    s.add(chr(bc[i]))
  writeFile(path, s)
  echo "  Bytecode written to ", path

proc testTileHelpers*() =
  echo "=== Test: Tile helper types ==="
  
  let scalar = tileScalar(ElemF32)
  doAssert scalar.shape.len == 0
  doAssert scalar.elemType == ElemF32
  
  let t1d = tile1D(128, ElemF32)
  doAssert t1d.shape.len == 1
  doAssert t1d.shape[0] == 128
  
  let t2d = tile2D(64, 64, ElemF16)
  doAssert t2d.shape.len == 2
  
  echo "✓ Tile helper types passed"

when isMainModule:
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  Layer 1: TileIR Bytecode Writer Tests               ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo ""
  
  testVarint()
  echo ""
  testHeader()
  echo ""
  testTypeSection()
  echo ""
  testStringSection()
  echo ""
  testOperationEncoding()
  echo ""
  testFullModule()
  echo ""
  testTileHelpers()
  echo ""
  echo "All Layer 1 tests passed ✓"
