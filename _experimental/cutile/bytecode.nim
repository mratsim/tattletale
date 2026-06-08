# cutile/bytecode.nim
# Layer 1: TileIR Bytecode Writer
#
# Pure Nim implementation of CUDA Tile IR bytecode format.
# Same binary format as cuda-tile's BytecodeWriter.cpp.
#
# Bytecode format:
#   header = magic[8] + version_major[1] + version_minor[1] + tag[2]
#   sections = (section_id[1] + length[varint] + data)*
#   end_marker = 0x00
#
# Test: tests/t1_bytecode.nim

import
  std/[tables, math]

# ############################################################
# Bytecode constants (from cuda-tile BytecodeEnums.h)
# ############################################################

const
  # Magic number: 0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00
  MagicNumber* = [0x7F'u8, 'T'.uint8, 'i'.uint8, 'l'.uint8,
                  'e'.uint8, 'I'.uint8, 'R'.uint8, 0x00'u8]

  # Bytecode version (13.1 = kCurrentCompatibilityVersion)
  BytecodeMajor* = 13
  BytecodeMinor* = 1
  BytecodeTag* = 0'u16

  # Section IDs
  SectionEnd* = 0x00'u8
  SectionString* = 0x01'u8
  SectionFunc* = 0x02'u8
  SectionDebug* = 0x03'u8
  SectionConstant* = 0x04'u8
  SectionType* = 0x05'u8
  SectionGlobal* = 0x06'u8
  SectionProducer* = 0x07'u8

  # Alignment padding byte
  AlignmentByte* = 0xCB'u8

# ############################################################
# Type tags (from BytecodeTypeOpcodes.td)
# ############################################################

type
  TypeTag* = enum
    TagInt1 = 0, TagInt8 = 1, TagInt16 = 2, TagInt32 = 3, TagInt64 = 4
    TagFloat16 = 5, TagBFloat16 = 6, TagFloat32 = 7, TagTFloat32 = 8, TagFloat64 = 9
    TagFloat8E4M3FN = 10, TagFloat8E5M2 = 11
    TagPointer = 12, TagTile = 13, TagTensorView = 14
    TagPartitionView = 15, TagFunctionType = 16, TagToken = 17
    TagFloat8E8M0FNU = 18, TagFloat4E2M1FN = 19
    TagGatherScatterView = 20, TagStridedView = 21, TagInt4 = 22

# ############################################################
# Opcodes (from BytecodeOpcodes.td - public ops only)
# ############################################################

type
  Opcode* = enum
    OpAbsF = 0x00, OpAbsI = 0x01, OpAddF = 0x02, OpAddI = 0x03
    OpAndI = 0x04, OpAssert = 0x05, OpAssume = 0x06
    OpAtomicCAS = 0x07, OpAtomicRMW = 0x08, OpBitcast = 0x09
    OpBreak = 0x0A, OpBroadcast = 0x0B, OpCat = 0x0C
    OpCeil = 0x0D, OpCmpF = 0x0E, OpCmpI = 0x0F
    OpConstant = 0x10, OpContinue = 0x11, OpCos = 0x12
    OpCosH = 0x13, OpDivF = 0x14, OpDivI = 0x15
    OpEntry = 0x16, OpExp = 0x17, OpExp2 = 0x18
    OpNegF = 0x1C, OpNegI = 0x1D, OpSin = 0x1E, OpSinH = 0x1F
    OpLog = 0x20, OpLog2 = 0x21, OpPow = 0x22, OpSqrt = 0x23
    OpRSqrt = 0x24, OpExtI = 0x25, OpExtract = 0x26, OpFloor = 0x27
    OpFma = 0x28, OpFor = 0x29
    OpSelect = 0x2A, OpIf = 0x2B
    OpMakeTensorView = 0x2C, OpMakeRangePartitionView = 0x2D
    OpMakeRangeStridedView = 0x2E, OpMakeRangeGatherScatterView = 0x2F
    OpGetTileBlockId = 0x30, OpGlobal = 0x31, OpGetGlobal = 0x32
    OpMakeRange = 0x33, OpMakeRangeReduce = 0x34, OpMakeRangeScan = 0x35
    OpMakeRangePermute = 0x36, OpMakeRangePack = 0x37, OpMakeRangeUnpack = 0x38
    OpMakeRangeCat = 0x39, OpIota = 0x3A, OpMakeRangeExtract = 0x3B
    OpMakeRangeBroadcast = 0x3C, OpLoadPtrTko = 0x3D, OpLoadViewTko = 0x3E
    OpMakeRangeIota = 0x3F, OpMakeRangeSelect = 0x40
    OpMakeRangeBitcast = 0x41, OpMakeRangeFtoF = 0x42
    OpMakeRangeFtoI = 0x43, OpMakeRangeItoF = 0x44
    OpMakeRangeExtI = 0x45, OpMakeRangeTruncI = 0x46
    OpMakeRangeIntToPtr = 0x47, OpMakeRangePtrToInt = 0x48
    OpMmaF = 0x49, OpMmaI = 0x4A, OpMakeRangeMakeTensorView = 0x4B
    OpMulF = 0x4C, OpMakeRangeMakePartitionView = 0x4D
    OpMakeRangeMakeStridedView = 0x4E, OpMakeRangeMakeGatherScatterView = 0x4F
    OpMakeRangeAlloca = 0x50, OpOffset = 0x51
    OpMakeRangeGetTensorShape = 0x52, OpMakeRangeGetIndexSpaceShape = 0x53
    OpMakeRangeAtomicCAS = 0x54, OpPrintTko = 0x55
    OpMakeRangeAtomicRMW = 0x56, OpMakeRangeAtomicRedView = 0x57
    OpMakeRangeMakeToken = 0x58, OpMakeRangeJoinTokens = 0x59
    OpMakeRangeCmpF = 0x5A, OpReshape = 0x5B, OpReturn = 0x5C
    OpMakeRangeReduce2 = 0x5D, OpMakeRangeScan2 = 0x5E
    OpMakeRangeLoop = 0x5F, OpMakeRangeYield = 0x60
    OpMakeRangeCall = 0x61, OpMakeRangeTerminator = 0x62
    OpMakeRangeUnreachable = 0x63, OpMakeRangeDebug = 0x64
    OpStorePtrTko = 0x65, OpStoreViewTko = 0x66
    OpSubF = 0x67, OpSubI = 0x68
    OpMakeRangeAddF = 0x69, OpMakeRangeAddI = 0x6A
    OpMakeRangeMulF = 0x6B, OpMakeRangeMulI = 0x6C
    OpMakeRangeDivF = 0x6D, OpMakeRangeDivI = 0x6E
    OpMakeRangeRemF = 0x6F, OpMakeRangeRemI = 0x70
    OpAlloca = 0x71, OpMmaFScaled = 0x72
    OpMakeRangeShrI = 0x73, OpMakeRangeShlI = 0x74
    OpMakeRangeOrI = 0x75, OpMakeRangeXorI = 0x76
    OpMakeRangeTan = 0x77, OpMakeRangeTanH = 0x78
    OpMakeRangeAtan2 = 0x79, OpMakeRangeRemF2 = 0x7A
    OpMakeRangeMulHI = 0x7B, OpMakeRangeMulHIU = 0x7C
    OpMakeRangeMulHISS = 0x7D, OpMakeRangeMulHISU = 0x7E
    OpMakeRangePtrToPtr = 0x7F

# ############################################################
# Memory ordering semantics
# ############################################################

type
  MemoryOrdering* = enum
    MemOrderWeak = 0, MemOrderAcquire = 1, MemOrderRelease = 2
    MemOrderAcquireRelease = 3, MemOrderSequentiallyConsistent = 4

# ############################################################
# Rounding mode
# ############################################################

type
  RoundingMode* = enum
    RoundNearestEven = 0, RoundTowardPositive = 1
    RoundTowardNegative = 2, RoundTowardZero = 3
    RoundUnnecessary = 4, RoundFull = 5

# ############################################################
# IR types
# ############################################################

type
  TileElemType* = enum
    ElemI1, ElemI4, ElemI8, ElemI16, ElemI32, ElemI64
    ElemF16, ElemBF16, ElemF32, ElemTF32, ElemF64
    ElemF8E4M3FN, ElemF8E5M2, ElemF8E8M0FNU, ElemF4E2M1FN
    ElemPointer

  TileType* = ref object
    shape*: seq[int64]
    elemType*: TileElemType

  FuncType* = ref object
    inputs*: seq[TileType]
    results*: seq[TileType]

  ValueIdx* = int

  BytecodeOp* = ref object
    opcode*: Opcode
    resultTypes*: seq[TileType]
    operandIndices*: seq[ValueIdx]
    attrs*: Table[string, seq[byte]]

  BytecodeFunction* = ref object
    name*: string
    funcType*: FuncType
    body*: seq[BytecodeOp]

# ############################################################
# Bytecode module
# ############################################################

type
  BytecodeModule* = ref object
    functions*: seq[BytecodeFunction]
    types*: seq[TileType]
    typeMap*: Table[int, int]
    strings*: seq[string]
    stringMap*: Table[string, int]
    funcTypes*: seq[seq[byte]]  ## Pre-serialized Func type entries

proc newBytecodeModule*(): BytecodeModule =
  result = BytecodeModule(
    functions: @[],
    types: @[],
    typeMap: initTable[int, int](),
    strings: @[],
    stringMap: initTable[string, int](),
    funcTypes: @[]
  )

proc getTypeIndex*(m: BytecodeModule, t: TileType): int =
  let hash = (t.shape.len shl 16) or (ord(t.elemType) and 0xFFFF)
  if m.typeMap.hasKey(hash):
    return m.typeMap[hash]
  let idx = m.types.len
  m.types.add(t)
  m.typeMap[hash] = idx
  return idx

proc getStringIndex*(m: BytecodeModule, s: string): int =
  if m.stringMap.hasKey(s):
    return m.stringMap[s]
  let idx = m.strings.len
  m.strings.add(s)
  m.stringMap[s] = idx
  return idx

# ############################################################
# Encoding helpers
# ############################################################

proc writeVarInt*(s: var seq[byte], value: uint64) =
  var v = value
  while v > 0x7F:
    s.setLen(s.len + 1)
    s[s.len - 1] = ((v and 0x7F) or 0x80).uint8
    v = v shr 7
  s.setLen(s.len + 1)
  s[s.len - 1] = v.uint8

proc writeSignedVarInt*(s: var seq[byte], value: int64) =
  var zigzag = (value shl 1) xor (value shr 63)
  writeVarInt(s, zigzag.uint64)

proc writeLE32*(s: var seq[byte], value: uint32) =
  s.setLen(s.len + 4)
  s[s.len - 4] = value.uint8
  s[s.len - 3] = (value shr 8).uint8
  s[s.len - 2] = (value shr 16).uint8
  s[s.len - 1] = (value shr 24).uint8

proc writeLE16*(s: var seq[byte], value: uint16) =
  s.setLen(s.len + 2)
  s[s.len - 2] = value.uint8
  s[s.len - 1] = (value shr 8).uint8

proc writeLE64*(s: var seq[byte], value: uint64) =
  s.setLen(s.len + 8)
  for i in 0 .. 7:
    s[s.len - 8 + i] = ((value shr (i * 8)) and 0xFF).uint8
proc alignTo*(s: var seq[byte], alignment: int) =
  if alignment < 2: return
  let currentPos = s.len
  let padding = (alignment - (currentPos mod alignment)) mod alignment
  for _ in 1 .. padding:
    s.setLen(s.len + 1)
    s[s.len - 1] = AlignmentByte

proc writeSectionHeader*(s: var seq[byte], sectionId: uint8, length: uint64,
                          alignment: uint64 = 1) =
  var idAndAligned = sectionId and 0x7F'u8
  if alignment > 1:
    idAndAligned = idAndAligned or 0x80'u8
  s.setLen(s.len + 1)
  s[s.len - 1] = idAndAligned
  writeVarInt(s, length)
  if alignment > 1:
    writeVarInt(s, alignment)
    alignTo(s, alignment.int)
proc scalarTypeTag(e: TileElemType): uint64 =
  ## Map TileElemType to Rust-compatible TypeTag for scalar types.
  case e
  of ElemI1:  ord(TagToken).uint64         # Token = 17 (not I1=0, for load/store tokens)
  of ElemI4:  ord(TagInt8).uint64          # Closest Rust type
  of ElemI8:  ord(TagInt8).uint64
  of ElemI16: ord(TagInt16).uint64
  of ElemI32: ord(TagInt32).uint64
  of ElemI64: ord(TagInt64).uint64
  of ElemF16: ord(TagFloat16).uint64
  of ElemBF16:ord(TagBFloat16).uint64
  of ElemF32: ord(TagFloat32).uint64
  of ElemTF32: ord(TagTFloat32).uint64
  of ElemF64: ord(TagFloat64).uint64
  of ElemF8E4M3FN: ord(TagFloat8E4M3FN).uint64
  of ElemF8E5M2: ord(TagFloat8E5M2).uint64
  of ElemPointer: ord(TagPointer).uint64 # Only for empty-shape pointer types
  else: ord(TagFloat32).uint64             # fallback

proc writeTileTypeData*(s: var seq[byte], t: TileType, m: BytecodeModule): int =
  ## Serialize a TileType entry in Rust-compatible TileIR bytecode format.
  ##
  ## Rust TypeTag dispatch:
  ##   shape=[], scalar elem → just TypeTag varint (F32=7, I32=3, etc.)
  ##   shape=[], elem=Pointer → TagPointer(12) + pointee_type_idx
  ##   shape=[], elem=I1     → TagToken(17)
  ##   shape non-empty       → TagTile(13) + elem_type_idx + shape
  let startLen = s.len
  case t.elemType
  of ElemPointer:
    if t.shape.len == 0:
      # Pointer type: TagPointer(12) + pointee_type_idx(varint)
      writeVarInt(s, ord(TagPointer).uint64)
      let pointeeTy = TileType(shape: @[], elemType: ElemF32)
      let pointeeIdx = getTypeIndex(m, pointeeTy)
      writeVarInt(s, pointeeIdx.uint64)
    else:
      # Tile of pointers: TagTile(13) + elem_type_idx + shape
      let elemTy = TileType(shape: @[], elemType: ElemPointer)
      let elemIdx = getTypeIndex(m, elemTy)
      writeVarInt(s, ord(TagTile).uint64)
      writeVarInt(s, elemIdx.uint64)
      writeVarInt(s, t.shape.len.uint64)
      for dim in t.shape:
        writeLE64(s, dim.uint64)
  of ElemI1:
    if t.shape.len == 0:
      # Token type: TagToken(17)
      writeVarInt(s, ord(TagToken).uint64)
    else:
      # Tile of I1: TagTile(13) + elem_type_idx + shape
      let elemTy = TileType(shape: @[], elemType: ElemI1)
      let elemIdx = getTypeIndex(m, elemTy)
      writeVarInt(s, ord(TagTile).uint64)
      writeVarInt(s, elemIdx.uint64)
      writeVarInt(s, t.shape.len.uint64)
      for dim in t.shape:
        writeLE64(s, dim.uint64)
  else:
    if t.shape.len == 0:
      # Scalar type: just the TypeTag varint
      writeVarInt(s, scalarTypeTag(t.elemType))
    else:
      # Tile of scalar: TagTile(13) + elem_type_idx + shape
      let elemTy = TileType(shape: @[], elemType: t.elemType)
      let elemIdx = getTypeIndex(m, elemTy)
      writeVarInt(s, ord(TagTile).uint64)
      writeVarInt(s, elemIdx.uint64)
      writeVarInt(s, t.shape.len.uint64)
      for dim in t.shape:
        writeLE64(s, dim.uint64)
  result = s.len - startLen
  result = s.len - startLen

# ############################################################
# Header
# ############################################################

proc writeHeader*(s: var seq[byte]) =
  for b in MagicNumber:
    s.setLen(s.len + 1)
    s[s.len - 1] = b
  s.setLen(s.len + 1); s[s.len - 1] = BytecodeMajor.uint8
  s.setLen(s.len + 1); s[s.len - 1] = BytecodeMinor.uint8
  writeLE16(s, BytecodeTag.uint16)

# ############################################################
# String section
# ############################################################

proc writeStringSection*(s: var seq[byte], strings: seq[string]) =
  if strings.len == 0: return
  var buf: seq[byte] = @[]
  writeVarInt(buf, strings.len.uint64)
  alignTo(buf, 4)
  let offsetsStart = buf.len
  for _ in strings:
    writeLE32(buf, 0)
  var runningOffset = 0'u32
  var offsets: seq[uint32] = @[]
  for str in strings:
    offsets.add(runningOffset)
    for ch in str:
      buf.setLen(buf.len + 1)
      buf[buf.len - 1] = ch.uint8
    runningOffset += str.len.uint32
  for i, off in offsets:
    let pos = offsetsStart + i * 4
    for j in 0 .. 3:
      buf[pos + j] = ((off shr (j * 8)) and 0xFF).uint8
  writeSectionHeader(s, SectionString, buf.len.uint64, 4)
  s &= buf

# ############################################################
# Type section
# ############################################################

proc writeTypeSection*(s: var seq[byte], m: BytecodeModule) =
  let numTileTypes = m.types.len
  let numFuncTypes = m.funcTypes.len
  if numTileTypes == 0 and numFuncTypes == 0: return
  let totalTypes = numTileTypes + numFuncTypes
  var buf: seq[byte] = @[]
  writeVarInt(buf, totalTypes.uint64)
  alignTo(buf, 4)
  let offsetsStart = buf.len
  for _ in 1 .. totalTypes:
    writeLE32(buf, 0)
  var runningOffset = 0'u32
  var offsets: seq[uint32] = @[]
  # Write TileType entries (indices 0 .. numTileTypes-1)
  for i in 0 ..< numTileTypes:
    let t = m.types[i]
    offsets.add(runningOffset)
    let bytesWritten = writeTileTypeData(buf, t, m)
    runningOffset += bytesWritten.uint32
  # Write FuncType entries (indices numTileTypes .. totalTypes-1)
  for i in 0 ..< numFuncTypes:
    offsets.add(runningOffset)
    let ft = m.funcTypes[i]
    for b in ft:
      buf.setLen(buf.len + 1); buf[buf.len - 1] = b
    runningOffset += ft.len.uint32
  # Patch offsets
  for i, off in offsets:
    let pos = offsetsStart + i * 4
    for j in 0 .. 3:
      buf[pos + j] = ((off shr (j * 8)) and 0xFF).uint8
  writeSectionHeader(s, SectionType, buf.len.uint64, 4)
  s &= buf

# ############################################################
# Operation encoding
# ############################################################

proc writeOperation*(s: var seq[byte], op: BytecodeOp, m: BytecodeModule) =
  ## Write operation in Rust-compatible TileIR bytecode format.
  ## Type order: opcode + [resultTypes] + [flags+inlineAttrs] + [operandIndices]
  writeVarInt(s, ord(op.opcode).uint64)
  for rt in op.resultTypes:
    let idx = getTypeIndex(m, rt)
    writeVarInt(s, idx.uint64)
  # Op-specific encoding (flags + inline attrs) after result types, before operands.
  case op.opcode
  of OpLoadPtrTko, OpStorePtrTko:
    # flags: bit 0 = memory_scope, bit 1 = optimization_hints
    var flags: uint64 = 0
    if "memory_scope" in op.attrs:
      flags = flags or 1
    if "optimization_hints" in op.attrs:
      flags = flags or 2
    writeVarInt(s, flags)
    # memory_ordering_semantics (required inline attr, default Weak=0)
    if "memory_ordering" in op.attrs and op.attrs["memory_ordering"].len > 0:
      writeVarInt(s, op.attrs["memory_ordering"][0].uint64)
    else:
      writeVarInt(s, 0)
    if (flags and 1) != 0:
      writeVarInt(s, op.attrs["memory_scope"][0].uint64)
    if (flags and 2) != 0:
      writeVarInt(s, op.attrs["optimization_hints"][0].uint64)
    # operands
    for operandIdx in op.operandIndices:
      writeVarInt(s, operandIdx.uint64)
  of OpFma:
    # Fma: flags(flush_to_zero) + rounding_mode + operands
    var flags: uint64 = 0
    if "flush_to_zero" in op.attrs:
      flags = flags or 1
    writeVarInt(s, flags)
    # rounding_mode (required inline attr, default RoundFull=5)
    if "rounding_mode" in op.attrs and op.attrs["rounding_mode"].len > 0:
      writeVarInt(s, op.attrs["rounding_mode"][0].uint64)
    else:
      writeVarInt(s, 5)
    for operandIdx in op.operandIndices:
      writeVarInt(s, operandIdx.uint64)
  else:
    # Generic: just operands (no flags/attrs)
    for operandIdx in op.operandIndices:
      writeVarInt(s, operandIdx.uint64)

proc writeFuncBodyBytes*(ops: seq[BytecodeOp], m: BytecodeModule): seq[byte] =
  result = @[]
  for op in ops:
    writeOperation(result, op, m)

proc writeFuncSection*(s: var seq[byte], m: BytecodeModule) =
  if m.functions.len == 0: return
  var buf: seq[byte] = @[]
  writeVarInt(buf, m.functions.len.uint64)
  for f in m.functions:
    let nameIdx = getStringIndex(m, f.name)
    writeVarInt(buf, nameIdx.uint64)
    # Create Func type entry for function signature.
    var funcTypeData: seq[byte] = @[]
    writeVarInt(funcTypeData, 16'u64)  # TypeTagFunc = 16
    writeVarInt(funcTypeData, f.funcType.inputs.len.uint64)
    for inp in f.funcType.inputs:
      let inpIdx = getTypeIndex(m, inp)
      writeVarInt(funcTypeData, inpIdx.uint64)
    writeVarInt(funcTypeData, f.funcType.results.len.uint64)
    for res in f.funcType.results:
      let resIdx = getTypeIndex(m, res)
      writeVarInt(funcTypeData, resIdx.uint64)
    let sigIdx = m.types.len + m.funcTypes.len
    m.funcTypes.add(funcTypeData)
    writeVarInt(buf, sigIdx.uint64)
    buf.setLen(buf.len + 1)
    buf[buf.len - 1] = 0x02'u8  # kernel entry flag
    writeVarInt(buf, 0)  # UnknownLoc
    let bodyBytes = writeFuncBodyBytes(f.body, m)
    writeVarInt(buf, bodyBytes.len.uint64)
    buf &= bodyBytes
  writeSectionHeader(s, SectionFunc, buf.len.uint64, 8)
  s &= buf

# ############################################################
# Full bytecode generation
# ############################################################

proc toBytecode*(m: BytecodeModule): seq[byte] =
  # Pre-register all scalar elemTypes so writeTileTypeData doesn't mutate m.types mid-iteration.
  # Each TileType with non-empty shape references a scalar elemType (shape=[]) that must
  # already exist in m.types when writeTypeSection captures numTileTypes.
  for t in m.types:
    if t.shape.len > 0:
      let scalarTy = TileType(shape: @[], elemType: t.elemType)
      discard getTypeIndex(m, scalarTy)  # idempotent lookup
  m.funcTypes = @[]
  result = @[]
  writeHeader(result)
  writeFuncSection(result, m)
  writeStringSection(result, m.strings)
  writeTypeSection(result, m)
  result.setLen(result.len + 1)
  result[result.len - 1] = SectionEnd
proc writeBytecodeToFile*(m: BytecodeModule, path: string) =
  let bc = toBytecode(m)
  var s = newStringOfCap(bc.len)
  for i in 0 ..< bc.len:
    s.add(chr(bc[i]))
  writeFile(path, s)
  echo "[cutile] Wrote ", bc.len, " bytes to ", path

# ############################################################
# Convenience helpers
# ############################################################

proc tileScalar*(elemType: TileElemType): TileType =
  TileType(shape: @[], elemType: elemType)

proc tile1D*(size: int64, elemType: TileElemType): TileType =
  TileType(shape: @[size], elemType: elemType)

proc tile2D*(rows, cols: int64, elemType: TileElemType): TileType =
  TileType(shape: @[rows, cols], elemType: elemType)

proc ptrType*(elemType: TileElemType = ElemF32): TileType =
  TileType(shape: @[], elemType: ElemPointer)

proc tokenType*(): TileType =
  TileType(shape: @[], elemType: ElemI1)
