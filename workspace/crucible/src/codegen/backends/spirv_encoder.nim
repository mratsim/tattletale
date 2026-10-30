# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ═══════════════════════════════════════════════════════════════════════
#  SPIR-V binary encoder
# ═══════════════════════════════════════════════════════════════════════

# Flavor-agnostic core, no Vulkan/OpenCL assumptions.
# Instruction encoding, first word: `wordCount shl 16 or opcode`.
# String operands: UTF-8 bytes + NUL, zero-padded to a word boundary, little-endian per word.

import std/algorithm
import std/macros
import std/tables

# ═══════════════════════════════════════════════════════════════════════
#  Opcode table
# ═══════════════════════════════════════════════════════════════════════

type
  SpvKind = enum
    ## Operand kinds, named after the core grammar's `kind` field
    spvIdResult
    spvIdResultType
    spvIdRef
    spvIdScope
    spvIdMemorySemantics
    spvLiteralInteger
    spvLiteralString
    spvLiteralNumber     ## grammar kind LiteralContextDependentNumber
    spvFPEncoding
    spvStorageClass
    spvCapability
    spvDecoration
    spvBuiltIn
    spvExecutionModel
    spvExecutionMode
    spvAddressingModel
    spvMemoryModel
    spvFunctionControl
    spvSelectionControl
    spvLoopControl
    spvMemoryAccess
    spvGroupOperation

  SpvQuant = enum
    ## Operand quantifier, named after the core grammar's `quantifier` field
    qOne   ## exactly one operand (grammar "")
    qOpt   ## zero or one operand (grammar "?")
    qMany  ## zero or more operands (grammar "*")

  SpvOperand = object
    kind*: SpvKind
    quant*: SpvQuant

func o(kind: SpvKind): SpvOperand {.inline.} =
  SpvOperand(kind: kind, quant: qOne)

func oOpt(kind: SpvKind): SpvOperand {.inline.} =
  SpvOperand(kind: kind, quant: qOpt)

func oMany(kind: SpvKind): SpvOperand {.inline.} =
  SpvOperand(kind: kind, quant: qMany)

macro genSpvOpcodeTable(defs: untyped) =
  ## Declarative opcode table in the glyph style, one row per opcode:
  ##   op OpName: <opcode>: [o(kind1), oOpt(kind2), ...]
  ## - generates the `SpvOp` enum, member value = opcode number
  ## - generates the `spvInstOperands` encoding lookup, rows sorted by opcode
  ## - a duplicate opcode or opcode name fails at macro expansion
  var
    seenOpcodes: Table[int, string]
    seenNames: Table[string, int]
    rows: seq[tuple[opcode: int, name: string, operands: NimNode]]
  for row in defs:
    row.expectKind nnkCommand
    doAssert row[0].eqIdent "op"
    row[1].expectKind nnkIdent
    row[2].expectKind nnkStmtList
    doAssert row.len == 3
    let name = $row[1]
    if name in seenNames:
      error "opcode name " & name & " declared twice", row[1]
    seenNames[name] = 0
    let body = row[2]
    var opcode: int
    var operands = newNimNode(nnkBracket, row)
    if body[0].kind == nnkIntLit:
      opcode = body[0].intVal.int
    else:
      body[0].expectKind nnkCall
      body[0][0].expectKind nnkIntLit
      opcode = body[0][0].intVal.int
      body[0][1].expectKind nnkStmtList
      let br = body[0][1][0]
      br.expectKind nnkBracket
      for k in br:
        operands.add k
    if opcode in seenOpcodes:
      error "opcode " & $opcode & " already used by " & seenOpcodes[opcode], row[2][0]
    seenOpcodes[opcode] = name
    rows.add (opcode, name, operands)
  rows.sort(proc(x, y: tuple[opcode: int, name: string, operands: NimNode]): int =
    cmp(x.opcode, y.opcode))
  var enumTy = nnkEnumTy.newTree(newEmptyNode())
  for r in rows:
    enumTy.add nnkEnumFieldDef.newTree(ident(r.name), newLit(r.opcode))
  let typeSection = nnkTypeSection.newTree(
    nnkTypeDef.newTree(
      nnkPostfix.newTree(ident"*", ident"SpvOp"),
      newEmptyNode(),
      enumTy))
  var table = nnkCaseStmt.newTree(ident"op")
  for r in rows:
    table.add nnkOfBranch.newTree(ident(r.name), nnkCall.newTree(ident"@", r.operands))
  let procSection = nnkProcDef.newTree(
    ident"spvInstOperands",
    newEmptyNode(), newEmptyNode(),
    nnkFormalParams.newTree(
      nnkBracketExpr.newTree(ident"seq", ident"SpvOperand"),
      nnkIdentDefs.newTree(ident"op", ident"SpvOp", newEmptyNode())),
    newEmptyNode(), newEmptyNode(),
    table)
  result = nnkStmtList.newTree(typeSection, procSection)

genSpvOpcodeTable:
  op OpNop: 0
  op OpUndef: 1: [o(spvIdResultType), o(spvIdResult)]
  op OpExtension: 10: [o(spvLiteralString)]
  op OpExtInstImport: 11: [o(spvIdResult), o(spvLiteralString)]
  op OpMemoryModel: 14: [o(spvAddressingModel), o(spvMemoryModel)]
  op OpEntryPoint: 15: [o(spvExecutionModel), o(spvIdRef), o(spvLiteralString), oMany(spvIdRef)]
  # Per-mode literal operands (LocalSize x y z, ...) are defined by the spec's
  # ExecutionMode enumerants and absent from the grammar's operand list
  op OpExecutionMode: 16: [o(spvIdRef), o(spvExecutionMode)]
  op OpCapability: 17: [o(spvCapability)]
  op OpConstantTrue: 41: [o(spvIdResultType), o(spvIdResult)]
  op OpConstantFalse: 42: [o(spvIdResultType), o(spvIdResult)]
  op OpConstant: 43: [o(spvIdResultType), o(spvIdResult), o(spvLiteralNumber)]
  op OpConstantNull: 46: [o(spvIdResultType), o(spvIdResult)]
  # Array Length is an <id> operand holding an integer OpConstant result
  op OpTypeArray: 28: [o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpTypeBool: 20: [o(spvIdResult)]
  # FPEncoding exists since SPIR-V 1.4, an absent word means IEEE754 encoding
  op OpTypeFloat: 22: [o(spvIdResult), o(spvLiteralInteger), oOpt(spvFPEncoding)]
  op OpTypeFunction: 33: [o(spvIdResult), o(spvIdRef), oMany(spvIdRef)]
  op OpTypeInt: 21: [o(spvIdResult), o(spvLiteralInteger), o(spvLiteralInteger)]
  op OpTypePointer: 32: [o(spvIdResult), o(spvStorageClass), o(spvIdRef)]
  op OpTypeRuntimeArray: 29: [o(spvIdResult), o(spvIdRef)]
  op OpTypeStruct: 30: [o(spvIdResult), oMany(spvIdRef)]
  op OpTypeVector: 23: [o(spvIdResult), o(spvIdRef), o(spvLiteralInteger)]
  op OpTypeVoid: 19: [o(spvIdResult)]
  # Per-decoration extra operands (BuiltIn, Offset) come from the spec's
  # Decoration enumerants, the grammar's operand list does not model them
  op OpDecorate: 71: [o(spvIdRef), o(spvDecoration)]
  op OpMemberDecorate: 72: [o(spvIdRef), o(spvLiteralInteger), o(spvDecoration)]
  op OpFunction: 54: [o(spvIdResultType), o(spvIdResult), o(spvFunctionControl), o(spvIdRef)]
  op OpFunctionParameter: 55: [o(spvIdResultType), o(spvIdResult)]
  # Arguments are IdRefs after the function id. Function-storage pointer args
  # (an OpVariable Function result passed directly) validate and execute
  # under the Vulkan target environment without any VariablePointers capability.
  # A pointer-to-pointer Function-storage VARIABLE is categorically invalid.
  op OpFunctionCall: 57: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), oMany(spvIdRef)]
  op OpFunctionEnd: 56
  op OpVariable: 59: [o(spvIdResultType), o(spvIdResult), o(spvStorageClass), oOpt(spvIdRef)]
  op OpLoad: 61: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), oOpt(spvMemoryAccess)]
  op OpStore: 62: [o(spvIdRef), o(spvIdRef), oOpt(spvMemoryAccess)]
  op OpAccessChain: 65: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), oMany(spvIdRef)]
  op OpCompositeConstruct: 80: [o(spvIdResultType), o(spvIdResult), oMany(spvIdRef)]
  op OpCompositeExtract: 81: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), oMany(spvLiteralInteger)]
  op OpSNegate: 126: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpFNegate: 127: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpIAdd: 128: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpFAdd: 129: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpISub: 130: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpFSub: 131: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpIMul: 132: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpFMul: 133: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpUDiv: 134: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpSDiv: 135: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpFDiv: 136: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpUMod: 137: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  # SRem: remainder of the dividend sign, Nim `mod` semantics. SMod: divisor sign
  op OpSRem: 138: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpSMod: 139: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpConvertFToU: 109: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpConvertFToS: 110: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpConvertSToF: 111: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpConvertUToF: 112: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpUConvert: 113: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpSConvert: 114: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpFConvert: 115: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpConvertUToPtr: 120: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpBitcast: 124: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  # OpLogical* operate on bool operands, OpIEqual / OpINotEqual require int
  op OpLogicalEqual: 164: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpLogicalNotEqual: 165: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpLogicalOr: 166: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpLogicalAnd: 167: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpLogicalNot: 168: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpSelect: 169: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef), o(spvIdRef)]
  op OpIEqual: 170: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpINotEqual: 171: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpUGreaterThan: 172: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpSGreaterThan: 173: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpUGreaterThanEqual: 174: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpSGreaterThanEqual: 175: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpULessThan: 176: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpSLessThan: 177: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpULessThanEqual: 178: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpSLessThanEqual: 179: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpFOrdEqual: 180: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpFOrdNotEqual: 182: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpFOrdLessThan: 184: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpFOrdGreaterThan: 186: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpFOrdLessThanEqual: 188: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpFOrdGreaterThanEqual: 190: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  # Bit and shift ops over integer operands, the shift amount is unsigned
  op OpShiftRightLogical: 194: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpShiftRightArithmetic: 195: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpShiftLeftLogical: 196: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpBitwiseOr: 197: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpBitwiseXor: 198: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpBitwiseAnd: 199: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdRef)]
  op OpNot: 200: [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]
  op OpControlBarrier: 224: [o(spvIdScope), o(spvIdScope), o(spvIdMemorySemantics)]
  op OpLabel: 248: [o(spvIdResult)]
  op OpBranch: 249: [o(spvIdRef)]
  op OpBranchConditional: 250: [o(spvIdRef), o(spvIdRef), o(spvIdRef), oMany(spvLiteralInteger)]
  op OpLoopMerge: 246: [o(spvIdRef), o(spvIdRef), o(spvLoopControl)]
  op OpSelectionMerge: 247: [o(spvIdRef), o(spvSelectionControl)]
  op OpReturn: 253
  op OpReturnValue: 254: [o(spvIdRef)]
  # GroupOperation enumerant (Reduce/InclusiveScan/ExclusiveScan) sits
  # between Scope and the value, the optional trailing IdRef is the ClusterSize
  op OpGroupNonUniformFAdd: 350: [o(spvIdResultType), o(spvIdResult), o(spvIdScope), o(spvGroupOperation), o(spvIdRef), oOpt(spvIdRef)]
  # Shuffle ops carry no GroupOperation word: the last operand is the lane id
  # (Shuffle) or the xor mask (ShuffleXor), both plain IdRef
  op OpGroupNonUniformShuffle: 345: [o(spvIdResultType), o(spvIdResult), o(spvIdScope), o(spvIdRef), o(spvIdRef)]
  op OpGroupNonUniformShuffleXor: 346: [o(spvIdResultType), o(spvIdResult), o(spvIdScope), o(spvIdRef), o(spvIdRef)]
  # SPV_EXT_shader_atomic_float_add: fp atomic add, the extension string pairs with the AtomicFloat32AddEXT capability
  op OpAtomicFAddEXT: 6035: [o(spvIdResultType), o(spvIdResult), o(spvIdRef), o(spvIdScope), o(spvIdMemorySemantics), o(spvIdRef)]

# ═══════════════════════════════════════════════════════════════════════
#  Enumerants
# ═══════════════════════════════════════════════════════════════════════

const
  ## Capability values (core grammar ValueEnum), compute subset
  capShader = 1'u32
  capFloat16 = 9'u32
  capFloat64 = 10'u32
  capInt64 = 11'u32
  capInt8 = 39'u32
  capInt16 = 22'u32
  capPhysicalStorageBufferAddresses* = 5347'u32
  capGroupNonUniformArithmetic = 63'u32
  capGroupNonUniformShuffle = 65'u32
  capAtomicFloat32AddEXT = 6033'u32

const
  ## Decoration values (core grammar ValueEnum), buffer subset
  decBlock = 2'u32
  decArrayStride = 6'u32
  decBuiltIn = 11'u32
  decOffset = 35'u32

const
  ## BuiltIn values (core grammar ValueEnum), compute subset
  biNumWorkgroups = 24'u32
  biWorkgroupSize = 25'u32
  biWorkgroupId = 26'u32
  biLocalInvocationId = 27'u32
  biGlobalInvocationId = 28'u32
  biLocalInvocationIndex = 29'u32

const
  ## StorageClass values (core grammar ValueEnum)
  scInput = 1'u32
  scFunction = 7'u32
  scPushConstant = 9'u32
  scPhysicalStorageBuffer = 5349'u32

const
  ## ExecutionModel values (core grammar ValueEnum)
  emGLCompute = 5'u32

const
  ## ExecutionMode values (core grammar ValueEnum)
  modeLocalSize* = 17'u32

const
  ## MemoryModel values (core grammar ValueEnum)
  mmGLSL450 = 1'u32

  ## AddressingModel values (core grammar ValueEnum)
  amLogical = 0'u32
  amPhysicalStorageBuffer64 = 5348'u32

const
  ## GroupOperation values (core grammar ValueEnum), OpGroupNonUniform* operands
  groupOpReduce = 0'u32

  ## Scope constants (grammar kind IdScope), OpGroupNonUniform* operand words
  scopeDevice = 1'u32
  scopeSubgroup = 3'u32

  ## MemoryAccess bit values (core grammar BitEnum), Aligned takes a trailing
  ## literal alignment word the grammar operand list does not model
  maAligned = 2'u32

const
  ## MemorySemantics bit values (core grammar BitEnum), the operand words
  ## for the OpAtomic* instructions
  semAcquireRelease = 0x8'u32
  semUniformMemory = 0x40'u32

const
  ## Control bit masks, all-None words for OpFunction / OpSelectionMerge /
  ## OpLoopMerge / memory operands
  fcNone = 0'u32
  selNone = 0'u32
  lcNone = 0'u32
  lcUnroll = 1'u32
  ## LoopControlUnroll bit of OpLoopMerge, `{.unroll.}` on the loop

# ═══════════════════════════════════════════════════════════════════════
#  Module sections and word stream
# ═══════════════════════════════════════════════════════════════════════

type
  SpvSection = enum
    ## Logical-layout sections in SPIR-V §2.4 order
    ssCapabilities
    ssExtensions
    ssExtInstImports
    ssMemoryModel
    ssEntryPoints
    ssExecutionModes
    ssDebug
    ssAnnotations
    ssTypesConstantsGlobals
    ssFunctions

  SpvModule = object
    version*: uint32      ## header version word, e.g. 0x00010300 for SPIR-V 1.3
    nextId*: uint32       ## next unallocated result id, ids start at 1
    sections*: array[SpvSection, seq[uint32]]

const spvMagic = 0x07230203'u32
# Module baseline: SPIR-V 1.3, encoded in the spvVersion13 version word.
# That baseline implies semantics beyond the number:
# - the entry-point interface lists Input/Output variables only
# - OpTypeFloat omits the FPEncoding operand, the 1.3 form
# - PhysicalStorageBufferAddresses is extension-gated
# - push-constant members narrower than 32 bits refuse, they need the SPIR-V 1.5 storage capabilities
#
# Bumping the constant to 1.4+ requires the entry-point interface to list
# every referenced global variable, the push-constant block variable included.
const spvVersion13 = 0x00010300'u32
const spvGeneratorNone = 0'u32
const spvSchemaNone = 0'u32

func initSpvModule(): SpvModule =
  SpvModule(version: spvVersion13, nextId: 1)

func addWord(m: var SpvModule, section: SpvSection, word: uint32) =
  m.sections[section].add word

func addWords(m: var SpvModule, section: SpvSection, words: openArray[uint32]) =
  for w in words:
    m.sections[section].add w

func getStringWords(s: string): int =
  (s.len + 4) div 4

func addString(m: var SpvModule, section: SpvSection, s: string) =
  var bytes = newSeq[uint8](s.len + 1)
  for i in 0 ..< s.len:
    bytes[i] = uint8(s[i])
  while bytes.len mod 4 != 0:
    bytes.add 0'u8
  var i = 0
  while i < bytes.len:
    let w = uint32(bytes[i]) or uint32(bytes[i + 1]) shl 8 or uint32(bytes[i + 2]) shl 16 or uint32(bytes[i + 3]) shl 24
    m.sections[section].add w
    i += 4

func allocId(m: var SpvModule): uint32 =
  result = m.nextId
  inc m.nextId

func startInst(m: var SpvModule, section: SpvSection, opcode: SpvOp, operandWords: int) =
  # string operands and mixed word/non-word operand lists append
  # their operand words manually after the header word
  m.addWord(section, uint32(operandWords + 1) shl 16 or uint32(opcode))

func emitInstSeq(m: var SpvModule, section: SpvSection, opcode: SpvOp, operands: openArray[uint32]) =
  m.startInst(section, opcode, operands.len)
  m.addWords(section, operands)

func finalize(m: SpvModule): seq[uint32] =
  # Header words: magic, version, generator (0), bound, schema (0).
  # Bound word is nextId at finalize time, covering all allocated ids.
  result = @[spvMagic, m.version, spvGeneratorNone, m.nextId, spvSchemaNone]
  for section in SpvSection:
    result.add m.sections[section]

# ═══════════════════════════════════════════════════════════════════════
#  Builder
# ═══════════════════════════════════════════════════════════════════════

type
  SpvMemoKey = tuple[opcode: SpvOp, words: seq[uint32]]
    ## Dedup key for a type or constant: opcode plus the operand words

  SpvBuilder = object
    module*: SpvModule
    memo*: Table[SpvMemoKey, uint32]
    scalarWidths*: Table[uint32, int]
      ## Bit width of every emitted OpTypeInt/OpTypeFloat, keyed by type id.
      ## OpConstant literal word counts derive from it

func initSpvBuilder(): SpvBuilder =
  SpvBuilder(module: initSpvModule())

func memoInst(b: var SpvBuilder, opcode: SpvOp, operands: openArray[uint32]): uint32 =
  let key: SpvMemoKey = (opcode, @operands)
  if key in b.memo:
    return b.memo[key]
  result = b.module.allocId()
  var words = @[result]
  for w in operands:
    words.add w
  b.module.emitInstSeq(ssTypesConstantsGlobals, opcode, words)
  b.memo[key] = result

func memoTypedInst(b: var SpvBuilder, opcode: SpvOp, operands: openArray[uint32]): uint32 =
  let key: SpvMemoKey = (opcode, @operands)
  if key in b.memo:
    return b.memo[key]
  result = b.module.allocId()
  var words = @[operands[0], result]
  for i in 1 ..< operands.len:
    words.add operands[i]
  b.module.emitInstSeq(ssTypesConstantsGlobals, opcode, words)
  b.memo[key] = result

func freshInst(b: var SpvBuilder, opcode: SpvOp, operands: openArray[uint32]): uint32 =
  result = b.module.allocId()
  var words = @[operands[0], result]
  for i in 1 ..< operands.len:
    words.add operands[i]
  b.module.emitInstSeq(ssFunctions, opcode, words)

func emitCapability(b: var SpvBuilder, capability: uint32) =
  b.module.emitInstSeq(ssCapabilities, OpCapability, @[capability])

func emitMemoryModel(b: var SpvBuilder, addressingModel, memoryModel: uint32) =
  b.module.emitInstSeq(ssMemoryModel, OpMemoryModel, @[addressingModel, memoryModel])

func emitEntryPoint(b: var SpvBuilder, executionModel, entryPoint: uint32, name: string, iface: openArray[uint32]) =
  # Interface lists the global Input/Output variables the entry point reads:
  # - Vulkan requires every such variable listed
  # - an empty list is valid when the entry point reads none
  # - the Input/Output-only interface matches the spvVersion13 baseline,
  #   SPIR-V 1.4+ requires listing every referenced global variable,
  #   push-constant block included
  b.module.startInst(ssEntryPoints, OpEntryPoint, 2 + getStringWords(name) + iface.len)
  b.module.addWords(ssEntryPoints, @[executionModel, entryPoint])
  b.module.addString(ssEntryPoints, name)
  b.module.addWords(ssEntryPoints, iface)

func emitExecutionMode(b: var SpvBuilder, entryPoint, mode: uint32, literals: openArray[uint32] = []) =
  var ops = @[entryPoint, mode]
  for w in literals:
    ops.add w
  b.module.emitInstSeq(ssExecutionModes, OpExecutionMode, ops)

func emitDecorate(b: var SpvBuilder, target, decoration: uint32, literals: openArray[uint32] = []) =
  var ops = @[target, decoration]
  for w in literals:
    ops.add w
  b.module.emitInstSeq(ssAnnotations, OpDecorate, ops)

func emitMemberDecorate(b: var SpvBuilder, target, member, decoration: uint32, literals: openArray[uint32] = []) =
  var ops = @[target, member, decoration]
  for w in literals:
    ops.add w
  b.module.emitInstSeq(ssAnnotations, OpMemberDecorate, ops)

func opVariable(b: var SpvBuilder, resultType, storageClass: uint32): uint32 =
  result = b.module.allocId()
  b.module.emitInstSeq(ssTypesConstantsGlobals, OpVariable, @[resultType, result, storageClass])

func tyVoid(b: var SpvBuilder): uint32 =
  b.memoInst(OpTypeVoid, @[])

func tyBool(b: var SpvBuilder): uint32 =
  b.memoInst(OpTypeBool, @[])

func recordScalarWidth(b: var SpvBuilder, ty: uint32, width: int) =
  b.scalarWidths[uint32(ty)] = width

func getScalarWidth(b: SpvBuilder, ty: uint32): int =
  doAssert ty in b.scalarWidths, "constant over a type id with no recorded width: " & $ty
  b.scalarWidths[ty]

func tyInt(b: var SpvBuilder, width: int, signedness: bool): uint32 =
  result = b.memoInst(OpTypeInt, @[uint32(width), uint32(signedness)])
  b.recordScalarWidth(result, width)

func tyFloat(b: var SpvBuilder, width: int): uint32 =
  result = b.memoInst(OpTypeFloat, @[uint32(width)])
  b.recordScalarWidth(result, width)

func tyVector(b: var SpvBuilder, componentType: uint32, componentCount: int): uint32 =
  b.memoInst(OpTypeVector, @[componentType, uint32(componentCount)])

func tyPointer(b: var SpvBuilder, storageClass, pointeeType: uint32): uint32 =
  b.memoInst(OpTypePointer, @[storageClass, pointeeType])

func tyFunction(b: var SpvBuilder, returnType: uint32, paramTypes: openArray[uint32]): uint32 =
  var ops = @[returnType]
  for p in paramTypes:
    ops.add p
  b.memoInst(OpTypeFunction, ops)

func constUint(b: var SpvBuilder, ty: uint32, value: uint64): uint32 =
  ## Emits OpConstant holding the bit pattern of `value`:
  ## - one literal word for 32-bit and narrower types
  ## - two words (low half first) for 64-bit types, even when the bit pattern
  ##   fits one word: the word count tracks the type width, never the value
  if b.getScalarWidth(ty) > 32:
    b.memoTypedInst(OpConstant, @[ty, uint32(value), uint32(value shr 32)])
  else:
    let width = b.getScalarWidth(ty)
    let mask = (1'u64 shl uint64(width)) - 1
    doAssert value <= mask, "constant bit pattern overflows the " & $width & "-bit type " & $ty
    b.memoTypedInst(OpConstant, @[ty, uint32(value)])

func constBool(b: var SpvBuilder, ty: uint32, value: bool): uint32 =
  let op = if value: OpConstantTrue else: OpConstantFalse
  b.memoTypedInst(op, @[ty])

# ═══════════════════════════════════════════════════════════════════════
#  Function body
# ═══════════════════════════════════════════════════════════════════════

func opFunction(b: var SpvBuilder, returnType, functionType: uint32): uint32 =
  b.freshInst(OpFunction, @[returnType, fcNone, functionType])

func opFunctionParameter(b: var SpvBuilder, resultType: uint32): uint32 =
  b.freshInst(OpFunctionParameter, @[resultType])

func opLabel(b: var SpvBuilder): uint32 =
  result = b.module.allocId()
  b.module.emitInstSeq(ssFunctions, OpLabel, @[result])

func opBranch(b: var SpvBuilder, targetLabel: uint32) =
  b.module.emitInstSeq(ssFunctions, OpBranch, @[targetLabel])

func opBranchConditional(b: var SpvBuilder, condition, trueLabel, falseLabel: uint32) =
  b.module.emitInstSeq(ssFunctions, OpBranchConditional, @[condition, trueLabel, falseLabel])

func opSelectionMerge(b: var SpvBuilder, mergeBlock: uint32, selectionControl: uint32) =
  b.module.emitInstSeq(ssFunctions, OpSelectionMerge, @[mergeBlock, selectionControl])

func opLoopMerge(b: var SpvBuilder, mergeBlock, continueTarget: uint32, loopControl: uint32) =
  b.module.emitInstSeq(ssFunctions, OpLoopMerge, @[mergeBlock, continueTarget, loopControl])

func opReturn(b: var SpvBuilder) =
  b.module.emitInstSeq(ssFunctions, OpReturn, @[])

func opReturnValue(b: var SpvBuilder, value: uint32) =
  b.module.emitInstSeq(ssFunctions, OpReturnValue, @[value])

func opFunctionEnd(b: var SpvBuilder) =
  b.module.emitInstSeq(ssFunctions, OpFunctionEnd, @[])

func opLoad(b: var SpvBuilder, resultType, pointer: uint32): uint32 =
  b.freshInst(OpLoad, @[resultType, pointer])

func opStore(b: var SpvBuilder, pointer, value: uint32) =
  b.module.emitInstSeq(ssFunctions, OpStore, @[pointer, value])

func opAccessChain(b: var SpvBuilder, resultType, base: uint32, indices: openArray[uint32]): uint32 =
  var ops = @[resultType, base]
  for i in indices:
    ops.add i
  b.freshInst(OpAccessChain, ops)

func opConvertUToPtr(b: var SpvBuilder, resultType, source: uint32): uint32 =
  b.freshInst(OpConvertUToPtr, @[resultType, source])

func opCompositeExtract(b: var SpvBuilder, resultType, composite: uint32, indices: openArray[int]): uint32 =
  var ops = @[resultType, composite]
  for i in indices:
    ops.add uint32(i)
  b.freshInst(OpCompositeExtract, ops)

func opSelect(b: var SpvBuilder, resultType, condition, object1, object2: uint32): uint32 =
  b.freshInst(OpSelect, @[resultType, condition, object1, object2])

func opVariableFn(b: var SpvBuilder, resultType: uint32): uint32 =
  ## Function-scope variables must sit in the entry block
  result = b.module.allocId()
  b.module.emitInstSeq(ssFunctions, OpVariable, @[resultType, result, scFunction])

func freshLabel(b: var SpvBuilder): uint32 =
  b.module.allocId()

func opLabelAt(b: var SpvBuilder, label: uint32) =
  b.module.emitInstSeq(ssFunctions, OpLabel, @[label])

func emitExtension(b: var SpvBuilder, name: string) =
  b.module.startInst(ssExtensions, OpExtension, getStringWords(name))
  b.module.addString(ssExtensions, name)

func opLoadAligned(b: var SpvBuilder, resultType, pointer, alignment: uint32): uint32 =
  ## Vulkan requires the Aligned memory access on every load through PhysicalStorageBuffer pointers
  b.freshInst(OpLoad, @[resultType, pointer, maAligned, alignment])

func opStoreAligned(b: var SpvBuilder, pointer, value, alignment: uint32) =
  b.module.emitInstSeq(ssFunctions, OpStore, @[pointer, value, maAligned, alignment])
