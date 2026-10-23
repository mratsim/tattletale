# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## SPIR-V encoder core (flavor-agnostic): grammar tables, word stream, builders
##
## Layer stack, bottom-up (all flavor-agnostic, no Vulkan/OpenCL assumptions):
## - opcode rows cover the compute subset, execution-model rows carry
##   the full grammar enumerant set (vertex/fragment included as vocabulary)
## - `verifyGrammar` checks each opcode row against spirv.core.grammar.json
## - `SpvModule` buffers instructions into sections (SPIR-V §2.4 layout order)
## - `finalize` flattens all sections behind the 5-word header, typed
##   wrappers sit over `memoInst` (dedup) and `freshInst` (body ops)
##
## Flavor modules drive emission, the Vulkan environment flavor lives in `spirv_shader_lang`.
##
## Instruction encoding (first word): `wordCount shl 16 or opcode`.
## String operands: UTF-8 bytes + NUL, zero-padded to a word boundary,
## little-endian per word.
import std/json
import std/tables

##
## spvInstTable transcribes the core grammar (spirv.core.grammar.json):
## - the opcode number per instruction
## - operand kinds and quantifiers per instruction
## - the GLCompute-kernel opcode subset only
## Enum-valued operands (Capability, StorageClass, ...) are raw uint32
## constants with values taken from the grammar.
## `verifyGrammar` checks the table against the grammar JSON at test time.


type
  SpvKind* = enum
    ## Operand kind, named after the grammar's `kind` field
    spvIdResult
    spvIdResultType
    spvIdRef
    spvIdScope
    spvIdMemorySemantics
    spvLiteralInteger
    spvLiteralString
    spvLiteralNumber     ## grammar kind LiteralContextDependentNumber
    spvFPEncoding
    spvPairLiteralIntegerIdRef
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

  SpvQuant* = enum
    ## Operand quantifier from the grammar's `quantifier` field
    qOne   ## exactly one operand (grammar "")
    qOpt   ## zero or one operand (grammar "?")
    qMany  ## zero or more operands (grammar "*")

  SpvOperand* = object
    kind*: SpvKind
    quant*: SpvQuant

  SpvInstDef* = object
    name*: string
    opcode*: uint32
    operands*: seq[SpvOperand]

const spvKindGrammarNames*: array[SpvKind, string] = [
  "IdResult", "IdResultType", "IdRef", "IdScope", "IdMemorySemantics",
  "LiteralInteger", "LiteralString", "LiteralContextDependentNumber",
  "FPEncoding", "PairLiteralIntegerIdRef", "StorageClass", "Capability", "Decoration",
  "BuiltIn", "ExecutionModel", "ExecutionMode", "AddressingModel",
  "MemoryModel", "FunctionControl", "SelectionControl", "LoopControl",
  "MemoryAccess", "GroupOperation"]

func o*(kind: SpvKind): SpvOperand {.inline.} =
  ## Operand required exactly once
  SpvOperand(kind: kind, quant: qOne)

func oOpt*(kind: SpvKind): SpvOperand {.inline.} =
  ## Optional operand
  SpvOperand(kind: kind, quant: qOpt)

func oMany*(kind: SpvKind): SpvOperand {.inline.} =
  ## Operand repeated zero or more times
  SpvOperand(kind: kind, quant: qMany)

func instDef*(name: static string, opcode: uint32, operands: varargs[SpvOperand]): SpvInstDef =
  ## Table row, the opcode number plus operand kinds with quantifiers
  SpvInstDef(name: name, opcode: opcode, operands: @operands)

const spvInstTable*: array[86, SpvInstDef] = [
  instDef("OpNop", 0),
  instDef("OpUndef", 1, [o(spvIdResultType), o(spvIdResult)]),
  instDef("OpExtInstImport", 11, [o(spvIdResult), o(spvLiteralString)]),
  instDef("OpExtension", 10, [o(spvLiteralString)]),
  instDef("OpMemoryModel", 14, [o(spvAddressingModel), o(spvMemoryModel)]),
  instDef("OpEntryPoint", 15, [o(spvExecutionModel), o(spvIdRef),
                               o(spvLiteralString), oMany(spvIdRef)]),
  # Per-mode literal operands (LocalSize x y z, ...) are defined by the spec's
  # ExecutionMode enumerants and absent from the grammar JSON's operand list
  instDef("OpExecutionMode", 16, [o(spvIdRef), o(spvExecutionMode)]),
  instDef("OpCapability", 17, [o(spvCapability)]),
  instDef("OpTypeVoid", 19, [o(spvIdResult)]),
  instDef("OpTypeBool", 20, [o(spvIdResult)]),
  instDef("OpTypeInt", 21, [o(spvIdResult), o(spvLiteralInteger),
                            o(spvLiteralInteger)]),
  # FPEncoding exists since SPIR-V 1.4, an absent word means IEEE754 encoding
  instDef("OpTypeFloat", 22, [o(spvIdResult), o(spvLiteralInteger),
                              oOpt(spvFPEncoding)]),
  instDef("OpTypeVector", 23, [o(spvIdResult), o(spvIdRef),
                               o(spvLiteralInteger)]),
  # Array Length is an <id> operand holding an integer OpConstant
  instDef("OpTypeArray", 28, [o(spvIdResult), o(spvIdRef), o(spvIdRef)]),
  instDef("OpTypeRuntimeArray", 29, [o(spvIdResult), o(spvIdRef)]),
  instDef("OpTypeStruct", 30, [o(spvIdResult), oMany(spvIdRef)]),
  instDef("OpTypePointer", 32, [o(spvIdResult), o(spvStorageClass),
                                o(spvIdRef)]),
  instDef("OpTypeFunction", 33, [o(spvIdResult), o(spvIdRef),
                                 oMany(spvIdRef)]),
  instDef("OpConstant", 43, [o(spvIdResultType), o(spvIdResult),
                             o(spvLiteralNumber)]),
  instDef("OpConstantNull", 46, [o(spvIdResultType), o(spvIdResult)]),
  instDef("OpVariable", 59, [o(spvIdResultType), o(spvIdResult),
                             o(spvStorageClass), oOpt(spvIdRef)]),
  # Per-decoration extra operands such as BuiltIn or Offset come from the spec's
  # Decoration enumerants, the grammar JSON's operand list does not model them
  instDef("OpDecorate", 71, [o(spvIdRef), o(spvDecoration)]),
  instDef("OpMemberDecorate", 72, [o(spvIdRef), o(spvLiteralInteger),
                                   o(spvDecoration)]),
  instDef("OpFunction", 54, [o(spvIdResultType), o(spvIdResult),
                             o(spvFunctionControl), o(spvIdRef)]),
  instDef("OpFunctionParameter", 55, [o(spvIdResultType), o(spvIdResult)]),
  # Arguments are IdRefs after the function id. Function-storage pointer args
  # (an OpVariable Function result passed directly) validate and execute under
  # the Vulkan target environment without any VariablePoints capability.
  # A pointer-to-pointer Function-storage VARIABLE is categorically invalid.
  instDef("OpFunctionCall", 57, [o(spvIdResultType), o(spvIdResult),
                                  o(spvIdRef), oMany(spvIdRef)]),
  instDef("OpFunctionEnd", 56),
  instDef("OpLabel", 248, [o(spvIdResult)]),
  instDef("OpBranch", 249, [o(spvIdRef)]),
  instDef("OpBranchConditional", 250, [o(spvIdRef), o(spvIdRef),
                                       o(spvIdRef), oMany(spvLiteralInteger)]),
  instDef("OpSelectionMerge", 247, [o(spvIdRef), o(spvSelectionControl)]),
  instDef("OpLoopMerge", 246, [o(spvIdRef), o(spvIdRef),
                               o(spvLoopControl)]),
  instDef("OpLoad", 61, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                         oOpt(spvMemoryAccess)]),
  instDef("OpStore", 62, [o(spvIdRef), o(spvIdRef), oOpt(spvMemoryAccess)]),
  instDef("OpAccessChain", 65, [o(spvIdResultType), o(spvIdResult),
                                o(spvIdRef), oMany(spvIdRef)]),
  instDef("OpCompositeConstruct", 80, [o(spvIdResultType), o(spvIdResult),
                                       oMany(spvIdRef)]),
  instDef("OpCompositeExtract", 81, [o(spvIdResultType), o(spvIdResult),
                                     o(spvIdRef), oMany(spvLiteralInteger)]),
  instDef("OpIAdd", 128, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                          o(spvIdRef)]),
  instDef("OpConvertUToPtr", 120, [o(spvIdResultType), o(spvIdResult),
                                   o(spvIdRef)]),
  instDef("OpConvertFToU", 109, [o(spvIdResultType), o(spvIdResult),
                                 o(spvIdRef)]),
  instDef("OpConvertFToS", 110, [o(spvIdResultType), o(spvIdResult),
                                 o(spvIdRef)]),
  instDef("OpConvertSToF", 111, [o(spvIdResultType), o(spvIdResult),
                                 o(spvIdRef)]),
  instDef("OpConvertUToF", 112, [o(spvIdResultType), o(spvIdResult),
                                 o(spvIdRef)]),
  instDef("OpUConvert", 113, [o(spvIdResultType), o(spvIdResult),
                              o(spvIdRef)]),
  instDef("OpSConvert", 114, [o(spvIdResultType), o(spvIdResult),
                              o(spvIdRef)]),
  instDef("OpFConvert", 115, [o(spvIdResultType), o(spvIdResult),
                              o(spvIdRef)]),
  instDef("OpBitcast", 124, [o(spvIdResultType), o(spvIdResult),
                             o(spvIdRef)]),
  instDef("OpFAdd", 129, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                          o(spvIdRef)]),
  instDef("OpISub", 130, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                          o(spvIdRef)]),
  instDef("OpFSub", 131, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                          o(spvIdRef)]),
  instDef("OpIMul", 132, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                          o(spvIdRef)]),
  instDef("OpFMul", 133, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                          o(spvIdRef)]),
  instDef("OpUDiv", 134, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                          o(spvIdRef)]),
  instDef("OpSDiv", 135, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                          o(spvIdRef)]),
  instDef("OpFDiv", 136, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                          o(spvIdRef)]),
  instDef("OpUMod", 137, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                          o(spvIdRef)]),
  instDef("OpSMod", 139, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                          o(spvIdRef)]),
  instDef("OpIEqual", 170, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                            o(spvIdRef)]),
  instDef("OpSLessThan", 177, [o(spvIdResultType), o(spvIdResult),
                               o(spvIdRef), o(spvIdRef)]),
  instDef("OpINotEqual", 171, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                               o(spvIdRef)]),
  # Ordered float comparisons, the float counterparts of the integer
  # comparison opcodes over unordered-capable FUnord* variants
  instDef("OpFOrdGreaterThan", 186, [o(spvIdResultType), o(spvIdResult),
                                     o(spvIdRef), o(spvIdRef)]),
  instDef("OpFOrdLessThan", 184, [o(spvIdResultType), o(spvIdResult),
                                  o(spvIdRef), o(spvIdRef)]),
  instDef("OpUGreaterThan", 172, [o(spvIdResultType), o(spvIdResult),
                                  o(spvIdRef), o(spvIdRef)]),
  instDef("OpSGreaterThan", 173, [o(spvIdResultType), o(spvIdResult),
                                  o(spvIdRef), o(spvIdRef)]),
  instDef("OpUGreaterThanEqual", 174, [o(spvIdResultType), o(spvIdResult),
                                       o(spvIdRef), o(spvIdRef)]),
  instDef("OpSGreaterThanEqual", 175, [o(spvIdResultType), o(spvIdResult),
                                       o(spvIdRef), o(spvIdRef)]),
  instDef("OpULessThan", 176, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                               o(spvIdRef)]),
  instDef("OpULessThanEqual", 178, [o(spvIdResultType), o(spvIdResult),
                                    o(spvIdRef), o(spvIdRef)]),
  instDef("OpSLessThanEqual", 179, [o(spvIdResultType), o(spvIdResult),
                                    o(spvIdRef), o(spvIdRef)]),
  # Logical equality over bool operands, the OpIEqual / OpINotEqual
  # counterparts whose operands must be scalar or vector int
  instDef("OpLogicalEqual", 164, [o(spvIdResultType), o(spvIdResult),
                                  o(spvIdRef), o(spvIdRef)]),
  instDef("OpLogicalNotEqual", 165, [o(spvIdResultType), o(spvIdResult),
                                     o(spvIdRef), o(spvIdRef)]),
  instDef("OpLogicalOr", 166, [o(spvIdResultType), o(spvIdResult),
                               o(spvIdRef), o(spvIdRef)]),
  instDef("OpLogicalAnd", 167, [o(spvIdResultType), o(spvIdResult),
                                o(spvIdRef), o(spvIdRef)]),
  instDef("OpSelect", 169, [o(spvIdResultType), o(spvIdResult), o(spvIdRef),
                            o(spvIdRef), o(spvIdRef)]),
  instDef("OpSNegate", 126, [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]),
  instDef("OpFNegate", 127, [o(spvIdResultType), o(spvIdResult), o(spvIdRef)]),
  instDef("OpLogicalNot", 168, [o(spvIdResultType), o(spvIdResult),
                                o(spvIdRef)]),
  instDef("OpConstantTrue", 41, [o(spvIdResultType), o(spvIdResult)]),
  instDef("OpConstantFalse", 42, [o(spvIdResultType), o(spvIdResult)]),
  # GroupOperation enumerant (Reduce/InclusiveScan/ExclusiveScan) sits between
  # Scope and the value. The optional trailing IdRef is the ClusterSize
  instDef("OpGroupNonUniformFAdd", 350, [o(spvIdResultType), o(spvIdResult),
                                         o(spvIdScope), o(spvGroupOperation),
                                         o(spvIdRef), oOpt(spvIdRef)]),
  # Subgroup shuffle instructions, the per-ladder-step reduction shape
  # for flavors that spell reductions without the arithmetic opcodes
  instDef("OpGroupNonUniformShuffle", 345, [o(spvIdResultType), o(spvIdResult),
                                            o(spvIdScope), o(spvIdRef),
                                            o(spvIdRef)]),
  instDef("OpGroupNonUniformShuffleXor", 346, [o(spvIdResultType), o(spvIdResult),
                                               o(spvIdScope), o(spvIdRef),
                                               o(spvIdRef)]),
  # SPV_EXT_shader_atomic_float_add: fp atomic add, pairing the extension string with the AtomicFloat32AddEXT capability
  instDef("OpAtomicFAddEXT", 6035, [o(spvIdResultType), o(spvIdResult),
                                    o(spvIdRef), o(spvIdScope),
                                    o(spvIdMemorySemantics), o(spvIdRef)]),
  instDef("OpControlBarrier", 224, [o(spvIdScope), o(spvIdScope),
                                    o(spvIdMemorySemantics)]),
  instDef("OpReturn", 253),
  instDef("OpReturnValue", 254, [o(spvIdRef)])
]

func spvOpcode*(name: static string): uint32 =
  ## Returns the opcode number of a table row, resolved from the grammar-checked table
  for row in spvInstTable:
    if row.name == name:
      return row.opcode
  doAssert false, "opcode not in spvInstTable: " & name

func spvOpcodeRt*(name: string): uint32 =
  ## Returns the opcode number of a table row, the runtime counterpart of `spvOpcode`
  ## for names picked at runtime (e.g. an operator-to-opcode mapping in a printer),
  ## fails on unknown names.
  for row in spvInstTable:
    if row.name == name:
      return row.opcode
  doAssert false, "opcode not in spvInstTable: " & name

const
  ## Capability values (grammar ValueEnum), compute subset
  capShader* = 1'u32
  capFloat16* = 9'u32
  capFloat64* = 10'u32
  capInt64* = 11'u32
  capInt8* = 39'u32
  capInt16* = 22'u32
  capVulkanMemoryModel* = 5345'u32
  capPhysicalStorageBufferAddresses* = 5347'u32
  capGroupNonUniformArithmetic* = 63'u32
  capGroupNonUniformShuffle* = 65'u32
  capVariablePointers* = 4442'u32
  capAtomicFloat32AddEXT* = 6033'u32

const
  ## Decoration values (grammar ValueEnum), buffer subset
  decBlock* = 2'u32
  decArrayStride* = 6'u32
  decBuiltIn* = 11'u32
  decNonWritable* = 24'u32
  decOffset* = 35'u32
  decBinding* = 33'u32
  decDescriptorSet* = 34'u32

const
  ## BuiltIn values (grammar ValueEnum), compute subset
  biNumWorkgroups* = 24'u32
  biWorkgroupSize* = 25'u32
  biWorkgroupId* = 26'u32
  biLocalInvocationId* = 27'u32
  biGlobalInvocationId* = 28'u32
  biLocalInvocationIndex* = 29'u32

const
  ## StorageClass values (grammar ValueEnum)
  scUniformConstant* = 0'u32
  scInput* = 1'u32
  scUniform* = 2'u32
  scOutput* = 3'u32
  scWorkgroup* = 4'u32
  scPrivate* = 6'u32
  scFunction* = 7'u32
  scPushConstant* = 9'u32
  scStorageBuffer* = 12'u32
  scPhysicalStorageBuffer* = 5349'u32

const
  ## ExecutionModel values (grammar ValueEnum)
  emVertex* = 0'u32
  emFragment* = 4'u32
  emGLCompute* = 5'u32

const
  ## ExecutionMode values (grammar ValueEnum)
  modeLocalSize* = 17'u32
  modeLocalSizeId* = 38'u32

const
  ## MemoryModel values (grammar ValueEnum)
  mmSimple* = 0'u32
  mmGLSL450* = 1'u32
  mmVulkan* = 3'u32

  ## AddressingModel values (grammar ValueEnum)
  amLogical* = 0'u32
  amPhysicalStorageBuffer64* = 5348'u32

const
  ## GroupOperation values (grammar ValueEnum), OpGroupNonUniform* operands
  groupOpReduce* = 0'u32
  groupOpInclusiveScan* = 1'u32
  groupOpExclusiveScan* = 2'u32

  ## Scope constants (grammar kind IdScope), operand words for OpControlBarrier
  scopeDevice* = 1'u32
  scopeWorkgroup* = 2'u32
  scopeSubgroup* = 3'u32
  scopeInvocation* = 4'u32

  ## MemoryAccess bit values (grammar BitEnum), Aligned carries a trailing
  ## literal alignment word the grammar operand list does not model
  maAligned* = 2'u32

const
  ## MemorySemantics bit values (grammar BitEnum), the operand words
  ## for OpControlBarrier and the OpAtomic* instructions
  semNone* = 0'u32
  semAcquire* = 0x2'u32
  semRelease* = 0x4'u32
  semAcquireRelease* = 0x8'u32
  semUniformMemory* = 0x40'u32
  semSubgroupMemory* = 0x80'u32
  semWorkgroupMemory* = 0x100'u32

const
  ## Control bit masks, all-None words for OpFunction / OpSelectionMerge /
  ## OpLoopMerge / memory operands
  fcNone* = 0'u32
  selNone* = 0'u32
  lcNone* = 0'u32
  maNone* = 0'u32

proc verifyGrammar*(grammarJsonText: string): seq[string] =
  ## Verifies the opcode table against core-grammar JSON text:
  ## - the opcode number per table row
  ## - operand kinds and quantifiers per table row
  ## Grammar quantifiers "" / "?" / "*" map to qOne / qOpt / qMany.
  ## Returns one human-readable mismatch per disagreeing row, an empty sequence
  ## means every row matches the grammar.
  let grammar = parseJson(grammarJsonText)
  var byName: Table[string, JsonNode]
  for instr in grammar["instructions"]:
    byName[instr["opname"].getStr()] = instr
  for row in spvInstTable:
    let g = byName.getOrDefault(row.name)
    if g.isNil:
      result.add row.name & ": not found in grammar"
      continue
    if g["opcode"].getInt().uint32 != row.opcode:
      result.add row.name & ": opcode " & $row.opcode & " != grammar " &
                  $g["opcode"].getInt()
    let gOps = if "operands" in g: g["operands"] else: nil
    let n = if gOps.isNil: 0 else: gOps.len
    if n != row.operands.len:
      result.add row.name & ": " & $row.operands.len &
                  " operands in table, " & $n & " in grammar"
      continue
    for i in 0 ..< n:
      let gKind = gOps[i]["kind"].getStr()
      let gQuant = if "quantifier" in gOps[i]: gOps[i]["quantifier"].getStr()
                   else: ""
      let tKind = spvKindGrammarNames[row.operands[i].kind]
      let tQuant = case row.operands[i].quant
                   of qOne: ""
                   of qOpt: "?"
                   of qMany: "*"
      if gKind != tKind or gQuant != tQuant:
        result.add row.name & " operand " & $i & ": table (" & tKind & ", \"" &
                    tQuant & "\") != grammar (\"" & gKind & "\", \"" & gQuant & "\")"

##
## SpvModule buffers instructions into named sections ordered per the SPIR-V
## §2.4 logical layout, then flattens all sections behind the 5-word header
## at `finalize`.
##
## Instruction encoding:
## - first word: `wordCount shl 16 or opcode`
## - string operand: UTF-8 bytes + NUL, zero-padded to a word boundary,
##   little-endian per word

type
  SpvSection* = enum
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

  SpvModule* = object
    version*: uint32      ## header version word, e.g. 0x00010300 for SPIR-V 1.3
    nextId*: uint32       ## next unallocated result id, ids start at 1
    sections*: array[SpvSection, seq[uint32]]

const spvMagic* = 0x07230203'u32
## Module baseline: SPIR-V 1.3, encoded in the spvVersion13 version word.
## The baseline carries semantics beyond the number:
## - the entry-point interface lists Input/Output variables only
## - OpTypeFloat omits the FPEncoding operand, the 1.3 form
## - PhysicalStorageBufferAddresses is extension-gated
## - push-constant members narrower than 32 bits refuse, they need the SPIR-V 1.5 storage capabilities
##
## Bumping the constant to 1.4+ requires the entry-point interface to list
## every referenced global variable, the push-constant block variable included.
const spvVersion13* = 0x00010300'u32
const spvGeneratorNone* = 0'u32
const spvSchemaNone* = 0'u32

proc initSpvModule*(): SpvModule =
  ## Empty module at the spvVersion13 baseline, result ids start at 1,
  ## id 0 is reserved
  SpvModule(version: spvVersion13, nextId: 1)

proc addWord*(m: var SpvModule, section: SpvSection, word: uint32) =
  ## Append one raw word to a section
  m.sections[section].add word

proc addWords*(m: var SpvModule, section: SpvSection, words: openArray[uint32]) =
  ## Append raw words to a section
  for w in words:
    m.sections[section].add w

proc stringWords*(s: string): int =
  ## Returns the word count of a string operand:
  ## UTF-8 bytes + NUL, padded up to a word boundary
  (s.len + 4) div 4

proc addString*(m: var SpvModule, section: SpvSection, s: string) =
  ## Append a string operand, little-endian per word:
  ## - UTF-8 bytes + terminating NUL
  ## - zero-padded up to a word boundary
  var bytes = newSeq[uint8](s.len + 1)
  for i in 0 ..< s.len:
    bytes[i] = uint8(s[i])
  while bytes.len mod 4 != 0:
    bytes.add 0'u8
  var i = 0
  while i < bytes.len:
    let w = uint32(bytes[i]) or uint32(bytes[i + 1]) shl 8 or
            uint32(bytes[i + 2]) shl 16 or uint32(bytes[i + 3]) shl 24
    m.sections[section].add w
    i += 4

proc allocId*(m: var SpvModule): uint32 =
  ## Returns the next sequential result id
  result = m.nextId
  inc m.nextId

proc startInst*(m: var SpvModule, section: SpvSection, opcode: uint32, operandWords: int) =
  ## Append an instruction header word before manually appended operand words,
  ## string operands and mixed word/non-word operand lists
  m.addWord(section, uint32(operandWords + 1) shl 16 or opcode)

proc emitInstSeq*(m: var SpvModule, section: SpvSection, opcode: uint32, operands: openArray[uint32]) =
  ## Append one instruction, `wordCount shl 16 or opcode` then the operand words
  m.startInst(section, opcode, operands.len)
  m.addWords(section, operands)

proc finalize*(m: SpvModule): seq[uint32] =
  ## Returns the flat module, the 5-word header then all sections
  ## in SPIR-V §2.4 order.
  ## Header words: magic, version, generator (0), bound, schema (0).
  ## The bound word is nextId at finalize time, covering all allocated ids.
  result = @[spvMagic, m.version, spvGeneratorNone, m.nextId, spvSchemaNone]
  for section in SpvSection:
    result.add m.sections[section]

##
## One thin wrapper per opcode of the spvInstTable, each wrapper writes
## one logical-layout section:
## - types and constants deduplicate via `memoInst`, repeated emissions share
##   one result id
## - body instructions (OpLoad, OpIAdd, ...) always allocate a fresh id
## - optional operands (MemoryAccess, initializer) are omitted, spec defaults
##   apply


type
  SpvMemoKey* = tuple[opcode: uint32, words: seq[uint32]]
    ## Dedup key for a type or constant: opcode plus the operand words

  SpvBuilder* = object
    module*: SpvModule
    memo*: Table[SpvMemoKey, uint32]
    scalarWidths*: Table[uint32, int]
      ## Bit width of every emitted OpTypeInt/OpTypeFloat, keyed by type id.
      ## OpConstant literal word counts derive from it

proc initSpvBuilder*(): SpvBuilder =
  ## Builder over an empty module at the spvVersion13 baseline
  SpvBuilder(module: initSpvModule())

proc memoInst*(b: var SpvBuilder, opcode: uint32, operands: openArray[uint32]): uint32 =
  ## Emit a result-carrying type or constant instruction, deduplicated on
  ## (opcode, operands), into the types-constants-globals section.
  ## Returns the instruction's result id.
  let key: SpvMemoKey = (opcode, @operands)
  if key in b.memo:
    return b.memo[key]
  result = b.module.allocId()
  var words = @[result]
  for w in operands:
    words.add w
  b.module.emitInstSeq(ssTypesConstantsGlobals, opcode, words)
  b.memo[key] = result

proc memoTypedInst*(b: var SpvBuilder, opcode: uint32, operands: openArray[uint32]): uint32 =
  ## Memoized emission for result-type-carrying instructions:
  ## - opcodes: OpConstant, OpConstantNull, OpUndef
  ## - `operands` leads with the result type, then the fresh result id
  ## - dedup key: opcode + operands, like `memoInst`
  ## Returns: the instruction's result id.
  let key: SpvMemoKey = (opcode, @operands)
  if key in b.memo:
    return b.memo[key]
  result = b.module.allocId()
  var words = @[operands[0], result]
  for i in 1 ..< operands.len:
    words.add operands[i]
  b.module.emitInstSeq(ssTypesConstantsGlobals, opcode, words)
  b.memo[key] = result

proc freshInst*(b: var SpvBuilder, opcode: uint32, operands: openArray[uint32]): uint32 =
  ## Emit a result-carrying instruction into the functions section.
  ## The fresh result id goes after the leading result type inside
  ## `operands`. Returns the fresh result id.
  result = b.module.allocId()
  var words = @[operands[0], result]
  for i in 1 ..< operands.len:
    words.add operands[i]
  b.module.emitInstSeq(ssFunctions, opcode, words)

# ── Non-result instructions ──

proc emitCapability*(b: var SpvBuilder, capability: uint32) =
  ## Append OpCapability
  b.module.emitInstSeq(ssCapabilities, spvOpcode("OpCapability"), @[capability])

proc emitMemoryModel*(b: var SpvBuilder, addressingModel, memoryModel: uint32) =
  ## Append OpMemoryModel
  b.module.emitInstSeq(ssMemoryModel, spvOpcode("OpMemoryModel"),
                       @[addressingModel, memoryModel])

proc emitEntryPoint*(b: var SpvBuilder, executionModel, entryPoint: uint32, name: string, iface: openArray[uint32]) =
  ## Append OpEntryPoint:
  ## - execution model, entry-point id, name string
  ## - interface ids, the global Input/Output variables the entry point reads
  ## Vulkan requires every such variable listed. An empty list is valid
  ## when the entry point reads none.
  ## The Input/Output-only interface matches the spvVersion13 baseline,
  ## SPIR-V 1.4+ requires listing every referenced global variable, push-constant block included.
  b.module.startInst(ssEntryPoints, spvOpcode("OpEntryPoint"),
                     2 + stringWords(name) + iface.len)
  b.module.addWords(ssEntryPoints, @[executionModel, entryPoint])
  b.module.addString(ssEntryPoints, name)
  b.module.addWords(ssEntryPoints, iface)

proc emitExecutionMode*(b: var SpvBuilder, entryPoint, mode: uint32, literals: openArray[uint32] = []) =
  ## Append OpExecutionMode with optional literal operands,
  ## LocalSize takes x y z
  var ops = @[entryPoint, mode]
  for w in literals:
    ops.add w
  b.module.emitInstSeq(ssExecutionModes, spvOpcode("OpExecutionMode"), ops)

proc emitDecorate*(b: var SpvBuilder, target, decoration: uint32, literals: openArray[uint32] = []) =
  ## Append OpDecorate with optional literal operands,
  ## BuiltIn needs the builtin literal, Offset the byte offset
  var ops = @[target, decoration]
  for w in literals:
    ops.add w
  b.module.emitInstSeq(ssAnnotations, spvOpcode("OpDecorate"), ops)

proc emitMemberDecorate*(b: var SpvBuilder, target, member, decoration: uint32, literals: openArray[uint32] = []) =
  ## Append OpMemberDecorate on member `member` of struct type `target`,
  ## Offset takes the byte offset literal
  var ops = @[target, member, decoration]
  for w in literals:
    ops.add w
  b.module.emitInstSeq(ssAnnotations, spvOpcode("OpMemberDecorate"), ops)

proc opVariable*(b: var SpvBuilder, resultType, storageClass: uint32): uint32 =
  ## Append OpVariable without initializer to the types-constants-globals section.
  ## Returns the variable result id.
  result = b.module.allocId()
  b.module.emitInstSeq(ssTypesConstantsGlobals, spvOpcode("OpVariable"),
                       @[resultType, result, storageClass])

proc opNop*(b: var SpvBuilder) =
  ## Append OpNop
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpNop"), @[])

# ── Types and constants, memoized ──

proc tyVoid*(b: var SpvBuilder): uint32 =
  ## Emit OpTypeVoid. Returns the memoized result id.
  b.memoInst(spvOpcode("OpTypeVoid"), @[])

proc tyBool*(b: var SpvBuilder): uint32 =
  ## Emit OpTypeBool. Returns the memoized result id.
  b.memoInst(spvOpcode("OpTypeBool"), @[])

proc tyArray*(b: var SpvBuilder, elementType, lengthId: uint32): uint32 =
  ## Emit OpTypeArray over `lengthId` elements, `lengthId` references an integer OpConstant result
  ## Returns the memoized result id.
  b.memoInst(spvOpcode("OpTypeArray"), @[elementType, lengthId])

proc recordScalarWidth(b: var SpvBuilder, ty: uint32, width: int) =
  ## Records the bit width of an emitted scalar type id
  b.scalarWidths[uint32(ty)] = width

proc scalarWidth(b: SpvBuilder, ty: uint32): int =
  ## Bit width of a scalar type id the builder emitted,
  ## every OpConstant operand type is one of those
  doAssert ty in b.scalarWidths,
    "constant over a type id with no recorded width: " & $ty
  b.scalarWidths[ty]

proc tyInt*(b: var SpvBuilder, width: int, signedness: bool): uint32 =
  ## Emit OpTypeInt: width in bits, signedness 1 = signed, 0 = unsigned.
  ## Returns the memoized result id.
  result = b.memoInst(spvOpcode("OpTypeInt"), @[uint32(width), uint32(signedness)])
  b.recordScalarWidth(result, width)

proc tyFloat*(b: var SpvBuilder, width: int): uint32 =
  ## Emit OpTypeFloat: width in bits, IEEE754 encoding by operand omission.
  ## Returns the memoized result id.
  result = b.memoInst(spvOpcode("OpTypeFloat"), @[uint32(width)])
  b.recordScalarWidth(result, width)

proc tyVector*(b: var SpvBuilder, componentType: uint32, componentCount: int): uint32 =
  ## Emit OpTypeVector over a scalar type. Returns the memoized result id.
  b.memoInst(spvOpcode("OpTypeVector"),
             @[componentType, uint32(componentCount)])

proc tyStruct*(b: var SpvBuilder, memberTypes: openArray[uint32]): uint32 =
  ## Emit OpTypeStruct over member types in declaration order.
  ## Returns the memoized result id.
  b.memoInst(spvOpcode("OpTypeStruct"), memberTypes)

proc tyRuntimeArray*(b: var SpvBuilder, elementType: uint32): uint32 =
  ## Emit OpTypeRuntimeArray, the unsized SSBO element array.
  ## Returns the memoized result id.
  b.memoInst(spvOpcode("OpTypeRuntimeArray"), @[elementType])

proc tyPointer*(b: var SpvBuilder, storageClass, pointeeType: uint32): uint32 =
  ## Emit OpTypePointer into a storage class. Returns the memoized result id.
  b.memoInst(spvOpcode("OpTypePointer"), @[storageClass, pointeeType])

proc tyFunction*(b: var SpvBuilder, returnType: uint32, paramTypes: openArray[uint32]): uint32 =
  ## Emit OpTypeFunction, paramTypes excludes the return type.
  ## Returns the memoized result id.
  var ops = @[returnType]
  for p in paramTypes:
    ops.add p
  b.memoInst(spvOpcode("OpTypeFunction"), ops)

proc constUint*(b: var SpvBuilder, ty: uint32, value: uint64): uint32 =
  ## Emit OpConstant holding the bit pattern of `value`:
  ## - one literal word for 32-bit and narrower types
  ## - two literal words, low half first, for 64-bit types
  ## Word count follows the type width, never the value:
  ## a 64-bit constant whose bit pattern fits one word still carries two.
  ## Returns the memoized result id.
  if b.scalarWidth(ty) > 32:
    b.memoTypedInst(spvOpcode("OpConstant"),
                    @[ty, uint32(value), uint32(value shr 32)])
  else:
    let width = b.scalarWidth(ty)
    let mask = (1'u64 shl uint64(width)) - 1
    doAssert value <= mask,
      "constant bit pattern overflows the " & $width & "-bit type " & $ty
    b.memoTypedInst(spvOpcode("OpConstant"), @[ty, uint32(value)])

proc constBool*(b: var SpvBuilder, ty: uint32, value: bool): uint32 =
  ## Emit OpConstantTrue / OpConstantFalse into the types-constants section:
  ## constant instructions are only valid in the graph definitions section,
  ## never inside a function body.
  ## Returns the memoized result id.
  let op = if value: spvOpcode("OpConstantTrue") else: spvOpcode("OpConstantFalse")
  b.memoTypedInst(op, @[ty])

proc constNull*(b: var SpvBuilder, ty: uint32): uint32 =
  ## Emit OpConstantNull, the zero of any type.
  ## Returns the memoized result id.
  b.memoTypedInst(spvOpcode("OpConstantNull"), @[ty])

proc undef*(b: var SpvBuilder, ty: uint32): uint32 =
  ## Emit OpUndef into the types-constants-globals section.
  ## Returns the memoized result id.
  b.memoTypedInst(spvOpcode("OpUndef"), @[ty])

# ── Function body ──

proc opFunction*(b: var SpvBuilder, returnType, functionType: uint32): uint32 =
  ## Append OpFunction with FunctionControl None.
  ## Returns the function result id.
  b.freshInst(spvOpcode("OpFunction"), @[returnType, fcNone, functionType])

proc opFunctionParameter*(b: var SpvBuilder, resultType: uint32): uint32 =
  ## Append OpFunctionParameter. Returns the parameter result id.
  b.freshInst(spvOpcode("OpFunctionParameter"), @[resultType])

proc opLabel*(b: var SpvBuilder): uint32 =
  ## Append OpLabel opening a basic block. Returns the label result id.
  result = b.module.allocId()
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpLabel"), @[result])

proc opBranch*(b: var SpvBuilder, targetLabel: uint32) =
  ## Append OpBranch, an unconditional jump to a label
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpBranch"), @[targetLabel])

proc opBranchConditional*(b: var SpvBuilder, condition, trueLabel, falseLabel: uint32) =
  ## Append OpBranchConditional, a conditional jump between two labels
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpBranchConditional"),
                       @[condition, trueLabel, falseLabel])

proc opSelectionMerge*(b: var SpvBuilder, mergeBlock: uint32, selectionControl: uint32) =
  ## Append OpSelectionMerge, declares the merge block of an OpBranchConditional
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpSelectionMerge"),
                       @[mergeBlock, selectionControl])

proc opLoopMerge*(b: var SpvBuilder, mergeBlock, continueTarget: uint32, loopControl: uint32) =
  ## Append OpLoopMerge, declares the merge and continue blocks of a loop
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpLoopMerge"),
                       @[mergeBlock, continueTarget, loopControl])

proc opReturn*(b: var SpvBuilder) =
  ## Append OpReturn, returns from a void function
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpReturn"), @[])

proc opReturnValue*(b: var SpvBuilder, value: uint32) =
  ## Append OpReturnValue, returns `value` from a non-void function
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpReturnValue"), @[value])

proc opFunctionEnd*(b: var SpvBuilder) =
  ## Append OpFunctionEnd, closes the current function
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpFunctionEnd"), @[])

proc opLoad*(b: var SpvBuilder, resultType, pointer: uint32): uint32 =
  ## Append OpLoad through a pointer without the MemoryAccess operand.
  ## Returns the loaded value result id.
  b.freshInst(spvOpcode("OpLoad"), @[resultType, pointer])

proc opStore*(b: var SpvBuilder, pointer, value: uint32) =
  ## Append OpStore through a pointer without the MemoryAccess operand
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpStore"), @[pointer, value])

proc opAccessChain*(b: var SpvBuilder, resultType, base: uint32, indices: openArray[uint32]): uint32 =
  ## Append OpAccessChain into a composite, indices are id refs.
  ## Returns the pointer result id.
  var ops = @[resultType, base]
  for i in indices:
    ops.add i
  b.freshInst(spvOpcode("OpAccessChain"), ops)

proc opIAdd*(b: var SpvBuilder, resultType, operand1, operand2: uint32): uint32 =
  ## Append OpIAdd, integer addition. Returns the sum result id.
  b.freshInst(spvOpcode("OpIAdd"), @[resultType, operand1, operand2])

proc opConvertUToPtr*(b: var SpvBuilder, resultType, source: uint32): uint32 =
  ## Append OpConvertUToPtr, unsigned integer to physical-pointer conversion.
  ## Returns the pointer result id.
  b.freshInst(spvOpcode("OpConvertUToPtr"), @[resultType, source])

proc opCompositeExtract*(b: var SpvBuilder, resultType, composite: uint32, indices: openArray[int]): uint32 =
  ## Append OpCompositeExtract, extracts a sub-value from a composite,
  ## indices are literal indexes. Returns the extracted value result id.
  var ops = @[resultType, composite]
  for i in indices:
    ops.add uint32(i)
  b.freshInst(spvOpcode("OpCompositeExtract"), ops)

proc opControlBarrier*(b: var SpvBuilder, execution, memory, semantics: uint32) =
  ## Append OpControlBarrier:
  ## - execution and memory scopes as scope operand words
  ## - memory semantics as the third operand word
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpControlBarrier"),
                       @[execution, memory, semantics])

proc opIEqual*(b: var SpvBuilder, resultType, operand1, operand2: uint32): uint32 =
  ## Append OpIEqual, integer equality yielding a boolean.
  ## Returns the boolean result id.
  b.freshInst(spvOpcode("OpIEqual"), @[resultType, operand1, operand2])

proc opSLessThan*(b: var SpvBuilder, resultType, operand1, operand2: uint32): uint32 =
  ## Append OpSLessThan, signed integer comparison yielding a boolean.
  ## Returns the boolean result id.
  b.freshInst(spvOpcode("OpSLessThan"), @[resultType, operand1, operand2])

proc opSelect*(b: var SpvBuilder, resultType, condition, object1, object2: uint32): uint32 =
  ## Append OpSelect, per-component selection between two same-typed objects.
  ## Returns the selected value result id.
  b.freshInst(spvOpcode("OpSelect"),
              @[resultType, condition, object1, object2])

# ── Infrastructure wrappers the kernel printers drive ──
# Value instructions (arithmetic, comparisons, conversions) have a uniform
# (IdResultType, IdResult, operands...) shape, the flavor modules emit them
# via `freshInst` + `spvOpcode` over the grammar-checked table.

proc opVariableFn*(b: var SpvBuilder, resultType: uint32): uint32 =
  ## Append OpVariable in the Function storage class to the functions section.
  ## Function-scope variables must sit in the entry block, so unlike
  ## `opVariable` this writes into ssFunctions. Returns the variable result id.
  result = b.module.allocId()
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpVariable"),
                       @[resultType, result, scFunction])

proc freshLabel*(b: var SpvBuilder): uint32 =
  ## Allocates a label result id without emitting OpLabel, for forward references
  ## (merge/continue targets) whose defining block appears later.
  ## Returns: the fresh label id. Pair with `opLabelAt`.
  b.module.allocId()

proc opLabelAt*(b: var SpvBuilder, label: uint32) =
  ## Append OpLabel for a label id allocated earlier via `freshLabel`
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpLabel"), @[label])

proc emitExtension*(b: var SpvBuilder, name: string) =
  ## Append OpExtension, declares a module extension.
  ## Args: `name`, e.g. SPV_KHR_physical_storage_buffer, required for
  ## PhysicalStorageBufferAddresses under the SPIR-V 1.3 baseline.
  b.module.startInst(ssExtensions, spvOpcode("OpExtension"),
                     stringWords(name))
  b.module.addString(ssExtensions, name)

proc opLoadAligned*(b: var SpvBuilder, resultType, pointer, alignment: uint32): uint32 =
  ## Append OpLoad with Aligned memory access, Vulkan requires Aligned for every load
  ## through a PhysicalStorageBuffer pointer. Args: `alignment` is the pointee alignment
  ## in bytes. Returns: the loaded value result id.
  b.freshInst(spvOpcode("OpLoad"),
              @[resultType, pointer, maAligned, alignment])

proc opStoreAligned*(b: var SpvBuilder, pointer, value, alignment: uint32) =
  ## Append OpStore with Aligned memory access, the store-side counterpart
  ## of `opLoadAligned`
  b.module.emitInstSeq(ssFunctions, spvOpcode("OpStore"),
                       @[pointer, value, maAligned, alignment])
