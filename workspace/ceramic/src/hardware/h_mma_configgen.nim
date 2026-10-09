## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## MMA atom registry generator + the atom datatypes.
import std/[macros, strutils]
import workspace/ceramic/src/int_tuples

# ═════════════════════════════════════════════════════════════════════════
#  Datatypes and SIMD ISAs
# ═════════════════════════════════════════════════════════════════════════

type
  MmaDType* = enum
    ## Matrix-Multiply-Accumulate (MMA) datatypes.
    mdtF32, mdtF64,
    mdtTF32,          ## specialized tensor float32, 32-bit with 10-bit mantissa in a 32-bit "opaque" blob
    mdtF16, mdtBF16,  ## 16-bit, packed 2-per-u32 in registers
    mdtFP8E4M3,       ## 8-bit, 1 sign + 4 exponent + 3 mantissa bits
    mdtFP8E5M2,       ## 8-bit, 1 sign + 5 exponent + 2 mantissa bits
    mdtInt8, mdtUint8, mdtInt16, mdtInt32

  SimdIsa* = enum
    ## CPU SIMD ISAs for the CPU atom ukernels.
    ## TODO, pending the CPU-atom registry at CPU-merge time.
    siAVX2, siAVX512, siNEON, siSVE, siI8MM, siVNNI, siSDOT

  MmaOperand* = enum
    ## Matrix operand in the standard GEMM description α·AB + β·C.
    opA, opB, opC

  NoLayout* = Int[-1]
    ## Sentinel layout, an Int[-1] placeholder.
    ## TODO, pending the CPU-atom registry at CPU-merge time.

# ═════════════════════════════════════════════════════════════════════════
#  declareAtoms — parser + generator
# ═════════════════════════════════════════════════════════════════════════

const AtomPropKeys* = ["m", "n", "k", "vpt", "threadCount",
                       "aLayout", "bLayout", "cLayout", "instr"]
  ## The property keys every atom must declare, in declaration order.
  ## The generated const name is `NAME_key`.

const OptionalAtomPropKeys = ["aType", "bType", "cType"]
  ## Optional property keys, declared only by the atoms that need them:
  ## Per-operand datatypes (A, B, accumulator/C), the Apple simdgroup
  ## atoms' MSL element names.
  ## Absent keys generate a default const ("" for string keys).

const DtypePropKeys = ["aType", "bType", "cType"]
  ## Optional per-operand datatype keys.
  ## Values: the MSL element names for the Apple and universal atoms
  ## ("float", "half", "bfloat"), the PTX mnemonics for the NVIDIA atoms
  ## ("f32", "f16", "bf16", "tf32", "s8", "s32", "e4m3", "e5m2").

const IntPropKeys = ["m", "n", "k", "vpt", "threadCount"]
  ## The scalar keys whose values must be positive int literals.

type
  AtomParams = object
    name: NimNode
    props: seq[(string, NimNode)]   ## (property key, value AST) as written

var atomDefs {.compileTime.}: seq[AtomParams]

proc parseAtomDecls*(defs: var seq[AtomParams]; body: NimNode) =
  ## Collects the atom declarations into `defs`, validating the keys.
  ##
  ## Args:
  ##
  ##   - defs, appended in declaration order
  ##   - body, the declareAtoms statement list
  ##
  ## Expected AST per atom:
  ##   Command(Ident"atom", Ident"NAME", StmtList(Call(Ident"key", StmtList(value)) …))
  body.expectKind(nnkStmtList)
  for atomDesc in body:
    atomDesc.expectKind(nnkCommand)
    doAssert atomDesc[0].eqIdent"atom", "expected `atom NAME:` declaration"
    let name = atomDesc[1]
    name.expectKind(nnkIdent)
    for existing in defs:
      doAssert $existing.name != $name,
        "declareAtoms: duplicate atom name `" & $name & "`"
    let propsNode = atomDesc[2]
    propsNode.expectKind(nnkStmtList)
    var params = AtomParams(name: name)
    var seen: seq[string]
    for prop in propsNode:
      prop.expectKind(nnkCall)
      let key = prop[0]
      key.expectKind(nnkIdent)
      let valNode = prop[1]
      valNode.expectKind(nnkStmtList)
      let keyStr = $key
      doAssert keyStr in AtomPropKeys or keyStr in OptionalAtomPropKeys,
        "declareAtoms: unknown property `" & keyStr & "` on atom `" & $name & "`"
      doAssert keyStr notin seen,
        "declareAtoms: duplicate property `" & keyStr & "` on atom `" & $name & "`"
      seen.add keyStr
      if keyStr in IntPropKeys:
        let v = valNode[0]
        v.expectKind(nnkIntLit)
        doAssert v.intVal > 0,
          "declareAtoms: property `" & keyStr & "` on atom `" & $name &
          "` must be positive, got " & $v.intVal
      elif keyStr == "instr":
        let v = valNode[0]
        v.expectKind(nnkStrLit)
        doAssert v.strVal == "" or v.strVal == "simdgroup_multiply_accumulate" or
                 v.strVal.startsWith("mma.sync.aligned."),
          "declareAtoms: invalid instruction `" & v.strVal & "` on atom `" & $name &
          "` (expected \"\", \"simdgroup_multiply_accumulate\", or an mma.sync.aligned.* mnemonic)"
      elif keyStr in DtypePropKeys:
        let v = valNode[0]
        v.expectKind(nnkStrLit)
        doAssert v.strVal in ["float", "half", "bfloat",
                              "f32", "f16", "bf16", "tf32",
                              "s8", "s32", "e4m3", "e5m2"],
          "declareAtoms: invalid operand datatype `" & v.strVal & "` on atom `" & $name &
          "` (expected an MSL element name \"float\"/\"half\"/\"bfloat\" or a PTX " &
          "mnemonic \"f32\"/\"f16\"/\"bf16\"/\"tf32\"/\"s8\"/\"s32\"/\"e4m3\"/\"e5m2\")"
      params.props.add (keyStr, valNode[0])
    for key in AtomPropKeys:
      doAssert key in seen,
        "declareAtoms: atom `" & $name & "` is missing property `" & key & "`"
    defs.add params

proc genAtomDecls(defs: seq[AtomParams]): NimNode =
  ## Generated declarations, one exported const per atom per property.
  ##
  ##   type MmaAtom* = enum NAME1, NAME2, …
  ##   const NAME1_m* = 8, NAME1_n* = 8, …
  ##
  ## Optional keys an atom does not declare emit a default const
  ## ("" for the per-operand datatype keys).
  result = newStmtList()
  var fields: seq[NimNode]
  for d in defs:
    fields.add d.name
  result.add newEnum(name = ident"MmaAtom", fields = fields, public = true, pure = false)
  for d in defs:
    let base = $d.name
    for (key, val) in d.props:
      result.add newConstStmt(
        nnkPostfix.newTree(ident"*", ident(base & "_" & key)),
        val)
    for key in OptionalAtomPropKeys:
      var declared = false
      for (k, _) in d.props:
        if k == key: declared = true
      if not declared:
        result.add newConstStmt(
          nnkPostfix.newTree(ident"*", ident(base & "_" & key)),
          newLit(""))

macro declareAtoms*(body: untyped): untyped =
  ## Parses the YAML-like atom registry block and expands to the enum
  ## plus the per-atom named consts.
  body.expectKind(nnkStmtList)
  atomDefs.setLen(0)  # a second declareAtoms expansion must not re-emit the first one's atoms
  atomDefs.parseAtomDecls(body)
  result = atomDefs.genAtomDecls()
