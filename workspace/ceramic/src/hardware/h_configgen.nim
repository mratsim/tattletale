## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## MMA atom registry generator + the atom datatypes.
##
## The atom is an enum member, not a data-carrying object: an enum static
## unifies across generic positions where a tuple-carrying object static
## does not (a tuple in the static value poisons unification). All atom
## properties are per-atom named consts resolved through getter macros
## (`h_properties.nim`).
##
## Lifecycle: `declareAtoms:` (invoked in `h_registry.nim`) parses the declarative atom block and generates
##   - `type MmaAtom = enum` with one member per atom,
##   - one exported const per atom per property: `NAME_m`, `NAME_n`,
##     `NAME_k`, `NAME_vpt`, `NAME_threadCount`, `NAME_aLayout`,
##     `NAME_bLayout`, `NAME_cLayout`, `NAME_instr`.
##
## Mirrors Constantine's `declareCurves` machinery
## (`constantine/named/deriv/parser_fields.nim`): the same AST shape
## (`Command(Ident"atom", Ident"NAME", StmtList(Call(Ident"key",
## StmtList(value)) …))`), the same two-step parse-then-generate.

import std/[macros, strutils]
import ../int_tuples

# ═════════════════════════════════════════════════════════════════════════
#  Datatypes and SIMD ISAs
# ═════════════════════════════════════════════════════════════════════════

type
  MmaDType* = enum
    ## Matrix-Multiply-Accumulate (MMA) datatypes.
    mdtF32, mdtF64,
    mdtTF32,          ## specialized tensor float32, 32-bit with 10-bit mantissa in a 32-bit "opaque" blob
    mdtF16, mdtBF16,  ## 16-bit, packed 2-per-u32 in registers
    mdtFP8E4M3,       ## 8-bit: 1 sign + 4 exponent + 3 mantissa bits
    mdtFP8E5M2,       ## 8-bit: 1 sign + 5 exponent + 2 mantissa bits
    mdtInt8, mdtUint8, mdtInt16, mdtInt32

  SimdIsa* = enum
    ## CPU SIMD ISAs for the CPU atom ukernels.
    ## TODO: pending the CPU-atom registry at CPU-merge time.
    siAVX2, siAVX512, siNEON, siSVE, siI8MM, siVNNI, siSDOT

  MmaOperand* = enum
    ## Matrix operand in the standard GEMM description α·AB + β·C.
    opA, opB, opC

  NoLayout* = Int[-1]
    ## Sentinel layout.
    ## TODO: pending the CPU-atom registry at CPU-merge time.

# ═════════════════════════════════════════════════════════════════════════
#  declareAtoms — parser + generator
# ═════════════════════════════════════════════════════════════════════════

const AtomPropKeys* = ["m", "n", "k", "vpt", "threadCount",
                       "aLayout", "bLayout", "cLayout", "instr"]
  ## The property keys every atom must declare, in declaration order.
  ## The generated const name is `NAME_key`.

const IntPropKeys = ["m", "n", "k", "vpt", "threadCount"]
  ## The scalar keys whose values must be positive int literals.

type
  AtomParams = object
    name: NimNode
    props: seq[(string, NimNode)]   ## (property key, value AST) as written

var atomDefs {.compileTime.}: seq[AtomParams]

proc parseAtomDecls*(defs: var seq[AtomParams]; body: NimNode) =
  ## Collects the atom declarations into `defs`, validating the keys.
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
      doAssert keyStr in AtomPropKeys,
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
      params.props.add (keyStr, valNode[0])
    doAssert seen.len == AtomPropKeys.len,
      "declareAtoms: atom `" & $name & "` declares " & $seen.len &
      " properties, expected " & $AtomPropKeys.len & " (" & AtomPropKeys.join(", ") & ")"
    defs.add params

proc genAtomDecls(defs: seq[AtomParams]): NimNode =
  ## Generates:
  ##   type MmaAtom* = enum NAME1, NAME2, …
  ##   const NAME1_m* = 8            (one exported const per atom per property)
  ##         NAME1_n* = 8
  ##         …
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

macro declareAtoms*(body: untyped): untyped =
  ## Parses the YAML-like atom registry block and expands to the enum
  ## plus the per-atom named consts.
  body.expectKind(nnkStmtList)
  atomDefs.setLen(0)  # a second declareAtoms: expansion must not re-emit the first one's atoms
  atomDefs.parseAtomDecls(body)
  result = atomDefs.genAtomDecls()
