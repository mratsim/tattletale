## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, strutils]
import workspace/crucible
import ./h_mma_registry

{.experimental: "dynamicBindSym".}

## Register-level MMA dispatch.

# TODO, pending AMD and Intel tensor cores, gemm_mma handles Nvidia asm,
# the Apple simdgroup intrinsic, and the universal 1×1×1 scalar FMA.

func constraintLetter(elemTypeName: string): string =
  ## Nim DSL register element type → GCC asm constraint letter.
  ## tf32/f16/bf16/int are mapped to integer registers ("r").
  ## f32/f64 accumulators to float registers ("f"/"d").
  case elemTypeName
  of "float32": "f"
  of "float64": "d"
  of "uint32", "uint16", "uint8", "int32", "int16", "int8": "r"
  else:
    raiseAssert "unsupported register element type: " & elemTypeName

func accumulatorElementName*(instr: string): string =
  ## Returns the accumulator's register element type for the instruction
  ## signature (PTX ISA, mma.sync, D and C share one register class).
  ## - the D-type dot token names the accumulator class
  ## - f32 → float32, the float registers ("f")
  ## - s32 → int32, the integer registers ("r")
  const prefix = "mma.sync.aligned."
  doAssert instr.startsWith(prefix), "unsupported instruction: " & instr
  let tokens = instr[len(prefix) ..< instr.len].split('.')
  doAssert tokens.len == 7, "unsupported instruction: " & instr
  result = case tokens[3]
    of "f32": "float32"
    of "s32": "int32"
    else: raiseAssert "unsupported accumulator type in " & instr

func regList(first, count: int): string =
  ## GCC operand register list "{%0,%1,...,%N-1}"
  result = "{"
  for i in first ..< first + count:
    if i > first: result.add ","
    result.add "%" & $i
  result.add "}"

func operandClause(names: openArray[string], clause: string): string =
  ## Constraint clause for the scalar registers named in `names`,
  ## `"clause"(`name0`), "clause"(`name1`), ...`.
  for i, name in names:
    if i > 0: result.add ", "
    result.add "\"" & clause & "\"(`" & name & "`)"

func mmaOperandWidth(operandType: string): int {.compileTime.} =
  ## Elements per .b32 mma.sync operand register, by operand dtype:
  ## PTX packs 16-bit operands two-per-register and 32-bit (tf32)
  ## one-per-register. The 8-bit dtypes pack four-per-register in PTX
  ## but that packing is not implemented here, they error like any
  ## unregistered operand dtype.
  case operandType
  of "tf32": 1
  of "f16", "bf16": 2
  else:
    error("unsupported mma.sync operand dtype: " & operandType)

func buildNvidiaMmaAsm*(instr: string, dNames, aNames, bNames: openArray[string], accType: string): string =
  ## GCC extended-asm builder for one Nvidia MMA instruction,
  ## D = A·B + D accumulated in place: the C operand is the D registers,
  ## modified ("+" read-write constraint).
  ##
  ## Expected input:
  ##   - `instr`: the mma.sync.aligned.* instruction string
  ##   - `dNames`: the staged accumulator scalars' names, one per value,
  ##     in the atom's per-lane value order
  ##   - `aNames`/`bNames`: the packed .b32 A/B operand register names,
  ##     one per register, 16-bit dtypes stage two scalars per uint32
  ##     local and one-value-per-register dtypes stage the raw scalar,
  ##     staged by gemm_mma
  ##   - `accType`: the accumulator register element type name
  ## Output: one extended-asm statement string.
  let vc = dNames.len
  let va = aNames.len
  let vb = bNames.len

  # the template part, "<instr> {D}, {A}, {B}, {D-modified}"
  let tpl =
    "\"" & instr &
    " " & regList(0, vc) &
    ", " & regList(vc, va) &
    ", " & regList(vc + va, vb) &
    ", " & regList(0, vc) & ";\""

  result = tpl &
    " : " & operandClause(dNames, "+" & constraintLetter(accType)) &
    " : " & operandClause(aNames, constraintLetter("uint32")) &
    ", " & operandClause(bNames, constraintLetter("uint32"))

func recastToMmaType(name, inType: string): string =
  ## Non-native host CPU types need to be reinterpreted
  ## from their integer buffer
  if inType == "float": result = "`" & name & "`"
  else: result = "as_type<" & inType & ">(`" & name & "`)"

func buildAppleSimdgroupAsm(dNames, aNames, bNames, cNames: openArray[string], accType, inType: string): string =
  ## Metal codegen for one simdgroup_multiply_accumulate builtin
  ## over caller-staged scalars, D = A·B + C.
  ##
  ## Expected input:
  ##   - `dNames`/`aNames`/`bNames`/`cNames`:
  ##        the staged scalars' names
  ##   - `accType`/`inType`: the MSL element names
  ##        the accumulator is fp32 ("float"),
  ##        A and B MUST be the atom's operand dtype
  result = "{\n"
  result.add "  simdgroup_" & accType & "8x8 sd = make_filled_simdgroup_matrix<" & accType &
            ", 8>(" & accType & "(0.0f));\n"
  for i, name in cNames:
    result.add "  sd.thread_elements()[" & $i & "] = `" & name & "`;\n"
  result.add "  simdgroup_" & inType & "8x8 sa = make_filled_simdgroup_matrix<" & inType &
            ", 8>(" & inType & "(0.0f));\n"
  for i, name in aNames:
    result.add "  sa.thread_elements()[" & $i & "] = " & recastToMmaType(name, inType) & ";\n"
  result.add "  simdgroup_" & inType & "8x8 sb = make_filled_simdgroup_matrix<" & inType &
            ", 8>(" & inType & "(0.0f));\n"
  for i, name in bNames:
    result.add "  sb.thread_elements()[" & $i & "] = " & recastToMmaType(name, inType) & ";\n"
  result.add "  simdgroup_multiply_accumulate(sd, sa, sb, sd);\n"
  for i, name in dNames:
    result.add "  `" & name & "` = sd.thread_elements()[" & $i & "];\n"
  result.add "}"

# ═════════════════════════════════════════════════════════════════════════
#  Atom registry access (macro-time)
# ═════════════════════════════════════════════════════════════════════════

template constStr*(atom: untyped, suffix: untyped): string =
  ## The string value of a per-atom registry const.
  bindSym($atom & "_" & suffix).getImpl()[2].strVal

proc assertStagedDtype(name: string, node: NimNode, dtype, operand: string) =
  let expected = case dtype
    of "float": bindSym"float32"
    of "half": bindSym"float16"
    of "bfloat": bindSym"bfloat16"
    else:
      error("gemm_mma: atom registry " & operand & " dtype `" & dtype &
            "` is not a staging dtype (\"float\"/\"half\"/\"bfloat\")")
      newEmptyNode()
  if not sameType(node.getTypeInst(), expected):
    error("gemm_mma: staged scalar `" & name & "` does not match the " & operand &
          " dtype the atom registry declares (" & dtype & "), got " &
          node.getTypeInst().repr & "; stage it in the registry dtype")

proc leafProduct(n: NimNode): int =
  ## Product of the int literals in a type-AST subtree (layout shape parts).
  case n.kind
  of nnkIntLit: result = int(n.intVal)
  of nnkTupleConstr, nnkPar, nnkBracketExpr, nnkTupleTy, nnkBracket,
     nnkIdentDefs:
    result = 1
    for ch in n:
      result *= leafProduct(ch)
  else: result = 1

template valuesPerThread*(atom: untyped, layoutKey: string): int =
  ## Values per thread of the atom operand named by `layoutKey`,
  ## the V component of the layout const's (T, V) shape type,
  ## read from the type alone.
  let layoutType = bindSym($atom & "_" & layoutKey).getTypeInst()
  doAssert layoutType.kind == nnkBracketExpr and layoutType.len == 3,
    "gemm_mma: expected a Layout[Shape, Stride] type, got " & layoutType.repr
  let shape = layoutType[1]
  doAssert shape.len >= 2,
    "gemm_mma: expected a (T, V) layout shape, got " & shape.repr
  leafProduct(shape[1])

# ═════════════════════════════════════════════════════════════════════════
#  gemm_mma: one register-level MMA call
# ═════════════════════════════════════════════════════════════════════════

macro gemm_mma*(atom: static MmaAtom, scalars: varargs[typed]): untyped =
  ## One register-level MMA call over caller-staged scalars: `d = a·b + c`,
  ## `atom.gemm_mma(d0, ..., a0, ..., b0, ..., c0, ...)`:
  ## - dV result scalars,
  ## - then aV A scalars,
  ## - then bV B scalars,
  ## - then cV C scalars.
  ## Order: the atom's per-lane value order.
  ## (dV/aV/bV/cV = the atom's values per thread per operand.)

  let instr = constStr(atom, "instr")
  let dV = atom.valuesPerThread("cLayout")
  let aV = atom.valuesPerThread("aLayout")
  let bV = atom.valuesPerThread("bLayout")

  var args = scalars
  if args.kind notin {nnkArgList, nnkBracket}:
    args = newTree(nnkArgList, args)
  if args.len != 2 * dV + aV + bV:
    error("gemm_mma: expected " & $(2 * dV + aV + bV) & " staged scalars (" &
          $dV & " result + " & $aV & " A + " & $bV & " B + " & $dV &
          " addend), got " & $args.len & ". Staging from tensor cells lives " &
          "in gemm_atom (k_layout_gemm); gemm_mma takes scalars only.")

  case instr
  of "simdgroup_multiply_accumulate":
    var dNames, aNames, bNames, cNames: seq[string]
    for i in 0 ..< dV: dNames.add($scalars[i])
    for i in 0 ..< aV: aNames.add($scalars[dV + i])
    for i in 0 ..< bV: bNames.add($scalars[dV + aV + i])
    for i in 0 ..< dV: cNames.add($scalars[dV + aV + bV + i])
    let aType = constStr(atom, "aType")
    let cType = constStr(atom, "cType")
    var staged: seq[(string, NimNode, string, string)]
    for i in 0 ..< dV: staged.add((dNames[i], scalars[i], "float", "accumulator"))
    for i in 0 ..< aV: staged.add((aNames[i], scalars[dV + i], aType, "A operand"))
    for i in 0 ..< bV: staged.add((bNames[i], scalars[dV + aV + i], aType, "B operand"))
    for i in 0 ..< dV: staged.add((cNames[i], scalars[dV + aV + bV + i], cType, "addend"))
    for (name, node, dtype, operand) in staged:
      assertStagedDtype(name, node, dtype, operand)
    result = newTree(nnkAsmStmt, newEmptyNode(),
      newLit(buildAppleSimdgroupAsm(dNames, aNames, bNames, cNames, "float", aType)))
  of "":
    # the universal 1x1x1 scalar atom: one plain FMA, the result cast
    # to its own type first (the product may be narrower, say half operands)
    if dV != 1 or aV != 1 or bV != 1:
      error("gemm_mma: universal atoms are the 1x1x1 scalar FMA (vpt 1), got " &
            $(dV) & "/" & $(aV) & "/" & $(bV))
    result = quote do:
      `scalars[0]` = typeof(`scalars[0]`)(`scalars[1]` * `scalars[2]`) + `scalars[3]`
  else:
    if not instr.startsWith("mma.sync.aligned."):
      error("gemm_mma: unsupported instruction `" & instr & "`")
    var dNames, aNames, bNames, cNames: seq[string]
    for i in 0 ..< dV: dNames.add($scalars[i])
    for i in 0 ..< aV: aNames.add($scalars[dV + i])
    for i in 0 ..< bV: bNames.add($scalars[dV + aV + i])
    for i in 0 ..< dV: cNames.add($scalars[dV + aV + bV + i])
    let accType = accumulatorElementName(instr)
    # PTX mma.sync reads and writes one register set (C == D),
    # Copy C into D before calling the low-level instruction
    result = newStmtList()
    for i in 0 ..< dV:
      if cNames[i] != dNames[i]:
        result.add nnkAsgn.newTree(scalars[i], scalars[dV + aV + bV + i])
    # PTX mma.sync A/B operands are packed .b32 registers holding two
    # 16-bit values, the low half carrying the smaller k element.
    # Each packed register stages as a uint32 local and the asm statement
    # references it by bare name: the asm lowering resolves backticked
    # names against the enclosing scope, keeping redeclarations across
    # gemm_atom expansions correct
    let packedWidth = mmaOperandWidth(constStr(atom, "aType"))
    doAssert aV mod packedWidth == 0 and bV mod packedWidth == 0,
      "gemm_mma: mma operand values must pack into whole registers, got " &
      $aV & " A / " & $bV & " B values of width " & $packedWidth
    var aRegs, bRegs: seq[string]
    for i in countup(0, aV - 1, packedWidth):
      case packedWidth
      of 1:
        # one value per register: the staged scalar is the register
        aRegs.add($scalars[dV + i])
      of 2:
        let lo = scalars[dV + i]
        let hi = scalars[dV + i + 1]
        let regIdent = ident("aReg" & $(i div 2))
        result.add quote do:
          var `regIdent`: uint32 =
            cast[uint32](`lo`) or (cast[uint32](`hi`) shl 16)
        aRegs.add($regIdent)
      else:
        error("gemm_mma: register packing unimplemented for width " & $packedWidth)
    for i in countup(0, bV - 1, packedWidth):
      case packedWidth
      of 1:
        bRegs.add($scalars[dV + aV + i])
      of 2:
        let lo = scalars[dV + aV + i]
        let hi = scalars[dV + aV + i + 1]
        let regIdent = ident("bReg" & $(i div 2))
        result.add quote do:
          var `regIdent`: uint32 =
            cast[uint32](`lo`) or (cast[uint32](`hi`) shl 16)
        bRegs.add($regIdent)
      else:
        error("gemm_mma: register packing unimplemented for width " & $packedWidth)
    let asmStr = buildNvidiaMmaAsm(instr, dNames, aRegs, bRegs, accType)
    result.add newTree(nnkAsmStmt, newEmptyNode(), newLit(asmStr))
  # labeled block: the Metal printer emits braces only for labeled blocks,
  # multiple calls in one function scope must not collide on the staged scalar names
  result = newTree(nnkBlockStmt, genSym(nskLabel, "gemmMma"), result)
