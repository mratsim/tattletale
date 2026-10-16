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

func buildNvidiaMmaAsm*(instr: string, dNames, aNames, bNames: openArray[string], accType: string): string =
  ## GCC extended-asm builder for one Nvidia MMA instruction,
  ## D = A·B + D accumulated in place: the C operand is the D registers,
  ## modified ("+" read-write constraint).
  ##
  ## Expected input:
  ##   - `instr`: the mma.sync.aligned.* instruction string
  ##   - `dNames`/`aNames`/`bNames`: the staged scalars' names, one
  ##     register per value, in the atom's per-lane value order
  ##   - `accType`: the accumulator register element type name, the A/B
  ##     operands are always packed uint32 registers
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

func buildAppleSimdgroupAsm(dNames, aNames, bNames: openArray[string], accType, inType: string): string =
  ## Metal codegen for one simdgroup_multiply_accumulate builtin over
  ## caller-staged scalars, D = A·B + D. The accumulator is always fp32
  ## (`simdgroup_float8x8`), A and B carry the atom's operand dtype.
  ##
  ## Expected input:
  ##   - `dNames`/`aNames`/`bNames`: the staged scalars' names, one
  ##     fragment value each, 2 per lane on the 8×8×8 atoms, in per-lane value order
  ##   - `accType`/`inType`: the MSL element names, the accumulator is fp32
  ##     ("float"), A and B carry the atom's operand dtype
  ## Output: one braced MSL statement string.
  ## The braced block keeps the `simdgroup_*8x8` declarations in one scope,
  ## several mma_AB payloads unroll into the same function scope
  ## and must not collide on the declaration names.
  result = "{\n"
  result.add "  simdgroup_" & accType & "8x8 sd = make_filled_simdgroup_matrix<" & accType &
            ", 8>(" & accType & "(0.0f));\n"
  for i, name in dNames:
    result.add "  sd.thread_elements()[" & $i & "] = `" & name & "`;\n"
  result.add "  simdgroup_" & inType & "8x8 sa = make_filled_simdgroup_matrix<" & inType &
            ", 8>(" & inType & "(0.0f));\n"
  for i, name in aNames:
    result.add "  sa.thread_elements()[" & $i & "] = `" & name & "`;\n"
  result.add "  simdgroup_" & inType & "8x8 sb = make_filled_simdgroup_matrix<" & inType &
            ", 8>(" & inType & "(0.0f));\n"
  for i, name in bNames:
    result.add "  sb.thread_elements()[" & $i & "] = `" & name & "`;\n"
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

macro gemm_mma*(atom: static MmaAtom, scalars: varargs[untyped]): untyped =
  ## One register-level MMA call over caller-staged scalars,
  ## `atom.gemm_mma(d0, ..., a0, ..., b0, ...)`: dV accumulator scalars,
  ## then aV A scalars, then bV B scalars, in the atom's per-lane value
  ## order (dV/aV/bV = the atom's values per thread per operand).
  ##
  ## Dispatch only. The scalars are read and written exactly as passed:
  ## no operand reads, no writebacks, no indexing. Staging a register
  ## tensor's cell values into the scalars is `gemm_atom`'s job
  ## (k_layout_gemm), which keeps the staging reads at its own call site.
  ##
  ## Contract:
  ##   - the accumulator scalars must be assignable locals, pre-loaded
  ##     with the accumulated value, overwritten with the result
  ##   - the asm paths (Apple, NVIDIA) reference the staged scalars by name,
  ##     so those scalars must be plain named locals, any name works
  ##
  ## Everything instruction-level is derived from the atom's registry
  ## consts (h_configgen), the mnemonic (`instr`), the per-operand
  ## values per thread (the layouts' V), and the MSL element names (`elem`).
  ##
  ## Dispatch by mnemonic:
  ##   - "simdgroup_multiply_accumulate" (Apple atoms), `nnkAsmStmt` block
  ##     (buildAppleSimdgroupAsm) rendered as raw MSL by the Metal printer
  ##   - "" (the universal 1×1×1 atoms), one plain scalar FMA
  ##   - "mma.sync.aligned.*" (NVIDIA atoms), the extended-asm path below
  ##
  ## Apple atoms always accumulate in fp32, the operand element names come
  ## from the atom's `aType` registry const (aType == bType asserted).

  let instr = constStr(atom, "instr")
  let dV = atom.valuesPerThread("cLayout")
  let aV = atom.valuesPerThread("aLayout")
  let bV = atom.valuesPerThread("bLayout")

  var args = scalars
  if args.kind != nnkArgList:
    args = newTree(nnkArgList, args)
  if args.len != dV + aV + bV:
    error("gemm_mma: expected " & $(dV + aV + bV) & " staged scalars (" &
          $dV & " accumulator + " & $aV & " A + " & $bV & " B), got " &
          $args.len & ". Staging from tensor cells lives in gemm_atom " &
          "(k_layout_gemm); gemm_mma takes scalars only.")

  case instr
  of "simdgroup_multiply_accumulate":
    var dNames, aNames, bNames: seq[string]
    for i in 0 ..< dV: dNames.add($scalars[i])
    for i in 0 ..< aV: aNames.add($scalars[dV + i])
    for i in 0 ..< bV: bNames.add($scalars[dV + aV + i])
    let aType = constStr(atom, "aType")
    let bType = constStr(atom, "bType")
    doAssert aType == bType,
      "gemm_mma: the Apple simdgroup intrinsic requires aType == bType on atom `" &
      $atom & "`"
    if aType.len == 0:
      error("gemm_mma: atom `" & $atom & "` is missing the `aType` registry " &
            "property (\"float\"/\"half\"/\"bfloat\") for the Apple staging payload")
    result = newTree(nnkAsmStmt, newEmptyNode(),
      newLit(buildAppleSimdgroupAsm(dNames, aNames, bNames, "float", aType)))
  of "":
    # the universal 1x1x1 scalar atom: one plain FMA, the accumulator cast
    # to its own type first (the product may be narrower, say half operands)
    if dV != 1 or aV != 1 or bV != 1:
      error("gemm_mma: universal atoms are the 1x1x1 scalar FMA (vpt 1), got " &
            $(dV) & "/" & $(aV) & "/" & $(bV))
    result = quote do:
      `scalars[0]` = typeof(`scalars[0]`)(`scalars[1]` * `scalars[2]`) + `scalars[0]`
  else:
    if not instr.startsWith("mma.sync.aligned."):
      error("gemm_mma: unsupported instruction `" & instr & "`")
    var dNames, aNames, bNames: seq[string]
    for i in 0 ..< dV: dNames.add($scalars[i])
    for i in 0 ..< aV: aNames.add($scalars[dV + i])
    for i in 0 ..< bV: bNames.add($scalars[dV + aV + i])
    let accType = accumulatorElementName(instr)
    let asmStr = buildNvidiaMmaAsm(instr, dNames, aNames, bNames, accType)
    result = newTree(nnkAsmStmt, newEmptyNode(), newLit(asmStr))
  # labeled block: the Metal printer emits braces only for labeled blocks,
  # multiple calls in one function scope must not collide on the staged scalar names
  result = newTree(nnkBlockStmt, genSym(nskLabel, "gemmMma"), result)
