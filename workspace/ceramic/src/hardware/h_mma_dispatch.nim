## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, strutils]
import workspace/crucible
import ./h_registry

{.experimental: "dynamicBindSym".}
# bindSym with a computed name (`$atom & "_suffix"`) from a static macro
# parameter needs this experimental mode (same as h_properties.nim).

## Register-level MMA dispatch (compile-time string builder / AST emitter).
##
## Public entries:
##   - `gemm_mma`: the atom-first register-level MMA macro — the atom's registry
##     consts (h_configgen) drive everything instruction-level: NVIDIA
##     `mma.sync` asm, the Apple simdgroup intrinsic on Metal, or the
##     universal software cross-lane shuffle reduction.
##   - `universalMma8x8x8`: the software 8×8×8 cross-lane shuffle reduction,
##     the universal FMA atoms' device path (the `gemm_mma` "universal" case
##     delegates here).
##   - `buildNvidiaMmaAsm`: the asm string for one NVIDIA register-level MMA.
##
# TODO: gemm_mma handles Nvidia asm + Apple simdgroup + the universal
# software reduction; AMD and Intel tensor cores are not implemented yet.

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

func regList(first, count: int): string =
  ## GCC operand register list "{%0,%1,...,%N-1}"
  result = "{"
  for i in first ..< first + count:
    if i > first: result.add ","
    result.add "%" & $i
  result.add "}"

func operandClause(name, letter: string, count: int): string =
  ## The constraint clause for `count` scalar registers,
  ## `"letter"(`name0`), "letter"(`name1`), ...`.
  result = ""
  for i in 0 ..< count:
    if result.len > 0: result.add ", "
    result.add "\"" & letter & "\"(`" & name & $i & "`)"

func buildNvidiaMmaAsm*(instr: string; va, vb, vc: int;
                        dName, aName, bName, cName: string;
                        dElem, aElem, bElem, cElem: string): string =
  ## GCC extended-asm builder for Nvidia Matrix-Multiply-Accumulate (MMA) instructions
  ##
  ## %N numbering follows the hardware operand order (V-order explode):
  ##   D = {%0..%vc-1}  A = {%vc..%vc+va-1}  B = {%vc+va..%vc+va+vb-1}
  ##   C = aliased ? {%0..%vc-1} (D, in-place accumulate: the mma.sync
  ##       output registers are the C operand) : {next %vc..}
  ##
  ## Args:
  ##   instr: atom mnemonic, e.g. "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
  ##   va, vb, vc: fragment register counts (V per thread per operand)
  ##   dName/aName/bName/cName: scalar register name stems. The asm
  ##     operands are `name0`, `name1`, ... (backtick identifiers)
  ##   dElem/aElem/bElem/cElem: fragment element type names, mapped to
  ##     constraint letters ("r" for integer regs, "f" for float acc)
  ##
  ## When dName == cName the caller aliases D and C (in-place accumulate):
  ## the output registers are the C operand, so the C list reuses %0..%vc-1.
  let aliased = dName == cName
  let cFirst = if aliased: 0 else: va + vb + vc

  # the template part: "<instr> {D}, {A}, {B}, {C};"
  let tpl =
    "\"" & instr &
    " " & regList(0, vc) &
    ", " & regList(vc, va) &
    ", " & regList(vc + va, vb) &
    ", " & regList(cFirst, vc) & ";\""

  result = tpl &
    " : " & operandClause(dName, (if aliased: "+" else: "=") & constraintLetter(dElem), vc) &
    " : " & operandClause(aName, constraintLetter(aElem), va) &
    ", " & operandClause(bName, constraintLetter(bElem), vb)
  if not aliased:
    result.add ", " & operandClause(cName, constraintLetter(cElem), vc)

func buildAppleSimdgroupAsm(dElem, aElem, bElem: string; dV, aV, bV: int): string =
  ## The MSL staging block for one Apple simdgroup MMA:
  ##   - one braced block per payload — mma_AB unrolls several payloads
  ##     into the same function scope, so the `simdgroup_*8x8`
  ##     declarations must not collide
  ##   - fragments are `make_filled` (uninitialized simdgroup vars trip
  ##     "used without initialization" diagnostics)
  ##   - `d0`, `a0`… backticked names are Nim asm symbols → plain MSL locals
  result = "{\n"
  result.add "  simdgroup_" & dElem & "8x8 sd = make_filled_simdgroup_matrix<" & dElem &
            ", 8>(" & dElem & "(0.0f));\n"
  for i in 0 ..< dV:
    result.add "  sd.thread_elements()[" & $i & "] = `d" & $i & "`;\n"
  result.add "  simdgroup_" & aElem & "8x8 sa = make_filled_simdgroup_matrix<" & aElem &
            ", 8>(" & aElem & "(0.0f));\n"
  for i in 0 ..< aV:
    result.add "  sa.thread_elements()[" & $i & "] = `a" & $i & "`;\n"
  result.add "  simdgroup_" & bElem & "8x8 sb = make_filled_simdgroup_matrix<" & bElem &
            ", 8>(" & bElem & "(0.0f));\n"
  for i in 0 ..< bV:
    result.add "  sb.thread_elements()[" & $i & "] = `b" & $i & "`;\n"
  result.add "  simdgroup_multiply_accumulate(sd, sa, sb, sd);\n"
  for i in 0 ..< dV:
    result.add "  `d" & $i & "` = sd.thread_elements()[" & $i & "];\n"
  result.add "}"

# ═════════════════════════════════════════════════════════════════════════
#  universalMma8x8x8: the 8×8×8 register-level MMA
# ═════════════════════════════════════════════════════════════════════════

proc universalMma8x8x8*[TD; TA; TB](
    d: var array[2, TD]; a: array[2, TA]; b: array[2, TB]) =
  ## One 8×8×8 FMA atom's cross-lane reduction: D = A·B + D.
  ##
  ## Each lane holds
  ##   A(m, n), A(m, n+1)
  ##   B(m, n), B(m, n+1)

  let lane = int(thread_index_in_threadgroup)
  let row = 4 * ((lane shr 4) and 1) + 2 * ((lane shr 2) and 1) + ((lane shr 1) and 1)
  let col = 4 * ((lane shr 3) and 1) + 2 * (lane and 1)
  let colBase = uint32((lane and 1) + 8 * ((lane shr 3) and 1))
  let srcABase = uint32(2 * row + 8 * (row div 4))
  for j in 0 ..< 4:
    let srcA = srcABase + uint32((j and 1) + (j shr 1) * 8)
    let srcB0 = colBase + uint32(4 * j + 8 * (j div 2))
    let srcB1 = colBase + uint32(4 * j + 2 + 8 * ((2 * j + 1) div 4))
    let a0 = simdShuffle(a[0], srcA)
    let a1 = simdShuffle(a[1], srcA)
    let b00 = simdShuffle(b[0], srcB0)
    let b01 = simdShuffle(b[1], srcB0)
    let b10 = simdShuffle(b[0], srcB1)
    let b11 = simdShuffle(b[1], srcB1)
    d[0] = d[0] + TD(a0) * TD(b00)   # k = 2j terms
    d[1] = d[1] + TD(a0) * TD(b01)
    d[0] = d[0] + TD(a1) * TD(b10)   # k = 2j+1 terms
    d[1] = d[1] + TD(a1) * TD(b11)

# ═════════════════════════════════════════════════════════════════════════
#  Atom registry access (macro-time)
# ═════════════════════════════════════════════════════════════════════════

template constStr(atom: untyped; suffix: untyped): string =
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

template vptOf(atom: untyped; layoutKey: untyped): int =
  ## The operand's values per thread: the layout const's (T, V) shape
  ## type's V component (the second part of the two-part shape tuple).
  ## Mirrors valuesPerThread (atoms_mma_partitioning) from the layout type
  ## alone — no layout-value evaluation needed.
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

macro gemm_mma*(atom: static MmaAtom; dFrag, aFrag, bFrag: untyped): untyped =
  ## One register-level MMA call — `atom.gemm_mma(dFrag, aFrag, bFrag)`.
  ##
  ## Everything instruction-level is derived from the atom's registry
  ## consts (h_configgen): the mnemonic (`instr`), the per-operand
  ## fragment counts (the layouts' V), and the MSL element names (`elem`).
  ##
  ## Dispatch by mnemonic:
  ##   - "simdgroup_multiply_accumulate" (Apple atoms): an `nnkAsmStmt`
  ##     staging block (buildAppleSimdgroupAsm), rendered by the Metal
  ##     printer as raw MSL. The accumulator is always fp32; the operand
  ##     element names come from the atom's `elem` registry const.
  ##   - "" (universal FMA atoms): a plain call to `universalMma8x8x8`.
  ##   - "mma.sync.aligned.*" (NVIDIA atoms): the extended-asm path below.
  ##
  ## Args:
  ##   atom: the MmaAtom enum member
  ##   dFrag: the accumulator fragment, seeded to the asm output and
  ##     written back (in-place accumulate)
  ##   aFrag, bFrag: the operand fragments, read-only
  let instr = constStr(atom, "instr")
  let dV = vptOf(atom, "cLayout")
  let aV = vptOf(atom, "aLayout")
  let bV = vptOf(atom, "bLayout")
  case instr
  of "simdgroup_multiply_accumulate":
    if dV != 2 or aV != 2 or bV != 2:
      error("gemm_mma: simdgroup_multiply_accumulate requires vpt 2 (Apple 8x8x8 atoms)")
    let aElem = constStr(atom, "elem")
    if aElem.len == 0:
      error("gemm_mma: atom `" & $atom & "` is missing the `elem` registry " &
            "property (\"float\"/\"half\"/\"bfloat\") for the Apple staging payload")
    # Scalar locals: asm operand names must be plain MSL identifiers, so
    # the fragment scalars are staged into Nim locals (bracket access: the
    # tile layer's fragments are plain arrays; legacy Tensors support `[]`).
    result = newStmtList()
    for i in 0 ..< dV:
      result.add newVarStmt(ident("d" & $i),
        newTree(nnkBracketExpr, dFrag, newLit(i)))
    for i in 0 ..< aV:
      result.add newLetStmt(ident("a" & $i),
        newTree(nnkBracketExpr, aFrag, newLit(i)))
    for i in 0 ..< bV:
      result.add newLetStmt(ident("b" & $i),
        newTree(nnkBracketExpr, bFrag, newLit(i)))
    result.add newTree(nnkAsmStmt, newEmptyNode(),
      newLit(buildAppleSimdgroupAsm("float", aElem, aElem, dV, aV, bV)))
    for i in 0 ..< dV:
      result.add newAssignment(
        newTree(nnkBracketExpr, dFrag, newLit(i)), ident("d" & $i))
    result = newTree(nnkBlockStmt, newEmptyNode(), result)
    return result
  of "":
    if dV != 2 or aV != 2 or bV != 2:
      error("gemm_mma: universal requires vpt 2 (universal 8x8x8 atoms)")
    result = newCall(bindSym("universalMma8x8x8"), dFrag, aFrag, bFrag)
    return result
  else:
    if not instr.startsWith("mma.sync.aligned."):
      error("gemm_mma: unsupported instruction `" & instr & "`")
  let dElem = "float32"
  let aElem = "uint32"
  let bElem = "uint32"
  let asmStr = buildNvidiaMmaAsm(instr, aV, bV, dV, "d", "a", "b", "d",
                                 dElem, aElem, bElem, dElem)

  # scalar register locals, one per fragment element:
  #   d0..d(dV-1): var float32, seeded from the accumulator, written back after
  #   a0..a(aV-1), b0..b(bV-1): let uint32, read from the operand tensors
  result = newStmtList()
  for i in 0 ..< dV:
    result.add newVarStmt(ident("d" & $i), newCall(dFrag, newLit(i)))
  for i in 0 ..< aV:
    result.add newLetStmt(ident("a" & $i), newCall(aFrag, newLit(i)))
  for i in 0 ..< bV:
    result.add newLetStmt(ident("b" & $i), newCall(bFrag, newLit(i)))
  result.add newTree(nnkAsmStmt, newEmptyNode(), newLit(asmStr))
  for i in 0 ..< dV:
    result.add newAssignment(newCall(dFrag, newLit(i)), ident("d" & $i))
  result = newTree(nnkBlockStmt, newEmptyNode(), result)
