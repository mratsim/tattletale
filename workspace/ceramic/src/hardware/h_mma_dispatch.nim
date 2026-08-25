## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, strutils]
import workspace/crucible

## Register-level MMA dispatch (compile-time string builder / AST emitter).
##
## Public entries:
##   - `gemm_mma`: the macro that emits the register-level MMA call — NVIDIA
##     `mma.sync` asm, or the Apple simdgroup intrinsic on Metal.
##   - `universalMma8x8x8`: the software 8×8×8 cross-lane shuffle reduction,
##     the universal FMA atoms' device path.
##   - `buildNvidiaMmaAsm`: the asm string for one NVIDIA register-level MMA.
##
# TODO: gemm_mma handles Nvidia asm + Apple simdgroup only; AMD and Intel
# tensor cores are not implemented yet.

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

# ═════════════════════════════════════════════════════════════════════════
#  gemm_mma: one register-level MMA call
# ═════════════════════════════════════════════════════════════════════════

macro gemm_mma*(instr: static string; dV, aV, bV: static int;
                dFrag, aFrag, bFrag: untyped): untyped =
  ## Tensor-core-level Matrix-Multiplication
  ##
  ## Args:
  ##   instr: the mma.sync mnemonic, or "simdgroup_multiply_accumulate"
  ##     for the Apple simdgroup atoms
  ##   dV, aV, bV: per-operand register counts (V per thread)
  ##   dFrag: the accumulator fragment tensor, seeded to the asm output
  ##     and written back (in-place accumulate)
  ##   aFrag, bFrag: the operand fragment tensors, read-only
  ##
  ## The Apple simdgroup atoms emit a call to the `simdgroupMultiplyAccumulate`
  ## builtin (the MSL printer maps it to the simdgroup intrinsic); the NVIDIA
  ## atoms keep the extended-asm path below.
  ##
  ## TODO:
  ##   Hardcoded element types:
  ##     float32 accumulator
  ##     uint32 operands (TF32)
  case instr
  of "simdgroup_multiply_accumulate":
    return newCall(ident("simdgroupMultiplyAccumulate"), dFrag, aFrag, bFrag)
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
