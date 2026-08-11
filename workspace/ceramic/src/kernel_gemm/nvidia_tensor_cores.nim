## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros

## NVIDIA tensor-core mma.sync asm construction (compile-time string builder).
##
## Public entries:
##   - `buildNvidiaMmaAsm`: the asm string for one register-level MMA.
##   - `gemm_mma`: the macro that emits the asm proc + the call.
## The Nvidia prefix keeps this builder distinct from AMD/Intel ones in the
## gemm dispatch.
##
## The fragment arrays are already register-typed by construction, so the
## constraint letters derive from their element types here.

func constraintLetter(elemTypeName: string): string =
  ## Nim DSL register element type → GCC asm constraint letter.
  ## tf32/f16/bf16/int fragments travel in integer registers ("r").
  ## f32/f64 accumulators in float registers ("f"/"d").
  case elemTypeName
  of "float32": "f"
  of "float64": "d"
  of "uint32", "uint16", "uint8", "int32", "int16", "int8": "r"
  else:
    raiseAssert "unsupported register element type: " & elemTypeName

func regList(first, count: int): string =
  ## "{%0,%1,...,%N-1}": the GCC operand register list for `count`
  ## registers starting at `first` (the %N numbering).
  result = "{"
  for i in first ..< first + count:
    if i > first: result.add ","
    result.add "%" & $i
  result.add "}"

func operandClause(name, letter: string; count: int; backtick: bool = false): string =
  ## The constraint clause for `count` registers.
  ## Array mode (default): `"letter"(name[0]), "letter"(name[1]), ...` with
  ## operands as fragment array elements (the gemm_atom form).
  ## Backtick mode: `"letter"(`name0`), ...` with operands as scalar
  ## register params (the gemm_mma proc formals). Nim resolves the
  ## backticked name to the param symbol.
  result = ""
  for i in 0 ..< count:
    if result.len > 0: result.add ", "
    if backtick:
      result.add "\"" & letter & "\"(`" & name & $i & "`)"
    else:
      result.add "\"" & letter & "\"(" & name & "[" & $i & "])"

func buildNvidiaMmaAsm*(instr: string; va, vb, vc: int;
                        dName, aName, bName, cName: string;
                        dElem, aElem, bElem, cElem: string;
                        backtick: bool = false): string =
  ## Full GCC extended-asm string for one NVIDIA register-level MMA.
  ##
  ## %N numbering follows the hardware operand order (V-order explode):
  ##   D = {%0..%vc-1}  A = {%vc..%vc+va-1}  B = {%vc+va..%vc+va+vb-1}
  ##   C = aliased ? {%0..%vc-1} (D, in-place accumulate: the mma.sync
  ##       output registers are the C operand) : {next %vc..}
  ##
  ## Args:
  ##   instr: atom mnemonic, e.g. "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
  ##   va, vb, vc: fragment register counts (V per thread per operand)
  ##   dName/aName/bName/cName: fragment array identifiers in the kernel
  ##   dElem/aElem/bElem/cElem: fragment element type names, mapped to
  ##     constraint letters ("r" for integer regs, "f" for float acc)
  ##   backtick: operand format: scalar register params (true, gemm_mma)
  ##     or fragment array elements (false, gemm_atom)
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
    " : " & operandClause(dName, (if aliased: "+" else: "=") & constraintLetter(dElem), vc, backtick) &
    " : " & operandClause(aName, constraintLetter(aElem), va, backtick) &
    ", " & operandClause(bName, constraintLetter(bElem), vb, backtick)
  if not aliased:
    result.add ", " & operandClause(cName, constraintLetter(cElem), vc, backtick)

# ═════════════════════════════════════════════════════════════════════════
#  gemm_mma: the explode equivalent, one register-level MMA call
# ═════════════════════════════════════════════════════════════════════════

macro gemm_mma*(instr: static string; aV, bV, dV: static int;
                dFrag, aFrag, bFrag: untyped): untyped =
  ## Tensor-core-level Matrix-Multiplication
  ##
  ## TODO:
  ##   Hardcoded element types:
  ##     float32 accumulator
  ##     uint32 operands (TF32)
  let dElem = "float32"
  let aElem = "uint32"
  let bElem = "uint32"
  let asmStr = buildNvidiaMmaAsm(instr, aV, bV, dV, "d", "a", "b", "d",
                                 dElem, aElem, bElem, dElem,
                                 backtick = true)

  # the asm proc: scalar register params
  #   d0..d(dV-1): var float32 (in-place accumulate, + constraint)
  #   a0..a(aV-1), b0..b(bV-1): uint32
  var formals = @[newEmptyNode()]
  for i in 0 ..< dV:
    formals.add newIdentDefs(ident("d" & $i), newTree(nnkVarTy, ident(dElem)), newEmptyNode())
  for i in 0 ..< aV:
    formals.add newIdentDefs(ident("a" & $i), ident(aElem), newEmptyNode())
  for i in 0 ..< bV:
    formals.add newIdentDefs(ident("b" & $i), ident(bElem), newEmptyNode())
  let asmStmt = newTree(nnkAsmStmt, newEmptyNode(), newLit(asmStr))
  let body = newStmtList(asmStmt)
  let procDef = newProc(ident"gemm_mma_impl", formals, body,
                        pragmas = nnkPragma.newTree(ident"inline"))

  # the call: element accesses on the passed tensors
  proc elemArgs(base: NimNode; v: int): seq[NimNode] =
    for i in 0 ..< v:
      result.add newCall(base, newLit(i))
  var args: seq[NimNode] = @[]
  args.add elemArgs(dFrag, dV)
  args.add elemArgs(aFrag, aV)
  args.add elemArgs(bFrag, bV)
  let call = newCall(ident"gemm_mma_impl", args)

  result = newTree(nnkBlockStmt, newEmptyNode(), newStmtList(procDef, call))
