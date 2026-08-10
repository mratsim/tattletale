## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## NVIDIA tensor-core mma.sync asm construction (compile-time string builder).
##
## Single public entry: `buildNvidiaMmaAsm`. AMD MFMA and Intel AMX get
## their own builders (buildAmdMmaAsm / buildAmx... in separate files) —
## the Nvidia prefix keeps them distinct in the gemm dispatch.
##
## The fragment arrays are already register-typed by construction, so the
## constraint letters derive from their element types here; only the
## mnemonic + register counts + names cross the module boundary.

func constraintLetter(elemTypeName: string): string =
  ## Nim DSL register element type → GCC asm constraint letter.
  ## tf32/f16/bf16/int fragments travel in integer registers ("r");
  ## f32/f64 accumulators in float registers ("f"/"d").
  case elemTypeName
  of "float32": "f"
  of "float64": "d"
  else: "r"

func regList(first, count: int): string =
  ## "{%0,%1,...,%N-1}" — the GCC operand register list for `count`
  ## registers starting at `first` (the %N numbering).
  result = "{"
  for i in first ..< first + count:
    if i > first: result.add ","
    result.add "%" & $i
  result.add "}"

func operandClause(name, letter: string; count: int): string =
  ## `"letter"(name[0]), "letter"(name[1]), ...` — the constraint clause
  ## for `count` registers. The operand C++ names are the fragment array
  ## identifiers (name[i]) — crucible emits the asm string verbatim, so
  ## these must be the C++ variable names.
  result = ""
  for i in 0 ..< count:
    if result.len > 0: result.add ", "
    result.add "\"" & letter & "\"(" & name & "[" & $i & "])"

func buildNvidiaMmaAsm*(instr: string; va, vb, vc: int;
                        dName, aName, bName, cName: string;
                        dElem, aElem, bElem, cElem: string): string =
  ## Full GCC extended-asm string for one NVIDIA register-level MMA.
  ##
  ## %N numbering follows the hardware operand order (V-order explode):
  ##   D = {%0..%vc-1}  A = {%vc..%vc+va-1}  B = {%vc+va..%vc+va+vb-1}
  ##   C = aliased ? {%0..%vc-1} (D, in-place accumulate — the mma.sync
  ##       output registers ARE the C operand) : {next %vc..}
  ##
  ## Args:
  ##   instr: atom mnemonic, e.g. "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
  ##   va, vb, vc: fragment register counts (V per thread per operand)
  ##   dName/aName/bName/cName: fragment array identifiers in the kernel
  ##   dElem/aElem/bElem/cElem: fragment array element types (uint32/float32/...)
  ##     → constraint letters ("r" for integer regs, "f" for float acc)
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
