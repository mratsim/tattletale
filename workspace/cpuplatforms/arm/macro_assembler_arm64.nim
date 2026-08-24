## Compile-time macro assembler for SVE/SME inline asm on Apple M4 (arm64).
##
## Record instructions and operand bindings, then `generate()` emits one
## `{.emit: "asm volatile(...)".}` pragma with the operand and clobber lists
## built from the recorded state. Operand bindings embed the Nim symbol nodes
## in the emit, so the C backend renders the real (mangled) C names.
##
## Register namespace: `x(n)`/`w(n)` general-purpose, `z(n)` SVE vectors,
## `p(n)` predicates, `pn(n)` predicate-as-counter, `v(n)` NEON vectors,
## `zaTile(n, shape)` ZA accumulators, `zaSlice(name, shape, idx, off, hi)`
## ZA tile slices.
##
## Apple clang rejects the textual SME2 multi-vector forms. Those instructions
## are emitted as `.inst` words by `smstart`, `smstop`, `zeroZa0`, `zeroZad`,
## `ptruePn9B`, `ld1w2`, and `incw`. M4 word formulas live in the doc
## comments of those procs. Clobber lists are explicit per kernel: `clobberZ`,
## `clobberP`, `clobberV`, plus named registers, `cc`, and `memory`.

import std/[macros, strutils]

type
  XReg* = distinct int  ## 64-bit general-purpose register, `x0`..`x30`.
  WReg* = distinct int  ## 32-bit view of a general-purpose register, `w0`..`w30`.
  ZReg* = distinct int  ## SVE vector register, `z0`..`z31`.
  PReg* = distinct int  ## SVE predicate register, `p0`..`p15`.
  PNReg* = distinct int ## SME2 predicate-as-counter register, `pn0`..`pn15`.
  VReg* = distinct int  ## NEON vector register, `v0`..`v31`.

  ZaTile* = object
    ## ZA accumulator reference: `zaN.<shape>` (the `fmopa` target).
    name: string
    shape: string

  ZaSlice* = object
    ## ZA tile slice: `zaN<o>.<shape>[wM, off]` single or `[wM, lo:hi]` range.
    ## `name` carries the orientation suffix (`za0v`, `za0h`, `za1h`).
    name: string
    shape: string
    idx: string
    off: int
    rangeHi: int

  ZRange* = object
    ## SVE vector range: `{zN.<shape>-zM.<shape>}`.
    lo: ZReg
    hi: ZReg
    shape: string

  VList* = object
    ## NEON consecutive-vector list: `{vN.<shape>, ..., vM.<shape>}`.
    lo: VReg
    hi: VReg
    shape: string

  MemKind = enum
    mkBase
    mkMulVL
    mkPostIndex

  MemAddr* = object
    ## Memory operand: `[xN]`, `[xN, #imm, MUL VL]`, or post-index `[xN], #imm`.
    base: XReg
    kind: MemKind
    imm: int

  AsmOperand* = object
    ## Named asm operand: rendered in text as `%[id]`/`%w[id]`/`%x[id]`,
    ## bound in the constraint list as `[id] "r"(<C expr>)`.
    asmId: string
    nimSymbol: NimNode
    prefix: string
    suffix: string

  AssemblerSME* = object
    ## Compile-time assembler state: instruction text plus operand and clobber lists.
    code: string
    inputs: seq[AsmOperand]
    clobbers: seq[string]

func init*(T: type AssemblerSME): AssemblerSME = discard

# Register constructors and rendering
# -----------------------------------------------------------------------------

func x*(n: int): XReg = XReg(n)
func w*(n: int): WReg = WReg(n)
func z*(n: int): ZReg = ZReg(n)
func p*(n: int): PReg = PReg(n)
func pn*(n: int): PNReg = PNReg(n)
func v*(n: int): VReg = VReg(n)

func `$`*(r: XReg): string = "x" & $int(r)
func `$`*(r: WReg): string = "w" & $int(r)
func `$`*(r: ZReg): string = "z" & $int(r)
func `$`*(r: PReg): string = "p" & $int(r)
func `$`*(r: PNReg): string = "pn" & $int(r)
func `$`*(r: VReg): string = "v" & $int(r)

func zaTile*(n: int, shape: string): ZaTile =
  ## ZA accumulator `zaN.<shape>` (n in 0..3 on M4's 4-tile file).
  ZaTile(name: "za" & $n, shape: shape)

func zaSlice*(name, shape, idx: string, off: int, rangeHi = -1): ZaSlice =
  ## ZA tile slice: `name.shape[idx, off]` or `name.shape[idx, off:rangeHi]`.
  ## `name` is the full tile prefix (`za0v`, `za0h`, `za1h`), `idx` a w
  ## register (`w13`, `w12`), `rangeHi` -1 for a single-element slice.
  ZaSlice(name: name, shape: shape, idx: idx, off: off, rangeHi: rangeHi)

func zrng*(lo, hi: int, shape: string): ZRange =
  ## SVE vector range `{zN.<shape>-zM.<shape>}` (M4 SME2 mova/fclamp forms).
  ZRange(lo: z(lo), hi: z(hi), shape: shape)

func vlist*(lo, hi: int, shape: string): VList =
  ## NEON vector list `{vN.<shape>, ..., vM.<shape>}`, consecutive registers.
  VList(lo: v(lo), hi: v(hi), shape: shape)

func mem*(base: XReg): MemAddr =
  ## Memory operand `[base]`.
  MemAddr(base: base, kind: mkBase)

func memVL*(base: XReg, imm: int): MemAddr =
  ## Memory operand `[base, #imm, MUL VL]` (imm in -8..7 vector lengths).
  MemAddr(base: base, kind: mkMulVL, imm: imm)

func memPost*(base: XReg, imm: int): MemAddr =
  ## Post-indexed memory operand `[base], #imm`.
  MemAddr(base: base, kind: mkPostIndex, imm: imm)

func `$`*(t: ZaTile): string = t.name & "." & t.shape
func `$`*(s: ZaSlice): string =
  result = s.name & "." & s.shape & "[" & s.idx & ", " & $s.off
  if s.rangeHi >= 0:
    result.add ":" & $s.rangeHi
  result.add "]"
func `$`*(r: ZRange): string =
  "{" & $r.lo & "." & r.shape & "-" & $r.hi & "." & r.shape & "}"
func `$`*(l: VList): string =
  result = "{"
  for n in int(l.lo) .. int(l.hi):
    if n > int(l.lo):
      result.add ", "
    result.add "v" & $n & "." & l.shape
  result.add "}"
func `$`*(m: MemAddr): string =
  case m.kind
  of mkBase:     "[" & $m.base & "]"
  of mkMulVL:    "[" & $m.base & ", #" & $m.imm & ", MUL VL]"
  of mkPostIndex: "[" & $m.base & "], #" & $m.imm

# Operand bindings
# -----------------------------------------------------------------------------

proc addOperand(a: var AssemblerSME, op: AsmOperand) =
  for existing in a.inputs:
    if existing.asmId == op.asmId:
      return
  a.inputs.add op

proc input*(a: var AssemblerSME, id: string, sym: NimNode): AsmOperand =
  ## Binds `[id] "r"(sym)` for a pointer or 64-bit value.
  result = AsmOperand(
    asmId: id, nimSymbol: sym,
    prefix: "[" & id & "] \"r\"(", suffix: ")")
  a.addOperand result

proc inputLong*(a: var AssemblerSME, id: string, sym: NimNode): AsmOperand =
  ## Binds `[id] "r"((long)sym)`. Aarch64 `"r"` is a 64-bit register,
  ## so a 32-bit `cint` value needs the widening cast to match the original
  ## kernels' operand text.
  result = AsmOperand(
    asmId: id, nimSymbol: sym,
    prefix: "[" & id & "] \"r\"((long)", suffix: ")")
  a.addOperand result

proc inputLongParen*(a: var AssemblerSME, id: string, sym: NimNode): AsmOperand =
  ## Binds `[id] "r"((long)(sym))` for a 32-bit value whose Nim source already
  ## uses parentheses (the alpha/beta bit patterns).
  result = AsmOperand(
    asmId: id, nimSymbol: sym,
    prefix: "[" & id & "] \"r\"((long)(", suffix: "))")
  a.addOperand result

proc inputLit*(a: var AssemblerSME, id: string, value: int64): AsmOperand =
  ## Binds `[id] "r"(value)` for a compile-time literal (the ReLU +inf word).
  result = AsmOperand(
    asmId: id, nimSymbol: newLit(value),
    prefix: "[" & id & "] \"r\"(", suffix: ")")
  a.addOperand result

func view*(op: AsmOperand): string = "%[" & op.asmId & "]"
func wview*(op: AsmOperand): string = "%w[" & op.asmId & "]"
func xview*(op: AsmOperand): string = "%x[" & op.asmId & "]"

# Clobber tracking
# -----------------------------------------------------------------------------

proc clobber*(a: var AssemblerSME, regs: varargs[string]) =
  ## Records registers for the clobber list, preserving call order.
  for r in regs:
    if r notin a.clobbers:
      a.clobbers.add r

proc clobberZ*(a: var AssemblerSME) =
  ## Clobbers `z0`..`z31` (smstart zeroes the Z/V register file on M4).
  for i in 0 .. 31:
    a.clobber "z" & $i

proc clobberP*(a: var AssemblerSME) =
  ## Clobbers `p0`..`p15`.
  for i in 0 .. 15:
    a.clobber "p" & $i

proc clobberV*(a: var AssemblerSME) =
  ## Clobbers `v0`..`v31` (the NEON register file, low halves of z0..z31).
  for i in 0 .. 31:
    a.clobber "v" & $i

proc clobberCC*(a: var AssemblerSME) =
  ## Clobbers the condition flags.
  a.clobber "cc"

proc clobberMemory*(a: var AssemblerSME) =
  ## Clobbers memory (the kernels read and write through raw pointers).
  a.clobber "memory"

func generate*(a: AssemblerSME): NimNode =
  ## Returns the `{.emit: ...}` pragma for the recorded code.
  ## String fragments and Nim symbol nodes are concatenated inside the emit,
  ## so the C backend renders each symbol's real C name.
  var frags: seq[NimNode]
  var body = "\nasm volatile(\n"
  for line in a.code.split('\n'):
    if line.len > 0:
      body.add "    \"" & line & "\\n\"\n"
  body.add "    :\n    : "
  frags.add newLit(body)
  for i, op in a.inputs:
    if i > 0:
      frags.add newLit(", ")
    frags.add newLit(op.prefix)
    frags.add op.nimSymbol
    frags.add newLit(op.suffix)
  var tail = "\n    : "
  for i, c in a.clobbers:
    if i > 0:
      tail.add ", "
    tail.add "\"" & c & "\""
  tail.add "\n);"
  frags.add newLit(tail)
  result = nnkPragma.newTree(
    nnkExprColonExpr.newTree(ident"emit", nnkBracket.newTree(frags)))
  result = nnkBlockStmt.newTree(newEmptyNode(), result)

# Raw words and SME2 escape hatch
# -----------------------------------------------------------------------------

proc inst*(a: var AssemblerSME, word: int) =
  ## Raw instruction word: `.inst 0x%08x`.
  ## Escape hatch for SME2-only instructions Apple clang rejects textually.
  a.code.add ".inst 0x" & toHex(word, 8).toLowerAscii & '\n'

proc smstart*(a: var AssemblerSME) =
  ## `smstart` as a raw word: `.inst 0xd503477f`.
  ## Full form on M4 (streaming + ZA). It zeroes the Z/V register file,
  ## whose NEON v-registers are the low 128 bits of the SVE z-registers.
  ## Kernels inside the bracket must clobber all z and p registers.
  a.inst 0xd503477f

proc smstop*(a: var AssemblerSME) =
  ## `smstop` as a raw word: `.inst 0xd503467f`.
  a.inst 0xd503467f

proc zeroZa0*(a: var AssemblerSME) =
  ## `zero {za0.s}` as a raw word: `.inst 0xc0080001` (single-tile kernels).
  a.inst 0xc0080001

proc zeroZad*(a: var AssemblerSME) =
  ## `zero {zad0..zad7}` as a raw word: `.inst 0xc00800ff`. Clears ZA0..ZA3.
  a.inst 0xc00800ff

proc ptruePn9B*(a: var AssemblerSME) =
  ## `ptrue pn9.b` as a raw word: `.inst 0x25207811` (SME2 predicate-as-counter).
  a.inst 0x25207811

proc ld1w2*(a: var AssemblerSME, zt1, xn, rm: int) =
  ## SME2 dual-vector load `ld1w {zt1, zt1+8}, pn9.b/Z, [xn, rm, LSL #2]`
  ## as a raw word. Word formula: `0xa1000000 | (Rm << 16) | (17 << 10)
  ## | (Xn << 5) | Zt1`. Second vector is implied Zt1 + 8.
  ## Apple clang rejects the textual form, so this raw word is the only
  ## rendering of the instruction.
  a.inst 0xa1000000 or (rm shl 16) or (17 shl 10) or (xn shl 5) or zt1

proc incw*(a: var AssemblerSME, xn: int) =
  ## `incw xn, ALL, MUL #2` as a raw word: `.inst 0x04b1e3f0 | Xn`.
  ## Increments a vector counter by 32 elements (two 64-B vectors).
  a.inst 0x04b1e3f0 or xn

# Instructions
# -----------------------------------------------------------------------------

proc ptrue*(a: var AssemblerSME, pr: PReg, shape: string) =
  ## Predicate-true: `ptrue <pr>.<shape>`, shape "s" (32-bit lanes) or "b".
  a.code.add "ptrue " & $pr & "." & shape & '\n'

proc label*(a: var AssemblerSME, name: string) =
  ## Local numeric label: `<name>:`. Branch targets append `b` (backward)
  ## or `f` (forward) to the same name.
  a.code.add name & ":\n"

proc mov*(a: var AssemblerSME, dst: XReg or WReg, src: string) =
  ## Move into a GPR: `mov <dst>, <src>`, where `src` is a rendered
  ## operand view (`%[pa]`, `%w[kc]`) or an immediate (`#0`).
  a.code.add "mov " & $dst & ", " & src & '\n'

proc mov*(a: var AssemblerSME, dst: XReg or WReg, src: XReg or WReg) =
  ## Move between GPRs: `mov <dst>, <src>`.
  a.code.add "mov " & $dst & ", " & $src & '\n'

proc mov*(a: var AssemblerSME, dst: ZReg, shape: string, imm: int) =
  ## SVE vector immediate move: `mov <dst>.<shape>, #<imm>`
  ## (the zeroing form, e.g. `mov z4.s, #0`. SVE `movi` is not accepted
  ## by Apple clang).
  a.code.add "mov " & $dst & "." & shape & ", #" & $imm & '\n'

proc mov*(a: var AssemblerSME, dst: ZRange, src: ZaSlice) =
  ## SME2 tile-slice read into vectors: `mov <dst>, <src>`.
  ## On M4 a `.h` column group of a `.s` tile holds one output row, which is
  ## what makes the transposed-tile extract work.
  a.code.add "mov " & $dst & ", " & $src & '\n'

proc add*(a: var AssemblerSME, dst: XReg or WReg, src: XReg or WReg, imm: int) =
  ## Add immediate: `add <dst>, <src>, #<imm>`.
  a.code.add "add " & $dst & ", " & $src & ", #" & $imm & '\n'

proc add*(a: var AssemblerSME, dst: XReg, src: XReg, rhs: XReg) =
  ## Add registers: `add <dst>, <src>, <rhs>`.
  a.code.add "add " & $dst & ", " & $src & ", " & $rhs & '\n'

proc addLsl*(a: var AssemblerSME, dst: XReg, src: XReg, rhs: XReg, shift: int) =
  ## Add with shifted register: `add <dst>, <src>, <rhs>, LSL #<shift>`.
  a.code.add "add " & $dst & ", " & $src & ", " & $rhs & ", LSL #" & $shift & '\n'

proc addvl*(a: var AssemblerSME, dst: XReg, src: XReg, imm: int) =
  ## Add vector length: `addvl <dst>, <src>, #<imm>` (imm in -32..31).
  a.code.add "addvl " & $dst & ", " & $src & ", #" & $imm & '\n'

proc sub*(a: var AssemblerSME, dst: XReg or WReg, src: XReg or WReg, imm: int) =
  ## Subtract immediate: `sub <dst>, <src>, #<imm>`.
  a.code.add "sub " & $dst & ", " & $src & ", #" & $imm & '\n'

proc sub*(a: var AssemblerSME, dst: XReg, src: XReg, rhs: XReg) =
  ## Subtract registers: `sub <dst>, <src>, <rhs>`.
  a.code.add "sub " & $dst & ", " & $src & ", " & $rhs & '\n'

proc subs*(a: var AssemblerSME, dst: WReg, src: WReg, imm: int) =
  ## Subtract immediate setting flags: `subs <dst>, <src>, #<imm>`
  ## (the k-loop counter decrement feeding `b.ne`).
  a.code.add "subs " & $dst & ", " & $src & ", #" & $imm & '\n'

proc lsr*(a: var AssemblerSME, dst: WReg, src: WReg, imm: int) =
  ## Logical shift right immediate: `lsr <dst>, <src>, #<imm>`.
  a.code.add "lsr " & $dst & ", " & $src & ", #" & $imm & '\n'

proc lsl*(a: var AssemblerSME, dst: XReg, src: string, imm: int) =
  ## Logical shift left immediate: `lsl <dst>, <src>, #<imm>`, where `src`
  ## is an operand view (`%x[cs]`).
  a.code.add "lsl " & $dst & ", " & src & ", #" & $imm & '\n'

proc `and`*(a: var AssemblerSME, dst: WReg, src: WReg, imm: int) =
  ## Bitwise and immediate: `and <dst>, <src>, #<imm>`.
  a.code.add "and " & $dst & ", " & $src & ", #" & $imm & '\n'

proc cmp*(a: var AssemblerSME, reg: WReg, imm: int) =
  ## Compare immediate setting flags: `cmp <reg>, #<imm>`
  ## (feeds the `b.lt`/`b.ne` pack branch guards).
  a.code.add "cmp " & $reg & ", #" & $imm & '\n'

proc cbz*(a: var AssemblerSME, reg: WReg, target: string) =
  ## Compare and branch if zero: `cbz <reg>, <target>`.
  a.code.add "cbz " & $reg & ", " & target & '\n'

proc cbz*(a: var AssemblerSME, reg: string, target: string) =
  ## Compare and branch if zero on a rendered operand view: `cbz <reg>, <target>`.
  a.code.add "cbz " & reg & ", " & target & '\n'

proc cbnz*(a: var AssemblerSME, reg: WReg, target: string) =
  ## Compare and branch if nonzero: `cbnz <reg>, <target>`.
  a.code.add "cbnz " & $reg & ", " & target & '\n'

proc cbnz*(a: var AssemblerSME, reg: string, target: string) =
  ## Compare and branch if nonzero on a rendered operand view: `cbnz <reg>, <target>`.
  a.code.add "cbnz " & reg & ", " & target & '\n'

proc bne*(a: var AssemblerSME, target: string) =
  ## Branch if not equal: `b.ne <target>` (the k-loop back edge after `subs`).
  a.code.add "b.ne " & target & '\n'

proc blt*(a: var AssemblerSME, target: string) =
  ## Branch if less than: `b.lt <target>` (the pack partial-row skip).
  a.code.add "b.lt " & target & '\n'

proc b*(a: var AssemblerSME, target: string) =
  ## Unconditional branch: `b <target>`.
  a.code.add "b " & target & '\n'

proc ld1w*(a: var AssemblerSME, zr: ZReg, shape: string, pr: PReg, mem: MemAddr) =
  ## Predicated single-vector load: `ld1w {<zr>.<shape>}, <pr>/z, <mem>`.
  a.code.add "ld1w {" & $zr & "." & shape & "}, " & $pr & "/z, " & $mem & '\n'

proc st1w*(a: var AssemblerSME, zr: ZReg, shape: string, pr: PReg, mem: MemAddr) =
  ## Streaming single-vector store: `st1w {<zr>.<shape>}, <pr>, <mem>`.
  ## Apple clang rejects the `/z` and `/m` suffixes on the store predicate,
  ## so the store form takes a bare predicate.
  a.code.add "st1w {" & $zr & "." & shape & "}, " & $pr & ", " & $mem & '\n'

proc st1w*(a: var AssemblerSME, slice: ZaSlice, pr: PReg, mem: MemAddr) =
  ## Tile-slice store: `st1w {<slice>}, <pr>, <mem>`.
  ## Index register needs the explicit `, 0` offset field. Bare `[wN]`
  ## form is rejected by Apple clang. M4 accepts a 2-entry source-
  ## address tracker only, so kernels must not alternate more than two base
  ## registers between consecutive tile-slice stores.
  a.code.add "st1w {" & $slice & "}, " & $pr & ", " & $mem & '\n'

proc fmopa*(a: var AssemblerSME, tile: ZaTile, pg1, pg2: PReg, zn, zm: ZReg) =
  ## Fused outer-product accumulate: `fmopa <tile>, <pg1>/m, <pg2>/m, <zn>.s, <zm>.s`.
  ## M4 computes `ZA[i][j] += Y[i] * X[j]` for operands `(X, Y)`,
  ## transposed vs the ARM-doc convention.
  ## AB-store kernels pass (B, A). The epi kernel passes (A, B), whose
  ## transposed tile feeds the column-wise mova extract.
  ## Swapping the order silently transposes every tile, detectable only with asymmetric data.
  a.code.add "fmopa " & $tile & ", " & $pg1 & "/m, " & $pg2 & "/m, " & $zn & ".s, " & $zm & ".s" & '\n'

proc mova*(a: var AssemblerSME, zrng: ZRange, slice: ZaSlice) =
  ## SME2 tile-slice read to vectors: `mova <zrng>, <slice>`.
  ## `slice` is a column group (`zaNh.h[w12, 0:3]`). On M4 a `.h` column
  ## of a `.s` tile holds one output row, which is the transposed-tile extract.
  a.code.add "mova " & $zrng & ", " & $slice & '\n'

proc mova*(a: var AssemblerSME, slice: ZaSlice, zrng: ZRange) =
  ## SME2 vector-to-tile write: `mova <slice>, <zrng>`.
  ## Column-group form (`zaNh.s[w12, 0:3]`, w12 in {0, 4, 8, 12}), used for
  ## the A-transpose pack: z-to-tile rows, then the tile-to-z read
  ## transposes.
  a.code.add "mova " & $slice & ", " & $zrng & '\n'

proc fmul*(a: var AssemblerSME, zd: ZReg, shape: string, pr: PReg, zn, zm: ZReg) =
  ## Predicated SVE multiply: `fmul <zd>.<shape>, <pr>/m, <zn>.<shape>, <zm>.<shape>`.
  a.code.add "fmul " & $zd & "." & shape & ", " & $pr & "/m, " & $zn & "." & shape & ", " & $zm & "." & shape & '\n'

proc fmul*(a: var AssemblerSME, vd: VReg, shape: string, vn, vm: VReg) =
  ## NEON multiply: `fmul <vd>.<shape>, <vn>.<shape>, <vm>.<shape>`.
  a.code.add "fmul " & $vd & "." & shape & ", " & $vn & "." & shape & ", " & $vm & "." & shape & '\n'

proc fadd*(a: var AssemblerSME, zd: ZReg, shape: string, pr: PReg, zn, zm: ZReg) =
  ## Predicated SVE add: `fadd <zd>.<shape>, <pr>/m, <zn>.<shape>, <zm>.<shape>`.
  a.code.add "fadd " & $zd & "." & shape & ", " & $pr & "/m, " & $zn & "." & shape & ", " & $zm & "." & shape & '\n'

proc fadd*(a: var AssemblerSME, vd: VReg, shape: string, vn, vm: VReg) =
  ## NEON add: `fadd <vd>.<shape>, <vn>.<shape>, <vm>.<shape>`.
  a.code.add "fadd " & $vd & "." & shape & ", " & $vn & "." & shape & ", " & $vm & "." & shape & '\n'

proc fmax*(a: var AssemblerSME, vd: VReg, shape: string, vn, vm: VReg) =
  ## NEON maximum: `fmax <vd>.<shape>, <vn>.<shape>, <vm>.<shape>`.
  ## NaN propagates through the NEON `fmax`, unlike the SVE `fclamp` ReLU
  ## which maps NaN to 0.
  a.code.add "fmax " & $vd & "." & shape & ", " & $vn & "." & shape & ", " & $vm & "." & shape & '\n'

proc fclamp*(a: var AssemblerSME, zrng: ZRange, zmin, zmax: ZReg) =
  ## SVE clamp: `fclamp <zrng>, <zmin>.s, <zmax>.s`.
  ## M4: `fclamp(NaN)` yields the min operand. The ReLU form with zmin = 0
  ## and zmax = +inf maps NaN to 0 like the scalar epilogue.
  a.code.add "fclamp " & $zrng & ", " & $zmin & ".s, " & $zmax & ".s" & '\n'

proc dup*(a: var AssemblerSME, zr: ZReg, shape: string, src: string) =
  ## SVE broadcast from a GPR: `dup <zr>.<shape>, <src>`, where `src` is
  ## an operand view (`%w[abits]`).
  a.code.add "dup " & $zr & "." & shape & ", " & src & '\n'

proc dup*(a: var AssemblerSME, vr: VReg, shape: string, src: string) =
  ## NEON broadcast from a GPR: `dup <vr>.<shape>, <src>`.
  a.code.add "dup " & $vr & "." & shape & ", " & src & '\n'

proc movi*(a: var AssemblerSME, vr: VReg, shape: string, imm: int) =
  ## NEON immediate move: `movi <vr>.<shape>, #<imm>` (the zeroing form).
  a.code.add "movi " & $vr & "." & shape & ", #" & $imm & '\n'

proc ld1*(a: var AssemblerSME, vl: VList, mem: MemAddr) =
  ## NEON multi-vector load: `ld1 <vl>, <mem>`, post-indexed form only
  ## (Apple clang rejects plain `[xN, #imm]` offsets for 4-register ld1).
  a.code.add "ld1 " & $vl & ", " & $mem & '\n'

proc st1*(a: var AssemblerSME, vl: VList, mem: MemAddr) =
  ## NEON multi-vector store: `st1 <vl>, <mem>`, post-indexed form only.
  a.code.add "st1 " & $vl & ", " & $mem & '\n'

proc stp*(a: var AssemblerSME, vd1, vd2: VReg, base: XReg, imm = 0) =
  ## NEON register-pair store: `stp q<vd1>, q<vd2>, [<base>]`,
  ## plus `[<base>, #<imm>]` when `imm != 0`. Renders the `q` (128-bit) register names.
  ## `imm` is a plain byte offset, not
  ## a `MUL VL` step. Operand is a base register plus immediate,
  ## not a `MemAddr`.
  a.code.add "stp q" & $int(vd1) & ", q" & $int(vd2) & ", [" & $base
  if imm != 0:
    a.code.add ", #" & $imm
  a.code.add "]\n"

proc trn1*(a: var AssemblerSME, vd: VReg, shape: string, vn, vm: VReg) =
  ## NEON transpose, first result: `trn1 <vd>.<shape>, <vn>.<shape>, <vm>.<shape>`.
  ## Half of the 8×8 transpose network used by the A-pack.
  a.code.add "trn1 " & $vd & "." & shape & ", " & $vn & "." & shape & ", " & $vm & "." & shape & '\n'

proc trn2*(a: var AssemblerSME, vd: VReg, shape: string, vn, vm: VReg) =
  ## NEON transpose, second result: `trn2 <vd>.<shape>, <vn>.<shape>, <vm>.<shape>`.
  a.code.add "trn2 " & $vd & "." & shape & ", " & $vn & "." & shape & ", " & $vm & "." & shape & '\n'
