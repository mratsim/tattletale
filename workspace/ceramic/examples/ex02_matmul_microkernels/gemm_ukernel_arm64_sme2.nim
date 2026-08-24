## SME2 and NEON asm micro-kernels + tile dispatch for the ex02 examples,
## float32, aarch64.
##
## Requires ARMv9.2 with FEAT_SME2 and assumes SVL = 64 B (16 f32 lanes per vector).
## arm64 CPUs without SME2 fault on these instructions.
## Larger-SVL implementations corrupt results (vector length mismatch).
## macOS SDK ships no `<arm_sme.h>`, so every SME instruction is inline asm.
## Tile-slice stores use the `st1w {zaNv.s[w13, 0]}, p1, [x4]` form.
## Assembler rejects the bare `[w13]` index and needs explicit `, 0`.
##
## Streaming-mode rules:
##   - `smstart`/`smstop` bracket every kernel call
##   - NEON instructions fault inside streaming mode, and M4 has no non-streaming SVE
##   - `.inst 0xd503477f` is a full `smstart` on M4 (streaming + ZA, zeroing the Z/V register file)
##
## Two fmopa conventions live in this file:
##   - AB-store kernels (16×16, 32×32): operands (B, A)
##   - epi kernels: operands (A, B), required by their column-wise mova extract
## Swapping the order silently transposes every tile. Only asymmetric data detects it.

{.localpassC: "-march=armv9-a+sme2 -fno-vectorize -fno-slp-vectorize".}
# SME2 is for the multi-vector `mova` extract. SVE auto-vectorization is
# disabled because emitted SVE instructions fault outside streaming mode.
import workspace/cpuplatforms/arm/macro_assembler_arm64

func memStep(base: XReg, imm: int): MemAddr =
  ## Memory operand for a k-step vector load: `[base]` when `imm == 0`,
  ## `[base, #imm, MUL VL]` otherwise.
  if imm == 0: mem(base) else: memVL(base, imm)

proc buildSme16x16(
    ctx: var AssemblerSME, pa, pb, ab, kcOp: NimNode) =
  ## Records the `sme_gemm_ukernel_16x16` instruction stream into `ctx`
  ## in source order: the k-loop over 16-lane vectors, then the 16-row
  ## tile-slice store. Operand bindings: pointers via `"r"`, the k counter
  ## via `"r"((long)...)`.
  let paOp = ctx.input("pa", pa)
  let pbOp = ctx.input("pb", pb)
  let abOp = ctx.input("ab", ab)
  let kcOp2 = ctx.inputLong("kc", kcOp)
  ctx.smstart()
  ctx.zeroZa0()
  ctx.ptrue(p(0), "s")
  ctx.ptrue(p(1), "s")
  ctx.mov(w(14), wview(kcOp2))
  ctx.cbz(w(14), "2f")
  ctx.mov(x(5), view(paOp))
  ctx.mov(x(6), view(pbOp))
  ctx.label("1")
  ctx.ld1w(z(0), "s", p(0), mem(x(5)))
  ctx.ld1w(z(4), "s", p(0), mem(x(6)))
  ctx.fmopa(zaTile(0, "s"), p(0), p(0), z(4), z(0))
  ctx.add(x(5), x(5), 64)
  ctx.add(x(6), x(6), 64)
  ctx.subs(w(14), w(14), 1)
  ctx.bne("1b")
  ctx.label("2")
  ctx.mov(x(4), view(abOp))
  for row in 0 .. 15:
    ctx.mov(w(13), "#" & $row)
    if row > 0:
      ctx.add(x(4), x(4), 64)
    ctx.st1w(zaSlice("za0v", "s", "w13", 0), p(1), mem(x(4)))
  ctx.smstop()
  ctx.clobber("x4", "x5", "x6", "w13", "w14")
  ctx.clobberZ()
  ctx.clobberP()
  ctx.clobberCC()
  ctx.clobberMemory()

macro genSme16x16(pa, pb, ab, kcOp: typed): untyped =
  ## Expands to the `{.emit: ...}` pragma for `sme_gemm_ukernel_16x16`
  var ctx = init(AssemblerSME)
  buildSme16x16(ctx, pa, pb, ab, kcOp)
  result = ctx.generate()

proc sme_gemm_ukernel_16x16*(
    packA, packB: ptr float32, AB: ptr float32, kc: cint) =
  ## 16×16 f32 SME outer-product micro-kernel:
  ## `AB[i][j] = Σ_k packA[k*16+i] * packB[k*16+j]`.
  ##
  ## Expected input:
  ##   - packA, packB: 16 contiguous f32 per k-step
  ##     (`packA[k*16 ..< k*16+16]`, same for B), the ex02a `(ir, kc, 16)`
  ##     packed layout
  ##   - kc: k-steps to contract, `kc >= 0`. A `kc == 0` call
  ##     stores a zeroed tile.
  ##   - AB: 16×16 f32 output tile, all 256 lanes written (overwritten,
  ##     never accumulated into)
  ##
  ## Output: the outer-product tile, one `fmopa` per k-step:
  ##
  ##   before (each k-step):               after (AB, 16×16):
  ##     AB += a_k ⊗ b_k                   AB[i][j] = Σ_k a_k[i] * b_k[j]
  ##     a_k = packA[k*16 ..< k*16+16]
  ##     b_k = packB[k*16 ..< k*16+16]
  ##
  ## Pointers need no alignment.
  genSme16x16(packA, packB, AB, kc)

proc storeQuadrant(
    ctx: var AssemblerSME, tileName: string, abOp: AsmOperand,
    baseOffset, rowAdvance: int) =
  ## Emits one 16-row ZA tile-slice store quadrant into AB: `mov x4, %[ab]`,
  ## an optional offset add (64/2048/2112 for the za1/za2/za3 quadrants),
  ## then per row `mov w13, #N`, `add x4, #rowAdvance`, and the slice
  ## store `st1w {zaNv.s[w13, 0]}, p1, [x4]`. Row 0 skips the advance.
  ctx.mov(x(4), view(abOp))
  if baseOffset > 0:
    ctx.add(x(4), x(4), baseOffset)
  for row in 0 .. 15:
    ctx.mov(w(13), "#" & $row)
    if row > 0:
      ctx.add(x(4), x(4), rowAdvance)
    ctx.st1w(zaSlice(tileName, "s", "w13", 0), p(1), mem(x(4)))

proc buildSme32x32(
    ctx: var AssemblerSME, pa, pb, ab, kcOp: NimNode) =
  ## Records the `sme_gemm_ukernel_32x32` instruction stream into `ctx`
  ## in source order: the block-pipelined k-loop over four ZA
  ## accumulators, the oddments tail, then the four quadrant stores.
  ## Operand bindings: pointers via `"r"`, the k counter via
  ## `"r"((long)...)`.
  let paOp = ctx.input("pa", pa)
  let pbOp = ctx.input("pb", pb)
  let abOp = ctx.input("ab", ab)
  let kcOp2 = ctx.inputLong("kc", kcOp)
  ctx.smstart()
  ctx.zeroZad()
  ctx.ptrue(p(0), "s")
  ctx.ptrue(p(1), "s")
  ctx.mov(x(5), view(paOp))
  ctx.mov(x(6), view(pbOp))
  ctx.mov(w(14), wview(kcOp2))
  ctx.cbz(w(14), "4f")
  ctx.lsr(w(16), w(14), 2)
  ctx.`and`(w(17), w(14), 3)
  ctx.cbz(w(16), "3f")
  # prologue: block 0 = A rows (z31..z24) and B cols (z23..z16)
  ctx.ld1w(z(31), "s", p(0), mem(x(5)))
  for i in 1 .. 7:
    ctx.ld1w(z(31 - i), "s", p(0), memVL(x(5), i))
  ctx.addvl(x(5), x(5), 8)
  ctx.ld1w(z(23), "s", p(0), mem(x(6)))
  for i in 1 .. 7:
    ctx.ld1w(z(23 - i), "s", p(0), memVL(x(6), i))
  ctx.addvl(x(6), x(6), 8)
  ctx.sub(w(16), w(16), 1)
  ctx.cbz(w(16), "2f")
  ctx.label("1")
  for step in 0 .. 3:
    for q in 0 .. 3:
      # Quadrant q of one k-step: B pair (23-2s, 22-2s), A pair
      # (31-2s, 30-2s). Each q slot refills one freed register
      # with next-block data:
      #   q0 (s > 0): z(24-2s) <- [x6, #2s-1]  (step 0: loop subs)
      #   q1:         z(31-2s) <- [x5, #2s]
      #   q2:         z(23-2s) <- [x6, #2s]
      #   q3:         z(30-2s) <- [x5, #2s+1]
      # The 16th block load, z16 <- [x6, #7], sits between the addvl calls.
      ctx.fmopa(zaTile(q, "s"), p(0), p(0),
                z(23 - 2 * step - (q mod 2)), z(31 - 2 * step - (q div 2)))
      case q
      of 0:
        if step == 0:
          ctx.subs(w(16), w(16), 1)
        else:
          ctx.ld1w(z(24 - 2 * step), "s", p(0), memVL(x(6), 2 * step - 1))
      of 1:
        ctx.ld1w(z(31 - 2 * step), "s", p(0), memStep(x(5), 2 * step))
      of 2:
        ctx.ld1w(z(23 - 2 * step), "s", p(0), memStep(x(6), 2 * step))
      of 3:
        ctx.ld1w(z(30 - 2 * step), "s", p(0), memVL(x(5), 2 * step + 1))
      else:
        discard
  ctx.addvl(x(5), x(5), 8)
  ctx.ld1w(z(16), "s", p(0), memVL(x(6), 7))
  ctx.addvl(x(6), x(6), 8)
  ctx.bne("1b")
  ctx.label("2")
  for step in 0 .. 3:
    for q in 0 .. 3:
      ctx.fmopa(zaTile(q, "s"), p(0), p(0),
                z(23 - 2 * step - (q mod 2)), z(31 - 2 * step - (q div 2)))
  ctx.label("3")
  ctx.cbz(w(17), "4f")
  ctx.label("5")
  # Oddments (kc % 4): loads put A in z16/z17 and B in z18/z19.
  # za<q> takes (B, A) = (z(18 + q mod 2), z(16 + q div 2)).
  for i in 0 .. 1:
    ctx.ld1w(z(16 + i), "s", p(0), memStep(x(5), i))
  for i in 0 .. 1:
    ctx.ld1w(z(18 + i), "s", p(0), memStep(x(6), i))
  for q in 0 .. 3:
    ctx.fmopa(zaTile(q, "s"), p(0), p(0),
              z(18 + q mod 2), z(16 + q div 2))
  ctx.add(x(5), x(5), 128)
  ctx.add(x(6), x(6), 128)
  ctx.subs(w(17), w(17), 1)
  ctx.bne("5b")
  ctx.label("4")
  # Quadrant tiles: za<q> covers rows 16*(q div 2) and cols 16*(q mod 2)
  # of the 32x32 AB tile, so its base offset is (q div 2)*2048 +
  # (q mod 2)*64 (row stride 128 B, col stride 4 B).
  for q in 0 .. 3:
    storeQuadrant(ctx, "za" & $q & "v", abOp,
                  (q div 2) * 2048 + (q mod 2) * 64, 128)
  ctx.smstop()
  ctx.clobber("x4", "x5", "x6", "w13", "w14", "x16", "x17")
  ctx.clobberZ()
  ctx.clobberP()
  ctx.clobberCC()
  ctx.clobberMemory()

macro genSme32x32(pa, pb, ab, kcOp: typed): untyped =
  ## Expands to the `{.emit: ...}` pragma for `sme_gemm_ukernel_32x32`
  var ctx = init(AssemblerSME)
  buildSme32x32(ctx, pa, pb, ab, kcOp)
  result = ctx.generate()

proc sme_gemm_ukernel_32x32*(
    packA, packB: ptr float32, AB: ptr float32, kc: cint) =
  ## 32×32 f32 SME outer-product micro-kernel with four independent ZA
  ## accumulators (ZA0..ZA3), one per 16×16 quadrant of the output tile:
  ## `AB[i][j] = Σ_k packA[k*32+i] * packB[k*32+j]`.
  ##
  ## Expected input:
  ##   - packA, packB: 32 contiguous f32 per k-step
  ##     (`packA[k*32 ..< k*32+32]`, same for B), the ex02a `(ir, kc, 32)`
  ##     packed layout
  ##   - kc: k-steps to contract, `kc >= 0`. A `kc == 0` call stores
  ##     zeroed tiles.
  ##   - AB: 32×32 f32 output tile, all 1024 lanes written (overwritten,
  ##     never accumulated into)
  ##
  ## Output: the 32×32 outer-product tile, one `fmopa` per quadrant per k-step.
  ## Quadrant mapping:
  ##
  ##   za0: rows 0-15,  cols 0-15     za1: rows 0-15,  cols 16-31
  ##   za2: rows 16-31, cols 0-15     za3: rows 16-31, cols 16-31
  ##
  ## Four independent quadrants give 4-way ILP, hiding the `fmopa`
  ## latency. The k-loop is block-double-buffered: 4-step blocks
  ## (16 loads + 16 fmopa), next-block loads interleaved
  ## after each register's last use. `kc % 4` leftover steps run
  ## unpipelined.
  ## Pointers need no alignment.
  genSme32x32(packA, packB, AB, kc)

proc emitRowPair(
    ctx: var AssemblerSME, lo, hi, dec: int,
    relu, aone, bzero, bone: AsmOperand) =
  ## Emits one 2-vector extract row (`z<lo>`, `z<hi>`) with label base `dec`:
  ## ReLU clamp, alpha scaling, C read-modify-write, then the streaming
  ## 2-vector store. The label order (dec, dec+1, dec+3, dec+2) is
  ## load-bearing for the forward branches.
  ctx.cbz(wview(relu), $dec & "f")
  ctx.fclamp(zrng(lo, hi, "s"), z(4), z(5))
  ctx.label($dec)
  ctx.cbnz(wview(aone), $(dec + 1) & "f")
  ctx.fmul(z(lo), "s", p(0), z(lo), z(14))
  ctx.fmul(z(hi), "s", p(0), z(hi), z(14))
  ctx.label($(dec + 1))
  ctx.cbnz(wview(bzero), $(dec + 2) & "f")
  ctx.ld1w(z(12), "s", p(0), mem(x(22)))
  ctx.ld1w(z(13), "s", p(0), memVL(x(22), 1))
  ctx.cbnz(wview(bone), $(dec + 3) & "f")
  ctx.fmul(z(12), "s", p(0), z(12), z(15))
  ctx.fmul(z(13), "s", p(0), z(13), z(15))
  ctx.label($(dec + 3))
  ctx.fadd(z(lo), "s", p(0), z(lo), z(12))
  ctx.fadd(z(hi), "s", p(0), z(hi), z(13))
  ctx.label($(dec + 2))
  ctx.st1w(z(lo), "s", p(0), mem(x(22)))
  ctx.st1w(z(hi), "s", p(0), memVL(x(22), 1))
  ctx.add(x(22), x(22), x(10))

proc extractRows8(
    ctx: var AssemblerSME, tile: static string,
    relu, aone, bzero, bone: AsmOperand) =
  ## Emits the fused 8-row C-extract body for one ZA tile half: four SME2
  ## `mova` slice reads (2 output rows each), then per row the epilogue
  ## math and a streaming 2-vector store. `tile` is `za0h` (output rows
  ## 0-15) or `za1h` (rows 16-31). In the transposed tile a mova column
  ## equals the output row. The paired vectors of each mova hold that
  ## row's left and right halves (za0/za2 for `za0h`, za1/za3 for `za1h`).
  ##
  ## Caller owns the loop counters: emits `mov w11, #2` and `mov w12, #0`
  ## before the `3:`/`4:` head and brackets the body with `b.ne`,
  ## decrementing w11 and advancing w12 by 8 after each mova pair
  ## (extracted row index = w12/2). Row pairs own label decades
  ## (z16/z17 → 40s, ..., z30/z31 → 110s), reused by both halves.
  ctx.mov(zrng(16, 19, "h"), zaSlice(tile, "h", "w12", 0, 3))
  ctx.mov(zrng(24, 27, "h"), zaSlice(tile, "h", "w12", 4, 7))
  ctx.add(w(12), w(12), 8)
  ctx.subs(w(11), w(11), 1)
  ctx.mov(zrng(8, 11, "h"), zaSlice(tile, "h", "w12", 0, 3))
  ctx.mov(zrng(28, 31, "h"), zaSlice(tile, "h", "w12", 4, 7))
  ctx.add(w(12), w(12), 8)
  # Row-pair groups, one per mova range in emission order: (range base,
  # first label decade). Two pairs per range. The decades step by 10.
  const extractGroups = [(16, 40), (24, 60), (8, 80), (28, 100)]
  for (loBase, decBase) in extractGroups:
    for pair in 0 .. 1:
      emitRowPair(ctx, loBase + 2 * pair, loBase + 2 * pair + 1,
                  decBase + 10 * pair, relu, aone, bzero, bone)

proc buildSme32x32Epi(
    ctx: var AssemblerSME,
    pa, pb, cOp, cs, kcOp, abits, bbits, reluOp, aone, bzero, bone: NimNode) =
  ## Records the `sme_gemm_ukernel_32x32_epi` instruction stream into
  ## `ctx` in source order: the single-vector k-loop over four ZA
  ## accumulators, the oddments tail, then the two fused extracts. Operand
  ## bindings. Stream shape equals
  ## the dual-vector kernel's minus its dual-load offsets.
  let paOp = ctx.input("pa", pa)
  let pbOp = ctx.input("pb", pb)
  let cOp2 = ctx.input("C", cOp)
  let csOp = ctx.inputLong("cs", cs)
  let kcOp2 = ctx.inputLong("kc", kcOp)
  let abitsOp = ctx.inputLongParen("abits", abits)
  let bbitsOp = ctx.inputLongParen("bbits", bbits)
  let infbitsOp = ctx.inputLit("infbits", 2143289344)
  let reluOp2 = ctx.inputLong("relu", reluOp)
  let aoneOp = ctx.inputLong("aone", aone)
  let bzeroOp = ctx.inputLong("bzero", bzero)
  let boneOp = ctx.inputLong("bone", bone)
  ctx.smstart()
  ctx.zeroZad()
  ctx.ptrue(p(0), "s")
  ctx.mov(x(5), view(paOp))
  ctx.mov(x(6), view(pbOp))
  ctx.mov(w(14), wview(kcOp2))
  ctx.dup(z(14), "s", wview(abitsOp))
  ctx.dup(z(15), "s", wview(bbitsOp))
  ctx.mov(z(4), "s", 0)
  ctx.dup(z(5), "s", wview(infbitsOp))
  ctx.cbz(w(14), "9f")
  ctx.lsr(w(16), w(14), 2)
  ctx.cbz(w(16), "5f")
  ctx.ld1w(z(31), "s", p(0), mem(x(5)))
  for i in 1 .. 7:
    ctx.ld1w(z(31 - i), "s", p(0), memVL(x(5), i))
  ctx.addvl(x(5), x(5), 8)
  ctx.ld1w(z(23), "s", p(0), mem(x(6)))
  for i in 1 .. 7:
    ctx.ld1w(z(23 - i), "s", p(0), memVL(x(6), i))
  ctx.addvl(x(6), x(6), 8)
  ctx.sub(w(16), w(16), 1)
  ctx.cbz(w(16), "2f")
  ctx.label("1")
  for step in 0 .. 3:
    for q in 0 .. 3:
      # Quadrant q of one k-step: A pair (31-2s, 30-2s), B pair
      # (23-2s, 22-2s). Each q slot refills one freed register
      # with next-block data:
      #   q0 (s > 0): z(24-2s) <- [x6, #2s-1]  (step 0: loop subs)
      #   q1:         z(23-2s) <- [x6, #2s]
      #   q2:         z(31-2s) <- [x5, #2s]
      #   q3:         z(30-2s) <- [x5, #2s+1]
      # The 16th block load, z16 <- [x6, #7], sits between the addvl calls.
      ctx.fmopa(zaTile(q, "s"), p(0), p(0),
                z(31 - 2 * step - (q mod 2)), z(23 - 2 * step - (q div 2)))
      case q
      of 0:
        if step == 0:
          ctx.subs(w(16), w(16), 1)
        else:
          ctx.ld1w(z(24 - 2 * step), "s", p(0), memVL(x(6), 2 * step - 1))
      of 1:
        ctx.ld1w(z(23 - 2 * step), "s", p(0), memStep(x(6), 2 * step))
      of 2:
        ctx.ld1w(z(31 - 2 * step), "s", p(0), memStep(x(5), 2 * step))
      of 3:
        ctx.ld1w(z(30 - 2 * step), "s", p(0), memVL(x(5), 2 * step + 1))
      else:
        discard
  ctx.addvl(x(5), x(5), 8)
  ctx.ld1w(z(16), "s", p(0), memVL(x(6), 7))
  ctx.addvl(x(6), x(6), 8)
  ctx.bne("1b")
  ctx.label("2")
  for step in 0 .. 3:
    for q in 0 .. 3:
      ctx.fmopa(zaTile(q, "s"), p(0), p(0),
                z(31 - 2 * step - (q mod 2)), z(23 - 2 * step - (q div 2)))
  ctx.label("5")
  ctx.`and`(w(17), w(14), 3)
  ctx.cbz(w(17), "9f")
  ctx.label("6")
  # Oddments (kc % 4): loads put A in z16/z17 and B in z18/z19.
  # za<q> takes (A, B) = (z(16 + q mod 2), z(18 + q div 2)).
  for i in 0 .. 1:
    ctx.ld1w(z(16 + i), "s", p(0), memStep(x(5), i))
  for i in 0 .. 1:
    ctx.ld1w(z(18 + i), "s", p(0), memStep(x(6), i))
  for q in 0 .. 3:
    ctx.fmopa(zaTile(q, "s"), p(0), p(0),
              z(16 + q mod 2), z(18 + q div 2))
  ctx.add(x(5), x(5), 128)
  ctx.add(x(6), x(6), 128)
  ctx.subs(w(17), w(17), 1)
  ctx.bne("6b")
  ctx.label("9")
  ctx.mov(x(22), view(cOp2))
  ctx.lsl(x(10), xview(csOp), 2)
  ctx.mov(w(11), "#2")
  ctx.mov(w(12), "#0")
  ctx.label("3")
  extractRows8(ctx, "za0h", reluOp2, aoneOp, bzeroOp, boneOp)
  ctx.bne("3b")
  ctx.mov(w(11), "#2")
  ctx.mov(w(12), "#0")
  ctx.label("4")
  extractRows8(ctx, "za1h", reluOp2, aoneOp, bzeroOp, boneOp)
  ctx.bne("4b")
  ctx.smstop()
  ctx.clobber("x5", "x6", "x10", "x11", "x22", "w12", "w14", "w16", "w17")
  ctx.clobberZ()
  ctx.clobberP()
  ctx.clobberCC()
  ctx.clobberMemory()

macro genSme32x32Epi(
    pa, pb, cOp, cs, kcOp, abits, bbits, reluOp, aone, bzero, bone: typed): untyped =
  ## Expands to the `{.emit: ...}` pragma for `sme_gemm_ukernel_32x32_epi`
  var ctx = init(AssemblerSME)
  buildSme32x32Epi(ctx, pa, pb, cOp, cs, kcOp, abits, bbits,
                   reluOp, aone, bzero, bone)
  result = ctx.generate()

proc sme_gemm_ukernel_32x32_epi*(
    packA, packB: ptr float32, C: ptr float32, cRowStride: cint, kc: cint,
    alpha, beta: float32, relu, alphaOne, betaZero, betaOne: cint) =
  ## 32×32 f32 SME micro-kernel with a fused in-streaming extract to C
  ## (single-vector inner loop). The dual-vector variant
  ## `sme_gemm_ukernel_32x32_epi_dv` is the hot path with the same
  ## contract.
  ##
  ## Expected input:
  ##   - packA, packB: 32 contiguous f32 per k-step (the ex02a `(ir, kc, 32)` packed layout)
  ##   - kc: k-steps to contract, `kc >= 0`
  ##   - C: 32×32 f32 tile, col stride 1, row stride `cRowStride`
  ##   - alpha, beta: scale factors
  ##   - relu, alphaOne, betaZero, betaOne: flags mirroring the scalar epilogue's comparisons
  ##
  ## Output: `C[i][j] = beta*C[i][j] + alpha*f(AB[i][j])`, f = identity
  ## or ReLU. The accumulator sits in ZA transposed (za row j = output col j),
  ## so the in-streaming mova extract reads output rows
  ## while applying the epilogue math:
  ##
  ##   before (ZA accumulator):            after (C, row-major):
  ##     za[j][i] = AB[i][j]                C[i][j] = beta*C[i][j]
  ##     (za row j = output col j)          + alpha*f(AB[i][j])
  ##
  ## ReLU `fclamp` maps NaN to 0 on M4, like the scalar epilogue.
  ## `beta == 0` stores `alpha*f(AB)`, possibly `-0.0` where the scalar epilogue stores `+0.0`.
  let alphaBits = cast[int32](alpha)
  let betaBits = cast[int32](beta)
  genSme32x32Epi(packA, packB, C, cRowStride, kc, alphaBits, betaBits,
                 relu, alphaOne, betaZero, betaOne)

proc buildSme32x32EpiDv(
    ctx: var AssemblerSME,
    pa, pb, cOp, cs, kcOp, abits, bbits, reluOp, aone, bzero, bone: NimNode) =
  ## Records the `sme_gemm_ukernel_32x32_epi_dv` instruction stream into
  ## `ctx` in source order: SME2 dual-vector loads in the 4-k-step blocks,
  ## single-vector oddments, then the two fused extracts. Operand bindings
  ## bindings: pointers and 64-bit values via
  ## `"r"`, 32-bit flags via `"r"((long)...)`, the alpha/beta bit patterns
  ## via `"r"((long)(...))`, and the ReLU +inf word as a literal.
  let paOp = ctx.input("pa", pa)
  let pbOp = ctx.input("pb", pb)
  let cOp2 = ctx.input("C", cOp)
  let csOp = ctx.inputLong("cs", cs)
  let kcOp2 = ctx.inputLong("kc", kcOp)
  let abitsOp = ctx.inputLongParen("abits", abits)
  let bbitsOp = ctx.inputLongParen("bbits", bbits)
  let infbitsOp = ctx.inputLit("infbits", 2143289344)
  let reluOp2 = ctx.inputLong("relu", reluOp)
  let aoneOp = ctx.inputLong("aone", aone)
  let bzeroOp = ctx.inputLong("bzero", bzero)
  let boneOp = ctx.inputLong("bone", bone)
  ctx.smstart()
  ctx.zeroZad()
  ctx.ptrue(p(0), "s")
  ctx.ptruePn9B()
  ctx.mov(x(5), view(paOp))
  ctx.mov(x(6), view(pbOp))
  ctx.mov(w(14), wview(kcOp2))
  ctx.dup(z(14), "s", wview(abitsOp))
  ctx.dup(z(15), "s", wview(bbitsOp))
  ctx.mov(z(4), "s", 0)
  ctx.dup(z(5), "s", wview(infbitsOp))
  ctx.cbz(w(14), "9f")
  ctx.lsr(w(16), w(14), 2)
  ctx.mov(x(21), "#0")
  ctx.mov(x(20), "#0")
  ctx.cbz(w(16), "5f")
  for i in 0 .. 3:
    ctx.ld1w2(23 - i, 5, 21)
    ctx.incw(21)
  for i in 0 .. 3:
    ctx.ld1w2(19 - i, 6, 20)
    ctx.incw(20)
  ctx.sub(w(16), w(16), 1)
  ctx.cbz(w(16), "2f")
  ctx.label("1")
  # Next-block dual-vector loads interleaved into the fmopa stream
  # at hand-tuned positions: (step, fmopa-in-step, ld1w2 first reg,
  # base, counter). Each entry refreshes one dual-vector pair
  # for the next block.
  const dvLoads = [
    (1, 2, 23, 5, 21), (1, 3, 19, 6, 20), (1, 4, 22, 5, 21),
    (2, 1, 18, 6, 20),
    (3, 2, 21, 5, 21), (3, 3, 17, 6, 20), (3, 4, 20, 5, 21), (3, 4, 16, 6, 20)]
  for step in 0 .. 3:
    for q in 0 .. 3:
      # Quadrant q of one k-step in emission order za0/za2/za1/za3: A pair
      # (23-s, 31-s), B pair (19-s, 27-s).
      ctx.fmopa(zaTile(2 * (q mod 2) + q div 2, "s"), p(0), p(0),
                z(23 - step + 8 * (q div 2)), z(19 - step + 8 * (q mod 2)))
      for (ls, pos, zt1, xn, rm) in dvLoads:
        if ls == step and pos == q + 1:
          ctx.ld1w2(zt1, xn, rm)
          ctx.incw(rm)
  ctx.subs(w(16), w(16), 1)
  ctx.bne("1b")
  ctx.label("2")
  for step in 0 .. 3:
    for q in 0 .. 3:
      ctx.fmopa(zaTile(2 * (q mod 2) + q div 2, "s"), p(0), p(0),
                z(23 - step + 8 * (q div 2)), z(19 - step + 8 * (q mod 2)))
  ctx.label("5")
  ctx.addLsl(x(5), x(5), x(21), 2)
  ctx.addLsl(x(6), x(6), x(20), 2)
  ctx.`and`(w(17), w(14), 3)
  ctx.cbz(w(17), "9f")
  ctx.label("6")
  # Oddments (kc % 4): loads put A in z16/z17 and B in z18/z19.
  # za<q> takes (A, B) = (z(16 + q mod 2), z(18 + q div 2)).
  for i in 0 .. 1:
    ctx.ld1w(z(16 + i), "s", p(0), memStep(x(5), i))
  for i in 0 .. 1:
    ctx.ld1w(z(18 + i), "s", p(0), memStep(x(6), i))
  for q in 0 .. 3:
    ctx.fmopa(zaTile(q, "s"), p(0), p(0),
              z(16 + q mod 2), z(18 + q div 2))
  ctx.add(x(5), x(5), 128)
  ctx.add(x(6), x(6), 128)
  ctx.subs(w(17), w(17), 1)
  ctx.bne("6b")
  ctx.label("9")
  ctx.mov(x(22), view(cOp2))
  ctx.lsl(x(10), xview(csOp), 2)
  ctx.mov(w(11), "#2")
  ctx.mov(w(12), "#0")
  ctx.label("3")
  extractRows8(ctx, "za0h", reluOp2, aoneOp, bzeroOp, boneOp)
  ctx.bne("3b")
  ctx.mov(w(11), "#2")
  ctx.mov(w(12), "#0")
  ctx.label("4")
  extractRows8(ctx, "za1h", reluOp2, aoneOp, bzeroOp, boneOp)
  ctx.bne("4b")
  ctx.smstop()
  ctx.clobber("x5", "x6", "x10", "x11", "x20", "x21", "x22",
              "w12", "w14", "w16", "w17")
  ctx.clobberZ()
  ctx.clobberP()
  ctx.clobberCC()
  ctx.clobberMemory()

macro genSme32x32EpiDv(
    pa, pb, cOp, cs, kcOp, abits, bbits, reluOp, aone, bzero, bone: typed): untyped =
  ## Expands to the `{.emit: ...}` pragma for `sme_gemm_ukernel_32x32_epi_dv`.
  var ctx = init(AssemblerSME)
  buildSme32x32EpiDv(ctx, pa, pb, cOp, cs, kcOp, abits, bbits,
                     reluOp, aone, bzero, bone)
  result = ctx.generate()

proc sme_gemm_ukernel_32x32_epi_dv*(
    packA, packB: ptr float32, C: ptr float32, cRowStride: cint, kc: cint,
    alpha, beta: float32, relu, alphaOne, betaZero, betaOne: cint) =
  ## 32×32 f32 SME micro-kernel with a fused in-streaming extract to C
  ## (dual-vector SME2 inner loop, the driver's hot path):
  ## `C[i][j] = beta*C[i][j] + alpha*f(AB[i][j])`, f = identity or ReLU.
  ##
  ## Expected input:
  ##   - packA, packB: 32 contiguous f32 per k-step (the ex02a `(ir, kc, 32)` packed layout)
  ##   - kc: k-steps to contract, `kc >= 0`
  ##   - C: 32×32 f32 tile, col stride 1, row stride `cRowStride`
  ##   - alpha, beta: scale factors
  ##   - relu, alphaOne, betaZero, betaOne: flags mirroring the scalar epilogue:
  ##     `alphaOne = alpha == 1`, `betaZero = beta == 0`,
  ##     `betaOne = beta == 1`
  ##
  ## Output: `C[i][j] = beta*C[i][j] + alpha*f(AB[i][j])`.
  ## The accumulator sits in ZA transposed (za row j = output col j),
  ## so the in-streaming mova extract reads output rows
  ## while applying the epilogue math:
  ##
  ##   before (ZA accumulator):            after (C, row-major):
  ##     za[j][i] = AB[i][j]                C[i][j] = beta*C[i][j]
  ##     (za row j = output col j)          + alpha*f(AB[i][j])
  ##
  ## ReLU `fclamp` maps NaN to 0 on M4, like the scalar epilogue.
  ## `beta == 0` stores `alpha*f(AB)`, possibly `-0.0` where the scalar epilogue stores `+0.0`.
  ## `ld1w2` dual-vector loads carry two k-steps each.
  let alphaBits = cast[int32](alpha)
  let betaBits = cast[int32](beta)
  genSme32x32EpiDv(packA, packB, C, cRowStride, kc, alphaBits, betaBits,
                   relu, alphaOne, betaZero, betaOne)

proc buildNeonEpilogue(
    ctx: var AssemblerSME,
    cOp, abOp, rowStrideOp, abits, bbits,
    reluOp, aone, bzero, bone: NimNode) =
  ## Records the `neon_epilogue_f32_32x32` instruction stream into `ctx`
  ## in source order: the beta==0 store loop, then the beta!=0
  ## read-modify-write loop. Operand bindings:
  ## constraint list: pointers via `"r"`, the row stride and flags via
  ## `"r"((long)...)`, the alpha/beta bit patterns via `"r"((long)(...))`.
  let cOp2 = ctx.input("C", cOp)
  let abOp2 = ctx.input("AB", abOp)
  let rsOp = ctx.inputLong("rowStride", rowStrideOp)
  let abitsOp = ctx.inputLongParen("alphaBits", abits)
  let bbitsOp = ctx.inputLongParen("betaBits", bbits)
  let reluOp2 = ctx.inputLong("relu", reluOp)
  let aoneOp = ctx.inputLong("alphaOne", aone)
  let bzeroOp = ctx.inputLong("betaZero", bzero)
  let boneOp = ctx.inputLong("betaOne", bone)
  ctx.mov(x(20), view(cOp2))
  ctx.mov(x(21), view(abOp2))
  ctx.lsl(x(22), xview(rsOp), 2)
  ctx.sub(x(22), x(22), 128)
  ctx.dup(v(16), "4s", wview(abitsOp))
  ctx.dup(v(17), "4s", wview(bbitsOp))
  ctx.movi(v(18), "16b", 0)
  ctx.mov(w(23), "#32")
  ctx.cbz(wview(bzeroOp), "2f")
  ctx.label("1")
  ctx.ld1(vlist(0, 3, "4s"), memPost(x(21), 64))
  ctx.ld1(vlist(4, 7, "4s"), memPost(x(21), 64))
  ctx.cbz(wview(reluOp2), "3f")
  for i in 0 .. 7:
    ctx.fmax(v(i), "4s", v(i), v(18))
  ctx.label("3")
  ctx.cbnz(wview(aoneOp), "4f")
  for i in 0 .. 7:
    ctx.fmul(v(i), "4s", v(i), v(16))
  ctx.label("4")
  ctx.st1(vlist(0, 3, "4s"), memPost(x(20), 64))
  ctx.st1(vlist(4, 7, "4s"), memPost(x(20), 64))
  ctx.add(x(20), x(20), x(22))
  ctx.subs(w(23), w(23), 1)
  ctx.bne("1b")
  ctx.b("9f")
  ctx.label("2")
  ctx.mov(x(19), x(20))
  ctx.ld1(vlist(0, 3, "4s"), memPost(x(21), 64))
  ctx.ld1(vlist(4, 7, "4s"), memPost(x(21), 64))
  ctx.cbz(wview(reluOp2), "5f")
  for i in 0 .. 7:
    ctx.fmax(v(i), "4s", v(i), v(18))
  ctx.label("5")
  ctx.cbnz(wview(aoneOp), "6f")
  for i in 0 .. 7:
    ctx.fmul(v(i), "4s", v(i), v(16))
  ctx.label("6")
  ctx.ld1(vlist(8, 11, "4s"), memPost(x(20), 64))
  ctx.ld1(vlist(12, 15, "4s"), memPost(x(20), 64))
  ctx.cbnz(wview(boneOp), "7f")
  for i in 8 .. 15:
    ctx.fmul(v(i), "4s", v(i), v(17))
  ctx.label("7")
  for i in 0 .. 7:
    ctx.fadd(v(8 + i), "4s", v(8 + i), v(i))
  ctx.st1(vlist(8, 11, "4s"), memPost(x(19), 64))
  ctx.st1(vlist(12, 15, "4s"), memPost(x(19), 64))
  ctx.add(x(20), x(19), x(22))
  ctx.subs(w(23), w(23), 1)
  ctx.bne("2b")
  ctx.label("9")
  ctx.clobber("x19", "x20", "x21", "x22", "w23")
  ctx.clobberV()
  ctx.clobberCC()
  ctx.clobberMemory()

macro genNeonEpilogue(
    cOp, abOp, rowStrideOp, abits, bbits,
    reluOp, aone, bzero, bone: typed): untyped =
  ## Expands to the `{.emit: ...}` pragma for `neon_epilogue_f32_32x32`
  var ctx = init(AssemblerSME)
  buildNeonEpilogue(ctx, cOp, abOp, rowStrideOp, abits, bbits,
                    reluOp, aone, bzero, bone)
  result = ctx.generate()

proc neon_epilogue_f32_32x32*(
    C: ptr float32, cRowStride: cint,   # C tile base, row stride in f32 elements
    AB: ptr float32,                    # full 32x32 AB tile (128 B per row)
    alpha, beta: float32,
    relu, alphaOne, betaZero, betaOne: cint) =
  ## Vectorized epilogue for a contiguous 32×32 f32 C tile (col stride 1,
  ## row stride ≥ 32): `C[i][j] = beta*C[i][j] + alpha*f(AB[i][j])`,
  ## f = identity or ReLU, 4 lanes per NEON op.
  ##
  ## Expected input:
  ##   - AB: 32×32 f32 accumulator tile (row stride 32, 128 B per row)
  ##   - C: 32×32 f32 tile, col stride 1, row stride ≥ 32
  ##   - alpha, beta: scale factors
  ##   - relu, alphaOne, betaZero, betaOne: flags mirroring the scalar epilogue's comparisons
  ##
  ## Output: the fused C tile:
  ##
  ##   before:                             after (C):
  ##     AB[i][j] = Σ_k a_i[k] b_j[k]       C[i][j] = beta*C[i][j]
  ##     C holds the prior values            + alpha*f(AB[i][j])
  ##
  ## Two divergences from the scalar epilogue:
  ##   - a NaN accumulator propagates through the NEON `fmax` ReLU
  ##     (scalar ReLU yields 0)
  ##   - `alpha == ±0.0` with `beta == 0` stores signed zero where the scalar path stores +0.0
  let alphaBits = cast[int32](alpha)
  let betaBits = cast[int32](beta)
  genNeonEpilogue(C, AB, cRowStride, alphaBits, betaBits,
                  relu, alphaOne, betaZero, betaOne)

# Tile dispatch
# ----------------------------------------------------------------------------

proc gemm_ukernel_sme*[MR, NR: static int](
    packA, packB: ptr UncheckedArray[float32],
    AB: var array[MR, array[NR, float32]],
    kc: int) =
  ## Computes `AB[ri][rj] = Σ_k packA[k*MR+ri] * packB[k*NR+rj]` with the SME
  ## outer-product unit. `AB` is overwritten, never accumulated into.
  ## ZA tiles are zeroed first, so callers may use `{.noInit.}`.
  ##
  ## Expected input:
  ##   - packA: MR contiguous f32 per k-step (the ex02a `(ir, kc, MR)` packed layout)
  ##   - packB: NR contiguous f32 per k-step (the ex02a `(ir, kc, NR)` packed layout)
  ##   - kc: k-steps to contract, `kc >= 0`
  ##   - AB: MR×NR f32 output tile
  ##
  ## Output: `AB` = the MR×NR outer-product tile, all lanes written.
  ##
  ## Two tile shapes, backed by two asm kernels in this file:
  ## - MR == NR == 16: one ZA0 accumulator (sme_gemm_ukernel_16x16).
  ## - MR == NR == 32: four ZA0..ZA3 accumulators, one per 16×16 quadrant
  ##   of the 32×32 output tile (sme_gemm_ukernel_32x32), 4-way ILP
  ##   hiding the `fmopa` latency.
  static: doAssert MR == NR and MR in {16, 32},
    "gemm_ukernel_sme requires square tiles of 16 or 32 (got MR=" & $MR & ", NR=" & $NR & ")"
  doAssert kc >= 0, "gemm_ukernel_sme: kc must be >= 0 (got " & $kc & ")"
  when MR == 16:
    sme_gemm_ukernel_16x16(
      cast[ptr float32](packA), cast[ptr float32](packB),
      addr AB[0][0], kc.cint)
  else:
    sme_gemm_ukernel_32x32(
      cast[ptr float32](packA), cast[ptr float32](packB),
      addr AB[0][0], kc.cint)

