## SIMD-accelerated pack kernels for the ex02a hand-tuned GEMM example.
##
## Packs transpose or copy row-major A/B panels into the `(ir, kc, 32)` layout
## consumed by the SME2 ukernels in `gemm_ukernel_arm64_sme2.nim`.
## `sme_` packs run inside an smstart/smstop bracket (streaming mode, SVE/SME2 only).
## NEON packs run outside the bracket, since NEON faults
## inside streaming mode. Requires ARMv9.2 with FEAT_SME2 and assumes
## SVL = 64 B (16 f32 lanes per vector).
##
## Every pack emits its instruction text through the compile-time macro
## assembler: a builder proc records the stream into an `AssemblerSME`,
## and a `gen*` macro expands `generate()` to one in-body `{.emit: "asm volatile(...)".}` pragma.

{.localpassC: "-march=armv9-a+sme2 -fno-vectorize -fno-slp-vectorize".}

import workspace/cpuplatforms/arm/macro_assembler_arm64

func memStep(base: XReg, imm: int): MemAddr =
  ## Memory operand for a vector load at `imm` vector lengths:
  ## `[base]` when `imm == 0`, `[base, #imm, MUL VL]` otherwise.
  if imm == 0: mem(base) else: memVL(base, imm)

proc emitTrnStore8x8(ctx: var AssemblerSME) =
  for cell in 0 .. 3:
    let a = 8 * (cell div 2) + cell mod 2
    ctx.trn1(v(24), "4s", v(a), v(a + 2))
    ctx.trn2(v(a + 2), "4s", v(a), v(a + 2))
    ctx.trn1(v(25), "4s", v(a + 4), v(a + 6))
    ctx.trn2(v(a + 6), "4s", v(a + 4), v(a + 6))
    ctx.trn1(v(a), "2d", v(24), v(25))
    ctx.trn2(v(a + 4), "2d", v(24), v(25))
    ctx.trn1(v(16 + cell), "2d", v(a + 2), v(a + 6))
    ctx.trn2(v(a + 6), "2d", v(a + 2), v(a + 6))
  # Each pair (lo, hi) holds one output column's rows 0-3 and 4-7.
  const storePairs = [(0, 8), (16, 18), (4, 12), (6, 14),
                      (1, 9), (17, 19), (5, 13), (7, 15)]
  for i, (lo, hi) in storePairs:
    ctx.stp(v(lo), v(hi), x(20), 128 * i)
  ctx.add(x(20), x(20), 1024)

proc buildNeonPackATranspose8rows(
    ctx: var AssemblerSME,
    dst, src, rowStrideOp, nGroupsOp, validOp: NimNode) =
  ## Records the `neon_packA_transpose_8rows` instruction stream into
  ## `ctx` in source order: the eight row pointers, the full 8-row transpose loop,
  ## then the cold partial-row loop. Operand bindings:
  ## pointers via `"r"`, the stride and counts via `"r"((long)...)`.
  let dstOp = ctx.input("dst", dst)
  let srcOp = ctx.input("src", src)
  let rsOp = ctx.inputLong("rowStride", rowStrideOp)
  let ngOp = ctx.inputLong("nGroups", nGroupsOp)
  let validOp2 = ctx.inputLong("valid", validOp)
  ctx.mov(x(20), view(dstOp))
  ctx.mov(x(21), view(srcOp))
  ctx.mov(x(22), xview(rsOp))
  ctx.mov(w(23), wview(ngOp))
  ctx.mov(w(24), wview(validOp2))
  ctx.lsl(x(22), $x(22), 2)
  ctx.mov(x(9), x(21))
  for r in 1 .. 7:
    ctx.add(x(9 + r), x(8 + r), x(22))
  ctx.cmp(w(24), 8)
  ctx.bne("4f")
  ctx.cbz(w(23), "8f")
  ctx.label("2")
  for r in 0 .. 7:
    ctx.ld1(vlist(2 * r, 2 * r + 1, "4s"), mem(x(9 + r)))
  for r in 0 .. 7:
    ctx.add(x(9 + r), x(9 + r), 32)
  emitTrnStore8x8(ctx)
  ctx.subs(w(23), w(23), 1)
  ctx.bne("2b")
  ctx.b("8f")
  ctx.label("4")
  ctx.cbz(w(23), "8f")
  for i in 0 .. 15:
    ctx.movi(v(i), "16b", 0)
  for r in 0 .. 7:
    ctx.cmp(w(24), r + 1)
    ctx.blt("5f")
    ctx.ld1(vlist(2 * r, 2 * r + 1, "4s"), mem(x(9 + r)))
  ctx.label("5")
  for r in 0 .. 7:
    ctx.add(x(9 + r), x(9 + r), 32)
  emitTrnStore8x8(ctx)
  ctx.subs(w(23), w(23), 1)
  ctx.bne("4b")
  ctx.label("8")
  ctx.clobber("x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16",
              "x20", "x21", "x22", "w23", "w24")
  ctx.clobberV()
  ctx.clobberCC()
  ctx.clobberMemory()

macro genNeonPackATranspose8rows(
    dst, src, rowStrideOp, nGroupsOp, validOp: typed): untyped =
  ## Expands to the `{.emit: ...}` pragma for
  ## `neon_packA_transpose_8rows`.
  var ctx = init(AssemblerSME)
  buildNeonPackATranspose8rows(ctx, dst, src, rowStrideOp, nGroupsOp,
                               validOp)
  result = ctx.generate()

proc neon_packA_transpose_8rows*(
    dst: ptr float32,      # packA + ir*kc*mr + g*8 (8-row group g, k = 0)
    src: ptr float32,      # pA_ptr + (srcRow + g*8) * pA_rs (group's first row, k = 0)
    srcRowStride: cint,    # pA_rs: elements between consecutive A rows
    nColGroups: cint,      # current_kc div 8: 8-column blocks to transpose
    validRows: cint) =
  ## NEON block-transpose pack for one 8-row group of the 32-row A tile.
  ##
  ## Expected input:
  ##   - src: first row of the 8-row group (row-major f32, rows
  ##     `srcRowStride` elements apart, `8 * nColGroups` contiguous f32 per row)
  ##   - srcRowStride: elements between consecutive source rows
  ##   - nColGroups: 8-column blocks to transpose (kc / 8)
  ##   - validRows: valid rows in `0..8`, rows past it zero-filled
  ##   - dst: group's packed slot (packA + ir*kc*mr + g*8)
  ##
  ## Output: one transposed 8×8 block per column group, columns 32 f32
  ##   apart: `dst[c*32 + r] = (r < validRows) ? src[r*rs + c] : 0`
  ##   for `c in 0 ..< 8 * nColGroups`, `r in 0 .. 7`. A zero group count
  ##   packs nothing.
  ##
  ## before (src, row-major):              after (dst, one 8×8 block):
  ##   r0: a b c d                           c0: a e i m      dst[0..7]
  ##   r1: e f g h                           c1: b f j n      dst[32..39]
  ##   r2: i j k l                           c2: c g k o      dst[64..71]
  ##   r3: m n o p                           c3: d h l p      dst[96..103]
  ##   (4×4 excerpt of each 8×8 block, whose columns land 32 f32 apart in dst)
  genNeonPackATranspose8rows(dst, src, srcRowStride, nColGroups,
                             validRows)

proc emitA16TransposeStore(ctx: var AssemblerSME) =
  for g in 0 .. 3:
    ctx.mov(w(12), "#" & $(4 * g))
    ctx.mova(zaSlice("za0h", "s", "w12", 0, 3),
             zrng(4 * g, 4 * g + 3, "s"))
  for g in 0 .. 3:
    ctx.mov(w(12), "#" & $(4 * g))
    ctx.mova(zrng(4 * g, 4 * g + 3, "s"),
             zaSlice("za0v", "s", "w12", 0, 3))
  const stBases = [16, 17, 23, 24]
  for g in 0 .. 3:
    for lane in 0 .. 3:
      ctx.st1w(z(4 * g + lane), "s", p(0),
               memVL(x(stBases[g]), 2 * lane))
  for base in stBases:
    ctx.add(x(base), x(base), 2048)

proc buildSmePackATranspose16rows(
    ctx: var AssemblerSME,
    dst, src, rowStrideOp, nColBlocksOp, validOp: NimNode) =
  ## Records the `sme_packA_transpose_16rows` instruction stream into
  ## `ctx` in source order: the smstart bracket, the 16 row pointers,
  ## the full 16-row transpose loop, then the cold partial-row loop.
  ## Operand bindings: pointers via `"r"`, the stride and counts via `"r"((long)...)`.
  let srcOp = ctx.input("src", src)
  let dstOp = ctx.input("dst", dst)
  let rsOp = ctx.inputLong("rs", rowStrideOp)
  let nbOp = ctx.inputLong("nb", nColBlocksOp)
  let validOp2 = ctx.inputLong("valid", validOp)
  ctx.smstart()
  ctx.ptrue(p(0), "s")
  ctx.mov(x(0), view(srcOp))
  ctx.lsl(x(11), xview(rsOp), 2)
  ctx.add(x(1), x(0), x(11))
  ctx.add(x(2), x(1), x(11))
  ctx.add(x(3), x(2), x(11))
  ctx.add(x(4), x(3), x(11))
  ctx.add(x(5), x(4), x(11))
  ctx.add(x(6), x(5), x(11))
  ctx.add(x(7), x(6), x(11))
  ctx.lsl(x(9), $x(11), 3)  # x9 = 8 rows in bytes
  ctx.sub(x(10), x(9), 64)  # x10 = x9 - one 16-col block
  ctx.mov(x(16), view(dstOp))
  ctx.add(x(17), x(16), 512)
  ctx.add(x(23), x(17), 512)
  ctx.add(x(24), x(23), 512)
  ctx.mov(w(27), wview(nbOp))
  ctx.mov(w(28), wview(validOp2))
  ctx.cmp(w(28), 16)
  ctx.bne("4f")
  ctx.cbz(w(27), "8f")
  ctx.label("2")
  for r in 0 .. 7:
    ctx.ld1w(z(r), "s", p(0), mem(x(r)))
  for r in 0 .. 7:
    ctx.add(x(r), x(r), x(9))
  for r in 8 .. 15:
    ctx.ld1w(z(r), "s", p(0), mem(x(r - 8)))
  for r in 0 .. 7:
    ctx.sub(x(r), x(r), x(10))
  emitA16TransposeStore(ctx)
  ctx.subs(w(27), w(27), 1)
  ctx.bne("2b")
  ctx.b("8f")
  ctx.label("4")
  ctx.cbz(w(27), "8f")
  for r in 0 .. 15:
    ctx.mov(z(r), "s", 0)
  for r in 0 .. 7:
    ctx.cmp(w(28), r + 1)
    ctx.blt("5f")
    ctx.ld1w(z(r), "s", p(0), mem(x(r)))
  ctx.label("5")
  for r in 0 .. 7:
    ctx.add(x(r), x(r), x(9))
  for r in 8 .. 15:
    ctx.cmp(w(28), r + 1)
    ctx.blt("6f")
    ctx.ld1w(z(r), "s", p(0), mem(x(r - 8)))
  ctx.label("6")
  for r in 0 .. 7:
    ctx.sub(x(r), x(r), x(10))
  emitA16TransposeStore(ctx)
  ctx.subs(w(27), w(27), 1)
  ctx.bne("4b")
  ctx.label("8")
  ctx.smstop()
  ctx.clobber("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
              "x9", "x10", "x11", "x16", "x17", "x23", "x24",
              "w12", "w27", "w28")
  ctx.clobberZ()
  ctx.clobberP()
  ctx.clobberCC()
  ctx.clobberMemory()

macro genSmePackATranspose16rows(
    dst, src, rowStrideOp, nColBlocksOp, validOp: typed): untyped =
  ## Expands to the `{.emit: ...}` pragma for
  ## `sme_packA_transpose_16rows`.
  var ctx = init(AssemblerSME)
  buildSmePackATranspose16rows(ctx, dst, src, rowStrideOp, nColBlocksOp,
                               validOp)
  result = ctx.generate()

proc sme_packA_transpose_16rows*(
    dst: ptr float32,      # packA + ir*kc*mr + g*16 (16-row group g, k = 0)
    src: ptr float32,      # pA_ptr + (srcRow + g*16) * pA_rs (group's first row, k = 0)
    srcRowStride: cint,    # pA_rs: elements between consecutive A rows
    nColBlocks: cint,      # current_kc div 16: 16-column blocks to transpose
    validRows: cint) =
  ## Streaming SME2 A-transpose pack for one 16-row group of the 32-row A tile.
  ##
  ## Expected input:
  ##   - src: first row of the 16-row group (row-major f32, rows
  ##     `srcRowStride` elements apart, `16 * nColBlocks` contiguous f32 per row)
  ##   - srcRowStride: elements between consecutive source rows
  ##   - nColBlocks: 16-column blocks to transpose (kc / 16)
  ##   - validRows: valid rows in `0..16`, rows past it zero-filled
  ##   - dst: group's packed slot (packA + ir*kc*mr + g*16)
  ##
  ## Output: one transposed 16×16 block per column group, columns 32 f32
  ##   apart: `dst[c*32 + r] = (r < validRows) ? src[r*rs + c] : 0`
  ##   for `c in 0 ..< 16 * nColBlocks`, `r in 0 .. 15`. A zero block count
  ##   packs nothing.
  ##
  ## before (src, row-major):              after (dst, one 16×16 block):
  ##   r0: a b c d                           c0: a e i m      dst[0..15]
  ##   r1: e f g h                           c1: b f j n      dst[32..47]
  ##   r2: i j k l                           c2: c g k o      dst[64..79]
  ##   r3: m n o p                           c3: d h l p      dst[96..111]
  ##   (4×4 excerpt of each 16×16 block, whose columns land 32 f32 apart in dst)
  ##
  ## Runs inside an smstart/smstop bracket (streaming mode, SVE/SME2 only).
  genSmePackATranspose16rows(dst, src, srcRowStride, nColBlocks,
                             validRows)

proc buildSmePackBCopy32f32X4(
    ctx: var AssemblerSME,
    dst0, dst1, dst2, dst3, src, rowStrideOp, nRowsOp: NimNode) =
  ## Records the `sme_packB_copy_32f32_x4` instruction stream into
  ## `ctx` in source order: the smstart bracket, then the row loop of eight `ld1w` loads
  ## and eight `st1w` stores. Operand bindings:
  ## pointers via `"r"`, the stride and row count via `"r"((long)...)`.
  let d0Op = ctx.input("d0", dst0)
  let d1Op = ctx.input("d1", dst1)
  let d2Op = ctx.input("d2", dst2)
  let d3Op = ctx.input("d3", dst3)
  let srcOp = ctx.input("src", src)
  let rsOp = ctx.inputLong("rs", rowStrideOp)
  let nOp = ctx.inputLong("n", nRowsOp)
  ctx.smstart()
  ctx.ptrue(p(0), "s")
  ctx.mov(x(26), view(srcOp))
  ctx.mov(x(9), view(d0Op))
  ctx.mov(x(10), view(d1Op))
  ctx.mov(x(11), view(d2Op))
  ctx.mov(x(12), view(d3Op))
  ctx.lsl(x(25), xview(rsOp), 2)
  ctx.mov(w(27), wview(nOp))
  ctx.label("1")
  for lane in 0 .. 7:
    ctx.ld1w(z(lane), "s", p(0), memStep(x(26), lane))
  ctx.add(x(26), x(26), x(25))
  for panel in 0 .. 3:
    ctx.st1w(z(2 * panel), "s", p(0), mem(x(9 + panel)))
    ctx.st1w(z(2 * panel + 1), "s", p(0), memVL(x(9 + panel), 1))
  for panel in 0 .. 3:
    ctx.add(x(9 + panel), x(9 + panel), 128)
  ctx.subs(w(27), w(27), 1)
  ctx.bne("1b")
  ctx.smstop()
  ctx.clobber("x9", "x10", "x11", "x12", "x25", "x26", "w27")
  ctx.clobberZ()
  ctx.clobberP()
  ctx.clobberCC()
  ctx.clobberMemory()

macro genSmePackBCopy32f32X4(
    dst0, dst1, dst2, dst3, src, rowStrideOp, nRowsOp: typed): untyped =
  ## Expands to the `{.emit: ...}` pragma for
  ## `sme_packB_copy_32f32_x4`.
  var ctx = init(AssemblerSME)
  buildSmePackBCopy32f32X4(ctx, dst0, dst1, dst2, dst3, src,
                           rowStrideOp, nRowsOp)
  result = ctx.generate()

proc sme_packB_copy_32f32_x4*(
    dst0, dst1, dst2, dst3: ptr float32,  # packB + (jr0+i)*kc*nr for i in 0..3
    src: ptr float32,                     # pB_ptr + jr0*nr (k = 0)
    srcRowStride: cint,                   # pB_rs: elements between consecutive B rows
    nRows: cint) =
  ## Streaming SME2 B-pack for four consecutive full `jr` panels.
  ##
  ## Expected input:
  ##   - src: first row of panel `jr0` (row-major f32, rows
  ##     `srcRowStride` elements apart, 32 contiguous f32 per row)
  ##   - srcRowStride: elements between consecutive source rows
  ##   - nRows: source rows per panel (k)
  ##   - dst0..dst3: panel bases (packB + (jr0+i)*kc*nr for i in 0..3)
  ##
  ## Output: row `k` of each panel copied into its packed slot:
  ##   rows 32 f32 apart: `dst_i[k*32 ..< k*32+32] = src[k*rs ..< k*rs+32]`
  ##   for each panel `i` in `0 .. 3`, `k in 0 ..< nRows` (bases dst0..dst3).
  ##   No zero-fill: every lane is copied.
  ##
  ## before (src, one panel):              after (dst_i, packed):
  ##   k0: [32 f32] at src + 0*rs           k0: [32 f32] at dst_i + 0*32
  ##   k1: [32 f32] at src + 1*rs           k1: [32 f32] at dst_i + 1*32
  ##   k2: [32 f32] at src + 2*rs           k2: [32 f32] at dst_i + 2*32
  ##   rows `rs` f32 apart                  rows 32 f32 apart
  ##
  ## Contract: all four panels have 32 valid lanes per row and the panel
  ## count is a multiple of 4.
  ## Runs inside an smstart/smstop bracket (streaming mode, SVE/SME2 only).
  genSmePackBCopy32f32X4(dst0, dst1, dst2, dst3, src, srcRowStride,
                         nRows)

proc buildNeonPackBCopy32f32(
    ctx: var AssemblerSME,
    dst, src, rowStrideOp, nRowsOp: NimNode) =
  ## Records the `neon_packB_copy_32f32` instruction stream into `ctx`
  ## in source order: the row loop of two 4-register `ld1`/`st1` pairs
  ## with the post-copy row advance. Operand bindings: pointers via `"r"`,
  ## the stride and row count via `"r"((long)...)`.
  let dstOp = ctx.input("dst", dst)
  let srcOp = ctx.input("src", src)
  let rsOp = ctx.inputLong("rs", rowStrideOp)
  let nOp = ctx.inputLong("n", nRowsOp)
  ctx.mov(x(20), view(dstOp))
  ctx.mov(x(21), view(srcOp))
  ctx.lsl(x(22), xview(rsOp), 2)
  ctx.sub(x(22), x(22), 128)
  ctx.mov(w(23), wview(nOp))
  ctx.label("1")
  ctx.ld1(vlist(0, 3, "4s"), memPost(x(21), 64))
  ctx.ld1(vlist(4, 7, "4s"), memPost(x(21), 64))
  ctx.add(x(21), x(21), x(22))
  ctx.st1(vlist(0, 3, "4s"), memPost(x(20), 64))
  ctx.st1(vlist(4, 7, "4s"), memPost(x(20), 64))
  ctx.subs(w(23), w(23), 1)
  ctx.bne("1b")
  ctx.clobber("x20", "x21", "x22", "w23")
  ctx.clobber("v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7")
  ctx.clobberCC()
  ctx.clobberMemory()

macro genNeonPackBCopy32f32(
    dst, src, rowStrideOp, nRowsOp: typed): untyped =
  ## Expands to the `{.emit: ...}` pragma for `neon_packB_copy_32f32`.
  var ctx = init(AssemblerSME)
  buildNeonPackBCopy32f32(ctx, dst, src, rowStrideOp, nRowsOp)
  result = ctx.generate()

proc neon_packB_copy_32f32*(
    dst: ptr float32,      # packB + jr*kc*nr (k = 0)
    src: ptr float32,      # pB_ptr + jr*nr (k = 0)
    srcRowStride: cint,    # pB_rs: elements between consecutive B rows
    nRows: cint) =
  ## NEON B-pack for contiguous B rows.
  ##
  ## Expected input:
  ##   - src: first row of panel `jr` (row-major f32, rows
  ##     `srcRowStride` elements apart, 32 contiguous f32 per row)
  ##   - srcRowStride: elements between consecutive source rows
  ##   - nRows: source rows to copy (k)
  ##   - dst: panel base (packB + jr*nr)
  ##
  ## Output: row `k` copied into its packed slot, rows 32 f32 apart:
  ##   `dst[k*32 ..< k*32+32] = src[k*rs ..< k*rs+32]`, `k in 0 ..< nRows`.
  ##
  ## before (src):                         after (dst, packed):
  ##   k0: [32 f32] at src + 0*rs           k0: [32 f32] at dst + 0*32
  ##   k1: [32 f32] at src + 1*rs           k1: [32 f32] at dst + 1*32
  ##   rows `rs` f32 apart                  rows 32 f32 apart
  ##
  ## Caller guarantees 32 valid lanes per source row and routes panels
  ## with `eff < 32` lanes to the scalar copyMem path.
  genNeonPackBCopy32f32(dst, src, srcRowStride, nRows)
