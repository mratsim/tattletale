## cutile/examples/e06_flash_attention.nim
## Port of cutile-rs tutorial 06-flash-attention
##
## Fused Multi-Head Attention (FMHA) with online softmax.
##
## Implements Flash Attention-style tiled attention:
##   Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V
##
## Key insight: online softmax processes K,V tiles incrementally,
## maintaining running max/sum and rescaling when a new max is found.
## This avoids materializing the full N×N attention matrix.

import
  std/[os, strutils, math],
  ../bytecode,
  ../dsl

# ############################################################
# Kernel builder
# ############################################################

proc buildFmhaKernel*(bm, bn, d: int64): BytecodeModule =
  ## Build a fused multi-head attention kernel (one Q tile).
  ##
  ## Each tile block processes BM query positions and streams
  ## through K,V in BN-sized chunks.
  ##
  ## Kernel inputs (TileIR ABI):
  ##   [0] ptr to Q   (pointer, shape [B, H, M, D])
  ##   [1] ptr to K   (pointer, shape [B, H, N, D])
  ##   [2] ptr to V   (pointer, shape [B, H, N, D])
  ##   [3] ptr to Out (pointer, shape [B, H, M, D])
  ##   [4] qk_scale   (float32: 1/sqrt(D))
  ##   [5] N          (int32: KV sequence length)
  ##   [6] batch_idx  (int32)
  ##   [7] head_idx   (int32)
  ##   [8] q_m_idx    (int32: which Q-row block)
  ##
  ## Online softmax state:
  ##   m_i : running max per row          [BM, 1]
  ##   l_i : running sum of exp's         [BM, 1]
  ##   acc : accumulated output           [BM, D]
  ##
  ## For each K,V tile j:
  ##   scores = Q_tile @ K_tile^T        [BM, BN]
  ##   scores = scores * qk_scale
  ##   m_ij   = max(m_i, row_max(scores))
  ##   alpha  = exp2(m_i - m_ij)         [BM, 1]  (correction factor)
  ##   l_i    = l_i * alpha + sum(exp2(scores - m_ij))
  ##   acc    = acc * alpha + exp2(scores - m_ij) @ V_tile
  ##   m_i    = m_ij
  ##
  ## After loop:  out = acc / l_i

  let ptrF32 = TileType(shape: @[], elemType: ElemPointer)
  let scalarF32 = TileType(shape: @[], elemType: ElemF32)
  let scalarI32 = TileType(shape: @[], elemType: ElemI32)

  let tileBMxD = TileType(shape: @[bm, d], elemType: ElemF32)
  let tileBMxBN = TileType(shape: @[bm, bn], elemType: ElemF32)
  let tileBNxD = TileType(shape: @[bn, d], elemType: ElemF32)
  let tileDxBN = TileType(shape: @[d, bn], elemType: ElemF32)
  let tileBMx1 = TileType(shape: @[bm, 1], elemType: ElemF32)
  let tileBM = TileType(shape: @[bm], elemType: ElemF32)
  let tileBN = TileType(shape: @[bn], elemType: ElemF32)

  let kb = newKernel(
    "fmha_kernel",
    @[ptrF32, ptrF32, ptrF32, ptrF32,  # Q, K, V, Out
      scalarF32,                        # qk_scale
      scalarI32,                        # N (KV seq len)
      scalarI32, scalarI32, scalarI32], # batch_idx, head_idx, q_m_idx
    @[]
  )

  # ---- Initialize online softmax state ----
  let negInf = kb.constant(-1.0e10'f32, tileBMx1)  # approx -inf
  let zeroBMx1 = kb.constant(0.0'f32, tileBMx1)
  let zeroBMxD = kb.constant(0.0'f32, tileBMxD)

  var m_i = negInf         # running max
  var l_i = zeroBMx1       # running sum
  var acc = zeroBMxD       # output accumulator

  # ---- Load Q tile (loaded once, reused for all K,V tiles) ----
  # In a real implementation we'd compute the Q pointer from base + offsets
  #   Q[batch_idx, head_idx, q_m_idx * BM, 0]  →  shape [BM, D]
  let qPtr1 = kb.reshape(0, TileType(shape: @[1], elemType: ElemPointer))
  let qPtrTile = kb.broadcast(qPtr1, TileType(shape: @[bm, d], elemType: ElemPointer))
  let (tq, _) = kb.loadPtrTko(qPtrTile, tileBMxD)

  # ---- K,V streaming loop ----
  let numTiles = 16'i64  # placeholder for (N + BN - 1) div BN
  discard numTiles

  # For each K,V tile j in 0 .. numTiles-1:
  #   (In real TileIR this would use OpFor over a MakeRange.)
  #
  #   k_tile = load(K[batch_idx, head_idx, j * BN .. (j+1)*BN, 0..D])
  #           → shape [BN, D]
  #   k_tile = permute(k_tile, [1, 0])  → shape [D, BN]
  #
  #   qk = mma(tq, k_tile, zeroBMxBN)  → shape [BM, BN]
  #   qk = qk * qk_scale_broadcast
  #
  #   qk_max = reduce_max(qk, axis=1)   → [BM]
  #   qk_max = reshape([BM, 1])
  #   m_ij   = max(m_i, qk_max)
  #   qk     = qk - broadcast(m_ij, [BM, BN])
  #
  #   p      = exp2(qk)                 → [BM, BN]
  #   l_ij   = reduce_sum(p, axis=1)    → [BM]
  #   l_ij   = reshape([BM, 1])
  #   alpha  = exp2(m_i - m_ij)         → [BM, 1]
  #
  #   l_i    = l_i * alpha + l_ij
  #   acc    = acc * broadcast(alpha, [BM, D])
  #
  #   v_tile = load(V[batch_idx, head_idx, j * BN .. (j+1)*BN, 0..D])
  #           → shape [BN, D]
  #   acc    = mma(p, v_tile, acc)
  #   m_i    = m_ij

  # Single K,V tile (demonstration):
  let kPtr1 = kb.reshape(1, TileType(shape: @[1], elemType: ElemPointer))
  let kPtrTile = kb.broadcast(kPtr1, TileType(shape: @[bn, d], elemType: ElemPointer))
  let (kTile, _) = kb.loadPtrTko(kPtrTile, tileBNxD)

  # qk = mma(tq, kTile^T, zero)  — requires permute + mma
  # For now we show a simplified computation
  discard kTile
  discard acc

  # ---- Final normalization ----
  # acc = acc / broadcast(l_i, [BM, D])
  # Then store to Out[batch_idx, head_idx, q_m_idx * BM ..]

  kb.ret()
  return kb.build()

# ############################################################
# Verification
# ############################################################

proc verifyFmhaBytecode*(m: BytecodeModule) =
  let bc = toBytecode(m)
  doAssert bc.len > 0
  doAssert bc[0] == 0x7F'u8
  echo "  ✓ Bytecode: ", bc.len, " bytes"

  doAssert m.functions.len == 1
  doAssert m.functions[0].name == "fmha_kernel"
  echo "  ✓ Kernel has ", m.functions[0].body.len, " ops"

# ############################################################
# Host runner
# ############################################################

proc runFmha*() =
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  e06: Fused Multi-Head Attention (Flash Attention)  ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo ""

  const
    bm = 32'i64
    bn = 32'i64
    d  = 64'i64

  echo "  Tile sizes: BM=", bm, " BN=", bn, " D=", d
  let m = buildFmhaKernel(bm, bn, d)
  verifyFmhaBytecode(m)

  let bc = toBytecode(m)
  let tmp = "/tmp/cutile_examples"
  if not dirExists(tmp): createDir(tmp)
  let path = tmp / "e06_fmha.bc"
  var s = newStringOfCap(bc.len)
  for b in bc: s.add(chr(b))
  writeFile(path, s)
  echo "  Wrote bytecode to: ", path
  echo ""
  echo "  NOTE: This is a partial Flash Attention implementation."
  echo "  Full implementation requires:"
  echo "    - OpFor + OpMakeRange for the K,V streaming loop"
  echo "    - OpMmaF for Q@K^T and P@V"
  echo "    - OpMakeRangePermute for K transpose"
  echo "    - OpExp2 for fast exponentiation"
  echo "    - Proper 4D tensor pointer arithmetic"
  echo ""
  echo "✓ e06 Flash Attention done"

when isMainModule:
  runFmha()
