# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests for k_tile_gemm_smem:
##
## - bank patterns of the stage store and of the mma fragment loads.
##   Pure functions, no GPU.
## - staged contents against a host reference, including zero-fill
##   for ragged bounds and the k-tail chunk's elementwise prefix copy
## - gemm_smem against the fp32 host reference over the highlight
##   and fuzz case lists, one operand pass per bf16 and f16
##
## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_gemm/t_kernel_gemm_smem.nim

import std/[random, times]
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/kernels/k_tile_gemm_smem

# ═══════════════════════════════════════════════════════════════════════
#  Bank-pattern analysis (pure, host side)
# ═══════════════════════════════════════════════════════════════════════

const StageK = 64
const StageRows = 64

proc chunkCoord(tid, i: int): (int, int) =
  ## Staging chunk (kc, rc) that thread `tid` copies at round `i`,
  ## 64 staging threads, chunks flat-numbered c = tid + i*64 over
  ## the (chunksPerRow, stageRows) chunk grid
  let c = tid + i * 64
  (c mod stageChunksPerRow(StageK), c div stageChunksPerRow(StageK))

proc testStoreBanks() =
  ## One row's 8 staged chunks stay 16-byte aligned and span 4
  ## consecutive words each, tiling that row's 32 bank words with no
  ## overlap inside the phase that writes them
  for phase in 0 ..< 8:
    var firsts: array[8, int]
    for t in 0 ..< 8:
      let (kc, rc) = chunkCoord(phase * 8 + t, 0)
      firsts[t] = stageBankWord(rc, kc * 8, StageK)
    for a in 0 ..< 8:
      for b in a + 1 ..< 8:
        # one chunk spans 4 consecutive words, the 8 chunks of a phase
        # must tile the 32-word bank range without overlap
        doAssert firsts[a] div 4 != firsts[b] div 4 or
          (firsts[a] mod 4) + 4 <= firsts[b] mod 4 or
          (firsts[b] mod 4) + 4 <= firsts[a] mod 4,
          "store phase " & $phase & ": chunks " & $a & "," & $b &
          " share banks (" & $firsts[a] & " vs " & $firsts[b] & ")"
  echo "  OK (stage store bank pattern, " & $StageK & "x" & $StageRows & ")"

proc fragmentTableA(atomMap: Layout): array[32, array[16, (int, int)]] =
  ## (row, k) of every (lane, value) A-fragment slot of one k16 step.
  ## The atom map sends (lane, value) to a col-major offset inside
  ## the (16, 16) atom cell, and the M repetition adds 16 rows.
  for t0 in 0 ..< 4:
    for t1 in 0 ..< 8:
      for v0 in 0 ..< 2:
        for v1 in 0 ..< 2:
          for v2 in 0 ..< 2:
            for r0 in 0 ..< 2:
              let off = int(crd2idx(atomMap, ((t0, t1), (v0, v1, v2))))
              result[t0 + 4*t1][v0 + 2*v1 + 4*v2 + 8*r0] =
                (16*r0 + off mod 16, off div 16)

proc fragmentTableB(atomMap: Layout): array[32, array[16, (int, int)]] =
  ## (col, k) of every (lane, value) B-fragment slot of one k16 step.
  ## The atom map sends (lane, value) to a col-major offset inside
  ## the (8, 16) atom cell, and the N repetition adds 8 columns.
  for t0 in 0 ..< 4:
    for t1 in 0 ..< 8:
      for v0 in 0 ..< 2:
        for v1 in 0 ..< 2:
          for r0 in 0 ..< 4:
            let off = int(crd2idx(atomMap, ((t0, t1), (v0, v1))))
            result[t0 + 4*t1][v0 + 2*v1 + 4*r0] =
              (8*r0 + off mod 8, off div 8)

proc testFragmentLoadBanks() =
  ## For every fragment register, one per value pair with adjacent
  ## values sharing a 4-byte word, the 32 lanes read bank-distinct
  ## words or else the same word when the value broadcasts.
  ##
  ## The check covers both warp rows, every k16 step of the stage,
  ## and the padded layout.
  # Host-side analysis, no DSL target block: the warp-tile mma
  # resolves against the SM80 bf16 atom directly
  const atom = SM80_16x8x16_F32BF16BF16F32_TN
  const coordsA = atom.getLayoutA().fragmentTableA()
  const coordsB = atom.getLayoutB().fragmentTableB()
  for warpRow in 0 ..< 2:
    for warpCol in 0 ..< 2:
      for kStep in 0 ..< StageK div 16:
        # A: 16 values per lane per step = 8 register pairs
        for pair in 0 ..< 8:
          var words: array[32, int]
          for t in 0 ..< 32:
            let (m, k) = coordsA[t][pair * 2]
            words[t] = stageBankWord(warpRow * 32 + m, kStep * 16 + k, StageK)
            let (_, kHi) = coordsA[t][pair * 2 + 1]
            doAssert words[t] == stageBankWord(warpRow * 32 + m, kStep * 16 + kHi, StageK),
              "A pair " & $pair & " splits its 4-byte word"
          for a in 0 ..< 32:
            for b in a + 1 ..< 32:
              doAssert (words[a] mod 32) != (words[b] mod 32) or words[a] == words[b],
                "A load pair " & $pair & " bank conflict: lanes " & $a & "," & $b &
                " words " & $words[a] & "," & $words[b]
        # B: 16 values per lane per step = 8 register pairs
        for pair in 0 ..< 8:
          var words: array[32, int]
          for t in 0 ..< 32:
            let (n, k) = coordsB[t][pair * 2]
            words[t] = stageBankWord(warpCol * 32 + n, kStep * 16 + k, StageK)
          for a in 0 ..< 32:
            for b in a + 1 ..< 32:
              doAssert (words[a] mod 32) != (words[b] mod 32) or words[a] == words[b],
                "B load pair " & $pair & " bank conflict: lanes " & $a & "," & $b &
                " words " & $words[a] & "," & $words[b]
  echo "  OK (fragment load bank pattern, warp grid 2x2, all k steps)"


# ═══════════════════════════════════════════════════════════════════════
#  The suite
# ═══════════════════════════════════════════════════════════════════════

proc main =
  echo "t_kernel_gemm_smem"
  testStoreBanks()
  testFragmentLoadBanks()
  # Kernel commit adds the GPU staging and gemm_smem checks

when isMainModule:
  main()
