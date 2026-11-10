# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Page pool write op suite (Metal): pool-coupled chunk decomposition and coordinates.
##
## Test that tn_pool_write places post-rope K/V entries at the right
## (page, layer, in-page) coordinates of PagePool slabs through one eval
## at multi-workgroup launch sizes.
##
## - single-chunk, multi-page, cached-prefix and remapped page-table
##   writes are exact against host-computed expectations
## - the grid-stride write-once regression lives in the tensorcery test
##   template (runPoolWriteTest)
##
## Run from the worktree root:
##   nim c -r -d:release --nimcache:nimcache/ttf --outdir:build/ttf --path:"$PWD" workspace/tattleformers/tests/stateful/test_pool_write.nim
##

import workspace/tensorcery/src/tensorcery
import workspace/tattleformers/src/stateful/kvcache
import workspace/tattleformers/src/stateful/page_pool

const
  NumPages = 4
  NumLayers = 2
  KvHeads = 2
  HeadDim = 4
  PageSize = TokensPerPage # 256
  SlotElems = PageSize * KvHeads * HeadDim
  SlabElems = NumPages * NumLayers * SlotElems

proc uploadSlab(slab: Tensor, seed: int): seq[bfloat16] =
  ## Fills the whole slab with known sentinel values, returns the host copy.
  result = newSeq[bfloat16](SlabElems)
  for i in 0 ..< SlabElems:
    result[i] = (float32((i * 31 + seed) mod 251) * 0.125'f32 - 8.0'f32).to(bfloat16)
  slab.upload(result)

proc srcRows(seqLen: int, seed: int): tuple[k, v: Tensor, kh, vh: seq[bfloat16]] =
  ## Builds (1, seq_len, kv_heads, head_dim) src entries, device and host copies.
  let shape = @[1, seqLen, KvHeads, HeadDim]
  result.kh = newSeq[bfloat16](seqLen * KvHeads * HeadDim)
  result.vh = newSeq[bfloat16](seqLen * KvHeads * HeadDim)
  for i in 0 ..< result.kh.len:
    result.kh[i] = (float32((i * 7 + seed) mod 97) * 0.25'f32 - 4.0'f32).to(bfloat16)
    result.vh[i] = (float32((i * 11 + seed) mod 89) * 0.25'f32 - 4.0'f32).to(bfloat16)
  result.k = toTensor(result.kh, shape)
  result.v = toTensor(result.vh, shape)

proc slabIdx(page, layer, pos, h, d: int): int =
  (((page * NumLayers + layer) * PageSize + pos) * KvHeads + h) * HeadDim + d

proc srcIdx(t, h, d: int): int =
  (t * KvHeads + h) * HeadDim + d

proc runWrite(pool: PagePool, layer: int, offset, seqLen, cachedTokens: int, kSrc, vSrc: Tensor, pageSlots: openArray[int]): Tensor =
  ## One eval of the pool write op over the pool slabs.
  ## The write enqueues when the pool-write node evaluates.
  ## Reading the slab leaf alone submits nothing.
  let w = pool_write(pool.kBuffer(), pool.vBuffer(), kSrc, vSrc, layer,
    offset, seqLen, cachedTokens, pageSlots)
  eval(w)
  w

proc runMain() =
  let pool = PagePool.init(NumPages, NumLayers, KvHeads, HeadDim)

  # single chunk: offset 0, 10 entries, one page, both layers write independently
  block:
    let kSlab = uploadSlab(pool.kBuffer(), 1)
    let vSlab = uploadSlab(pool.vBuffer(), 2)
    let (kSrc, vSrc, kh, vh) = srcRows(10, 3)
    discard runWrite(pool, 0, 0, 10, 0, kSrc, vSrc, @[0, 1, 2, 3])
    # Layer 0 stores the entries, layer 1 keeps its sentinels
    let gotK = pool.kBuffer().toSeq(bfloat16)
    let gotV = pool.vBuffer().toSeq(bfloat16)
    for t in 0 ..< 10:
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          doAssert gotK[slabIdx(0, 0, t, h, d)].to(float32) == kh[srcIdx(t, h, d)].to(float32),
            "K layer0 (" & $t & "," & $h & "," & $d & ")"
          doAssert gotV[slabIdx(0, 0, t, h, d)].to(float32) == vh[srcIdx(t, h, d)].to(float32),
            "V layer0 (" & $t & "," & $h & "," & $d & ")"
          doAssert gotK[slabIdx(0, 1, t, h, d)].to(float32) == kSlab[slabIdx(0, 1, t, h, d)].to(float32),
            "K layer1 sentinel (" & $t & "," & $h & "," & $d & ")"
          doAssert gotV[slabIdx(0, 1, t, h, d)].to(float32) == vSlab[slabIdx(0, 1, t, h, d)].to(float32),
            "V layer1 sentinel (" & $t & "," & $h & "," & $d & ")"
    # Positions outside the write range keep their sentinels
    for pos in 10 ..< 16:
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          doAssert gotK[slabIdx(0, 0, pos, h, d)].to(float32) == kSlab[slabIdx(0, 0, pos, h, d)].to(float32),
            "K sentinel past range (" & $pos & "," & $h & "," & $d & ")"

  # multi-page: offset 200, 600 entries crossing pages 0..2, 19 threadgroups
  block:
    let kSlab = uploadSlab(pool.kBuffer(), 4)
    let vSlab = uploadSlab(pool.vBuffer(), 5)
    let (kSrc, vSrc, kh, vh) = srcRows(600, 6)
    discard runWrite(pool, 1, 200, 600, 0, kSrc, vSrc, @[0, 1, 2, 3])
    let gotK = pool.kBuffer().toSeq(bfloat16)
    let gotV = pool.vBuffer().toSeq(bfloat16)
    for t in 0 ..< 600:
      let globalPos = 200 + t
      let page = globalPos div PageSize
      let pos = globalPos mod PageSize
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          doAssert gotK[slabIdx(page, 1, pos, h, d)].to(float32) == kh[srcIdx(t, h, d)].to(float32),
            "K multi (" & $t & " @" & $globalPos & "," & $h & "," & $d & "): got " &
            $gotK[slabIdx(page, 1, pos, h, d)].to(float32) & " want " & $kh[srcIdx(t, h, d)].to(float32)
          doAssert gotV[slabIdx(page, 1, pos, h, d)].to(float32) == vh[srcIdx(t, h, d)].to(float32),
            "V multi (" & $t & " @" & $globalPos & "," & $h & "," & $d & ")"
    # layer 0 keeps its sentinels across all touched pages
    for page in 0 .. 2:
      for pos in 0 ..< PageSize:
        for h in 0 ..< KvHeads:
          for d in 0 ..< HeadDim:
            doAssert gotK[slabIdx(page, 0, pos, h, d)].to(float32) == kSlab[slabIdx(page, 0, pos, h, d)].to(float32),
              "K multi layer0 sentinel (" & $page & "," & $pos & ")"

  # cached-prefix skip: offset 100, 10 entries, cachedTokens 105 -> entries 0..5 skipped
  block:
    let kSlab = uploadSlab(pool.kBuffer(), 7)
    let (kSrc, vSrc, kh, vh) = srcRows(10, 8)
    discard runWrite(pool, 0, 100, 10, 105, kSrc, vSrc, @[0, 1, 2, 3])
    let gotK = pool.kBuffer().toSeq(bfloat16)
    let gotV = pool.vBuffer().toSeq(bfloat16)
    # writeStart = max(0, 105 - 100) = 5: global positions 105..109 written,
    # 100..104 keep their sentinels
    for t in 0 ..< 10:
      let globalPos = 100 + t
      let pos = globalPos mod PageSize
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          if t < 5:
            doAssert gotK[slabIdx(0, 0, pos, h, d)].to(float32) == kSlab[slabIdx(0, 0, pos, h, d)].to(float32),
              "K cached sentinel (" & $t & "," & $h & "," & $d & ")"
          else:
            doAssert gotK[slabIdx(0, 0, pos, h, d)].to(float32) == kh[srcIdx(t, h, d)].to(float32),
              "K cached (" & $t & "," & $h & "," & $d & ")"
            doAssert gotV[slabIdx(0, 0, pos, h, d)].to(float32) == vh[srcIdx(t, h, d)].to(float32),
              "V cached (" & $t & "," & $h & "," & $d & ")"

  # remapped page table case:
  # - sequence page 0 targets pool slot 2
  # - sequence page 1 targets pool slot 0
  # - the write uses the table over the slot order
  block:
    let kSlab = uploadSlab(pool.kBuffer(), 9)
    let vSlab = uploadSlab(pool.vBuffer(), 10)
    let (kSrc, vSrc, kh, vh) = srcRows(PageSize + 10, 11)
    discard runWrite(pool, 1, 0, PageSize + 10, 0, kSrc, vSrc, @[2, 0, 1, 3])
    let gotK = pool.kBuffer().toSeq(bfloat16)
    let gotV = pool.vBuffer().toSeq(bfloat16)
    for t in 0 ..< PageSize + 10:
      let seqPage = t div PageSize
      let poolPage = [2, 0, 1, 3][seqPage]
      let pos = t mod PageSize
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          doAssert gotK[slabIdx(poolPage, 1, pos, h, d)].to(float32) == kh[srcIdx(t, h, d)].to(float32),
            "K remapped (" & $t & " seqPage " & $seqPage & " slot " & $poolPage & "," & $h & "," & $d & ")"
          doAssert gotV[slabIdx(poolPage, 1, pos, h, d)].to(float32) == vh[srcIdx(t, h, d)].to(float32),
            "V remapped (" & $t & " seqPage " & $seqPage & " slot " & $poolPage & "," & $h & "," & $d & ")"

  echo "pool write op (Metal): all checks passed"

runMain()
