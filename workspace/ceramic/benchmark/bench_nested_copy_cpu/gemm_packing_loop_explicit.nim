## Reference: Explicit blocked copy loops for GEMM packing.
##
## These are the hand-written loops from the autoresearch session
## that achieved 142 GFlop/s GEMM performance. They use precomputed
## tile offsets, copyMem for contiguous rows, and assume_aligned hints.
##
## These serve as the "ground truth" against which generic
## iteration methods are compared.

proc copy_autoresearch_B*(
    pB_aligned: var openArray[float32];
    panel: openArray[float32]; pB_rs, pB_cs: int;
    num_jr, kc, nr: int) =
  ## B-packing: from column-major panel to LayoutRight pack buffer.
  if pB_cs == 1:
    # Contiguous rows — use copyMem
    for jr in 0 ..< num_jr:
      for k in 0 ..< kc:
        let dstOff = jr * kc * nr + k * nr
        let srcOff = k * pB_rs + jr * nr
        copyMem(addr pB_aligned[dstOff], addr panel[srcOff], nr * sizeof(float32).int)
  else:
    for jr in 0 ..< num_jr:
      for k in 0 ..< kc:
        let dstOff = jr * kc * nr + k * nr
        let srcOff = k * pB_rs + jr * nr * pB_cs
        for jj in 0 ..< nr:
          pB_aligned[dstOff + jj] = panel[srcOff + jj * pB_cs]

proc copy_autoresearch_A*(
    packBuf: var openArray[float32]; pA_aligned: var openArray[float32];
    panel: openArray[float32]; pA_rs, pA_cs: int;
    num_ir, kc, mr, current_mc: int) =
  ## A-packing: from column-major panel to LayoutRight pack buffer,
  ## with edge-case handling for partial tiles.
  let num_ir_eff = if current_mc > 0: (current_mc + mr - 1) div mr else: 0
  if pA_rs == 1:
    # Contiguous rows — use copyMem
    for ir in 0 ..< num_ir_eff:
      let srcRow = ir * mr
      let lastTile = (srcRow + mr) > current_mc
      for k in 0 ..< kc:
        let dstOff = ir * kc * mr + k * mr
        let srcOff = srcRow + k * pA_cs
        if lastTile:
          let valid = current_mc - srcRow
          copyMem(addr pA_aligned[dstOff], addr panel[srcOff], valid * sizeof(float32).int)
          for ii in valid ..< mr:
            pA_aligned[dstOff + ii] = 0'f32
        else:
          copyMem(addr pA_aligned[dstOff], addr panel[srcOff], mr * sizeof(float32).int)
  else:
    for ir in 0 ..< num_ir_eff:
      let srcRow = ir * mr
      for k in 0 ..< kc:
        let dstOff = ir * kc * mr + k * mr
        let srcOff = srcRow * pA_rs + k * pA_cs
        for ii in 0 ..< mr:
          if (srcRow + ii) < current_mc:
            pA_aligned[dstOff + ii] = panel[srcOff + ii * pA_rs]
          else:
            pA_aligned[dstOff + ii] = 0'f32

# ── Generic helpers ──

proc size[Rank: static int](shape: array[Rank, int]; rank: int): int =
  result = shape[0]
  for i in 1 ..< rank:
    result *= shape[i]
