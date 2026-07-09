## Generic (scalar) micro-kernel — fallback for non-SIMD targets.
##
## Same structure as ex01's gemm_ukernel.

proc gemm_ukernel_generic*[T; MR, NR: static int](
    packA, packB: ptr UncheckedArray[T];
    AB: var array[MR, array[NR, T]];
    kc: int) =
  # Zero the accumulator first (AB is {.noInit.} in caller)
  for ri in 0 ..< MR:
    for rj in 0 ..< NR:
      AB[ri][rj] = T(0)
  for k in 0 ..< kc:
    for ri in 0 ..< MR:
      let ai = packA[k * MR + ri]
      for rj in 0 ..< NR:
        AB[ri][rj] += ai * packB[k * NR + rj]
