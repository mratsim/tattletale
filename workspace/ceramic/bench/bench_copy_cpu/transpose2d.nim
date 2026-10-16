## Transpose implementations.
##
## `transpose2D_laser`: 2D blocked transpose (from Laser swapaxes.nim)
## `transpose2D_naive`:  simple double for-loop

when defined(openmp):
  {.passC: "-fopenmp".}
  {.passL: "-fopenmp".}

func transpose2D_laser*[T](
    dst, src: ptr (T or UncheckedArray[T]),
    NR, NC: Natural) =
  ## Efficient physical transposition of a contiguous 2D matrix
  ## Output:
  ##   - dst: a pointer to an allocated buffer of size NC * NR
  ##     dst does not need to be initialized and will be overwritten
  ## Input:
  ##   - src: a pointer to the source matrix of shape [NR, NC]
  ##   - NR, NC: the number of rows and columns respectively in the source matrix.
  ##
  ## Uses 32×32 blocked tiling with OpenMP collapse(2) + #pragma omp simd.
  ## Writes to dst contiguously (scatter source) — scatters cheaper than gathers.

  const blck = 32

  {.emit: "/*transpose_laser*/".}
  {.emit: """
    `T` (* __restrict pd)[`NR`] = (void*)`dst`;
    `T` (* __restrict ps)[`NC`] = (void*)`src`;

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < `NC`; j+=`blck`)
      for (int i = 0; i < `NR`; i+=`blck`)
        for (int jj = j; jj<j+`blck` && jj < `NC`; jj++)
          #pragma omp simd
          for (int ii = i; ii<i+`blck` && ii < `NR`; ii++)
            pd[jj][ii] = ps[ii][jj];
  """.}

func transpose2D_naive*(
    dst, src: ptr float32,
    NR, NC: Natural) =
  ## Simple double for-loop transpose.
  ## Writes to dst contiguously: dst[j*NR + i] = src[i*NC + j]
  let pDst = cast[ptr UncheckedArray[float32]](dst)
  let pSrc = cast[ptr UncheckedArray[float32]](src)
  for i in 0 ..< NR:
    for j in 0 ..< NC:
      pDst[j * NR + i] = pSrc[i * NC + j]

func transpose2D_cacheBlock*(
    dst, src: ptr float32,
    NR, NC: Natural) =
  ## 1D cache-blocked transpose (blck=64).
  ## Blocks on the source row dimension for cache efficiency.
  const blck = 64
  {.emit: "/*cache_block*/".}
  {.emit: """
    NF32* __restrict pd = (NF32*)`dst`;
    NF32* __restrict ps = (NF32*)`src`;
    for (int i = 0; i < `NR`; i+=`blck`)
      for (int j = 0; j < `NC`; ++j)
        for (int ii = i; ii < i+`blck` && ii < `NR`; ++ii)
          pd[ii+j*`NR`] = ps[j+ii*`NC`];
  """.}

func transpose2D_cacheBlockPrefetch*(
    dst, src: ptr float32,
    NR, NC: Natural) =
  ## 1D cache-blocked transpose with prefetch (blck=32).
  ## Prefetches the next row-block for read.
  const blck = 32
  {.emit: "/*cache_blk_prefetch*/".}
  {.emit: """
    NF32* __restrict pd = (NF32*)`dst`;
    NF32* __restrict ps = (NF32*)`src`;
    for (int i = 0; i < `NR`; i+=`blck`) {
      for (int j = 0; j < `NC`; ++j)
        for (int ii = i; ii < i+`blck` && ii < `NR`; ++ii)
          pd[ii+j*`NR`] = ps[j+ii*`NC`];
      __builtin_prefetch(&ps[(i+`blck`)*`NC`], 0, 1);
    }
  """.}
