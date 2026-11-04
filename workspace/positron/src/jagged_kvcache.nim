# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

type
  JaggedKv* = object
    ## Kernel descriptor of the paged KV cache: the metadata the kernel walks
    ## to gather each sequence's pages out of the pool, the page storage
    ## belonging to the PagePool.
    ## Pointer fields bind whole at launch, batchSize bounds the walk.
    ##
    ##   sequence 0 = "The cat sat on the mat and slept" (8 tokens)
    ##   sequence 1 = "The cat sat on the dog and ran away" (9 tokens)
    ##
    ##   pool (PagePool's page storage)
    ##   [ The cat sat on ][ the dog and ran ][ the mat and slept ][ away ]
    ##          page 0             page 1                page 2       page 3
    ##                ^                 ^                    ^           ^
    ##                .                 .                    .           +-- sequence 1's third page
    ##                .                 .                    +-- sequence 0's second page
    ##                .                 +-- sequence 1's second page
    ##                +-- both sequences' first page, shared prefix
    ##
    ##   indices  len = capacity                             offsets  len = capacity + 1
    ##   [ 0 ][ 2 ][ 0 ][ 1 ][ 3 ]                           [ 0 ][ 2 ][ 5 ][   ][   ]
    ##
    ##   seqlens  len = capacity
    ##   [ 8 ][ 9 ][   ][   ][   ]
    ##
    ## - the page ids of a sequence `s` ∈ [0, batchSize) are indices[offsets[s] ..< offsets[s + 1]]
    ##   sequence 0 = page ids [0, 2] = "The cat sat on the mat and slept"
    ##   sequence 1 = page ids [0, 1, 3] = "The cat sat on the dog and ran away"
    ## - page 0 serves both sequences, the page ids interleave, each new
    ##   page goes to whichever sequence appends next
    ## - page p stores the key and value vectors of its tokens in buf[p * page_size ..< (p + 1) * page_size],
    ##   page_size is a kernel static
    ## - offsets[batchSize] = 5 is where the newest sequence ends and the next one starts there
    offsets*: ptr UncheckedArray[int32]
    indices*: ptr UncheckedArray[int32]
    seqlens*: ptr UncheckedArray[int32]
    batchSize*: int
