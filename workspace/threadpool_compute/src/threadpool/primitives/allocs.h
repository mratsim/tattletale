/* Aligned allocation, Windows/POSIX/C11 backends. */
#ifndef TPC_ALLOCS_H
#define TPC_ALLOCS_H

#include <stdlib.h>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <malloc.h>
#endif

/** Allocates `size` bytes at an `alignment` boundary.
 *
 *  The size is rounded up to a multiple of the alignment, the C11
 *  aligned_alloc contract requires it and the Windows and POSIX
 *  backends adopt it for a uniform result.
 *
 *  Parameters:
 *  In:      alignment:   a power-of-two alignment, at least sizeof(void*).
 *           size:        the allocation size, 0 returns NULL.
 *
 *  Returns: the allocation or NULL on exhaustion, release it with
 *           tpc_alignedFree.
 */
static inline void* tpc_alignedAlloc(size_t alignment, size_t size) {
    size = (size + alignment - 1) / alignment * alignment;
#if defined(_WIN32)
    /* _aligned_malloc takes the size first, the alignment second */
    return _aligned_malloc(size, alignment);
#elif defined(__APPLE__)
    void* p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) {
        return NULL;
    }
    return p;
#else
    return aligned_alloc(alignment, size);
#endif
}

/** Releases an allocation from tpc_alignedAlloc.
 *
 *  Parameters:
 *  In:      p: the allocation, NULL is a no-op.
 */
static inline void tpc_alignedFree(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

#endif /* TPC_ALLOCS_H */
