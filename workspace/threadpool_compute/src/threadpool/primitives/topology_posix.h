/* Hardware topology, posix backend.
 *
 * Kernel APIs provide the physical core count on macOS and FreeBSD,
 * other OSes including Linux, BSDs and Solaris do not.
 * /proc/cpuinfo is avoided: the output varies a lot depending on topology
 * so a parser needs several assumptions and attracts edge cases
 * file descriptors are painful to support in trusted enclaves like SGX
 */
#ifndef TPC_TOPOLOGY_POSIX_H
#define TPC_TOPOLOGY_POSIX_H

#include <unistd.h>

static inline int32_t tpc_topology_getNumThreadsOS(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int32_t)n : 1;
}

#endif /* TPC_TOPOLOGY_POSIX_H */
