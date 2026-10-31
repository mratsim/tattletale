/* Hardware topology, Windows backend. */
#ifndef TPC_TOPOLOGY_WINDOWS_H
#define TPC_TOPOLOGY_WINDOWS_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static inline int32_t tpc_topology_getNumThreadsOS(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int32_t)si.dwNumberOfProcessors : 1;
}

#endif /* TPC_TOPOLOGY_WINDOWS_H */
