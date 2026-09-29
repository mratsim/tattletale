## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Hardware copy instructions

import ./h_copy_configgen

declareCopyAtoms:
  # Universal blocking copy
  # - a dst element <- src element assignment with no asm
  # - any memory spaces, every backend, the atom for every non-NVIDIA path
  # - synchronous, the commit/wait slots discard
  # - a false predicate zero-fills the chunk, the GEMM ragged-K contract
  atom UNIVERSAL_COPY:
    kind: ckCopy
    srcSpace: spAny
    dstSpace: spAny
    vecBytes: 16
    minAlign: 1
    zeroFill: true
    cache: cache_default
    instr: ""
  atom SM80_CP_ASYNC_CG_16B:
    kind: ckCopy
    srcSpace: spGmem
    dstSpace: spSmem
    vecBytes: 16
    minAlign: 16
    cache: cache_global_bypass_L1
    minCudaArch: 800
    instr: "cp.async.cg.shared.global.L2::128B [%0], [%1], %2"
  atom SM80_CP_ASYNC_CG_16B_ZFILL:
    kind: ckCopy
    srcSpace: spGmem
    dstSpace: spSmem
    vecBytes: 16
    minAlign: 16
    cache: cache_global_bypass_L1
    zeroFill: true
    minCudaArch: 800
    instr: "cp.async.cg.shared.global.L2::128B [%0], [%1], %2"
  atom SM80_CP_ASYNC_CA_16B:
    kind: ckCopy
    srcSpace: spGmem
    dstSpace: spSmem
    vecBytes: 16
    minAlign: 16
    cache: cache_always
    minCudaArch: 800
    instr: "cp.async.ca.shared.global.L2::128B [%0], [%1], %2"
  atom SM80_CP_ASYNC_CA_16B_ZFILL:
    kind: ckCopy
    srcSpace: spGmem
    dstSpace: spSmem
    vecBytes: 16
    minAlign: 16
    cache: cache_always
    zeroFill: true
    minCudaArch: 800
    instr: "cp.async.ca.shared.global.L2::128B [%0], [%1], %2"
  atom SM80_CP_ASYNC_CA_8B:
    kind: ckCopy
    srcSpace: spGmem
    dstSpace: spSmem
    vecBytes: 8
    minAlign: 8
    cache: cache_always
    minCudaArch: 800
    instr: "cp.async.ca.shared.global.L2::128B [%0], [%1], %2"
  atom SM80_CP_ASYNC_CA_8B_ZFILL:
    kind: ckCopy
    srcSpace: spGmem
    dstSpace: spSmem
    vecBytes: 8
    minAlign: 8
    cache: cache_always
    zeroFill: true
    minCudaArch: 800
    instr: "cp.async.ca.shared.global.L2::128B [%0], [%1], %2"
  atom SM80_CP_ASYNC_CA_4B:
    kind: ckCopy
    srcSpace: spGmem
    dstSpace: spSmem
    vecBytes: 4
    minAlign: 4
    cache: cache_always
    minCudaArch: 800
    instr: "cp.async.ca.shared.global.L2::128B [%0], [%1], %2"
  atom SM80_CP_ASYNC_CA_4B_ZFILL:
    kind: ckCopy
    srcSpace: spGmem
    dstSpace: spSmem
    vecBytes: 4
    minAlign: 4
    cache: cache_always
    zeroFill: true
    minCudaArch: 800
    instr: "cp.async.ca.shared.global.L2::128B [%0], [%1], %2"

  # sm80 cp.async group bookkeeping
  # - commit closes a prepared-copies group
  # - wait blocks until all but `waitDepth` recent groups completed
  atom SM80_CP_ASYNC_COMMIT:
    kind: ckCommit
    instr: "cp.async.commit_group"
  atom SM80_CP_ASYNC_WAIT:
    kind: ckWait
    instr: "cp.async.wait_group %0"
    waitDepth: 2

  # Universal commit/wait slots for the blocking tier, no asm, discards
  atom UNIVERSAL_COMMIT:
    kind: ckCommit
    instr: ""
  atom UNIVERSAL_WAIT:
    kind: ckWait
    instr: ""
