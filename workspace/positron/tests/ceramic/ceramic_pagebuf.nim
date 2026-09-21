# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

## Page-aligned host buffers for the ceramic kernel tests on Metal.
## The engine's no-copy binding contract needs a page-aligned pointer with a page-multiple byte length,
## any other binding copies and the kernel's in-place writes are lost.
##
## Shared by the ceramic-vs-naive suites here, no `t_`/`test_` prefix so the test task never
## compiles it standalone.
##
## Untouched-memory checks:
##   - `assertTailZero`       elements past the logical extent stay zero
##   - `assertReadUnchanged`  a kernel-read buffer stays bit-identical

import workspace/crucible

const HostPageSize* = 16384  ## Metal no-copy binding alignment

type PageBuf*[T] = object
  ## Page-aligned buffer over owned storage.
  ## - `elems` covers the requested extent plus the guaranteed tail page, the byte
  ##   extent rounds up to a `HostPageSize` multiple
  ## - the kernel's writes stay inside the caller's requested extent, bytes past the extent stay zero scratch
  data*: pointer
  elems*: int

proc posixMemalign(memptr: ptr pointer; alignment, size: csize_t): cint
  {.importc: "posix_memalign", header: "<stdlib.h>".}
proc freeShared(p: pointer) {.importc: "free", header: "<stdlib.h>".}

proc allocPageBuf*[T](elems: int): PageBuf[T] =
  ## Page-aligned zero-filled buffer, byte extent rounded up to a `HostPageSize` multiple
  ## (posix_memalign does not zero), plus one full page of zero tail beyond the extent.
  ##
  ## Contract:
  ## - an extent that is itself page-exact would otherwise round to its own byte length,
  ##   leaving `assertTailZero` an empty tail on exactly-page-sized buffers
  ## - the tail page keeps the sentinel teeth without changing the kernel-visible addresses,
  ##   the binding byte length stays a `HostPageSize` multiple
  let nbytes = (elems + HostPageSize div sizeof(T)) * sizeof(T)
  let rounded = (nbytes + HostPageSize - 1) div HostPageSize * HostPageSize
  var p: pointer = nil
  doAssert posixMemalign(addr p, csize_t(HostPageSize), csize_t(rounded)) == 0
  zeroMem(p, rounded)
  PageBuf[T](data: p, elems: rounded div sizeof(T))

func hostPtr*[T](buf: PageBuf[T]): ptr UncheckedArray[T] =
  ## Raw view of the buffer's storage, for copyMem in and out and for the element checks.
  cast[ptr UncheckedArray[T]](buf.data)

func pa*[T](buf: PageBuf[T]): PtrArg[T] =
  ## Pointer-argument form of the buffer, byte length a `HostPageSize` multiple, so the binding takes the no-copy path.
  PtrArg[T](buf: buf.hostPtr, len: buf.elems, off: 0)

proc freePageBuf*[T](buf: PageBuf[T]) =
  ## Frees the page-aligned storage.
  freeShared(buf.data)

proc assertTailZero*[T: uint16|float32](buf: PageBuf[T], usedElems: int) =
  ## Elements past `usedElems` must still hold the zero initialization, kernel writes stay inside the logical extent.
  for i in usedElems ..< buf.elems:
    doAssert buf.hostPtr[i] == 0, "buffer written past its extent at element " & $i

proc assertReadUnchanged*[T: uint16|float32](buf: PageBuf[T], want: seq[T]) =
  ## A kernel-read buffer must stay bit-identical, the host memory is the device memory under the no-copy binding.
  for i in 0 ..< want.len:
    doAssert buf.hostPtr[i] == want[i], "kernel-read buffer modified at element " & $i
