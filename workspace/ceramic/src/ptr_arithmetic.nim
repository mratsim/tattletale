## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

template `+%`*[T](p: ptr T, offset: SomeInteger): ptr T =
  ## Pointer arithmetic | increment
  cast[ptr T](cast[uint](p) + uint(offset) * uint(sizeof(T)))

template `+%`*[T](p: ptr UncheckedArray[T], offset: SomeInteger): ptr UncheckedArray[T] =
  cast[ptr UncheckedArray[T]](cast[uint](p) + cast[uint](offset)*uint(sizeof(T)))
