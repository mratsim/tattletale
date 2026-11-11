# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Swizzle concretization: shared-memory bank-permutation layouts.
##
## A swizzle permutes the offsets a layout produces, spreading accesses across shared-memory banks.
##
## Two equivalent forms, ported from the vendored pycute (`pycute/swizzle.py`):
## - F2 arithmetic, int level: XOR addition and carry-less multiplication
## - the Swizzle functor `SW_<bits>_<base>_<shift>`, an offset permutation
##
## SwizzledLayout applies the functor on top of any algebra Layout over the codomain, call(crd) = swizzle.apply(layout.call(crd)).
## It stays out of the core Layout type, whose stride scalar keeps integer and coordinate-axis leaves only.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/b8 --outdir:build/b8
##     workspace/tattletile/tests/test_layout_swizzle.nim

import ./htuple
import ./layout

export layout

# ═══════════════════════════════════════════════════════════════════════
#  F2 arithmetic: the offsets of an F2-strided layout
# ═══════════════════════════════════════════════════════════════════════

proc f2Add*(a, b: int): int =
  ## F2 addition: XOR, carry-free
  a xor b

proc f2Mul*(a, b: int): int =
  ## Carry-less multiplication over GF(2): the bits of `a` times the bits of `b`, without cross-term carry propagation.
  ##
  ## Examples: `f2Mul(0b11, 0b11) == 0b101` (where integer 3 * 3 == 9),
  ## `f2Mul(3, 0b1010) == 0b11110` (an integer operand acts through its bits).
  result = 0
  var x = a
  var y = b
  var bit = 0
  while y != 0:
    if (y and 1) != 0:
      result = result xor (x shl bit)
    y = y shr 1
    inc bit

proc f2SpanBits*(lengths, strides: seq[int]): int =
  ## Bits the image of an F2-strided layout spans: the union of the bit positions every
  ## mode stride-by-coordinate product reaches.
  ##
  ## An F2 codomain bound is a power of two spanning the modes' bit-fields:
  ## XOR is not monotone, summing maxima under-counts.
  ##
  ## Examples: dims (8, 8), strides (1, 9) span 6 bits -> bound 64
  ## - dims (4, 4), strides (1, 1) span 2 bits -> bound 4
  ## - stride 0 reaches nothing -> bound 1
  var seen = 0
  for i in 0 ..< lengths.len:
    for x in 0 ..< lengths[i]:
      var w = f2Mul(strides[i], x)
      var pos = 0
      while w != 0:
        if (w and 1) != 0:
          seen = seen or (1 shl pos)
        w = w shr 1
        inc pos
  var bits = 0
  while (1 shl bits) <= seen and seen != 0:
    inc bits
  if seen == 0:
    return 1
  1 shl bits

# ═══════════════════════════════════════════════════════════════════════
#  The Swizzle functor
# ═══════════════════════════════════════════════════════════════════════

type Swizzle* = object
  ## Offset permutation `SW_<bits>_<base>_<shift>`: the `bits`-wide offset
  ## field at `base + shift` XORs into the field at `base`.
  ##
  ## Bit layout of an offset:
  ##
  ##   0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
  ##                                 ^--^  base: least-sig bits kept constant
  ##                    ^-^       ^-^      bits: the YYY mask width
  ##                       ^---------^     shift: YYY-to-ZZZ distance
  ##
  ## The result XORs AA = ZZ xor YY into the ZZZ field. An involution when the YYY and ZZZ
  ## fields do not overlap, abs(shift) >= bits.
  bits*, base*, shift*: int
  yyyMsk*, zzzMsk*: int

proc initSwizzle*(bits, base, shift: int): Swizzle =
  ## Swizzle from its three parameters, preconditions checked loudly.
  ##
  ## Raises:
  ## - LayoutError on negative bits or base, or a shift with abs(shift) < bits:
  ##   the fields would overlap, the map is no involution
  if bits < 0 or base < 0 or abs(shift) < bits:
    raise layoutError("Swizzle(" & $bits & ", " & $base & ", " & $shift & ")" &
      ": bits and base must be non-negative, a shift needs " &
      "abs(shift) >= bits")
  let mask = (1 shl bits) - 1
  result.bits = bits
  result.base = base
  result.shift = shift
  result.yyyMsk = mask shl (base + max(0, shift))
  result.zzzMsk = mask shl (base - min(0, shift))

proc apply*(sw: Swizzle, offset: int): int =
  ## Swizzle one integer offset
  let shifted = if sw.shift >= 0: (offset and sw.yyyMsk) shr sw.shift
                else: (offset and sw.yyyMsk) shl (-sw.shift)
  offset xor shifted

proc `$`*(sw: Swizzle): string =
  ## Print form matching the pycute functor: `SW_<bits>_<base>_<shift>`
  "SW" & $sw.bits & "_" & $sw.base & "_" & $sw.shift

# ═══════════════════════════════════════════════════════════════════════
#  Swizzled layouts: the functor composed over an algebra layout
# ═══════════════════════════════════════════════════════════════════════

type SwizzledLayout* = object
  ## A shared-memory tile layout with the swizzle composed over its codomain:
  ## `call(crd) = swizzle.apply(layout.call(crd))`.
  ##
  ## The inner layout is the unswizzled tile, shape and strides through the algebra, swizzled across banks.
  ## Sizes agree, the permutation is a bijection over the reached offsets.
  ##
  ## Integration point:
  ## - a shared-memory staging pass emits swizzled offset arithmetic for smem tile addressing
  ## - `SwizzledLayout.call` lowers like a Layout call
  ## - no current fixture stages through shared memory, this module ships as the layer plus property tests
  swizzle*: Swizzle
  layout*: Layout

proc swizzleLayout*(l: Layout, sw: Swizzle): SwizzledLayout =
  ## Compose a swizzle functor over a tile layout's codomain
  SwizzledLayout(swizzle: sw, layout: l)

proc call*(s: SwizzledLayout, crd: IntTuple): int =
  ## Swizzled offset of one coordinate in any form (integral, flat, natural)
  ##
  ## Raises:
  ## - LayoutError for a codomain over coordinate axes, the swizzle applies
  ##   to integer offsets only
  let v = s.layout.call(crd)
  if v.kind != skInt:
    raise layoutError("SwizzledLayout.call: non-integer offset " & ssStr(v))
  s.swizzle.apply(v.intVal)

proc callInt*(s: SwizzledLayout, i: int): int =
  ## Swizzled offset of one flat coordinate
  s.call(itv(i))

proc size*(s: SwizzledLayout): int =
  ## Number of coordinates in the domain, the inner layout's size
  s.layout.size

proc `$`*(s: SwizzledLayout): string =
  ## Constructor form: `SW3_3_3 o Layout((8, 64), (64, 1))`
  $s.swizzle & " o " & $s.layout
