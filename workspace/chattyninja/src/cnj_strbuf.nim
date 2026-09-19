# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Byte accumulator over a caller-owned presized buffer. Render code writes derived values
# (stringified containers, JSON, concatenations) into scratch instead of building
# intermediate strings, so a derived value costs no per-part allocation.
#
# - an append that does not fit raises, reporting capacity and shortfall, never growing
#   or reallocating the buffer
# - a nil backing pointer selects measure mode, where appends only advance `len`, letting
#   a caller presize a result string in one pass before rendering into it
#
# Scratch sizing:
#   the JSON rendering of the largest context value is the working upper
# bound for the largest single derived value. Underestimation is safe, a too-small append
# raising and the caller growing scratch and repulling. The repr of a string holding many
# single quotes exceeds its JSON rendering, which keeps each quote as one byte.

import std/unicode
import cnj_errors

type
  StrBuf* = object
    ## Cursor over a caller-owned byte buffer. A nil `buf` selects measure mode.
    buf*: ptr UncheckedArray[char]
    cap*: int
      ## writable bytes behind `buf`, ignored in measure mode
    len*: int
      ## bytes appended so far, the measured length in measure mode

func over*(s: var string): StrBuf =
  ## Returns a buffer over `s`'s bytes, capacity the string's length, an empty string
  ## yielding measure mode, which a zero-length render target needs.
  if s.len == 0:
    StrBuf()
  else:
    StrBuf(buf: cast[ptr UncheckedArray[char]](addr s[0]), cap: s.len)

func spanString*(s: openArray[char]): string =
  ## Returns a fresh string holding the bytes of `s`, one allocation bounded by the span.
  result = newString(s.len)
  if s.len > 0:
    copyMem(addr result[0], unsafeAddr s[0], s.len)

proc scratchShort(sb: StrBuf, need: int) {.noreturn.} =
  ## Raises the typed overflow an unfitting append reports, naming capacity and shortfall.
  var e = ScratchError(capacity: sb.cap, shortfall: max(0, need - (sb.cap - sb.len)))
  e.msg = "render scratch capacity " & $sb.cap & " exceeded, " & $e.shortfall &
      " more bytes needed"
  raise e

proc add*(sb: var StrBuf, c: char) =
  ## Appends one byte, raising when the buffer cannot hold it.
  if sb.buf == nil:
    inc sb.len
    return
  if sb.len >= sb.cap:
    scratchShort(sb, 1)
  sb.buf[sb.len] = c
  inc sb.len

proc add*(sb: var StrBuf, s: openArray[char]) =
  ## Appends a byte span, reading `s` in place, raising when it does not fit.
  if s.len == 0:
    return
  if sb.buf == nil:
    sb.len += s.len
    return
  if sb.len + s.len > sb.cap:
    scratchShort(sb, s.len)
  copyMem(addr sb.buf[sb.len], unsafeAddr s[0], s.len)
  sb.len += s.len

proc addInt*(sb: var StrBuf, i: int64) =
  ## Appends the decimal form of `i`, matching `$i`.
  var digits: array[20, char]
  var n = 0
  let neg = i < 0
  # Two's-complement negation in unsigned space, so int64.low negates without overflow.
  var u = if neg: 0'u64 - cast[uint64](i) else: cast[uint64](i)
  while true:
    digits[n] = char(ord('0') + int(u mod 10'u64))
    inc n
    u = u div 10'u64
    if u == 0:
      break
  if neg:
    sb.add '-'
  for j in countdown(n - 1, 0):
    sb.add digits[j]

proc addFloat*(sb: var StrBuf, f: float64) =
  ## Appends Python's `str()` for a float, integral values keeping one decimal place,
  ## the shortest float repr coming from `$f`, one allocation per append.
  let s = $f
  sb.add s
  if '.' notin s and 'e' notin s and 'E' notin s and 'n' notin s and 'i' notin s:
    sb.add ".0"

proc addRune*(sb: var StrBuf, r: Rune) =
  ## Appends `r` as its UTF-8 bytes, matching `toUTF8` for every reachable codepoint.
  let c = ord(r)
  if c > 0x10FFFF:
    # Invalid UTF-8 decodes to out-of-range codepoints, whose `$` round-trip is not
    # standard UTF-8, so defer to `toUTF8` and render exactly as before.
    sb.add toUTF8(r)
  elif c < 0x80:
    sb.add char(c)
  elif c < 0x800:
    sb.add char(0xC0 or (c shr 6))
    sb.add char(0x80 or (c and 0x3F))
  elif c < 0x10000:
    sb.add char(0xE0 or (c shr 12))
    sb.add char(0x80 or ((c shr 6) and 0x3F))
    sb.add char(0x80 or (c and 0x3F))
  else:
    sb.add char(0xF0 or (c shr 18))
    sb.add char(0x80 or ((c shr 12) and 0x3F))
    sb.add char(0x80 or ((c shr 6) and 0x3F))
    sb.add char(0x80 or (c and 0x3F))
