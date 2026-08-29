# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Config JSON readers: strictly typed, loudly failing key access shared
## across checkpoint configuration parsers.

import
  std/options,
  pkg/packedjson

proc parseIntList*(json: JsonNode, key: string): seq[int] =
  ## Parse a JSON array of ints, or a single int as a one-element list.
  ## A node that is neither an array nor an int yields an empty seq. Array
  ## elements must be `JInt`: any other kind raises `ValueError` naming `key`
  ## and the index, so token id 0 cannot enter a stop set.
  result = newSeq[int]()
  case json.kind
  of JArray:
    for i in 0 ..< json.len:
      let elem = json[i]
      if elem.kind != JInt:
        raise newException(ValueError,
          "[ttt] " & key & "[" & $i & "]: expected an int, found " & $elem.kind)
      result.add elem.getInt().int
  of JInt:
    result.add json.getInt().int
  else:
    discard

proc reqInt*(json: JsonNode, key: string, note = ""): int =
  ## Read a key whose value must be an int. Raises `ValueError` naming `key`
  ## when the node is absent, null, a string, a float or a list. The optional
  ## `note` points at the file that holds the value instead.
  if json.kind != JInt:
    raise newException(ValueError,
      "[ttt] " & key & ": expected an int, found " & $json.kind &
      (if note.len > 0: ". " & note else: ""))
  json.getInt().int

proc reqPosInt*(json: JsonNode, key: string): int =
  ## Read a positive int: absent, null, wrong-typed and non-positive values
  ## raise `ValueError` naming `key`.
  result = json.reqInt(key)
  if result <= 0:
    raise newException(ValueError,
      "[ttt] " & key & ": expected a positive value, found " & $result)

proc reqPosFloat*(json: JsonNode, key: string): float64 =
  ## Read a positive number, accepting the `JInt` spelling a JSON writer emits
  ## for a whole value. Raises `ValueError` naming `key` for any other kind
  ## and for a non-positive value.
  case json.kind
  of JInt, JFloat:
    result = json.getFloat()
  else:
    raise newException(ValueError,
      "[ttt] " & key & ": expected a number, found " & $json.kind)
  if result <= 0:
    raise newException(ValueError,
      "[ttt] " & key & ": expected a positive value, found " & $result)

proc optInt*(json: JsonNode, key: string): Option[int] =
  ## Read an optional int key: `JNull` (absent or null) maps to `none`, `JInt`
  ## maps to `some`. Any other kind raises `ValueError` naming `key`, which keeps
  ## a wrong-typed value distinct from absence.
  case json.kind
  of JNull:
    none(int)
  of JInt:
    some(json.getInt().int)
  else:
    raise newException(ValueError,
      "[ttt] " & key & ": expected an int or null, found " & $json.kind)

