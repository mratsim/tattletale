## Phase 0: Base58 short hash collision test
##
## Tests the base58 encoding mechanism for generating short,
## collision-resistant identifiers from 64-bit signature hashes.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_base58.nim

import std/[random, sequtils, sets, strutils]

const Base58* = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
  ## Base58 alphabet (no 0, O, I, l for readability and ambiguity avoidance).

func shortHash*(sigHash: int64): string =
  ## Encode a 64-bit signature hash as a 7-character base58 string.
  ## 58^7 = 2,204,715,403,072 (~2.2T namespace), sufficient for collision
  ## avoidance across all symbols in a GPU compilation unit.
  ##
  ## Algorithm: treat the int64 as unsigned, repeatedly divide by 58,
  ## prepending base58 chars for each remainder (least significant first,
  ## then reversed at the end).
  var n = uint64(sigHash)
  if n == 0:
    return "1111111"
  var chars: array[7, char]
  for i in countdown(6, 0):
    let rem = int(n mod 58)
    chars[i] = Base58[rem]
    n = n div 58
    if n == 0 and i == 0:
      break
    if n == 0:
      # Pad remaining higher positions with '1' (Base58 '0' equivalent)
      for j in 0 ..< i:
        chars[j] = '1'
      break
  result = cast[string](@chars)

when isMainModule:
  const expected = "1111111"
  let zeroHash = shortHash(0)
  doAssert zeroHash == expected,
    "shortHash(0) expected '" & expected & "', got '" & zeroHash & "'"
  echo "  OK — shortHash(0) = ", zeroHash

  # ── Test: Determinism ──
  block:
    let a = shortHash(42)
    let b = shortHash(42)
    doAssert a == b, "shortHash is not deterministic"
  echo "  OK — deterministic"

  # ── Test: 7-char output length ──
  block:
    randomize(42)
    for i in 0 ..< 100:
      let h = rand(int64.high)
      let encoded = shortHash(h)
      doAssert encoded.len == 7,
        "shortHash length expected 7, got " & $encoded.len & " for hash " & $h
  echo "  OK — 7-char output length"

  # ── Test: Valid base58 characters only ──
  block:
    randomize(42)
    for i in 0 ..< 100:
      let h = rand(int64.high)
      let encoded = shortHash(h)
      for c in encoded:
        doAssert c in Base58,
          "Invalid character '" & c & "' in shortHash for hash " & $h
  echo "  OK — valid base58 characters only"

  # ── Test: 10K random hashes produce unique short hashes ──
  block:
    randomize(42)
    var seen = initHashSet[string]()
    const N = 10_000
    for i in 0 ..< N:
      let h = rand(int64.high)
      let encoded = shortHash(h)
      doAssert encoded notin seen,
        "Collision detected at iteration " & $i & ": hash " & $h &
        " maps to '" & encoded & "'"
      seen.incl encoded
    echo "  OK — 0 collisions in ", N, " random hashes"

  # ── Test: Sequential small numbers produce different short hashes ──
  block:
    var seen = initHashSet[string]()
    for i in 0'i64 .. 999:
      let encoded = shortHash(i)
      doAssert encoded notin seen,
        "Collision for small sequential value " & $i & ": '" & encoded & "'"
      seen.incl encoded
    echo "  OK — 0 collisions in 1000 sequential small values"

  echo ""
  echo "  All base58 tests passed."
