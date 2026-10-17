# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/typetraits
import ../../codegen/ir/gpu_types
import ../engines_common

# ═════════════════════════════════════════════════════════════════════════
# ▸ Compile-time binding
# ═════════════════════════════════════════════════════════════════════════

func argWidth[T](_: typedesc[T], backend: static BackendKind): int {.compileTime.} =
  ## Crucible-lowered byte width of scalar `T`
  when T is int|uint|enum: 4
  elif T is bool: (when backend == bkCuda: 1 else: 4)
  else: sizeof(T)

func isStaticArg[T](_: typedesc[T]): bool {.compileTime.} =
  ## True when the value lives in the type (phantom objects like Int[V]), so it occupies no runtime bytes
  when T is object|tuple:
    for f in fields(default(T)):
      if not isStaticArg(typeof(f)): return false
    result = true
  elif T is array:
    result = default(T).len == 0 or isStaticArg(typeof(default(T)[0]))
  else: result = false

func hasBufferLeaf[T, H](_: typedesc[T], _: typedesc[H]): bool {.compileTime.} =
  ## True when `T` holds a pointer or handle anywhere
  when T is H or T is ptr or T is pointer: true
  elif T is object|tuple:
    for f in fields(default(T)):
      if hasBufferLeaf(typeof(f), H): return true
    result = false
  elif T is array:
    result = default(T).len > 0 and hasBufferLeaf(typeof(default(T)[0]), H)
  else: result = false

func roundStepUp(x, step: int): int {.inline.} =
  ((x + step - 1) div step) * step

func deviceLayout[T](_: typedesc[T], backend: static BackendKind): tuple[size, align: int] {.compileTime.} =
  ## Padded and aligned type on device
  ## This skips phnatom types, static values
  ## and handle Nim int being lowered to 4-byte integer even on 64-bit.
  when T is object|tuple:
    when isStaticArg(T):
      result = (0, 1)
    else:
      result.align = 1
      for f in fields(default(T)):
        let fl = deviceLayout(typeof(f), backend)
        result.size = roundStepUp(result.size, fl.align) + fl.size
        result.align = max(result.align, fl.align)
      result.size = roundStepUp(result.size, result.align)
  elif T is array:
    let ea = if default(T).len > 0: argWidth(typeof(default(T)[0]), backend) else: 1
    result = (sizeof(T), ea)
  else:
    result = (argWidth(T, backend), argWidth(T, backend))

func argCount[T, H](_: typedesc[T], _: typedesc[H], backend: static BackendKind): int {.compileTime.} =
  ## Device args one launch value delivers
  when T is H or T is ptr or T is pointer: 1
  elif T is object|tuple:
    when isStaticArg(T): 0
    elif hasBufferLeaf(T, H):
      for f in fields(default(T)):
        result += argCount(typeof(f), H, backend)
    else: 1
  elif T is array: 1
  else: 1

# ═════════════════════════════════════════════════════════════════════════
# ▸ Runtime packing
# ═════════════════════════════════════════════════════════════════════════

proc putBytes[T](dst: var openArray[byte], off: int, v: T, w: int) {.inline.} =
  ## One scalar lowered to `w` bytes, wider values truncate to 32-bit
  when T is bool:
    if w == 1: dst[off] = byte(v)
    else:
      let x = int32(v)
      copyMem(dst[off].addr, x.unsafeAddr, 4)
  elif T is enum:
    let x = uint32(ord(v))
    copyMem(dst[off].addr, x.unsafeAddr, 4)
  else:
    if w < sizeof(T):
      # only Nim int/uint lower to 32-bit, explicit widths stay exact
      when T is int:
        let x = int32(v)
        copyMem(dst[off].addr, x.unsafeAddr, 4)
      elif T is uint:
        let x = uint32(v)
        copyMem(dst[off].addr, x.unsafeAddr, 4)
      else: doAssert false, "putBytes: unsupported narrowing"
    else:
      copyMem(dst[off].addr, v.unsafeAddr, w)

proc packInto[T](dst: var openArray[byte], off: var int, v: T, backend: static BackendKind) {.inline.} =
  ## Serializes to the device layout.
  ## Nim's int/uint are lowered to 32-bit.
  ## bool to 4 bytes (1 on CUDA).
  when T is object|tuple:
    when not isStaticArg(T):
      let a = deviceLayout(T, backend).align
      off = (off + a - 1) div a * a
      for f in fields(v):
        packInto(dst, off, f, backend)
      off = (off + a - 1) div a * a
  elif T is array:
    let a = deviceLayout(T, backend).align
    off = (off + a - 1) div a * a
    copyMem(dst[off].addr, v.addr, sizeof(v))
    off += sizeof(v)
  else:
    const a = argWidth(T, backend)
    off = (off + a - 1) div a * a
    putBytes(dst, off, v, a)
    off += a

proc serialize[H, C](ctx: var C, pos: var int, v: auto,
                     serializer: proc(ctx: var C, bufferHandle: H, p: pointer, len: int, argIdx: int, viaBuffer: bool) {.nimcall.},
                     backend: static BackendKind, ptrAsBuffer: static bool) {.inline.} =
  ## Serializes one value (or value subtree) through `serializer`
  when not supportsCopyMem(typeof(v)):
    {.error: "kernel args are trivially copyable types only, " &
        $typeof(v) & " does not support copyMem".}
  when typeof(v) is H:
    serializer(ctx, v, nil, 0, pos, true)
    inc pos
  elif typeof(v) is ptr or typeof(v) is pointer:
    when ptrAsBuffer:
      serializer(ctx, cast[H](v), nil, 0, pos, true)
    else:
      var copy = v
      serializer(ctx, default(H), copy.addr, sizeof(pointer), pos, false)
    inc pos
  elif typeof(v) is object|tuple:
    when isStaticArg(typeof(v)):
      discard
    elif hasBufferLeaf(typeof(v), H):
      var copy = v
      for f in fields(copy):
        serialize(ctx, pos, f, serializer, backend, ptrAsBuffer)
    else:
      const size = deviceLayout(typeof(v), backend).size
      var blob: array[size, byte]
      var off = 0
      packInto(blob, off, v, backend)
      serializer(ctx, default(H), blob.addr, off, pos, false)
      inc pos
  elif typeof(v) is array:
    var copy = v
    serializer(ctx, default(H), copy.addr, sizeof(copy), pos, false)
    inc pos
  else:
    const w = argWidth(typeof(v), backend)
    var staging: array[w, byte]
    const n = if sizeof(typeof(v)) < w: sizeof(typeof(v)) else: w
    copyMem(staging[0].addr, v.unsafeAddr, n)
    serializer(ctx, default(H), staging.addr, w, pos, false)
    inc pos

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

func launchArgCount*[O, A, H](_: typedesc[O], _: typedesc[A], _: typedesc[H], backend: static BackendKind): int {.compileTime.} =
  ## Device args one launch delivers, output first
  result = argCount(O, H, backend)
  for f in fields(default(A)):
    result += argCount(typeof(f), H, backend)

proc bindArgs*[H, C](ctx: var C, output: auto, args: tuple,
                     serializer: proc(ctx: var C, bufferHandle: H, p: pointer, len: int, argIdx: int, viaBuffer: bool) {.nimcall.},
                     backend: static BackendKind, ptrAsBuffer: static bool) =
  ## Sinks the output then the args tuple, positional
  var pos = 0
  serialize(ctx, pos, output, serializer, backend, ptrAsBuffer)
  var argsCopy = args
  for f in fields(argsCopy):
    serialize(ctx, pos, f, serializer, backend, ptrAsBuffer)
