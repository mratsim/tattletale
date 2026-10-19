# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/typetraits
import ../../codegen/ir/gpu_types
import ../engines/engines_common

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

func hasBufferLeaf*[T, H](_: typedesc[T], _: typedesc[H]): bool {.compileTime.} =
  ## True when `T` holds a pointer, handle or DeviceMem anywhere
  when T is H or T is DeviceMem or T is ptr or T is pointer: true
  elif T is object|tuple:
    for f in fields(default(T)):
      if hasBufferLeaf(typeof(f), H): return true
    result = false
  elif T is array:
    result = default(T).len > 0 and hasBufferLeaf(typeof(default(T)[0]), H)
  else: result = false

func explodesStructArgs(backend: static BackendKind): bool {.compileTime.} =
  backend in {bkMetal, bkWGSL, bkOpenCL}

func roundStepUp*(x, step: int): int {.inline.} =
  ((x + step - 1) div step) * step

func deviceLayout*[T](_: typedesc[T], backend: static BackendKind): tuple[size, align: int] {.compileTime.} =
  ## Device size and alignment, phantom/static types excluded,
  ##  Nim int/uint lowered to 4 bytes
  when T is DeviceMem: result = (8, 8)  # the device address slot
  elif T is object|tuple:
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
  when T is H or T is DeviceMem or T is ptr or T is pointer: 1
  elif T is object|tuple:
    when isStaticArg(T): 0
    elif hasBufferLeaf(T, H) and explodesStructArgs(backend):
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

proc packFrom*[T](dst: var openArray[byte], off: var int, v: T, backend: static BackendKind) {.inline.} =
  ## Packs one value at its device layout, aligned, ints lowered to 32-bit
  when T is object|tuple:
    when not isStaticArg(T):
      let a = deviceLayout(T, backend).align
      off = roundStepUp(off, a)
      for f in fields(v):
        dst.packFrom(off, f, backend)
      off = roundStepUp(off, a)
  elif T is array:
    let a = deviceLayout(T, backend).align
    off = roundStepUp(off, a)
    copyMem(dst[off].addr, v.addr, sizeof(v))
    off += sizeof(v)
  else:
    const a = argWidth(T, backend)
    off = roundStepUp(off, a)
    putBytes(dst, off, v, a)
    off += a

macro serialize(ctx: var auto, v: auto, pos: var int, store: untyped, H: typedesc, backend: static BackendKind) =
  ## Only whole-argument packing requires trivially-copyable args.
  ## Struct-explode packing writes the buffer handle plus trivial fields,
  ## so owning wrappers with destructors serialize fine there.
  result = quote do:
    when typeof(`v`) is `H` or typeof(`v`) is DeviceMem or typeof(`v`) is ptr|pointer:
      when typeof(`v`) is DeviceMem:
        `store`(`ctx`, `v`.mem, `pos`)
      else:
        `store`(`ctx`, `v`, `pos`)
      inc `pos`
    elif typeof(`v`) is object|tuple:
      when isStaticArg(typeof(`v`)):
        discard
      elif hasBufferLeaf(typeof(`v`), `H`) and explodesStructArgs(`backend`):
        for f in fields(`v`):
          serialize(`ctx`, f, `pos`, `store`, `H`, `backend`)
      else:
        when not supportsCopyMem(typeof(`v`)):
          {.error: "kernel args are trivially copyable types only, " &
              $typeof(`v`) & " does not support copyMem".}
        `store`(`ctx`, `v`, `pos`)
        inc `pos`
    else:
      when not supportsCopyMem(typeof(`v`)):
        {.error: "kernel args are trivially copyable types only, " &
            $typeof(`v`) & " does not support copyMem".}
      `store`(`ctx`, `v`, `pos`)
      inc `pos`

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

func launchArgCount*[O, A, H](_: typedesc[O], _: typedesc[A], _: typedesc[H], backend: static BackendKind): int {.compileTime.} =
  result = argCount(O, H, backend)
  for f in fields(default(A)):
    result += argCount(typeof(f), H, backend)

template bindArgs*(ctx: var auto, output: auto, args: var tuple, store: untyped, H: typedesc, backend: static BackendKind) =
  var pos = 0
  serialize(ctx, output, pos, store, H, backend)
  for f in fields(args):
    serialize(ctx, f, pos, store, H, backend)
