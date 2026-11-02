# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/typetraits
import workspace/crucible/src/crucible_codegen
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/backends/runtime_utils

# ═══════════════════════════════════════════════════════════════════════
# ▸ Compile-time binding
# ═══════════════════════════════════════════════════════════════════════

func getArgWidth[T](_: typedesc[T], backend: static CodegenKind): int {.compileTime.} =
  ## Crucible-lowered byte width of scalar `T`
  when T is int|uint|enum:
    4
  elif T is bool:
    when backend == cgCuda:
      1
    else:
      4
  else:
    sizeof(T)

func isStaticArg[T](_: typedesc[T]): bool {.compileTime.} =
  ## True when the value lives in the type (phantom objects like Int[V]), so it occupies no runtime bytes
  when T is object|tuple:
    for f in fields(default(T)):
      if not isStaticArg(typeof(f)):
        return false
    result = true
  elif T is array:
    result = default(T).len == 0 or isStaticArg(typeof(default(T)[0]))
  else:
    result = false

func hasBufferLeaf*[T, H](_: typedesc[T], _: typedesc[H]): bool {.compileTime.} =
  ## True when `T` contains a pointer, handle or DeviceMem anywhere
  when T is H or T is DeviceMem or T is ptr or T is pointer:
    true
  elif T is object|tuple:
    for f in fields(default(T)):
      if hasBufferLeaf(typeof(f), H):
        return true
    result = false
  elif T is array:
    result = default(T).len > 0 and hasBufferLeaf(typeof(default(T)[0]), H)
  else:
    result = false

func hasBoolLeaf[T](_: typedesc[T]): bool {.compileTime.} =
  ## True when `T` contains a bool anywhere. GLSL block layout gives every
  ## boundary bool a 4-byte std430 word, while a Nim bool is 1 byte.
  ## Arrays that contain bools cannot pack as raw Nim bytes on GLSL.
  when T is bool:
    true
  elif T is object|tuple:
    for f in fields(default(T)):
      if hasBoolLeaf(typeof(f)):
        return true
    result = false
  elif T is array:
    result = default(T).len > 0 and hasBoolLeaf(typeof(default(T)[0]))
  else:
    result = false

func isBlobPackable[T, H](_: typedesc[T], _: typedesc[H]): bool {.compileTime.} =
  ## True when every non-leaf field of `T` is trivially copyable: the backend store
  ## flattens leaves individually, and the wrapper is never copyMem'd whole.
  when T is object|tuple:
    for f in fields(default(T)):
      when typeof(f) is H or typeof(f) is DeviceMem or typeof(f) is ptr or typeof(f) is pointer:
        discard
      elif not supportsCopyMem(typeof(f)):
        return false
    result = true
  else: result = false

func explodesStructArgs(backend: static CodegenKind): bool {.compileTime.} =
  backend in {cgMetal, cgWebGPU, cgOpenCL}

func computeDeviceLayout*[T](_: typedesc[T], backend: static CodegenKind): tuple[size, align: int] {.compileTime.} =
  ## Device size and alignment of `T` on `backend`: phantom and static types
  ## contribute zero bytes, Nim int/uint lower to 4 bytes
  when T is DeviceMem:
    result = (8, 8)  # 8 bytes, one device address slot
  elif T is object|tuple:
    when isStaticArg(T):
      result = (0, 1)
    else:
      result.align = 1
      for f in fields(default(T)):
        let fl = computeDeviceLayout(typeof(f), backend)
        result.size = round_multiple_up(result.size, fl.align) + fl.size
        result.align = max(result.align, fl.align)
      result.size = round_multiple_up(result.size, result.align)
  elif T is array:
    when backend == cgGlsl and hasBoolLeaf(T):
      # one 4-byte std430 word per bool, array elements stride at the device element layout, never Nim's 1-byte bool
      let el = computeDeviceLayout(typeof(default(T)[0]), backend)
      result = (default(T).len * el.size, el.align)
    else:
      let ea = if default(T).len > 0: getArgWidth(typeof(default(T)[0]), backend) else: 1
      result = (sizeof(T), ea)
  else:
    result = (getArgWidth(T, backend), getArgWidth(T, backend))

func countArgs[T, H](_: typedesc[T], _: typedesc[H], backend: static CodegenKind): int {.compileTime.} =
  ## Device args one launch value delivers
  when T is H or T is DeviceMem or T is ptr or T is pointer:
    1
  elif T is object|tuple:
    when isStaticArg(T):
      0
    elif hasBufferLeaf(T, H) and explodesStructArgs(backend):
      for f in fields(default(T)):
        result += countArgs(typeof(f), H, backend)
    else:
      1
  elif T is array:
    1
  else:
    1

# ═══════════════════════════════════════════════════════════════════════
# ▸ Runtime packing
# ═══════════════════════════════════════════════════════════════════════

proc putBytes[T](dst: var openArray[byte], off: int, v: T, w: int) {.inline.} =
  ## One scalar lowered to `w` bytes: Nim int/uint narrow to 32-bit and every
  ## other narrowing asserts
  when T is bool:
    if w == 1:
      dst[off] = byte(v)
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

proc packFrom*[T](dst: var openArray[byte], off: var int, v: T, backend: static CodegenKind) {.inline.} =
  ## Packs one value at its device layout, aligned, ints lowered to 32-bit
  when T is object|tuple:
    when not isStaticArg(T):
      let a = computeDeviceLayout(T, backend).align
      off = round_multiple_up(off, a)
      for f in fields(v):
        dst.packFrom(off, f, backend)
      off = round_multiple_up(off, a)
  elif T is array:
    when backend == cgGlsl and hasBoolLeaf(T):
      # element-wise packing, one 4-byte std430 word per bool, same as the scalar getArgWidth rule
      let a = computeDeviceLayout(T, backend).align
      off = round_multiple_up(off, a)
      for el in v:
        dst.packFrom(off, el, backend)
    else:
      let a = computeDeviceLayout(T, backend).align
      off = round_multiple_up(off, a)
      copyMem(dst[off].addr, v.addr, sizeof(v))
      off += sizeof(v)
  else:
    const a = getArgWidth(T, backend)
    off = round_multiple_up(off, a)
    putBytes(dst, off, v, a)
    off += a

macro serialize(ctx: var auto, v: auto, pos: var int, store: untyped, H: typedesc, backend: static CodegenKind) =
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
      elif hasBufferLeaf(typeof(`v`), `H`):
        when explodesStructArgs(`backend`):
          for f in fields(`v`):
            serialize(`ctx`, f, `pos`, `store`, `H`, `backend`)
        else:
          # leaves flatten to device addresses, non-leaf fields must stay copyable
          when not supportsCopyMem(typeof(`v`)) and
              not isBlobPackable(typeof(`v`), `H`):
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
    else:
      when not supportsCopyMem(typeof(`v`)):
        {.error: "kernel args are trivially copyable types only, " &
            $typeof(`v`) & " does not support copyMem".}
      `store`(`ctx`, `v`, `pos`)
      inc `pos`

# ═══════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

func launchArgCount*[O, A, H](_: typedesc[O], _: typedesc[A], _: typedesc[H], backend: static CodegenKind): int {.compileTime.} =
  result = countArgs(O, H, backend)
  for f in fields(default(A)):
    result += countArgs(typeof(f), H, backend)

template bindArgs*(ctx: var auto, output: auto, args: untyped, store: untyped, H: typedesc, backend: static CodegenKind) =
  var pos = 0
  serialize(ctx, output, pos, store, H, backend)
  for f in fields(args):
    serialize(ctx, f, pos, store, H, backend)
