# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/typetraits
import workspace/crucible/src/codegen/backends/targets
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/backends/runtime_utils

# ═══════════════════════════════════════════════════════════════════════
# ▸ Compile-time binding
# ═══════════════════════════════════════════════════════════════════════

func argWidth[T](_: typedesc[T], backend: static BackendKind): int {.compileTime.} =
  ## Crucible-lowered byte width of scalar `T`
  when T is int|uint|enum:
    4
  elif T is bool:
    (when backend == bkCuda: 1 else: 4)
  else: sizeof(T)

func isStaticArg[T](_: typedesc[T]): bool {.compileTime.} =
  ## True when the value lives in the type (phantom objects like Int[V]), so it occupies no runtime bytes
  when T is object|tuple:
    for f in fields(default(T)):
      if not isStaticArg(typeof(f)):
        return false
    result = true
  elif T is array:
    result = default(T).len == 0 or isStaticArg(typeof(default(T)[0]))
  else: result = false

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
  else: result = false

func blobPackable[T, H](_: typedesc[T], _: typedesc[H]): bool {.compileTime.} =
  ## True when every non-leaf field of `T` is trivially copyable: the backend store
  ## flattens leaves individually, the wrapper is never copyMem'd whole.
  when T is object|tuple:
    for f in fields(default(T)):
      when typeof(f) is H or typeof(f) is DeviceMem or typeof(f) is ptr or typeof(f) is pointer:
        discard
      elif not supportsCopyMem(typeof(f)):
        return false
    result = true
  else: result = false

func explodesStructArgs(backend: static BackendKind): bool {.compileTime.} =
  backend in {bkMetal, bkWebGPU, bkOpenCL}

func deviceLayout*[T](_: typedesc[T], backend: static BackendKind): tuple[size, align: int] {.compileTime.} =
  ## Device size and alignment, phantom/static types excluded,
  ##  Nim int/uint lowered to 4 bytes
  when T is DeviceMem:
    # device address slot
    result = (8, 8)
  elif T is object|tuple:
    when isStaticArg(T):
      result = (0, 1)
    else:
      result.align = 1
      for f in fields(default(T)):
        let fl = deviceLayout(typeof(f), backend)
        result.size = round_multiple_up(result.size, fl.align) + fl.size
        result.align = max(result.align, fl.align)
      result.size = round_multiple_up(result.size, result.align)
  elif T is array:
    let ea = if default(T).len > 0: argWidth(typeof(default(T)[0]), backend) else: 1
    result = (sizeof(T), ea)
  else:
    result = (argWidth(T, backend), argWidth(T, backend))

func argCount[T, H](_: typedesc[T], _: typedesc[H], backend: static BackendKind): int {.compileTime.} =
  ## Device args one launch value delivers
  when T is H or T is DeviceMem or T is ptr or T is pointer:
    1
  elif T is object|tuple:
    when isStaticArg(T):
      0
    elif hasBufferLeaf(T, H) and explodesStructArgs(backend):
      for f in fields(default(T)):
        result += argCount(typeof(f), H, backend)
    else: 1
  elif T is array:
    1
  else: 1

# ═══════════════════════════════════════════════════════════════════════
# ▸ Runtime packing
# ═══════════════════════════════════════════════════════════════════════

proc putBytes[T](dst: var openArray[byte], off: int, v: T, w: int) {.inline.} =
  ## One scalar lowered to `w` bytes, wider values truncate to 32-bit
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

proc packFrom*[T](dst: var openArray[byte], off: var int, v: T, backend: static BackendKind) {.inline.} =
  ## Packs one value at its device layout, aligned, ints lowered to 32-bit
  when T is object|tuple:
    when not isStaticArg(T):
      let a = deviceLayout(T, backend).align
      off = round_multiple_up(off, a)
      for f in fields(v):
        dst.packFrom(off, f, backend)
      off = round_multiple_up(off, a)
  elif T is array:
    let a = deviceLayout(T, backend).align
    off = round_multiple_up(off, a)
    copyMem(dst[off].addr, v.addr, sizeof(v))
    off += sizeof(v)
  else:
    const a = argWidth(T, backend)
    off = round_multiple_up(off, a)
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
      elif hasBufferLeaf(typeof(`v`), `H`):
        when explodesStructArgs(`backend`):
          for f in fields(`v`):
            serialize(`ctx`, f, `pos`, `store`, `H`, `backend`)
        else:
          # leaves flatten to device addresses, non-leaf fields must stay copyable
          when not supportsCopyMem(typeof(`v`)) and
              not blobPackable(typeof(`v`), `H`):
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
# ▸ HLSL constant-buffer layout (the descriptor route's uniform block)
# ═══════════════════════════════════════════════════════════════════════
#
# HLSL printer packs the kernel's scalar and value-struct parameters into
# one `struct Push` block behind the buffers. The engine stages the value
# leaves into a uniform buffer whose byte layout must match the member
# offsets the HLSL compiler computes for that block. The rules below were
# against the compiler's SPIR-V output (OpMemberDecorate offsets):
# - scalars align to their byte width: bool/int/uint/enum 4, float64 8
# - vectors align 4 and never straddle a 16-byte register boundary
# - structs align 16 at any nesting depth, size rounded up to 16
# - array elements each start on a register boundary

func hlslRound16(v: int): int =
  (v + 15) div 16 * 16

func hlslPushLayout*[T](_: typedesc[T]): tuple[size, align: int] {.compileTime.} =
  ## HLSL constant-buffer layout of one value type, per the verified rules.
  ## Static-arg (phantom) types occupy nothing.
  when T is object|tuple:
    when isStaticArg(T):
      result = (0, 1)
    else:
      var off = 0
      for f in fields(default(T)):
        when isStaticArg(typeof(f)):
          discard
        else:
          let fl = hlslPushLayout(typeof(f))
          when typeof(f) is object|tuple:
            off = hlslRound16(off)
            off += fl.size
          elif typeof(f) is array:
            off = hlslRound16(off)
            off += hlslRound16(max(fl.size, 4)) * default(f).len
          else:
            off = (off + fl.align - 1) div fl.align * fl.align
            if off mod 16 + fl.size > 16:
              off = hlslRound16(off)
            off += fl.size
      result.align = 16
      result.size = hlslRound16(off)
  elif T is array:
    let el = hlslPushLayout(typeof(default(T)[0]))
    result = (hlslRound16(max(el.size, 4)) * default(T).len, 16)
  elif T is bool|enum|int|uint:
    result = (4, 4)
  else:
    result = (sizeof(T), sizeof(T))

func hlslPushLeafSize[T, H](_: typedesc[T], _: typedesc[H]): int {.compileTime.} =
  ## Blob bytes one argument tree occupies.
  ## - a buffer leaf: an 8-byte placeholder slot, it binds as a storage
  ##   descriptor instead
  ## - a value leaf: its full constant-buffer layout
  when T is H or T is DeviceMem or T is ptr or T is pointer:
    8
  elif T is object|tuple:
    when isStaticArg(T):
      0
    else:
      hlslPushLayout(T).size
  else:
    hlslPushLayout(T).size

func hlslUniformBlobCap*[O, A, H](_: typedesc[O], _: typedesc[A], _: typedesc[H]): int {.compileTime.} =
  ## Uniform-blob byte capacity for one launch: the value leaves of the output
  ## and the argument tuple, plus 16 bytes of headroom for the materialized
  ## dispatch counts the engine appends when the kernel reads them.
  ##
  ## Returns: the capacity in bytes, rounded up to a 16-byte boundary.
  result = hlslPushLeafSize(O, H)
  for f in fields(default(A)):
    result += hlslPushLeafSize(typeof(f), H)
  result = hlslRound16(result + 16)

proc packHlslPush*[T](dst: var openArray[byte], off: var int, v: T) {.inline.} =
  ## Packs one value at its HLSL constant-buffer layout, the offsets match
  ## the member offsets the HLSL compiler computes for the printed Push block.
  ##
  ## Args:
  ## - `dst`: the blob, sized by `hlslUniformBlobCap`
  ## - `off`: the write offset, advanced by the packed byte count
  when T is object|tuple:
    when not isStaticArg(T):
      off = hlslRound16(off)
      for f in fields(v):
        dst.packHlslPush(off, f)
      off = hlslRound16(off)
  elif T is array:
    for el in v:
      off = hlslRound16(off)
      dst.packHlslPush(off, el)
    off = hlslRound16(off)
  else:
    const lay = hlslPushLayout(T)
    off = (off + lay.align - 1) div lay.align * lay.align
    if off mod 16 + lay.size > 16:
      off = hlslRound16(off)
    putBytes(dst, off, v, lay.size)
    off += lay.size

# ═══════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

func launchArgCount*[O, A, H](_: typedesc[O], _: typedesc[A], _: typedesc[H], backend: static BackendKind): int {.compileTime.} =
  result = argCount(O, H, backend)
  for f in fields(default(A)):
    result += argCount(typeof(f), H, backend)

template bindArgs*(ctx: var auto, output: auto, args: untyped, store: untyped, H: typedesc, backend: static BackendKind) =
  var pos = 0
  serialize(ctx, output, pos, store, H, backend)
  for f in fields(args):
    serialize(ctx, f, pos, store, H, backend)
