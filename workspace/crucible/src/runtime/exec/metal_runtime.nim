# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal (MSL) runtime execution primitives for the `bkMetal` engine.
##
## Every public engine call wraps its body in an `NSAutoreleasePool`
## (`newPool`/`drainPool`). Metal methods return autoreleased objects
## (command buffers, encoders) that a pool drain releases.
## The objects the engine caches (device, queue, library, pipeline states)
## come from `new*`/Create methods and are +1 owned.
## They survive the drain without a `retain` and must be `release`d by the owner.
##
## Tested ABI (macOS, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.

import std/strutils
import workspace/crucible/src/abis/objc_abi

# ═════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════

const
  ## Shared storage (0) + untracked hazard tracking (1 << 8): the storage mode of Apple's MLX allocator.
  ## CPU and GPU share a single memory region, so `contents()` needs no staging or map.
  MTLResourceStorageModeShared = 0
  MTLResourceHazardTrackingModeUntracked = 1 shl 8
  SharedUntrackedOptions* = NSUInteger(MTLResourceStorageModeShared or
                                       MTLResourceHazardTrackingModeUntracked)

  ## `MTLLanguageVersion` enum values (MTLLibrary.h): 3_1 = macOS 14,
  ## 3_2 = macOS 15, 4_0 = macOS 26.
  MTLLanguageVersion3_1 = NSUInteger(0x30001)
  MTLLanguageVersion3_2 = NSUInteger(0x30002)
  MTLLanguageVersion4_0 = NSUInteger(0x40000)

# ═════════════════════════════════════════════════
# msgSend casts beyond the ABI surface
# ═════════════════════════════════════════════════

type
  MsgSendBool = proc (self: ID; op: SEL; a: BOOL): ID {.cdecl.}
  MsgSendUInt1 = proc (self: ID; op: SEL; a: NSUInteger): ID {.cdecl.}
  MsgSendUInt0 = proc (self: ID; op: SEL): NSUInteger {.cdecl.}

# `objc_abi` keeps the varargs symbol private,
# so this module re-declares the same libobjc symbol to build the fixed-signature casts below.
proc objc_msgSend(self: ID; op: SEL): ID {.importc, cdecl, varargs, dynlib: "libobjc.A.dylib".}

## Selectors with one BOOL argument: setFastMathEnabled:.
template msgSend*(self: ID; op: SEL; a: BOOL): ID =
  cast[MsgSendBool](objc_msgSend)(self, op, a)

## Selectors with one NSUInteger argument: setLanguageVersion:.
template msgSend*(self: ID; op: SEL; a: NSUInteger): ID =
  cast[MsgSendUInt1](objc_msgSend)(self, op, a)

## Selectors returning an NSUInteger with no extra arguments: status.
template msgSendUInt*(self: ID; op: SEL): NSUInteger =
  cast[MsgSendUInt0](objc_msgSend)(self, op)

# ═════════════════════════════════════════════════
# Error policy and autorelease pools
# ═════════════════════════════════════════════════

template failLoud*(msg: string) =
  ## Unified error policy: stacktrace + stderr + quit(1) with the caller's location.
  ## A template so instantiationInfo() reports the call site.
  writeStackTrace()
  stderr.write($instantiationInfo() & " exited with error: " & msg & '\n')
  quit 1

proc newPool*(): ID =
  ## Alloc/init an `NSAutoreleasePool`. Public engine calls wrap their body in one.
  ## Without a pool, autoreleased Metal objects trip OBJC_DEBUG_MISSING_POOLS=YES.
  result = msgSend(ID(objc_getClass("NSAutoreleasePool")), $$"alloc")
  discard msgSend(result, $$"init")

proc drainPool*(pool: ID) =
  ## Drains the pool: releases everything autoreleased since `newPool`.
  discard msgSend(pool, $$"drain")

proc releaseObjC*(obj: ID) =
  ## Releases a +1 Objective-C object, balanced against a `new*`/Create acquisition. Nil-safe.
  if not obj.isNil:
    discard msgSend(obj, $$"release")

# ═════════════════════════════════════════════════
# MSL version ladder (uname)
# ═════════════════════════════════════════════════

type
  UtSName = object
    ## libc `struct utsname`: every field is 256 chars on macOS.
    sysname: array[256, char]
    nodename: array[256, char]
    release: array[256, char]
    version: array[256, char]
    machine: array[256, char]

proc uname(uts: ptr UtSName): cint {.importc, header: "<sys/utsname.h>".}

proc mslLanguageVersion*(): NSUInteger =
  ## MSL language version for this OS, from the `uname()` release ladder.
  ## `uname()` reports the Darwin kernel release (25.6.0 on macOS 26.6.1),
  ## not the marketing version, so the ladder maps Darwin majors:
  ## < 24 → 3.1 (macOS < 15), 24 → 3.2 (macOS 15), ≥ 25 → 4.0 (macOS ≥ 26).
  ## This reproduces MLX's marketing-version ladder.
  ## `NSProcessInfo.operatingSystemVersion` is banned because it returns a 24-byte struct
  ## through the x8 sret register, which is UB through an ID-returning msgSend.
  var uts: UtSName
  if uname(addr uts) != 0:
    failLoud("uname failed, cannot determine the MSL language version")
  var major = 0
  for c in uts.release:
    if c in {'0' .. '9'}:
      major = major * 10 + (ord(c) - ord('0'))
    else:
      break
  if major == 0:
    failLoud("uname release '" & $cast[cstring](addr uts.release[0]) &
             "' has no leading digit, cannot determine the MSL language version")
  if major < 24:
    result = MTLLanguageVersion3_1
  elif major < 25:
    result = MTLLanguageVersion3_2
  else:
    result = MTLLanguageVersion4_0

# ═════════════════════════════════════════════════
# Compilation
# ═════════════════════════════════════════════════

proc compileOptions*(): ID =
  ## MTLCompileOptions: fast math disabled (correctness-first) and the OS's MSL language version.
  ## Returns a +1 object. The caller releases it after `newLibraryWithSource`.
  let cls = objc_getClass("MTLCompileOptions")
  if cls.isNil:
    failLoud("objc_getClass(MTLCompileOptions) is nil, Metal.framework not loaded")
  result = msgSend(ID(cls), $$"alloc")
  discard msgSend(result, $$"init")
  discard msgSend(result, $$"setFastMathEnabled:", BOOL(0))
  discard msgSend(result, $$"setLanguageVersion:", mslLanguageVersion())

proc compileLibrary*(device: ID, source: string, opts: ID): ID =
  ## Compiles `source` via `newLibraryWithSource:options:error:`.
  ## Returns the +1 library, or quits loudly with the NSError `localizedDescription` when the compiler provides one.
  var compileError: ID = ID(nil)
  result = msgSend(device, $$"newLibraryWithSource:options:error:",
                   nsStringFromNimString(source), opts, addr compileError)
  if result.isNil:
    var detail = "no NSError object provided"
    if not compileError.isNil:
      detail = nsStringToNimString(msgSend(compileError, $$"localizedDescription"))
    failLoud("Metal library compilation failed: " & detail)

proc compilePipelineState*(device, library: ID, kernel: string): ID =
  ## Creates the compute pipeline state for `kernel` via `newComputePipelineStateWithFunction:error:`.
  ## The function object is +1 from `newFunctionWithName:` and is released here.
  ## The pipeline state retains it internally.
  let fn = msgSend(library, $$"newFunctionWithName:", nsStringFromNimString(kernel))
  if fn.isNil:
    failLoud("newFunctionWithName(" & kernel & ") returned nil")
  defer:
    releaseObjC(fn)
  var psoError: ID = ID(nil)
  result = msgSend(device, $$"newComputePipelineStateWithFunction:error:",
                   fn, addr psoError)
  if result.isNil:
    var detail = "no NSError object provided"
    if not psoError.isNil:
      detail = nsStringToNimString(msgSend(psoError, $$"localizedDescription"))
    failLoud("compute pipeline state creation failed for " & kernel & ": " & detail)

# ═════════════════════════════════════════════════
# Device, command queue, buffers
# ═════════════════════════════════════════════════

type
  MetalCtx* = object
    ## Live Metal device and command queue, +1 objects owned by the caller.
    device*: ID
    queue*: ID

  MetalBuffer* = object
    ## Shared-storage buffer: CPU and GPU share one memory region.
    ## `contents()` is therefore directly CPU-visible after `waitUntilCompleted`.
    buffer*: ID
    data*: pointer   # `contents()`, valid for the buffer's lifetime

proc initMetal*(): MetalCtx =
  ## The default Metal device (`MTLCreateSystemDefaultDevice`) and its command queue.
  ## Both are +1 (Create/new ownership rules) and survive the caller's autorelease-pool drain.
  ## Release them via `releaseObjC`.
  result.device = MTLCreateSystemDefaultDevice()
  if result.device.isNil:
    failLoud("MTLCreateSystemDefaultDevice returned nil (no Metal device)")
  result.queue = msgSend(result.device, $$"newCommandQueue")
  if result.queue.isNil:
    failLoud("newCommandQueue returned nil")

proc allocBuffer*(device: ID, size: int): MetalBuffer =
  ## Allocates a shared|untracked buffer of `size` bytes (options 256).
  ## Returns a +1 object. The caller releases it via `releaseBuffer`.
  if size <= 0:
    failLoud("allocBuffer: size must be positive, got " & $size)
  result.buffer = msgSend(device, $$"newBufferWithLength:options:",
                          NSUInteger(size), SharedUntrackedOptions)
  if result.buffer.isNil:
    failLoud("newBufferWithLength:options: returned nil (length " & $size & ")")
  result.data = cast[pointer](msgSend(result.buffer, $$"contents"))
  if result.data.isNil:
    failLoud("contents() returned nil for a shared buffer")

proc releaseBuffer*(buffer: var MetalBuffer) =
  ## Releases the +1 buffer object. Nil-safe.
  if not buffer.buffer.isNil:
    releaseObjC(buffer.buffer)
    buffer.buffer = ID(nil)
    buffer.data = nil
