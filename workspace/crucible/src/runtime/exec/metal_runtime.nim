# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal (MSL) runtime execution primitives for the `bkMetal` engine.
##
## Every public engine call wraps its body in an `NSAutoreleasePool`
## (`objc.withMemPool`). Metal methods return autoreleased objects
## (command buffers, encoders) that a pool drain releases.
## The objects the engine caches (device, queue, library, pipeline states)
## come from `new*`/Create methods and are +1 owned.
## They survive the drain without a `retain` and must be `release`d by the owner.
##
## Tested ABI (macOS, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## The Objective-C bridge lives in `objc_abi`, imported here as `objc`.
## This module keeps only its own concerns: compile options, buffers,
## and the `failLoud` error policy.
##
## The execution primitives are `when defined(macosx)`-guarded. On other
## platforms, only the platform-neutral constants and `failLoud` remain,
## along with the `MetalCtx`/`MetalBuffer` records. The engine entry point
## (`newMetalEngine` in `runtime/engines/metal`) quits loudly there.

import workspace/crucible/src/abis/objc_abi as objc

# ═════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════

const
  ## Shared storage (0) + untracked hazard tracking (1 << 8): the storage mode of Apple's MLX allocator.
  ## CPU and GPU share a single memory region, so `contents()` needs no staging or map.
  MTLResourceStorageModeShared = 0
  MTLResourceHazardTrackingModeUntracked = 1 shl 8
  SharedUntrackedOptions* = objc.NSUInteger(MTLResourceStorageModeShared or
                                           MTLResourceHazardTrackingModeUntracked)

  ## `MTLLanguageVersion` enum values (MTLLibrary.h): 3_1 = macOS 14,
  ## 3_2 = macOS 15, 4_0 = macOS 26.
  MTLLanguageVersion3_1 = objc.NSUInteger(0x30001)
  MTLLanguageVersion3_2 = objc.NSUInteger(0x30002)
  MTLLanguageVersion4_0 = objc.NSUInteger(0x40000)

# ═════════════════════════════════════════════════
# Error policy and records
# ═════════════════════════════════════════════════

template failLoud*(msg: string) =
  ## Unified error policy: stacktrace + stderr + quit(1) with the caller's location.
  ## A template so instantiationInfo() reports the call site.
  writeStackTrace()
  stderr.write($instantiationInfo() & " exited with error: " & msg & '\n')
  quit 1

type
  MetalCtx* = object
    ## Live Metal device and command queue, +1 objects owned by the caller.
    device*: objc.ID
    queue*: objc.ID

  MetalBuffer* = object
    ## Shared-storage buffer: CPU and GPU share one memory region.
    ## `contents()` is therefore directly CPU-visible after `waitUntilCompleted`.
    buffer*: objc.ID
    data*: pointer   # `contents()`, valid for the buffer's lifetime

# ═════════════════════════════════════════════════
# macOS-only execution primitives
# ═════════════════════════════════════════════════

when defined(macosx):
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

  proc mslLanguageVersion*(): objc.NSUInteger =
    ## MSL language version for this OS, from the `uname()` release ladder.
    ## `uname()` reports the Darwin kernel release (25.6.0 on macOS 26.6.1),
    ## not the marketing version, so the ladder maps Darwin majors:
    ## < 24 → 3.1 (macOS < 15), 24 → 3.2 (macOS 15), ≥ 25 → 4.0 (macOS ≥ 26).
    ## This reproduces MLX's marketing-version ladder.
    ## `NSProcessInfo.operatingSystemVersion` is banned because it returns a 24-byte struct
    ## through the x8 sret register, which is UB for the ID-returning objc.msgSend family.
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

  proc compileOptions*(): objc.ID =
    ## MTLCompileOptions: fast math disabled (correctness-first) and the OS's MSL language version.
    ## Returns a +1 object. The caller releases it after `newLibraryWithSource`.
    let cls = objc.getClass("MTLCompileOptions")
    if objc.isNil(cls):
      failLoud("getClass(MTLCompileOptions) is nil, Metal.framework not loaded")
    result = objc.msgSend(objc.ID(cls), objc.`$$`("alloc"))
    # `init` may return a different object than `alloc` — keep the init result.
    result = objc.msgSend(result, objc.`$$`("init"))
    discard objc.msgSend(result, objc.`$$`("setFastMathEnabled:"), objc.BOOL(0))
    discard objc.msgSend(result, objc.`$$`("setLanguageVersion:"), mslLanguageVersion())

  proc compileLibrary*(device: objc.ID, source: string, opts: objc.ID): objc.ID =
    ## Compiles `source` via `newLibraryWithSource:options:error:`.
    ## Returns the +1 library, or quits loudly with the NSError `localizedDescription` when the compiler provides one.
    var compileError: objc.ID = objc.ID(nil)
    result = objc.msgSend(device, objc.`$$`("newLibraryWithSource:options:error:"),
                          objc.nsStringFromNimString(source), opts, addr compileError)
    if objc.isNil(result):
      var detail = "no NSError object provided"
      if not objc.isNil(compileError):
        detail = objc.nsStringToNimString(objc.msgSend(compileError, objc.`$$`("localizedDescription")))
      failLoud("Metal library compilation failed: " & detail)

  proc compilePipelineState*(device, library: objc.ID, kernel: string): objc.ID =
    ## Creates the compute pipeline state for `kernel` via `newComputePipelineStateWithFunction:error:`.
    ## The function object is +1 from `newFunctionWithName:` and is released here.
    ## The pipeline state retains it internally.
    let fn = objc.msgSend(library, objc.`$$`("newFunctionWithName:"), objc.nsStringFromNimString(kernel))
    if objc.isNil(fn):
      failLoud("newFunctionWithName(" & kernel & ") returned nil")
    defer:
      objc.release(fn)
    var psoError: objc.ID = objc.ID(nil)
    result = objc.msgSend(device, objc.`$$`("newComputePipelineStateWithFunction:error:"),
                          fn, addr psoError)
    if objc.isNil(result):
      var detail = "no NSError object provided"
      if not objc.isNil(psoError):
        detail = objc.nsStringToNimString(objc.msgSend(psoError, objc.`$$`("localizedDescription")))
      failLoud("compute pipeline state creation failed for " & kernel & ": " & detail)

  # ═════════════════════════════════════════════════
  # Device, command queue, buffers
  # ═════════════════════════════════════════════════

  proc initMetal*(): MetalCtx =
    ## The default Metal device and its command queue.
    ## Both are +1 (Create/new ownership rules) and survive the caller's autorelease-pool drain.
    ## Release them via `objc.release`.
    result.device = objc.MTLCreateSystemDefaultDevice()
    if objc.isNil(result.device):
      failLoud("default Metal device lookup returned nil (no Metal device)")
    result.queue = objc.msgSend(result.device, objc.`$$`("newCommandQueue"))
    if objc.isNil(result.queue):
      failLoud("newCommandQueue returned nil")

  proc allocBuffer*(device: objc.ID, size: int): MetalBuffer =
    ## Allocates a shared|untracked buffer of `size` bytes (options 256).
    ## Returns a +1 object. The caller releases it via `releaseBuffer`.
    if size <= 0:
      failLoud("allocBuffer: size must be positive, got " & $size)
    result.buffer = objc.msgSend(device, objc.`$$`("newBufferWithLength:options:"),
                                 objc.NSUInteger(size), SharedUntrackedOptions)
    if objc.isNil(result.buffer):
      failLoud("newBufferWithLength:options: returned nil (length " & $size & ")")
    result.data = cast[pointer](objc.msgSend(result.buffer, objc.`$$`("contents")))
    if result.data == nil:
      failLoud("contents() returned nil for a shared buffer")

  proc releaseBuffer*(buffer: var MetalBuffer) {.inline.} =
    ## Releases the +1 buffer object. Nil-safe.
    ## Inlined: called per dispatch for the output and every input buffer.
    if not objc.isNil(buffer.buffer):
      objc.release(buffer.buffer)
      buffer.buffer = objc.ID(nil)
      buffer.data = nil
