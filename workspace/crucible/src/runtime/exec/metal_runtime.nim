# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/os
import workspace/crucible/src/abis/objc_abi as objc
import ../kernel_args

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

# ═════════════════════════════════════════════════
# macOS-only execution primitives
# ═════════════════════════════════════════════════

when defined(macosx):
  # ═════════════════════════════════════════════════
  # MSL version ladder (uname)
  # ═════════════════════════════════════════════════

  type
    UtSName {.importc: "struct utsname", header: "<sys/utsname.h>".} = object
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
    ## not the marketing version, the ladder maps Darwin majors:
    ## < 24 → 3.1 (macOS < 15), 24 → 3.2 (macOS 15), ≥ 25 → 4.0 (macOS ≥ 26).
    ## `NSProcessInfo.operatingSystemVersion` returns a 24-byte sret struct, UB for the ID-returning objc.msgSend family.
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
    # `init` may return a different object than `alloc`, keep the init result.
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
    ## Returns the +1 pipeline state, the function object (+1 from `newFunctionWithName:`)
    ## is released here, the pipeline state retains it internally.
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
  # Argument reflection
  # ═════════════════════════════════════════════════

  const
    ## `MTLArgumentType` values (MTLArgument.h)
    MTLArgumentTypeBuffer = 0
    MTLArgumentTypeThreadgroupMemory = 1
    ## `MTLDataType` values (MTLDataType.h) with a host scalar binding
    MTLDataTypeStruct = 1
    MTLDataTypeArray = 2
    MTLDataTypeFloat = 3
    MTLDataTypeHalf = 16
    MTLDataTypeInt = 29
    MTLDataTypeUInt = 33
    MTLDataTypeShort = 37
    MTLDataTypeUShort = 41
    MTLDataTypeChar = 45
    MTLDataTypeUChar = 49
    MTLDataTypeBool = 53
    MTLDataTypeLong = 81
    MTLDataTypeULong = 85
    MTLDataTypeBFloat = 121
    ## `MTLPipelineOption` bits (MTLDevice.h): binding info plus buffer
    ## type info, the latter fills the MTLStructType member layouts
    MTLPipelineOptionBindingAndBufferType = 3

  proc mtlScalarLayout(dt: objc.NSUInteger): tuple[kind: GpuTypeKind, width: int] =
    ## Returns the Crucible IR kind and byte width of a declared MSL
    ## scalar type from `MTLDataType`. `char` maps to the unsigned byte kind,
    ## the one-byte storage is what binds.
    case dt
    of MTLDataTypeFloat: (gtFloat32, 4)
    of MTLDataTypeHalf: (gtFloat16, 2)
    of MTLDataTypeBFloat: (gtBf16, 2)
    of MTLDataTypeInt: (gtInt32, 4)
    of MTLDataTypeUInt: (gtUint32, 4)
    of MTLDataTypeShort: (gtInt16, 2)
    of MTLDataTypeUShort: (gtUint16, 2)
    of MTLDataTypeChar, MTLDataTypeUChar: (gtUint8, 1)
    of MTLDataTypeBool: (gtBool, 1)
    of MTLDataTypeLong: (gtInt64, 8)
    of MTLDataTypeULong: (gtUint64, 8)
    else:
      failLoud("Metal reflection: declared MSL type " & $dt &
               " has no host scalar binding (vectors, matrices, textures and" &
               " pointers inside structs cannot bind from launch values)")

  proc flattenDeclaredStruct(st: objc.ID, base: int, members: var seq[KernelArgMember]) =
    ## Flattens an MTLStructType into scalar members with absolute offsets:
    ## nested structs recurse, array members expand one member per element
    ## at the declared stride, non-scalar element types fail loudly
    let ms = objc.msgSend(st, objc.`$$`("members"))
    let n = objc.msgSendUInt(ms, objc.`$$`("count"))
    for i in 0 ..< n:
      let m = objc.msgSend(ms, objc.`$$`("objectAtIndex:"), objc.NSUInteger(i))
      let off = base + int(objc.msgSendUInt(m, objc.`$$`("offset")))
      let dt = objc.msgSendUInt(m, objc.`$$`("dataType"))
      case dt
      of MTLDataTypeStruct:
        flattenDeclaredStruct(objc.msgSend(m, objc.`$$`("structType")), off, members)
      of MTLDataTypeArray:
        let at = objc.msgSend(m, objc.`$$`("arrayType"))
        let length = int(objc.msgSendUInt(at, objc.`$$`("arrayLength")))
        let stride = int(objc.msgSendUInt(at, objc.`$$`("stride")))
        let eldt = objc.msgSendUInt(at, objc.`$$`("elementType"))
        if eldt == MTLDataTypeStruct:
          let est = objc.msgSend(at, objc.`$$`("elementStructType"))
          for j in 0 ..< length:
            flattenDeclaredStruct(est, off + j * stride, members)
        else:
          let (kind, width) = mtlScalarLayout(eldt)
          for j in 0 ..< length:
            members.add(KernelArgMember(irKind: kind, offset: off + j * stride,
                                        width: width))
      else:
        let (kind, width) = mtlScalarLayout(dt)
        members.add(KernelArgMember(irKind: kind, offset: off, width: width))

  proc reflectedKernelLayout*(device, library: objc.ID, kernel: string): tuple[pso: objc.ID, layout: KernelArgLayout] =
    ## Builds the compute pipeline state for `kernel` and reflects
    ## the kernel's declared argument signature via
    ## MTLComputePipelineReflection (options: binding + buffer type info):
    ## - a scalar pointee records one kaBytes entry at its declared
    ##   width and a struct pointee one kaStruct entry whose members
    ##   carry the declared offsets and widths, a top-level array one
    ##   raw kaBytes chunk
    ## - threadgroup-memory args reserve their own index space, bind no
    ##   launch value and record no entry, textures and samplers fail
    ##   loudly like every other unsupported arg kind
    ## - each entry records the kernel's declared arg index, [[buffer(n)]]
    ##   or the position Metal assigns unattributed args, the value binds
    ##   at that index
    ##
    ## Returns the +1 pipeline state and the declared layout.
    ## The reflection object is autoreleased, drained with the caller's pool.
    ##
    ## Usage:
    ##   let (pso, declared) = reflectedKernelLayout(device, library, "saxpy")
    let fn = objc.msgSend(library, objc.`$$`("newFunctionWithName:"),
                          objc.nsStringFromNimString(kernel))
    if objc.isNil(fn):
      failLoud("newFunctionWithName(" & kernel & ") returned nil")
    defer:
      objc.release(fn)
    var refl: objc.ID = objc.ID(nil)
    var psoError: objc.ID = objc.ID(nil)
    result.pso = objc.msgSend(device,
      objc.`$$`("newComputePipelineStateWithFunction:options:reflection:error:"),
      fn, objc.NSUInteger(MTLPipelineOptionBindingAndBufferType),
      addr refl, addr psoError)
    if objc.isNil(result.pso):
      var detail = "no NSError object provided"
      if not objc.isNil(psoError):
        detail = objc.nsStringToNimString(objc.msgSend(psoError, objc.`$$`("localizedDescription")))
      failLoud("compute pipeline state creation failed for " & kernel & ": " & detail)
    let args = objc.msgSend(refl, objc.`$$`("arguments"))
    let n = objc.msgSendUInt(args, objc.`$$`("count"))
    for i in 0 ..< n:
      let a = objc.msgSend(args, objc.`$$`("objectAtIndex:"), objc.NSUInteger(i))
      let name = objc.nsStringToNimString(objc.msgSend(a, objc.`$$`("name")))
      let typ = objc.msgSendUInt(a, objc.`$$`("type"))
      if typ == MTLArgumentTypeThreadgroupMemory:
        continue
      if typ != MTLArgumentTypeBuffer:
        failLoud("Metal reflection: kernel '" & kernel & "' declares argument '" &
                 name & "' of kind " & $typ & ", only buffer and" &
                 " threadgroup-memory arguments bind from launch values")
      let idx = int(objc.msgSendUInt(a, objc.`$$`("index")))
      let bdt = objc.msgSendUInt(a, objc.`$$`("bufferDataType"))
      if bdt == MTLDataTypeStruct:
        var members: seq[KernelArgMember]
        flattenDeclaredStruct(objc.msgSend(a, objc.`$$`("bufferStructType")), 0, members)
        result.layout.args.add(KernelArg(kind: kaStruct,
          width: int(objc.msgSendUInt(a, objc.`$$`("bufferDataSize"))),
          members: members, index: idx))
      elif bdt == MTLDataTypeArray:
        # a top-level array arg: length and stride live on the pointer type
        let ptype = objc.msgSend(a, objc.`$$`("bufferPointerType"))
        let at = if objc.isNil(ptype): objc.ID(nil)
                 else: objc.msgSend(ptype, objc.`$$`("elementArrayType"))
        if objc.isNil(at):
          failLoud("Metal reflection: kernel '" & kernel & "' declares array" &
                   " argument '" & name & "' with no element array info")
        if objc.msgSendUInt(at, objc.`$$`("elementType")) == MTLDataTypeStruct:
          failLoud("Metal reflection: kernel '" & kernel &
                   "' declares array-of-struct argument '" & name &
                   "', element structs do not bind as one raw chunk")
        let count = int(objc.msgSendUInt(at, objc.`$$`("arrayLength")))
        result.layout.args.add(KernelArg(kind: kaBytes, irKind: gtArray,
          count: count,
          width: int(objc.msgSendUInt(a, objc.`$$`("bufferDataSize"))), index: idx))
      else:
        let (kind, width) = mtlScalarLayout(bdt)
        result.layout.args.add(KernelArg(kind: kaBytes, irKind: kind,
          width: int(objc.msgSendUInt(a, objc.`$$`("bufferDataSize"))), index: idx))

  # ═════════════════════════════════════════════════
  # Device, command queue, buffers
  # ═════════════════════════════════════════════════

  proc defaultMetalDevice*(): objc.ID =
    ## Returns the default Metal device, +1 (Create ownership rules), survives
    ## the caller's autorelease-pool drain, release via `objc.release`.
    ## A lookup, not an init: the engine owns its context construction.
    result = objc.MTLCreateSystemDefaultDevice()
    if objc.isNil(result):
      failLoud("default Metal device lookup returned nil (no Metal device)")

  proc getpagesize(): cint {.importc: "getpagesize", header: "<unistd.h>".}

  proc hostPageSize*(): int =
    ## Host page size (16 KiB on Apple Silicon).
    int(getpagesize())
