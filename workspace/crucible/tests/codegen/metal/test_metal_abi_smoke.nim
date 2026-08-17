## Metal ABI smoke POC: dispatches a trivial MSL a+b kernel
## purely through `objc_abi.nim` (no engine, no DSL)
## and verifies the result on the device (2 + 3 == 5).
##
## Also proves:
## - Foundation auto-load (NSAutoreleasePool is visible to `objc_getClass`)
## - the invalid-MSL error path surfaces the NSError `localizedDescription`
##   as a Nim string
## - autorelease-pool discipline: run with `OBJC_DEBUG_MISSING_POOLS=YES`
##   prints no pool warnings
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_abi_smoke.nim

import std/os
import workspace/crucible/src/abis/objc_abi

# ── MSL kernel sources ──────────────────────────────────────────────────────

const addMsl = """
#include <metal_stdlib>
using namespace metal;

kernel void addKernel(device uint* output [[buffer(0)]],
                      device const uint* a [[buffer(1)]],
                      device const uint* b [[buffer(2)]])
{
  output[0] = a[0] + b[0];
}
"""

const invalidMsl = """
kernel void broken( {
"""

# ── Test-local ObjC/Metal glue ─────────────────────────────────────────────

template failLoud(msg: string) =
  ## Unified error policy: stacktrace + stderr + quit(1) with the caller's location.
  ## A template so instantiationInfo() reports the call site.
  writeStackTrace()
  stderr.write($instantiationInfo() & " exited with error: " & msg & '\n')
  quit 1

# ── Smoke run ───────────────────────────────────────────────────────────────

proc runTest() =
  # Autorelease pool wraps the whole run.
  # Metal objects are autoreleased, so a missing pool would trip OBJC_DEBUG_MISSING_POOLS=YES.
  let pool = msgSend(ID(objc_getClass("NSAutoreleasePool")), $$"alloc")
  discard msgSend(pool, $$"init")

  # Acceptance: Foundation must be loaded (NS* classes are nil until then).
  doAssert not objc_getClass("NSAutoreleasePool").isNil,
    "Foundation not loaded: objc_getClass(NSAutoreleasePool) is nil"
  doAssert not objc_getClass("NSString").isNil,
    "Foundation not loaded: objc_getClass(NSString) is nil"
  doAssert not objc_lookUpClass("NSString").isNil,
    "objc_lookUpClass(NSString) is nil — class lookup broken"
  doAssert not objc_lookUpClass("NSAutoreleasePool").isNil,
    "objc_lookUpClass(NSAutoreleasePool) is nil — class lookup broken"
  let probeName = "NoSuchClass_" & $os.getCurrentProcessId()
  doAssert objc_lookUpClass(probeName.cstring).isNil,
    "objc_lookUpClass returned non-nil for an unregistered class"

  # Acceptance: the `class_$1` auto-prefix import resolves to class_setVersion,
  # never class_class_setVersion.
  objc_getClass("NSObject").setVersion(3)

  # Device
  let device = MTLCreateSystemDefaultDevice()
  if device.isNil:
    failLoud("MTLCreateSystemDefaultDevice returned nil (no Metal device)")

  # Compile the a+b kernel
  let src = nsStringFromNimString(addMsl)
  var compileError: ID = ID(nil)
  let library = msgSend(device, $$"newLibraryWithSource:options:error:",
                        src, ID(nil), addr compileError)
  if library.isNil:
    if compileError.isNil:
      failLoud("Metal library compilation failed: no NSError object provided")
    let desc = msgSend(compileError, $$"localizedDescription")
    failLoud("Metal library compilation failed: " & nsStringToNimString(desc))

  let kernelFn = msgSend(library, $$"newFunctionWithName:",
                         nsStringFromNimString("addKernel"))
  if kernelFn.isNil:
    failLoud("newFunctionWithName(addKernel) returned nil")

  var psoError: ID = ID(nil)
  let pso = msgSend(device, $$"newComputePipelineStateWithFunction:error:",
                    kernelFn, addr psoError)
  if pso.isNil:
    if psoError.isNil:
      failLoud("compute pipeline state creation failed: no NSError object provided")
    let desc = msgSend(psoError, $$"localizedDescription")
    failLoud("compute pipeline state creation failed: " & nsStringToNimString(desc))

  # Buffers (shared storage, CPU-visible straight from contents())
  type Elem = uint32   # buffer element: size and CPU pointer derive from it, and it must match MSL `device uint*`
  let outBuf = msgSend(device, $$"newBufferWithLength:options:",
                       NSUInteger(sizeof(Elem)), NSUInteger(0))
  let aBuf = msgSend(device, $$"newBufferWithLength:options:",
                     NSUInteger(sizeof(Elem)), NSUInteger(0))
  let bBuf = msgSend(device, $$"newBufferWithLength:options:",
                     NSUInteger(sizeof(Elem)), NSUInteger(0))
  if outBuf.isNil or aBuf.isNil or bBuf.isNil:
    failLoud("newBufferWithLength:options: returned nil")

  let outPtr = cast[ptr Elem](msgSend(outBuf, $$"contents"))
  let aPtr = cast[ptr Elem](msgSend(aBuf, $$"contents"))
  let bPtr = cast[ptr Elem](msgSend(bBuf, $$"contents"))
  outPtr[] = 0'u32
  aPtr[] = 2'u32
  bPtr[] = 3'u32

  # Command queue / buffer / encoder
  let queue = msgSend(device, $$"newCommandQueue")
  if queue.isNil:
    failLoud("newCommandQueue returned nil")
  let cmdBuf = msgSend(queue, $$"commandBuffer")
  let encoder = msgSend(cmdBuf, $$"computeCommandEncoder")

  # Dispatch: one threadgroup of one thread
  discard msgSend(encoder, $$"setComputePipelineState:", pso)
  discard msgSend(encoder, $$"setBuffer:offset:atIndex:", outBuf, NSUInteger(0), NSUInteger(0))
  discard msgSend(encoder, $$"setBuffer:offset:atIndex:", aBuf, NSUInteger(0), NSUInteger(1))
  discard msgSend(encoder, $$"setBuffer:offset:atIndex:", bBuf, NSUInteger(0), NSUInteger(2))
  let one = MTLSize(width: NSUInteger(1), height: NSUInteger(1), depth: NSUInteger(1))
  discard msgSend(encoder, $$"dispatchThreadgroups:threadsPerThreadgroup:", one, one)
  discard msgSend(encoder, $$"endEncoding")
  discard msgSend(cmdBuf, $$"commit")
  discard msgSend(cmdBuf, $$"waitUntilCompleted")

  # Readback
  let got = cast[ptr Elem](msgSend(outBuf, $$"contents"))[]
  echo "  addKernel: 2 + 3 = ", got
  doAssert got == Elem(5), "expected 5, got " & $got

  # Error path: invalid MSL surfaces the NSError description as a Nim string
  let badSrc = nsStringFromNimString(invalidMsl)
  var badError: ID = ID(nil)
  let badLibrary = msgSend(device, $$"newLibraryWithSource:options:error:",
                           badSrc, ID(nil), addr badError)
  if not badLibrary.isNil:
    failLoud("invalid MSL unexpectedly compiled")
  if badError.isNil:
    failLoud("invalid MSL produced no NSError")
  let errDesc = nsStringToNimString(msgSend(badError, $$"localizedDescription"))
  echo "  invalid-MSL NSError description: ", errDesc
  doAssert errDesc.len > 0, "NSError localizedDescription came back empty"

  # NUL guard: `stringWithUTF8String:` truncates at the first NUL byte,
  # so the bridge must refuse NUL-containing strings instead of silently truncating.
  block:
    var guardFired = false
    try:
      discard nsStringFromNimString("a\0b")
    except AssertionDefect:
      guardFired = true
    doAssert guardFired,
      "nsStringFromNimString accepted a NUL-containing string (silent truncation)"

  discard msgSend(pool, $$"drain")
  echo "Metal ABI smoke test passed"

when isMainModule:
  runTest()
