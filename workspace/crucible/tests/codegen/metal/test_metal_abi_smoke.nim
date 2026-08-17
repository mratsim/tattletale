## Metal ABI smoke POC: dispatches a trivial MSL a+b kernel
## purely through `objc_abi.nim` (no engine, no DSL)
## and verifies the result on the device (2 + 3 == 5).
##
## Also proves:
## - Foundation auto-load (NSAutoreleasePool is visible to `getClass`)
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
import workspace/crucible/src/abis/objc_abi as objc

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
  let pool = objc.msgSend(objc.ID(objc.getClass("NSAutoreleasePool")), objc.`$$`("alloc"))
  discard objc.msgSend(pool, objc.`$$`("init"))

  # Acceptance: Foundation must be loaded (NS* classes are nil until then).
  doAssert not objc.isNil(objc.getClass("NSAutoreleasePool")),
    "Foundation not loaded: getClass(NSAutoreleasePool) is nil"
  doAssert not objc.isNil(objc.getClass("NSString")),
    "Foundation not loaded: getClass(NSString) is nil"
  doAssert not objc.isNil(objc.lookUpClass("NSString")),
    "lookUpClass(NSString) is nil — class lookup broken"
  doAssert not objc.isNil(objc.lookUpClass("NSAutoreleasePool")),
    "lookUpClass(NSAutoreleasePool) is nil — class lookup broken"
  let probeName = "NoSuchClass_" & $os.getCurrentProcessId()
  doAssert objc.isNil(objc.lookUpClass(probeName.cstring)),
    "lookUpClass returned non-nil for an unregistered class"

  # Acceptance: the `class_$1` auto-prefix import resolves to class_setVersion,
  # never class_class_setVersion.
  objc.setVersion(objc.getClass("NSObject"), 3)

  # Device
  let device = objc.MTLCreateSystemDefaultDevice()
  if objc.isNil(device):
    failLoud("default Metal device lookup returned nil (no Metal device)")

  # Compile the a+b kernel
  let src = objc.nsStringFromNimString(addMsl)
  var compileError: objc.ID = objc.ID(nil)
  let library = objc.msgSend(device, objc.`$$`("newLibraryWithSource:options:error:"),
                             src, objc.ID(nil), addr compileError)
  if objc.isNil(library):
    if objc.isNil(compileError):
      failLoud("Metal library compilation failed: no NSError object provided")
    let desc = objc.msgSend(compileError, objc.`$$`("localizedDescription"))
    failLoud("Metal library compilation failed: " & objc.nsStringToNimString(desc))

  let kernelFn = objc.msgSend(library, objc.`$$`("newFunctionWithName:"),
                              objc.nsStringFromNimString("addKernel"))
  if objc.isNil(kernelFn):
    failLoud("newFunctionWithName(addKernel) returned nil")

  var psoError: objc.ID = objc.ID(nil)
  let pso = objc.msgSend(device, objc.`$$`("newComputePipelineStateWithFunction:error:"),
                         kernelFn, addr psoError)
  if objc.isNil(pso):
    if objc.isNil(psoError):
      failLoud("compute pipeline state creation failed: no NSError object provided")
    let desc = objc.msgSend(psoError, objc.`$$`("localizedDescription"))
    failLoud("compute pipeline state creation failed: " & objc.nsStringToNimString(desc))

  # Buffers (shared storage, CPU-visible straight from contents())
  type Elem = uint32   # buffer element: size and CPU pointer derive from it, and it must match MSL `device uint*`
  let outBuf = objc.msgSend(device, objc.`$$`("newBufferWithLength:options:"),
                            objc.NSUInteger(sizeof(Elem)), objc.NSUInteger(0))
  let aBuf = objc.msgSend(device, objc.`$$`("newBufferWithLength:options:"),
                          objc.NSUInteger(sizeof(Elem)), objc.NSUInteger(0))
  let bBuf = objc.msgSend(device, objc.`$$`("newBufferWithLength:options:"),
                          objc.NSUInteger(sizeof(Elem)), objc.NSUInteger(0))
  if objc.isNil(outBuf) or objc.isNil(aBuf) or objc.isNil(bBuf):
    failLoud("newBufferWithLength:options: returned nil")

  let outPtr = cast[ptr Elem](objc.msgSend(outBuf, objc.`$$`("contents")))
  let aPtr = cast[ptr Elem](objc.msgSend(aBuf, objc.`$$`("contents")))
  let bPtr = cast[ptr Elem](objc.msgSend(bBuf, objc.`$$`("contents")))
  outPtr[] = 0'u32
  aPtr[] = 2'u32
  bPtr[] = 3'u32

  # Command queue / buffer / encoder
  let queue = objc.msgSend(device, objc.`$$`("newCommandQueue"))
  if objc.isNil(queue):
    failLoud("newCommandQueue returned nil")
  let cmdBuf = objc.msgSend(queue, objc.`$$`("commandBuffer"))
  let encoder = objc.msgSend(cmdBuf, objc.`$$`("computeCommandEncoder"))

  # Dispatch: one threadgroup of one thread
  discard objc.msgSend(encoder, objc.`$$`("setComputePipelineState:"), pso)
  discard objc.msgSend(encoder, objc.`$$`("setBuffer:offset:atIndex:"), outBuf, objc.NSUInteger(0), objc.NSUInteger(0))
  discard objc.msgSend(encoder, objc.`$$`("setBuffer:offset:atIndex:"), aBuf, objc.NSUInteger(0), objc.NSUInteger(1))
  discard objc.msgSend(encoder, objc.`$$`("setBuffer:offset:atIndex:"), bBuf, objc.NSUInteger(0), objc.NSUInteger(2))
  let one = objc.MTLSize(width: objc.NSUInteger(1), height: objc.NSUInteger(1), depth: objc.NSUInteger(1))
  discard objc.msgSend(encoder, objc.`$$`("dispatchThreadgroups:threadsPerThreadgroup:"), one, one)
  discard objc.msgSend(encoder, objc.`$$`("endEncoding"))
  discard objc.msgSend(cmdBuf, objc.`$$`("commit"))
  discard objc.msgSend(cmdBuf, objc.`$$`("waitUntilCompleted"))

  # Readback
  let got = cast[ptr Elem](objc.msgSend(outBuf, objc.`$$`("contents")))[]
  echo "  addKernel: 2 + 3 = ", got
  doAssert got == Elem(5), "expected 5, got " & $got

  # Error path: invalid MSL surfaces the NSError description as a Nim string
  let badSrc = objc.nsStringFromNimString(invalidMsl)
  var badError: objc.ID = objc.ID(nil)
  let badLibrary = objc.msgSend(device, objc.`$$`("newLibraryWithSource:options:error:"),
                                badSrc, objc.ID(nil), addr badError)
  if not objc.isNil(badLibrary):
    failLoud("invalid MSL unexpectedly compiled")
  if objc.isNil(badError):
    failLoud("invalid MSL produced no NSError")
  let errDesc = objc.nsStringToNimString(objc.msgSend(badError, objc.`$$`("localizedDescription")))
  echo "  invalid-MSL NSError description: ", errDesc
  doAssert errDesc.len > 0, "NSError localizedDescription came back empty"

  # NUL guard: `stringWithUTF8String:` truncates at the first NUL byte,
  # so the bridge must refuse NUL-containing strings instead of silently truncating.
  block:
    var guardFired = false
    try:
      discard objc.nsStringFromNimString("a\0b")
    except AssertionDefect:
      guardFired = true
    doAssert guardFired,
      "nsStringFromNimString accepted a NUL-containing string (silent truncation)"

  discard objc.msgSend(pool, objc.`$$`("drain"))
  echo "Metal ABI smoke test passed"

when isMainModule:
  runTest()
