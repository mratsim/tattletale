## Vulkan: kernel-arg copyability contract for blob-style packing.
##
## Test that DeviceMem-holding wrappers pack through the whole-argument
## blob path while genuinely non-copyable args still fail loudly
## at compile time, never silently packed.
##
## Run:
##   nim c --hints:off --warnings:off --nimcache:nimcache/tests \
##     tests/codegen/vulkan/serialization/t_vulkan_arg_copyability.nim

import workspace/crucible/src/runtime/launch/launch_serialization
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/exec/vulkan_runtime

type
  Sink = object
    bytes: int

  DmBuf = object
    ## Owning wrapper over a DeviceMem handle, the shape the RAII
    ## migration produces: non-copyable as a whole, leaf-packable.
    p: DeviceMem[VulkanBuffer, float32]
    n: int

  BadNoLeaf = object
    ## No buffer leaf at all, destructor on board: genuinely non-copyable.
    xs: seq[uint32]

  BadWithLeaf = object
    ## Buffer leaf present but a non-leaf field owns memory: the blob path
    ## would copyMem only the non-leaf fields, so it must refuse loudly.
    p: DeviceMem[VulkanBuffer, float32]
    xs: seq[uint32]

proc take[T](c: var Sink, v: T, i: int) {.inline.} =
  ## Fake backend store: counts bytes, never runs (compile-only test).
  when T is DeviceMem:
    c.bytes += 8
  elif T is object|tuple:
    for f in fields(v):
      when typeof(f) is DeviceMem:
        c.bytes += 8
      else:
        c.bytes += 4
  else:
    c.bytes += 4

static:
  # The DeviceMem wrapper packs: the blob path flattens the leaf
  # to the device address, no whole-object copyMem needed.
  doAssert compiles(block:
    var sink: Sink
    let v = DmBuf(p: DeviceMem[VulkanBuffer, float32](), n: 3)
    bindArgs(sink, (), (v,), take, VulkanBuffer, bkVulkan)
  ), "DeviceMem-holding wrapper must pack through the blob path"

  # Negative controls: genuinely non-copyable args stay loud.
  doAssert not compiles(block:
    var sink: Sink
    let v = BadNoLeaf(xs: @[1'u32])
    bindArgs(sink, (), (v,), take, VulkanBuffer, bkVulkan)
  ), "non-copyable arg without a buffer leaf must be rejected loudly"

  doAssert not compiles(block:
    var sink: Sink
    let v = BadWithLeaf(p: DeviceMem[VulkanBuffer, float32](), xs: @[1'u32])
    bindArgs(sink, (), (v,), take, VulkanBuffer, bkVulkan)
  ), "non-copyable non-leaf field next to a buffer leaf must be rejected loudly"

echo "[OK] Vulkan blob packing: DeviceMem wrappers pack, non-copyable args rejected loudly"
