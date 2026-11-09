# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Op registry: one handler bundle per op, registration at import
##
## - an op file registers its handlers in a static block, one per engine
##   it supports
## - the registry key is the opName string each node reports
## - the dispatcher (eval.nim) imports the op files, freezes the registry
##   into a const after them, and dispatches by lookup
## - a node whose op has no handler for the running engine is an error

import std/tables
import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import ../tensors_internals

type
  WalkFn*[E: HwEngine] = proc(engine: E, batch: Batch[E], n: TensorNode) {.nimcall.}
    ## Walk re-entry an op handler recurses into for its inputs

  MetalHandler* = proc(engine: MetalEngine, batch: Batch[MetalEngine],
      n: TensorNode, walk: WalkFn[MetalEngine]): bool {.nimcall.}
  VulkanHandler* = proc(engine: VulkanEngine, batch: Batch[VulkanEngine],
      n: TensorNode, walk: WalkFn[VulkanEngine]): bool {.nimcall.}
  OpenCLHandler* = proc(engine: OpenCLEngine, batch: Batch[OpenCLEngine],
      n: TensorNode, walk: WalkFn[OpenCLEngine]): bool {.nimcall.}
  WebgpuHandler* = proc(engine: WebgpuEngine, batch: Batch[WebgpuEngine],
      n: TensorNode, walk: WalkFn[WebgpuEngine]): bool {.nimcall.}

  OpHandlers* = object
    ## One op's handlers, a field per engine the op supports
    metal*: MetalHandler
    vulkan*: VulkanHandler
    opencl*: OpenCLHandler
    webgpu*: WebgpuHandler

var OpRegistry* {.compileTime.}: Table[string, OpHandlers]
  ## Populated by op files' static blocks, frozen by the dispatcher

proc registerEnqueuer*(op: string, handlers: OpHandlers) =
  ## Registration entry of an op file, called once from a static block
  OpRegistry[op] = handlers

proc handler*(h: OpHandlers, E: typedesc[MetalEngine]): MetalHandler = h.metal
proc handler*(h: OpHandlers, E: typedesc[VulkanEngine]): VulkanHandler = h.vulkan
proc handler*(h: OpHandlers, E: typedesc[OpenCLEngine]): OpenCLHandler = h.opencl
proc handler*(h: OpHandlers, E: typedesc[WebgpuEngine]): WebgpuHandler = h.webgpu
