# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Add: z = x + y, two inputs one output
##
## - k_add: one threadgroup per tile, the register tile map tiles all inputs
## - elements pair by coordinate, 32 lanes x split per tile
## - add: z.node = an AddNode over a.node and b.node
## - the handler enqueues k_add over the node's bridges into the batch

import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/crucible/src/builtins/builtins_gpu_types
import workspace/iface
import workspace/math
import ../../tensors_internals
import ../dispatch

# ############################################################
#
#                   Add
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc k_add*(z, x, y: distinct TensorView) =
  let thrID = getLocalThreadID()
  var reg: tile(z.T)
  let tiled = make_layout(z.size()).logical_divide(reg.getLayout())
  let grid = tiled.getTileGrid()
  let
    cSize = grid.size()
    gridStride = getGridSize()
  for tileID in countup(grid.getLocalTileID(), cSize - 1, gridStride):
    let rx = tile(x.T).load(x, thrID, tileID)
    let ry = tile(y.T).load(y, thrID, tileID)
    var rz: tile(z.T)
    for i in 0 ..< rx.size():
      when typeof(rx[i]) is bfloat16:
        rz[i] = (rx[i].to(float32) + ry[i].to(float32)).to(bfloat16)
      else:
        rz[i] = rx[i] + ry[i]
    z.store(rz, thrID, tileID)

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  AddNode* = ref object
    r*: AnyTensorImpl
    a*, b*: TensorNode

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*(n: AddNode): AnyTensorImpl =
  n.r

proc opName*(n: AddNode): string =
  "add"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_add*(engine: AnyEngine, a, b: TensorNode): TensorNode =
  let ia = a.getImpl()
  let ib = b.getImpl()
  if ia.shape() != ib.shape():
    raise ValueError.newException("tn_add: shape mismatch")
  if ia.dtype() != ib.dtype():
    raise ValueError.newException("tn_add: dtype mismatch")
  AddNode(a: a, b: b, r: engine.newTensorNode(ia.dtype(), ia.shape()).getImpl()).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueAdd*[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: TensorNode): bool =
  let asAdd = n.to(AddNode)
  if asAdd.isNil:
    return false
  let rz = asAdd.r.to(TensorImpl[Mem, Layout[Sh, Sh]])
  engine.evalNode(batch, asAdd.a)
  engine.evalNode(batch, asAdd.b)
  let
    xa = asAdd.a.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    xb = asAdd.b.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    numel = rz.layout.shape.getNumElems()
  let
    zv = initTensorBridge(rz.data[], rz.offset, rz.layout)
    xv = initTensorBridge(xa.data[], xa.offset, xa.layout)
    yv = initTensorBridge(xb.data[], xb.offset, xb.layout)
  batch.enqueue<<(ceil_div(numel, 128), 32)>> k_add(zv, xv, yv)
  true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("add", enqueueAdd)
