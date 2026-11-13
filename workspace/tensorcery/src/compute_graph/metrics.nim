# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Inert unless compiled with -d:TTT_tensorgraphMetrics.

type
  GraphCounters* = object
    evals*: int
    nodesDispatched*: int
    nodesSkippedAllocated*: int
    kernelsEnqueued*: int

var gGraphCounters: GraphCounters

template metrics*(body: untyped): untyped =
  when defined(TTT_tensorgraphMetrics):
    block: {.noSideEffect, gcsafe.}: body

template incMetric*(name: untyped{ident}, amount = 1) =
  bind name
  metrics:
    gGraphCounters.name += amount

proc graphCounters*(): GraphCounters {.inline.} =
  when defined(TTT_tensorgraphMetrics):
    gGraphCounters
  else:
    raiseAssert("graphCounters: counters are not compiled in, use -d:TTT_tensorgraphMetrics")

proc printGraphMetrics*() =
  metrics:
    echo "graph metrics:"
    echo "  evals:                 ", gGraphCounters.evals
    echo "  nodesDispatched:       ", gGraphCounters.nodesDispatched
    echo "  nodesSkippedAllocated: ", gGraphCounters.nodesSkippedAllocated
    echo "  kernelsEnqueued:       ", gGraphCounters.kernelsEnqueued
