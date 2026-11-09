## Unsanctioned helpers, pending rationalization.
##
## Helpers landing here bypass the sanctioned op surface.
## The funnel pass later promotes, keeps, or kills each one.
##
## - sanctioned modules may import this file while a helper awaits its home
## - what survives rationalization is re-exported through the family module
import std/macros, std/typetraits
import ./int_tuples_compiletime
import ./int_tuples_datatypes
import workspace/ceramic/src/integer_semi_modules
import ./int_tuples_streams

template toInt*(x: int): int = x
template toInt*[V: static int](x: Int[V]): int = V

func cosizeLeaf*(leaf, stride: int or Int): int =
  ## One cosize term of the cosize sum: (leaf - 1) * |stride|.
  ##
  ## Args:
  ## - leaf, stride: one shape leaf and its stride, `int` or `Int[N]`
  ## Returns: the term as a scalar int, `Int[N]` leaves fold to N.
  ##
  ## Emitted by cosizeImpl as a bound symbol: the macro expansion resolves
  ## the call in this module's scope, valid at instantiation sites that do
  ## not see the Int operator matrix.
  (leaf.toInt() - 1) * abs(stride.toInt())

func toSeqStaticInts*(t: NimNode): seq[int] {.compileTime.} =
  ## Recursively extract Int[N] values from a (possibly nested) tuple type AST node.
  ## Returns low(int) (DynamicSentinel) for non-static (dynamic int) elements.
  ##
  ## Handles:
  ## - ((Int[1], Int[16]), (Int[512], Int[64]))  → @[1, 16, 512, 64]
  ## - ((int, int), (int, int))                  → @[DynamicSentinel, DynamicSentinel, ...]
  ## - Int[64]                                   → @[64]
  for (_, ty) in t.tupleStream().leaves():
    result.add ty.getStaticInt()
