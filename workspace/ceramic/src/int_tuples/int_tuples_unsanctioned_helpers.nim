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

func cosizeLeaf*[L, S](leaf: L, stride: S): auto =
  ## One cosize term of the cosize sum: (leaf - 1) * |stride|.
  ##
  ## Args:
  ## - leaf, stride: one shape leaf and its stride, `int` or `Int[N]`,
  ##   never a tuple
  ## Returns: the term, folding at compile time for `Int[N]` leaves
  ## through the Int operator matrix, runtime ints compute the term.
  ##
  ## Emitted by cosizeImpl as a bound symbol: the macro expansion resolves
  ## each call in this module's scope, valid at instantiation sites lacking
  ## the Int operator matrix. Static layouts fold to compile-time terms here.
  ## A runtime cosize never reads an `Int[N]` phantom's storage.
  (leaf - 1) * abs(stride)

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
