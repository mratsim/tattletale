## Unsanctioned helpers, pending rationalization.
##
## Helpers landing here bypass the sanctioned op surface, the funnel pass
## later promotes, keeps, or kills each one.
##
## - sanctioned modules may import this file while a helper awaits its home
## - what survives rationalization is re-exported through the family module
import std/macros, std/typetraits
import ./int_tuples_compiletime
import ./int_tuples_streams

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
