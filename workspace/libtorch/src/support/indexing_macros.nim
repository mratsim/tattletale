# Arraymancer
# Copyright (c) 2017 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/macros,
  workspace/libtorch/src/support/ast_utils,
  workspace/libtorch/src/abi/[
    torch_tensors,
    std_cpp
  ]

# #######################################################################
#
#                      Slicing syntactic sugar
#
# #######################################################################
# https://github.com/mratsim/Arraymancer/blob/bdcdfe1/src/arraymancer/tensor/accessors_macros_syntax.nim

# Using a shifted Vandermonde matrix as an example v[i, j] = i^(j+1)
# torch.arange(1, 6).reshape(-1, 1) ** torch.arange(1, 6)
# [[   1    1    1    1    1]
#  [   2    4    8   16   32]
#  [   3    9   27   81  243]
#  [   4   16   64  256 1024]
#  [   5   25  125  625 3125]]
#
#
# Slicing syntax:
#
# Basic indexing - foo[2, 3]
# Basic indexing - foo[1+1, 2*2*1]
# Basic slicing - foo[1..2, 3]
# Basic slicing - foo[1+1..4, 3-2..2]
# Span slices - foo[_, 3]
# Span slices - foo[1.._, 3]
# Span slices - foo[_..3, 3]
# Span slices - foo[_.._, 3]
# Stepping - foo[1..3|2, 3]
# Span stepping - foo[_.._|2, 3]
# Span stepping - foo[_.._|+2, 3]
# Span stepping - foo[1.._|1, 2..3]
# Span stepping - foo[_..<4|2, 3]
# Slicing until at n from the end - foo[0..-4, 3]
# Span Slicing until at n from the end - foo[_..-2, 3]
# Stepped Slicing until at n from the end - foo[1..-1|2, 3]
#
# TODO: C++ libtorch doesn't support negative stride
#       PyTorch does preprocessing with flip before indexing.
# Slice from the end - foo[-1..0|-1, 3]
# Slice from the end - expect non-negative step error - foo[-1..0, 3]
# Slice from the end - foo[-(2*2)..2*2, 3]
# Slice from the end - foo[-3..-2, 3]
#
# Important: Nim slices are inclusive while TorchSlice are exclusive!
#
# Note: This syntax sugar is actually never generated
#       When desugaring we directly generate the proper TorchSlice.
#       However, detecting whether we have integers, slices or tensors
#       for dispatch requires help for the type system, and so
#       all the sigils must be defined as properly typed procedures.

# #######################################################################
#
#          Python Slice Semantics to libtorch Slice Conversion
#
# #######################################################################
#
# Python slicing uses EXCLUSIVE upper bound (like C++ standard).
# Negative indices are normalized by adding the dimension size.
#
# Examples for a 5-element array (indices 0, 1, 2, 3, 4):
#
# Python             | Start | Stop  | Result indices | Nim equivalent
# ------------------+-------+-------+---------------+----------------
# a[:]              | 0     | 5     | 0,1,2,3,4     | a[_.._]
# a[1:]              | 1     | 5     | 1,2,3,4      | a[1.._]
# a[:3]              | 0     | 3     | 0,1,2        | a[_..<3]
# a[1:3]             | 1     | 3     | 1,2          | a[1..<3]
# a[:-1]             | 0     | 4     | 0,1,2,3      | a[_..-1]
# a[-3:]             | 2     | 5     | 2,3,4        | a[-3.._]
# a[-3:-1]           | 2     | 4     | 2,3          | a[-3..-1]
#
# Key insight: In Python, -1 as STOP means "up to but NOT including the last element"
# -1 as START means "starting at the last element"
#
# libtorch Slice(start, stop) has EXCLUSIVE upper bound, matching Python.
# Negative values are normalized by adding size at runtime.
#
# Translation rules:
# - `_` (underscore) means "full span" → cpp_nullopt
# - `..<` (exclusive) → stop stays as-is
# - `..-` (end-relative, Python exclusive) → stop becomes size + (-N) = size - N
# - `..` (inclusive, rarely used) → stop becomes size + (-N) + 1 = size - N + 1

type OptInt = int | Nullopt_t

template normalizeNegativeIndex[T: int|Nullopt_t](idx: T, axisLen: int): T =
  when idx is Nullopt_t:
    idx
  else:
    if idx < 0:
      idx + axisLen
    else:
      idx

func normalizedSlice*(
        start: distinct OptInt,
        stop: distinct OptInt = cpp_nullopt(),
        step: distinct OptInt = cpp_nullopt(),
        axisLen: int): TorchSlice {.inline.} =
  ## Convert Python-style slice with step to C++ libtorch slices.

  let normStart = normalizeNegativeIndex(start, axisLen)
  let normStop =
    when stop is int:
      normalizeNegativeIndex(stop, axisLen)
    else:
      stop
  when step is int:
    # 0 repeats the first item of the axis, unsure why it would be useful but it shouldn't be an issue
    doAssert step >= 0, "C++ libtorch backend does not support negative steps" # TODO, flip tensor to support negative steps

  # debugEcho "normalizedStep:"
  # debugEcho "  start:     ", start
  # debugEcho "  stop:      ", stop
  # debugEcho "  step:      ", step
  # debugEcho "  axisLen:   ", axisLen
  # debugEcho "  ---"
  # debugEcho "  normStart: ", normStart
  # debugEcho "  normStop:  ", normStop
  # debugEcho "  ---"

  torchSlice(normStart, normStop, step)

template handleInclusiveDotDot(stop: OptInt): OptInt =
  # For positive numbers we want the classic Nim inclusive stop slice
  # For negative numbers we want equivalence of a ..- b and a .. -b
  # Ergo
  #   - Inclusive for positive
  #   - Exclusive for negative, Nim negative slices are inclusive with ^n so no collision
  when stop is Nullopt_t:
    stop
  else:
    if stop >= 0: succ(stop)
    else: stop

template handleExclusiveDotDot(stop: OptInt): OptInt =
  # For positive numbers we want the classic Nim exclusive stop slice
  # For negative numbers a ..<- b doesn't exist
  #   For now let's throw because it likely might reveal a logic problem
  when stop is Nullopt_t:
    stop
  else:
    if stop >= 0: stop
    else: raise newException(
      ValueError,
      "Slicing with exclusive stop ` * ..< " & $stop & "`.\n" &
      "This is currently not allowed as it's likely a logic bug that would be hard to debug otherwise."
    )

# #######################################################################
#
#                     Slicing Syntax Sugar
#
# #######################################################################

type Step = object
  ## Internal: Workaround to build ``TorchSlice`` without using parenthesis.
  ##
  ## Expected syntax is ``tensor[0..10|1]``.
  ##
  ## Due to operator precedence of ``|`` over ``..`` [0..10|1] is interpreted as [0..(10|1)]
  b: int
  step: int

func `|`*(s: Slice[int], step: int): TorchSlice {.inline.} =
  ## Internal: A ``TorchSlice`` constructor
  ## Input:
  ##     - a slice
  ##     - a step
  ## Returns:
  ##     - a ``TorchSlice``
  return torchSlice(s.a, s.b, step)

func `|`*(b, step: int): Step {.inline.} =
  ## Internal: A ``Step`` constructor
  ##
  ## ``Step`` is a workaround due to operator precedence.
  ##
  ## [0..10|1] is interpreted as [0..(10|1)]
  ## Input:
  ##     - the end of a slice range
  ##     - a step
  ## Returns:
  ##     - a ``Step``
  return Step(b: b, step: step)

func `|`*(ss: TorchSlice, step: int): TorchSlice {.inline.} =
  ## Internal: Modifies the step of a ``TorchSlice``
  ## Input:
  ##     - a ``TorchSlice``
  ##     - the new stepping
  ## Returns:
  ##     - a ``TorchSlice``
  return torchSlice(ss.start, ss.stop, step)

func `|+`*(s: Slice[int], step: int): TorchSlice {.inline.} =
  ## Internal: Alias for ``|``
  return `|`(s, step)

func `|+`*(b, step: int): Step {.inline.} =
  ## Internal: Alias for ``|``
  return `|`(b, step)

func `|+`*(ss: TorchSlice, step: int): TorchSlice {.inline.} =
  ## Internal: Alias for ``|``
  return `|`(ss, step)

func `|-`*(s: Slice[int], step: int): TorchSlice {.inline.} =
  ## Internal: A ``TorchSlice`` constructor
  ##
  ## Workaround to tensor[slice|-1] being interpreted as [slice `|-` 1]
  ##
  ## Properly create ``TorchSlice`` with negative stepping
  return torchSlice(s.a, s.b, -step)

func `|-`*(b, step: int): Step {.inline.} =
  ## Internal: A ``TorchSlice`` constructor
  ##
  ## Workaround to tensor[0..10|-1] being intepreted as [0 .. (10 `|-` 1)]
  ##
  ## Properly create ``TorchSlice`` with negative stepping
  return Step(b: b, step: -step)

func `|-`*(ss: TorchSlice, step: int): TorchSlice {.inline.} =
  ## Internal: Modifies the step of a ``TorchSlice``
  ##
  ## Workaround to tensor[slice|-1] being interpreted as [slice `|-` 1]
  ##
  ## Properly create ``TorchSlice`` with negative stepping
  return torchSlice(ss.start, ss.stop, -step)

func `|`*(step: int): TorchSlice {.inline.} =
  ## Create a full-span slice with step.
  ## Equivalent to Python's `::step` or `[::step]`
  ##
  ## Usage:
  ##   tensor[|2]       -> Slice(None, None, 2)  (every 2nd element)
  ##   tensor[|2, 3]    -> Slice(None, None, 2), 3
  ##   tensor[1, |2, _] -> 1, Slice(None, None, 2), Slice()
  ##
  ## This is cleaner than the older `_.._|step` syntax.
  return torchSlice(cpp_nullopt(), cpp_nullopt(), step)

func `|+`*(step: int): TorchSlice {.inline.} =
  ## Alias for ``|`` - positive step
  return `|`(step)

func `..|`*(a: int, step: int): TorchSlice {.inline.} =
  ## Internal: Build a TorchSlice from [a..|step]
  ## Input:
  ##     - the beginning of the slice range
  ##     - a `step` stride
  ## Returns:
  ##     - a ``TorchSlice``, end of range will be inclusive
  return torchSlice(a, cpp_nullopt(), step)

func `..`*(a: int, s: Step): TorchSlice {.inline.} =
  ## Internal: Build a TorchSlice from [a .. (b|step)] (workaround to operator precedence)
  ## Input:
  ##     - the beginning of the slice range
  ##     - a ``Step`` workaround object
  ## Returns:
  ##     - a ``TorchSlice``, end of range will be inclusive
  return torchSlice(a, s.b + 1, s.step)

func `..<`*(a: int, s: Step): TorchSlice {.inline.} =
  ## Internal: Build a TorchSlice from [a ..< (b|step)] (workaround to operator precedence)
  ## Input:
  ##     - the beginning of the slice range
  ##     - a ``Step`` workaround object
  ## Returns:
  ##     - a ``TorchSlice``, end of range will be exclusive.
  return torchSlice(a, s.b, s.step)

func `..-`*(a: int, s: Step): TorchSlice {.inline.} =
  ## Internal: Build a TorchSlice from [a ..- (b|step)] (workaround to operator precedence)
  ## Input:
  ##     - the beginning of the slice range
  ##     - a ``Step`` workaround object
  ## Returns:
  ##     - a ``TorchSlice``, end of range will be at negative position from end
  return torchSlice(a, -s.b, s.step)

func `-`*(s: TorchSlice): TorchSlice {.inline.} =
  ## Internal: Prefix to a slice to indicate starting at negative position from end
  return torchSlice(-s.start, s.stop, s.step)

func `-`*(s: Slice): TorchSlice {.inline.} =
  ## Internal: Prefix to a slice to indicate starting at negative position from end
  return torchSlice(-s.a, s.b, 1)

# #######################################################################
#
#                          Slicing desugaring
#
# #######################################################################
# https://github.com/mratsim/Arraymancer/blob/bdcdfe1/src/arraymancer/tensor/private/p_accessors_macros_desugar.nim

# We need to catch all `sym` (resolved symbol)
# in the AST to untype them
# so we can delay generics/template symbol resolution
# for after the indexing macro is done.
#
# Not doing so and having a type mismatch also results
# in cryptic `undeclared identifier` errors.

func sliceSpan(): NimNode =
  newCall(bindSym"SliceSpan")

func sliceNone(): NimNode =
  newCall(bindSym"cpp_nullopt")

func sliceEllipsis(): NimNode =
  newCall(bindSym"ellipsis")

func dotdotIncl(node: NimNode): NimNode =
  newCall(bindsym"handleInclusiveDotDot", node)

func dotdotExcl(node: NimNode): NimNode =
  newCall(bindsym"handleExclusiveDotDot", node)

func `-`(node: NimNode): NimNode =
  newCall(bindsym"-", node)

func Slice(nodes: varargs[NimNode]): NimNode =
  # We model with torchSlices first
  # then in the latest step we change to normalizedSlice and append the axis-length
  # to handle negative indices.
  # C++ libtorch slices with a.index({None, -1}) will slice to -1 inclusive
  # Pytorch with a[:-1] will slice to -1 exclusive
  #
  # The solution is to pass a.index({None, axisLen-1}) which will slice to axisLen-1 exclusive
  result = newCall(bindSym"torchSlice")
  for node in nodes:
    result.add node

macro desugarSlices*(args: untyped): void =
  ## Transform all syntactic sugar into Slice(start, stop, step)
  ## or integers
  ##
  ## This is necessary otherwise something like `[_, _]`
  ## will try to be matched against FancyIndexing functions
  ## that can't handle it
  ## AND due to template/macro early name resolution rule
  ## the error message will be a confusing `undeclared identifier`

  # echo "\n------------------\nOriginal tree"
  # echo args.treerepr
  var r = newNimNode(nnkArglist)

  for nnk in children(args):
    ###### Traverse top tree nodes and one-hot-encode the different conditions

    # Node is "_"
    let nnk_joker = eqIdent(nnk, "_")

    # Node is of the form "* .. *"
    let nnk0_inf_dotdot = (nnk.kind == nnkInfix and eqIdent(nnk[0], ".."))

    # Node is of the form "* ..< *" or "* ..- *" or "* ..| *"
    let nnk0_inf_dotdot_alt = (nnk.kind == nnkInfix and (
      eqIdent(nnk[0], "..<") or
      eqident(nnk[0], "..-") or
      eqident(nnk[0], "..|")
    ))

    # Node is of the form "* .. *", "* ..< *" or "* ..- *" or "* ..| *"
    let nnk0_inf_dotdot_all = (nnk0_inf_dotdot or nnk0_inf_dotdot_alt)

    # Node is of the form "- *" (minus prefix)
    let nnk0_pre_minus = (nnk.kind == nnkPrefix and eqIdent(nnk[0], "-"))

    # Node is of the form "_ `op` *"
    let nnk1_joker = (nnk.kind == nnkInfix and eqIdent(nnk[1], "_"))

    # Node is of the form "* `op` - *"
    let nnk10_minus = (nnk.kind == nnkInfix and nnk[1].kind == nnkPrefix and eqident(nnk[1][0], "-"))

    # Node is of the form "* `op` _"
    let nnk2_joker = (nnk.kind == nnkInfix and eqident(nnk[2], "_"))

    # Node is of the form "* `op` * | *" or "* `op` * |+ *"
    let nnk20_bar_pos =
      (nnk.kind == nnkInfix and nnk[2].kind == nnkInfix and (eqident(nnk[2][0], "|") or eqIdent(nnk[2][0], "|+")))

    # Node is of the form "* `op` * |- *"
    let nnk20_bar_min = (nnk.kind == nnkInfix and nnk[2].kind == nnkInfix and eqIdent(nnk[2][0], "|-"))

    # Node is of the form "* `op` * | *" or "* `op` * |+ *" or "* `op` * |- *"
    let nnk20_bar_all = nnk20_bar_pos # or nnk20_bar_min

    # Node is of the form "* `op1` _ `op2` *"
    let nnk21_joker = (nnk.kind == nnkInfix and nnk[2].kind == nnkInfix and eqIdent(nnk[2][1], "_"))

    # Node is of the form "|step" (unary pipe operator for stepped span)
    # e.g., tensor[|2, 3] -> tensor[Slice(None, None, 2), 3]
    let nnk_pre_bar = (nnk.kind == nnkPrefix and (eqIdent(nnk[0], "|") or eqIdent(nnk[0], "|+")))

    ###### Core desugaring logic
    if nnk_joker:
      ## [_, 3] into [Slice(), 3]  # Arraymancer span, equivalent to Python ":"
      ## This is NOT Ellipsis - it should be Slice() (full span for one dimension)
      r.add(sliceSpan())
    elif nnk_pre_bar:
      ## [|2] into [Slice(None, None, 2)]
      ## [|2, 3] into [Slice(None, None, 2), 3]
      ## Equivalent to Python's [::2] or [::2, 3]
      r.add(Slice(sliceNone(), sliceNone(), nnk[1]))
    elif nnk20_bar_min:
      error "Negative steps are not supported when indexing in torch::Tensor. The use of flip() is recommended."
    elif nnk0_inf_dotdot and nnk1_joker and nnk2_joker:
      ## [_.._, 3] into [Slice(), 3]  # Full span, not Ellipsis
      ## PyTorch Slice() is equivalent to Python ":", "::", not "..."
      r.add(sliceSpan())
    elif nnk0_inf_dotdot and nnk1_joker and nnk20_bar_all and nnk21_joker:
      ## [_.._|2, 3] into [{Slice(None, None, 2), 3}]
      ## [_.._|+2, 3] into [{Slice(None, None, 2), 3}]
      ## [_.._|-2 doesn't make sense and will throw out of bounds
      r.add(Slice(sliceNone(), sliceNone(), nnk[2][2]))
    elif nnk0_inf_dotdot_all and nnk1_joker and nnk20_bar_all:
      ## [_..10|1, 3] into [{Slice(None, 10+1, 1), 3}] (inclusive end)
      ## [_..-10|1, 3] into [{Slice(None, -10, 1), 3}] (negative handled at runtime)
      ## [_..<10|1, 3] into [{Slice(None, 10, 1), 3}] (exclusive end)
      if nnk[0].eqident(".."):
        r.add Slice(sliceNone(), dotdotIncl(nnk[2][1]), nnk[2][2])
      elif nnk[0].eqident("..-"):
        r.add Slice(sliceNone(), -nnk[2][1], nnk[2][2])  # negative handled by normalizedSlice
      elif nnk[0].eqident("..<"):
        r.add Slice(sliceNone(), dotdotExcl(nnk[2][1]), nnk[2][2])
      else:
        error "Unreachable"
    elif nnk0_inf_dotdot_all and nnk1_joker:
      ## [_..10, 3] into [{Slice(None, 10+1), 3}] (inclusive end)
      ## [_..-10, 3] into [{Slice(None, -10), 3}] (negative handled at runtime)
      ## [_..<10, 3] into [{Slice(None, 10), 3}] (exclusive end)
      ## [_..|2, 3] into [{Slice(None, None, 2), 3}]
      if nnk[0].eqident(".."):
        r.add Slice(sliceNone(), dotdotIncl(nnk[2]))
      elif nnk[0].eqident("..-"):
        r.add Slice(sliceNone(), -nnk[2])  # negative handled by normalizedSlice
      elif nnk[0].eqident("..<"):
        r.add Slice(sliceNone(), dotdotExcl(nnk[2]))
      elif nnk[0].eqident("..|"):
        r.add Slice(sliceNone(), sliceNone(), nnk[2])
      else:
        error "Unreachable"
    elif nnk0_inf_dotdot and nnk2_joker:
      ## [1.._, 3] into [{Slice(1, None, None), 3}]
      r.add Slice(nnk[1])
    elif nnk0_inf_dotdot and nnk20_bar_pos and nnk21_joker:
      ## [1.._|1, 3] into [{Slice(1, None, 1), 3}]
      ## [1.._|+1, 3] into [{Slice(1, None, 1), 3}]
      r.add Slice(nnk[1], sliceNone(), nnk[2][2])
    elif nnk0_inf_dotdot and nnk20_bar_min and nnk21_joker:
      # TODO : Remove ? This is actually unreachable because nnk20_bar_min is disallowed
      ## Raise error on [5.._|-1, 3]
      raise newException(
        IndexDefect, "Please use explicit end of range " & "instead of `_` " & "when the steps are negative"
      )
    elif nnk0_inf_dotdot_all and nnk10_minus and nnk20_bar_all:
      # TODO disable negative step at CT
      ## [-1..2|-1, 3] into [{Slice(-1, 2, -1), 3}]
      # r.add Slice(nnk[1][1], nnk[2][1], -nnk[2][2])
      error "Slicing Tensor in reverse is equivalent to using negative steps. Negative steps are not supported when indexing torch::tensor. Use flip() instead."
    elif nnk0_inf_dotdot_all and nnk10_minus:
      # TODO disable negative step at CT
      ## [-1..2*3, 3] into [{Slice(-1, 2*3 + 1), 3}]
      ## [-1..0, 3] into [{Slice(-1, 0 + 1), 3}]
      ## [-1..<10, 3] into [{Slice(-1, 10), 3}]
      ## [-10..^1, 3] into [{Slice(-10, -1), 3}]
      if nnk[0].eqident(".."):
        # a[-1..3]
        error "Slicing Tensor in reverse is equivalent to using negative steps. Negative steps are not supported when indexing torch::tensor. Use flip() instead."
      elif nnk[0].eqident("..<"):
        # a[-1..<3]
        # r.add Slice(nnk[1][1], nnk[2])
        error "Slicing Tensor in reverse is equivalent to using negative steps. Negative steps are not supported when indexing torch::tensor. Use flip() instead."
      elif nnk[0].eqident("..-"):
        # a[-3..-1]
        # TODO currently we allow string literals but not variables or expressions
        if nnk[1][1].toStrLit.strVal[0] > nnk[2].toStrLit.strVal[0]:
          r.add Slice(nnk[1][1], -nnk[2])
        else:
          error "Slicing Tensor in reverse is equivalent to using negative steps. Negative steps are not supported when indexing torch::tensor. Use flip() instead."
      else:
        error "Unreachable"
    elif nnk0_inf_dotdot_all and nnk20_bar_all:
      ## [1..10|1] into [{Slice(1, 10 + 1, 1)}]
      ## [1..-10|1] into [{Slice(1, -10, 1)}]
      ## [1..<10|1] into [{Slice(1, 10, 1)}]
      if nnk[0].eqident(".."):
        r.add Slice(nnk[1], dotdotIncl(nnk[2][1]), nnk[2][2])
      elif nnk[0].eqident("..-"):
        r.add Slice(nnk[1], -nnk[2][1], nnk[2][2])
      elif nnk[0].eqident("..<"):
        r.add Slice(nnk[1], dotdotExcl(nnk[2][1]), nnk[2][2])
      else:
        error "Unreachable"
    elif nnk0_inf_dotdot_all:
      ## [1..10] into [{Slice(1, 10 + 1)}]
      ## [1..-10] into [{Slice(1, -10)}]
      ## [1..<10] into [{Slice(1, 10)}]
      ## [1..|2] into [{Slice(1, None, 2)}]
      if nnk[0].eqident(".."):
        r.add Slice(nnk[1], dotdotIncl(nnk[2]))
      elif nnk[0].eqident("..-"):
        r.add Slice(nnk[1], -nnk[2])
      elif nnk[0].eqident("..<"):
        r.add Slice(nnk[1], dotdotExcl(nnk[2]))
      elif nnk[0].eqident("..|"):
        r.add Slice(nnk[1], sliceNone(), nnk[2])
      else:
        error "Unreachable"
    elif nnk0_pre_minus:
      ## [-2, 3] into [-2..-2|1, 3]
      r.add(-nnk[1])
    else:
      r.add(nnk)
  # echo "\nAfter modif"
  # echo r.treerepr
  return r

# #######################################################################
#
#                          Slicing dispatch
#
# #######################################################################
# https://github.com/mratsim/Arraymancer/blob/bdcdfe1/src/arraymancer/tensor/private/p_accessors_macros_read.nim
# https://github.com/mratsim/Arraymancer/blob/bdcdfe1/src/arraymancer/tensor/private/p_accessors_macros_write.nim

type FancySelectorKind* = enum
  FancyNone
  FancyIndex
  FancyMaskFull
  FancyMaskAxis
  # Workaround needed for https://github.com/nim-lang/Nim/issues/14021
  FancyUnknownFull
  FancyUnknownAxis

proc getFancySelector(ast: NimNode, axis: var int, selector: var NimNode): FancySelectorKind =
  ## Detect indexing in the form
  ##   - "tensor[_, _, [0, 1, 4], _, _]
  ##   - "tensor[_, _, [0, 1, 4], `...`]
  ##  or with the index selector being a tensor
  result = FancyNone
  var foundNonSpanOrEllipsis = false
  var ellipsisAtStart = false

  template checkNonSpan(): untyped {.dirty.} =
    doAssert not foundNonSpanOrEllipsis,
      "Fancy indexing is only compatible with full spans `_` on non-indexed dimensions" & " and/or ellipsis `...`"

  var i = 0
  while i < ast.len:
    let cur = ast[i]
    # Important: sameType doesn't work for generic type like Array, Seq or Tensors ...
    #            https://github.com/nim-lang/Nim/issues/14021
    if (cur.kind == nnkCall and cur[0].eqIdent"SliceSpan"):
      # Found a span
      discard
    elif (cur.kind == nnkCall and cur[0].eqIdent"torchSlice") or cur.isInt():
      doAssert result == FancyNone,
        "Internal FancyIndexing Error: Expected FancyNone, but got " & $result & " for AST: " & cur.repr()
      foundNonSpanOrEllipsis = true
    elif (cur.kind == nnkCall and cur[0].eqIdent"ellipsis"):
      if i == ast.len - 1: # t[t.sum(axis = 1) >. 0.5, `...`]
        doAssert not ellipsisAtStart,
          "Cannot deduce the indexed/sliced dimensions due to ellipsis at the start and end of indexing."
        ellipsisAtStart = false
      elif i == 0: # t[`...`, t.sum(axis = 0) >. 0.5]
        ellipsisAtStart = true
      else:
        # t[0 ..< 10, `...`, t.sum(axis = 0) >. 0.5] is unsupported
        # so we tag as "foundNonSpanOrEllipsis"
        foundNonSpanOrEllipsis = true
    elif cur.kind == nnkBracket:
      checkNonSpan()
      axis = i
      if cur[0].kind == nnkIntLit:
        result = FancyIndex
        selector = cur
      elif cur[0].isBool():
        let full = i == 0 and ast.len == 1
        result = if full: FancyMaskFull else: FancyMaskAxis
        selector = cur
      else:
        # byte, char, enums are all represented by integers in the VM
        error "Fancy indexing is only possible with integers or booleans"
    else:
      checkNonSpan()
      axis = i
      let full = i == 0 and ast.len == 1
      result = if full: FancyUnknownFull else: FancyUnknownAxis
      selector = cur
    inc i

  # Handle ellipsis at the start
  if result != FancyNone and ellipsisAtStart:
    axis = ast.len - axis

  # replace all possible `nnkSym` by `idents` because we otherwise might get
  # type mismatches
  selector = replaceSymsByIdents(selector)

macro slice_typed_dispatch*(t: typed, args: varargs[typed]): untyped =
  ## Typed macro so that isAllInt has typed context and we can dispatch.
  ## If args are all int, we dispatch to atIndex and return T
  ## Else, all ints are converted to SteppedSlices and we return a Tensor.
  ## Note, normal slices and `_` were already converted in the `[]` macro
  ## TODO in total we do 3 passes over the list of arguments :/. It is done only at compile time though

  # Point indexing
  # -----------------------------------------------------------------
  if isAllInt(args):
    # Point indexing: all indices are integers
    # PyTorch's index() will handle bounds checking internally
    let indexCall = newCall(bindSym"index", t)
    for slice in args:
      indexCall.add(slice)
    result = indexCall
    # echo result.repr
    # echo "--------------------"
    return

  # Fancy indexing
  # -----------------------------------------------------------------
  # Cannot depend/bindSym the "selectors.nim" proc
  # Due to recursive module dependencies
  var selector: NimNode
  var axis: int
  let fancy = args.getFancySelector(axis, selector)

  if fancy == FancyIndex:
    return newCall(ident"index_select", t, newLit axis, selector)
  if fancy == FancyMaskFull:
    return newCall(ident"masked_select", t, selector)
  elif fancy == FancyMaskAxis:
    return newCall(ident"masked_axis_select", t, selector, newLit axis)

  # Slice indexing
  # -----------------------------------------------------------------
  if fancy == FancyNone:
    result = newCall(bindSym"index", t)
    for i in 0 ..< args.len:
      let slice = args[i]
      if slice.kind == nnkCall and slice[0].eqIdent"torchSlice":
        # Call normalizedSlice(start, stop, step, axisLen)
        # instead of torchSlice(start, stop, step)
        let normalizedSlice = newCall(bindSym"normalizedSlice")
        for j in 1 ..< slice.len:
          normalizedSlice.add(slice[j])
        let axisLen = nnkExprEqExpr.newTree(
          # use named argument
          # axisLen = `t`.shape[`i`]
          ident"axisLen",
          nnkBracketExpr.newTree(
            nnkDotExpr.newTree(
              t,
              bindSym"shape"
            ),
            newLit i
          )
        )
        normalizedSlice.add axisLen
        result.add(normalizedSlice)
      else:
        result.add(slice)
    return

  # Fancy bug in Nim compiler
  # -----------------------------------------------------------------
  # We need to drop down to "when a is T" to infer what selector to call
  # as `getType`/`getTypeInst`/`getTypeImpl`/`sameType`
  # are buggy with generics
  # due to https://github.com/nim-lang/Nim/issues/14021
  let lateBind_masked_select = ident"masked_select"
  let lateBind_masked_axis_select = ident"masked_axis_select"
  let lateBind_index_select = ident"index_select"

  result = quote:
    type FancyType = typeof(`selector`)
    when FancyType is (array or seq):
      type FancyTensorType = typeof(toTensor(`selector`))
    else:
      type FancyTensorType = FancyType
    if `t`.scalarType == kBool:
      when FancySelectorKind(`fancy`) == FancyUnknownFull:
        `lateBind_masked_select`(`t`, `selector`)
      elif FancySelectorKind(`fancy`) == FancyUnknownAxis:
        `lateBind_masked_axis_select`(`t`, `selector`, `axis`)
      else:
        {.error: "Unreachable".}
    else:
      `lateBind_index_select`(`t`, `axis`, `selector`)

macro slice_typed_dispatch_mut*(t: typed, args: varargs[typed], val: typed): untyped =
  ## Assign `val` to Tensor T at slice/position `args`

  # Point indexing
  # -----------------------------------------------------------------
  if isAllInt(args):
    # Point indexing: all indices are integers
    # PyTorch's index_put() will handle bounds checking internally
    let putCall = newCall(bindSym"index_put", t)
    for slice in args:
      putCall.add(slice)
    putCall.add(val)
    result = putCall
    return

  # Fancy indexing
  # -----------------------------------------------------------------
  # Cannot depend/bindSym the "selectors.nim" proc
  # Due to recursive module dependencies
  var selector: NimNode
  var axis: int
  let fancy = args.getFancySelector(axis, selector)
  if fancy == FancyIndex:
    return newCall(ident"index_fill_mut", t, newLit axis, selector, val)
  if fancy == FancyMaskFull:
    return newCall(ident"masked_fill_mut", t, selector, val)
  elif fancy == FancyMaskAxis:
    return newCall(ident"masked_axis_fill_mut", t, selector, newLit axis, val)

  # Slice indexing
  # -----------------------------------------------------------------
  if fancy == FancyNone:
    result = newCall(bindSym"index_put", t)
    for slice in args:
      result.add(slice)
    result.add(val)
    return

  # Fancy bug in Nim compiler
  # -----------------------------------------------------------------
  # We need to drop down to "when a is T" to infer what selector to call
  # as `getType`/`getTypeInst`/`getTypeImpl`/`sameType`
  # are buggy with generics
  # due to https://github.com/nim-lang/Nim/issues/14021
  let lateBind_masked_fill = ident"masked_fill"
  let lateBind_masked_axis_fill = ident"masked_axis_fill"
  let lateBind_index_fill = ident"index_fill"

  result = quote:
    type FancyType = typeof(`selector`)
    when FancyType is (array or seq):
      type FancyTensorType = typeof(toTensor(`selector`))
    else:
      type FancyTensorType = FancyType
    if `t`.scalarType == kBool:
      when FancySelectorKind(`fancy`) == FancyUnknownFull:
        `lateBind_masked_fill`(`t`, `selector`, `val`)
      elif FancySelectorKind(`fancy`) == FancyUnknownAxis:
        `lateBind_masked_axis_fill`(`t`, `selector`, `axis`, `val`)
      else:
        {.error: "Unreachable".}
    else:
      `lateBind_index_fill`(`t`, `axis`, `selector`, `val`)
