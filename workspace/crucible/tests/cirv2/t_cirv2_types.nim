## CIRv2 typing invariants: asgn agreement, bool conditions, case ranges, assume shape
##
## The verifier rejects these violations:
##
## | Violation                  | Detail                                              |
## | -------------------------- | --------------------------------------------------- |
## | asgn type mismatch         | operand types disagree with the destination         |
## | non-bool ite condition     | the condition does not evaluate to bool             |
## | bad case range             | overlapping, negative, or empty range               |
## | oversized literal          | the value does not fit its stated type              |
## | malformed assume condition | not a comparison or an and/or tree of comparisons   |
## | ret disagreement           | the returned type differs from the proc return type |
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/cirv2 --nimcache:nimcache/tests/cirv2 \
##     workspace/crucible/tests/cirv2/t_cirv2_types.nim

import
  workspace/crucible/src/codegen/ir/cirv2_lang,
  workspace/crucible/src/codegen/ir/cirv2_build,
  workspace/crucible/src/codegen/ir/cirv2_verify,
  workspace/crucible/tests/cirv2/cirv2_testutils

static:
  let
    f32 = tFloat(32)
    i32t = tInt(32)
    u32t = tUint(32)
    i = sym("i", i32t)

  # asgn sides from different lattices (f32 destination, i32 value)
  block:
    let fn = procD("asgn_mismatch", @[], tVoid(), @[
      varD("x", f32),
      asgnS(sym("x", f32), addE(i32t, i, i)),
    ])
    fn.expectVerifyError("asgn at asgn_mismatch/body[1].rhs" &
      ": right-hand side type differs from the destination type")

  # an ite condition of int type
  block:
    let fn = procD("ite_nonbool", @[], tVoid(), @[
      iteS(addE(i32t, i, i), @[retS()]),
    ])
    fn.expectVerifyError("ite at ite_nonbool/body[0]: condition is not bool")

  # a typed op whose operands disagree with its stated type
  block:
    let fn = procD("op_type_lie", @[], tVoid(), @[
      varD("x", f32),
      asgnS(sym("x", f32), addE(f32, i, i)),
    ])
    fn.expectVerifyError("add at op_type_lie/body[1].rhs" &
      ": left operand type differs from the spelled type")

  # an untyped arithmetic op (no stated type)
  block:
    var bad = addE(i32t, i, i)
    bad.opType = nil
    let fn = procD("op_untyped", @[], tVoid(), @[
      varD("x", i32t),
      asgnS(sym("x", i32t), bad),
    ])
    fn.expectVerifyError("add at op_untyped/body[1].rhs" &
      ": right-hand side has no derivable type")

  # overlapping case ranges across branches
  block:
    let fn = procD("case_overlap", @[], tVoid(), @[
      caseS(i, @[
        branch(@[rng(0, 5)], @[retS()]),
        branch(@[rng(3, 9)], @[retS()]),
      ]),
    ])
    fn.expectVerifyError("case at case_overlap/body[0]" &
      ": overlapping ranges 0..5 and 3..9")

  # a negative range over an unsigned selector
  block:
    let fn = procD("case_negative_uint", @[], tVoid(), @[
      varD("u", u32t, intLit(0, u32t)),
      caseS(sym("u", u32t), @[
        branch(@[rng(-1, 3)], @[retS()]),
      ]),
    ])
    fn.expectVerifyError("negative range bound over an unsigned selector")

  # an empty (inverted) range
  block:
    let fn = procD("case_empty_range", @[], tVoid(), @[
      caseS(i, @[
        branch(@[rng(5, 1)], @[retS()]),
      ]),
    ])
    fn.expectVerifyError("empty range 5..1")

  # a literal that does not fit its stated type
  block:
    let fn = procD("literal_overflow", @[], tVoid(), @[
      varD("x", i32t, intLit(1'i64 shl 40, i32t)),
    ])
    fn.expectVerifyError("int literal at literal_overflow/body[0].init" &
      ": value 1099511627776 does not fit (i 32)")

  # an assume condition that is not comparison-shaped
  block:
    let fn = procD("assume_unshaped", @[], tVoid(), @[
      assumeS(addE(i32t, i, i)),
    ])
    fn.expectVerifyError("assume at assume_unshaped/body[0]" &
      ": condition is not a comparison or a conjunction over comparisons")

  # a ret whose payload disagrees with the proc return type
  block:
    let fn = procD("ret_mismatch", @[], f32, @[
      retS(addE(i32t, i, i)),
    ])
    fn.expectVerifyError("ret at ret_mismatch/body[0].value" &
      ": return type differs from the proc return type")

  # an empty ret from a non-void proc
  block:
    let fn = procD("ret_empty_nonvoid", @[], f32, @[])
    fn.body.add(retS())
    fn.expectVerifyError("empty return from a non-void proc")

  # a conjunction over comparisons is a legal assume shape
  block:
    let fn = procD("assume_ok", @[], tVoid(), @[
      assumeS(andE(leE(i, intLit(10, i32t)), ltE(intLit(0, i32t), i))),
    ])
    verify(fn)
