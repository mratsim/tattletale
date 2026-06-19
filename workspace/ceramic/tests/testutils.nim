## Test utilities — type-safe comparison with constant-folding verification.

template check*(got: untyped; expected: typed; expectedType: typedesc): untyped =
  ## Assert `got` has type `expectedType` and equals `expected` via `===`.
  ##
  ## Fails with a compile-time error if the type doesn't match,
  ## directing the developer to check constant-folding.
  block:
    let tmp = got
    type TmpType = typeof(tmp)
    when TmpType is expectedType:
      doAssert tmp === expected
    else:
      static:
        doAssert false, "[ttt] Please check constant-folding: type is " &
          $TmpType & ", expected " & $expectedType
