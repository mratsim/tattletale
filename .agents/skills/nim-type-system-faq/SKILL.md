---
name: nim-type-system-faq
description: Nim type system patterns and pitfalls
license: MIT
compatibility: opencode
metadata:
  audience: nim-developers
  topic: types
---

## Problem: Union types in generics require same concrete type

When you define a generic with a union type like `T: int|Nullopt_t`, Nim requires ALL parameters of that type to be the SAME concrete type:

```nim
template handleNegativeIndex[T: int|Nullopt_t](idx: T, axisLen: int): T =
  when idx is Nullopt_t:
    idx
  else:
    if idx < 0:
      idx + axisLen
    else:
      idx
```

This fails if you call `handleNegativeIndex(start, len)` where `start` is `int` and you want to pass `nullopt` for `stop`. The compiler complains because `int` and `Nullopt_t` are different types, even though both belong to the union.

## Solution: Use `distinct` type

Define a distinct wrapper type that "unifies" the union:

```nim
type OptInt* = distinct int | Nullopt_t

template handleNegativeIndex*[T: int|Nullopt_t](idx: T, axisLen: int): T =
  when idx is Nullopt_t:
    idx
  else:
    if idx < 0:
      idx + axisLen
    else:
      idx

func normalizedSlice*(
        start, stop: distinct OptInt,
        step: OptInt = nullopt, axisLen: int): TorchSlice {.inline.} =
  let normStart = handleNegativeIndex(start, axisLen)
  let normStop = handleNegativeIndex(stop, axisLen)
  torchSlice(normStart, normStop, step)
```

The `distinct` keyword creates a new type that:
1. Is compatible with all types in the union at runtime
2. Allows parameters to have DIFFERENT concrete types from the same union
3. Preserves type safety while enabling flexible APIs

## When to use this pattern

- API functions that accept either a value OR "none"/"default"
- Slice/indexing functions where parameters can be int or nullopt
- Callbacks that may receive typed or untyped values

## Related patterns

- `option[T]` from stdlib for explicit optional values
- `nullopt` singleton for "no value provided"
- `when defined(T)` branches for type-specific logic
