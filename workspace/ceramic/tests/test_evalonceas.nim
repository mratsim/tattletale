import ../src/int_tuples

# ═══════════════════════════════════════════════════════════════
# evalOnceAs test suite
# ═══════════════════════════════════════════════════════════════

block: # Sym reuse — let indirection
  let x = 42
  evalOnceAs(a, x)
  doAssert a() === 42
  echo "✅ let indirection"

block: # Sym reuse — const indirection (gets constant-folded to nnkIntLit)
  const y = 16
  evalOnceAs(b, y)
  doAssert b() === 16
  echo "✅ const indirection"

block: # Sym reuse — proc parameter
  proc test(p: int): auto =
    evalOnceAs(c, p)
    c()
  doAssert test(99) === 99
  echo "✅ param indirection"

block: # Compile-time — int literal
  evalOnceAs(e, 1024)
  doAssert e() === 1024
  echo "✅ int literal"

block: # Compile-time — all-args-CT call (max of CT values)
  evalOnceAs(f, max(1, 16))
  doAssert f() === 16
  echo "✅ all-args-CT call"

block: # Runtime — dynamic proc call
  proc rtAdd(a, b: int): int = a + b
  let v = 5
  evalOnceAs(g, rtAdd(v, 10))
  doAssert g() === 15
  echo "✅ dynamic proc call"

block: # Runtime — mixed CT/RT args
  proc rtAdd2(a, b: int): int = a + b
  let w = 5
  evalOnceAs(h, rtAdd2(1, w))
  doAssert h() === 6
  echo "✅ mixed CT/RT args"

block: # Runtime — no-arg proc
  proc getVal(): int = 99
  evalOnceAs(i, getVal())
  doAssert i() === 99
  echo "✅ no-arg proc"

block: # Constant-folding — runtime func with CT args folded to Int[N]
  func square(x: int): int = x * x

  func squareWithStaticDetection(x: int): int = x * x
  func squareWithStaticDetection(x: static int): static int = x * x

  evalOnceAs(bar, squareWithStaticDetection(square(3)))
  # square(3) = 9 → `isCompileTime` sees all-args-CT → const-folded
  # squareWithStaticDetection(9) with static int → 9 * 9 = 81
  doAssert typeof(bar()) is Int, "runtime func + CT args → const fold → Int[N]"
  doAssert bar() === 81
  echo "✅ constant-folding"

block: # Field access — const object
  type Obj = object
    x: int
    y: int
  const obj = Obj(x: 42, y: 16)
  evalOnceAs(j, obj.x)
  doAssert j() === 42
  echo "✅ field access (const object)"

block: # Field access — let object
  type Obj2 = object
    x: int
  let obj2 = Obj2(x: 99)
  evalOnceAs(k, obj2.x)
  doAssert k() === 99
  echo "✅ field access (let object)"

block: # Field access — const tuple
  const tup = (a: 7, b: 8)
  evalOnceAs(l, tup.a)
  doAssert l() === 7
  echo "✅ field access (const tuple)"

block: # Field access — nested const object
  type Inner = object
    val: int
  type Outer = object
    inner: Inner
  const outer = Outer(inner: Inner(val: 99))
  evalOnceAs(m, outer.inner.val)
  doAssert m() === 99
  echo "✅ field access (nested const object)"

echo "\n✅ All evalOnceAs tests passed!"
