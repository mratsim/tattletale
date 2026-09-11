---
series: "Understanding Transducers through Python (8 articles)"
author: Rob Smallshire
source: sixty-north.com/blog
published: "2015-01-19 through 2015-03-27"
contents: Deriving Transducers from First Principles · Improving Transducer Composition · Enriching the Transducer Protocol · Stateful Transducers · Terminating Transducers · Item Injecting Transducers · Lazy Transducer Evaluation · Event Processing with Transducers
license-note: "Blog posts © Rob Smallshire / Sixty North; web-published copies reproduced for private research/study with attribution and links. All rights remain with the author."
---

# Understanding Transducers through Python — Rob Smallshire

*The complete eight-article series deriving transducers from first principles: plain reducers → transducers → composition → enrichment → state → early termination → injecting transducers → laziness → event processing.*

## Article 1 of 8 — Deriving Transducers from First Principles

*Mon 19 January 2015 — https://sixty-north.com/blog/deriving-transducers-from-first-principles.html*

## What is a transducer?

Transducers - a portmanteau of ‘transform reducers’ - are a new functional programming concept introduced into the Clojure programming language. Although transducers are actually pretty straightforward in retrospect, wrapping your brain around them, especially if you’re not already a competent Clojureist, can be challenging. In this series of articles we'll explain transducers by implementing them from first principles in Python.

To understand transducers, we must first understand reducing functions.
In a nutshell, a *reducing function* is any function which takes a
partial result and a new piece of information to produce a new result.
Reducers are what we pass to the `reduce()` function in Python and can
be thought of as having the following structure:

(result, input) -> result

A *transducer* is a function which accepts a reducing function and
returns a new reducing function:

((result, input) -> result)  ->  ((result, input) -> result)

A transducer function can be used to augment an existing reducing
function with some additional reducing behaviour, producing a new
reducing function which applies both reducing behaviours in a specific
order. In this way, we can use transducers to chain arbitrary reducing
functions together. This in turn is useful, because it allows us to
perform complex reduction operations using only a single call to
`reduce()`. In other words, transducers are primarily useful because
they provide a means of composing reducers.

As we will see, transducers also facilitate a clear separation of concerns between the algorithmic essence of a particular reduction, and the details of from where the input data are coming from and where the output data are going to.

In this series, we'll introduce transducers by implementing them from
scratch in everybody’s favourite executable pseudocode, Python. We’ll
start with the familiar staples of functional programming, `map()`,
`filter()` and `reduce()`, and derive transducers from first
principles. We’ll work towards a set of general tools which work with
eager collections, lazy ‘pull’ sequences, and ‘push’ event streams.
Along the way we’ll cover stateful transducers and transducer
composition, demonstrating that transducers are both more general, and
more fundamental, than the functional programming tools baked into
Python and many other languages.

By the end of this series, not only should transducers make sense to you, but you’ll have a recipe for implementing transducers in your own favourite programming language.

The main objective of this exercise is to understand transducers. A
secondary objective is to see how some more advanced functional
programming techniques can be realised in a language like Python. We're
not particularly advocating that you should widely adopt transducers in
Python instead of, say, generators. For the time being, we expect
transducers in Python to remain something of curiosity. That said, if
you *do* find any interesting applications of transducers in Python, do
please let us know.

## The origin of transducers

Transducers were introduced by Rich Hickey, creator of the Clojure programming language, in a blog post "Transducers are Coming" [1] on August 6th, 2014.

The best way to understand transducers is to implement them yourself. Unfortunately from an educational point of view, the Clojure implementations are mixed up with some other details of Clojure which are pretty much irrelevant to transducers, and Clojure transducers are heavily implemented using anonymous functions which makes them excessively difficult to discuss.

## How does this relate to Python?

Python is a pretty straightforward language - at least on the surface,
and although Python supports lambdas and other forms of anonymous
functions, in this presentation we have striven to use *named* functions
to make the intent of the code more obvious, rather than writing Python
in the style of Clojure.

Both Clojure and Python happen to be dynamically typed – or uni-typed – languages, and although that's not particularly important here, it does mean that the Python version can stay fairly close to the Clojure original, and we don't need to get sidetracked into a discussion of how transducers are typed.

## Reviewing Python's functional programming tools

Let's briefly review three key functional programming tools included
with Python, chiefly for the benefit of those people new to Python, and
specifically Python 3. We'll look at `map()`, `filter()` and
`reduce()` in turn.

### map()

First, the `map()` function:

```
>>> help(map)
Help on class map in module builtins:
class map(object)
 |  map(func, *iterables) --> map object
 |
 |  Make an iterator that computes the function using arguments from
 |  each of the iterables.  Stops when the shortest iterable is exhausted.
```
The `map()` function accepts a function - or more strictly in Python,
any callable object – as its first argument, and then any number of
iterable series. In Python an *iterable* is a sort of lazy sequence
which supports only forward iteration. For now, we'll pass the built-in
function `len` as the first argument, and we'll stick to using one
iterable, which will be a list of words, produced by calling `split()`
on a string object:

```
>>> map(len, "Compute the length of each word".split())
<map object at 0x102beaf28>
```
In fact, in Python 3, which we're using here, `map()` is lazy and
produces an iterator. To force evaluation, we'll pass this iterator to
the `list` constructor:

```
>>> list(map(len, "Compute the length of each word".split()))
[7, 3, 6, 2, 4, 4]
```
As expected, we get the length of each word.

### filter()

Secondly, we'll look at the `filter()` function:

```
>>> help(filter)
Help on class filter in module builtins:
class filter(object)
 |  filter(function or None, iterable) --> filter object
 |
 |  Return an iterator yielding those items of iterable for which function(item)
 |  is true. If function is None, return the items that are true.
```
The `filter()` function accepts a predicate function – a function
which returns `True` or `False` – and an iterable series. Each item
from the series is passed in turn to the predicate, and `filter()`
returns an iterator over a series which contains only those items from
the source series which match the predicate:

```
>>> list(filter(lambda w: 'o' in w,
... "the quick brown fox jumps over the lazy dog".split()))
['brown', 'fox', 'over', 'dog']
```
Neither `map()` nor `filter()` see much use in contemporary Python
owing to the capabilities of list comprehensions, which provide a more
natural syntax for mapping functions over, and filtering items from,
iterable series:

```
>>> [len(w) for w in "jackdaws love my big sphinx of quartz".split()
... if 'a' in w]
[8, 6]
```
### reduce()

The built-in `reduce()` function has had a troubled life [2]  in the history of Python, and in the
transition from Python 2 to Python 3, it was moved out of the language
proper into the Python standard library. In fact, Guido was ready to
throw `lambda` out with the bathwater!

Let's take a closer look at `reduce()`, which will set us firmly on
the road to understanding and implementing transducers:

```
>>> from functools import reduce
>>> help(reduce)
Help on built-in function reduce in module _functools:
reduce(...)
    reduce(function, sequence[, initial]) -> value
    Apply a function of two arguments cumulatively to the items of a sequence,
    from left to right, so as to reduce the sequence to a single value.
    For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
    ((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
    of the sequence in the calculation, and serves as a default when the
    sequence is empty.
```
When we execute reduce, we can see that it evaluates its arguments eagerly, immediately producing a result.

```
>>> reduce(lambda x, y: x+y, [1, 2, 3, 4, 5])
15
```
The most common uses of reduction in Python, such as summing a series of
items, have been replaced by the `sum()`, `any()` and `all()`
functions built into the language, so `reduce()` doesn't see much use
either.

We can also supply a initial item to be used in front of the supplied sequence, which is especially useful if the sequence is empty:

```
>>> reduce(lambda x, y: x+y, [1, 2, 3, 4, 5], 10)
25
>>> reduce(lambda x, y: x+y, [], 10)
10
>>> reduce(lambda x, y: x+y, [])
Traceback (most recent call last):
  File "", line 1, in
TypeError: reduce() of empty sequence with no initial value
>>>
```
Notice that with an empty series, the initial value becomes necessary to avoid an error.

It can sometimes be helpful to think of the initial value as being the
*identity* of an operation. This value depends on the operation itself;
for example, the additive identity is zero, whereas the multiplicative
identity is one.

You might be tempted into thinking that the initial value should be of
the same type as the values in the data series, but this isn't
necessarily so. For example, using the following function we can
accumulate a histogram into a dictionary item by item. In this case the
identity value is an empty dictionary (*i.e.* histogram), not an
integer:

```
>>> def count(hist, item):
...     if item not in hist:
...         hist[item] = 0
...     hist[item] += 1
...     return hist
...
>>> reduce(count, [1, 1, 5, 6, 5, 2, 1], {})
{1: 3, 2: 1, 5: 2, 6: 1}
```
Note that in production code, this use of `reduce()` with `count()`
would be much better handled using the `Counter` type [3].
from the Python Standard Library `collections` module!

Our lambda expression for adding two numbers and our `count()`
function serve as so-called "reducing functions", or simply 'reducers'
for short. In fact, this is what during this session we'll refer to any
function that can be passed to `reduce()` in this way.

Those of you familiar with functional programming techniques in other
languages will recognise Python's `reduce()` as being equivalent to
Haskell's `foldl`, Clojure's `reduce`, F#'s `reduce` or `fold`,
Ruby's `inject`, C++'s `std::accumulate()`, or the `Aggregate()`
extension method from .NET's LINQ.

### Reduce is more fundamental than map or filter

Reduce is interesting because it's is more fundamental than `map()` or
`filter()`. This is because we can implement `map()` and
`filter()` in terms of `reduce()`, but not vice-versa. To do so, we
just need to cook up the right reducing functions and supply a suitable
initial value. Here is `map()` implemented in terms of `reduce()`:

```
"""A module t.py in which we'll do live coding."""
from functools import reduce
def my_map(transform, iterable):
    def map_reducer(sequence, item):
        sequence.append(transform(item))
        return sequence
    return reduce(map_reducer, iterable, [])
```
Then in the REPL:

```
>>> my_map(str.upper, "Reduce can perform mapping too!".split())
['REDUCE', 'CAN', 'PERFORM', 'MAPPING', 'TOO!']
```
Similarly we can make an implementation of `filter()` in terms of
`reduce()`:

```
def my_filter(predicate, iterable):
    def filter_reducer(sequence, item):
        if predicate(item):
            sequence.append(item)
        return sequence
    return reduce(filter_reducer, iterable, [])
```
Then in the REPL:

```
>>> my_filter(lambda x: x % 2 != 0, [1, 2, 3, 4, 5, 6, 7, 8])
[1, 3, 5, 7]
```
### Reducers

In the two previous examples, the local functions `map_reducer()` and
`filter_reducer()` were used as the reducing functions with
`reduce()`. Our `my_map()` and `my_filter()` functions *reduce*
one collection to another, transforming values and collecting the
results together. For the time being, we'll stick with the notion that a
reducer is any operation you can pass to the `reduce()` function,
although we'll extend and refine this notion later.

Composability is perhaps the most important quality of any programming
abstraction. How composable are `my_map()` and `my_filter()`? Well,
of course, given a predicate function `is_prime()` and transforming
function `square()` defined like this:

```
from math import sqrt
def is_prime(x):
  if x < 2:
      return False
  for i in range(2, int(sqrt(x)) + 1):
      if x % i == 0:
          return False
  return True
def square(x):
    return x*x
```
We can compose this filtering expression,

```
>>> my_filter(is_prime, range(100))
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
67, 71, 73, 79, 83, 89, 97]
```
with a mapping expression using regular function composition:

```
>>> my_map(square, my_filter(is_prime, range(100)))
[4, 9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369, 1681, 1849,
2209, 2809, 3481, 3721, 4489, 5041, 5329, 6241, 6889, 7921, 9409]
```
One issue here, is that since `my_map()` and `my_filter()` are
implemented in terms of `reduce()` they are *eager* and so a full
intermediate list of elements is created. Notably, if we'd used the
built-in Python 3 versions of `map()` and `filter()` evaluation
would have been lazy, and consequently more memory efficient, so we've
lost an important quality here.

Furthermore, these sort of functional pipelines can get hard to read in a language like Python as the arguments to the outermost (last applied) function become ever more widely separated. Other "real" functional languages such as Haskell and F# have better support for partial function application and pipelining results through a chain of functions. To some extent, this can be solved in Python by using named intermediate results, although this trades off the readability problem against the difficulty of coming up with meaningful names for the intermediate values.

### Progressively abstracting reducers away from iterables

Notice that although `reduce()` abstracts away from some of the
details of the operation - specifically how the items are transformed
and combined, it is fundamentally coupled to the notion of iterable
series. Furthermore, our `my_map()` and `my_filter()` functions are
closely tied to the notion of mutable sequences, since they both call
`append()`:

```
my_filter(is_prime, my_map(prime_factors, range(32)))
```
Iterables in Python can be lazy, but our `my_filter()` and
`my_map()` functions eagerly return fully realised `list` sequences.

Let's return to our function definitions to see why:

```
def my_map(transform, iterable):
    def map_reducer(sequence, item):
        sequence.append(transform(item))
        return sequence
    return reduce(map_reducer, iterable, [])
def my_filter(predicate, iterable):
    def filter_reducer(sequence, item):
        if predicate(item):
            sequence.append(item)
        return sequence
    return reduce(filter_reducer, iterable, [])
```
The `my_map()` and `my_filter()` functions have quite a lot in
common:

- Both `my_map()` and`my_filter()` call`reduce()`
- Both `my_map()` and`my_filter()` supply a mutable sequence –
specifically an empty list – as the initial value to`reduce()`
- Both `map_reducer()` and`filter_reducer()` call`append()` ,
thereby expecting a mutable sequence

Let's progressively refactor these functions, extracting the common
elements as we go. We'll start by moving the use of `reduce()` out of
the functions completely. Instead we'll simply return the inner reducing
function object, which can then be used externally. Our outer
`my_map()` and `my_filter()` functions now become what are
essentially reducer factories; we'll reflect this in the code by
renaming them to `make_mapper()` and `make_filterer()`:

```
def make_mapper(transform):
    def map_reducer(sequence, item):
        sequence.append(transform(item))
        return sequence
    return map_reducer
def make_filterer(predicate):
    def filter_reducer(sequence, item):
        if predicate(item):
            sequence.append(item)
        return sequence
    return filter_reducer
```
These can be used with individually with `reduce()` like this:

```
>>> from t import *
>>> reduce(make_mapper(square), range(10), [])
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
>>> reduce(make_filterer(is_prime), range(100), [])
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
67, 71, 73, 79, 83, 89, 97]
```
Unfortunately, it's not possible to compose the mapper and filterer into
a single function which can be used with a single application of
`reduce()`. If we want to perform both mapping and filtering we must
still call `reduce()` twice.

We've pointed out that, both of our inner reducing functions depend on
their first argument supporting an `append()` method, so we *must*
pass a suitable mutable sequence type – such as a `list` – as the seed
item for `reduce()`. In this case, `append()` is just a tool for
combining an existing partial result (the list) with a new piece of data
(the new item).

Let's abstract away that dependency on sequences, by arranging to pass in this "combining function", which in the case of appending to a sequence, will look like this:

```
def appender(result, item):
    result.append(item)
    return result
```
Look carefully at our combining function and recall that 'reducing' is
just a synonymn for 'combining'. This innocuous function, which does
nothing more than wrap a method call and return the modified sequence,
is more interesting than it might first appear, because it too is a
reducing-function that takes an intermediate result (the sequence) and
the next piece of input (the item) and returns the next result for any
subsequent iterations. Our combining function `appender()` is itself a
reducer! We can prove this to ourselves by, using this roundabout way of
constructing a `list` from a `range`:

```
>>> from t import *
>>> reduce(appender, range(5), [])
[0, 1, 2, 3, 4]
```
Having identified `appender()` as a reducer, we'll add another layer
of abstraction, by using another layer of nested functions. The new
layer affords us the opportunity to pass in a reducing function, such as
`appender()`. We use a separate layer of inner functions, rather than
adding an argument to the outer function, because the point in our
client code at which we want to specify the `transform` or
`predicate` is different from the point at which we want to specify
the `reducer`:

We've made the following changes to `make_mapper()`:

```
def mapping(transform):
    def make_mapping_reducer(reducer):
        def map_reducer(result, item):
            return reducer(result, transform(item))
        return map_reducer
    return make_mapping_reducer
```
1. Renamed `make_mapper()` to`mapping()`
2. Added a new inner function *around*`map_reducer()` called`make_mapping_reducer()` . This function accepts the 'combining'
reducer and returns the`map_reducer` function.
3. Adjusted the outer `mapping()` function to return the`map_reducer` function.
4. Reworked the implementation of `map_reducer()` to be in terms of
the passed-in combining reducer instead of`append()` , thereby
breaking the dependency on sequence behaviour.
5. Renamed the `sequence` argument of`map_reducer` to`result` ,
since it contains the partial result and is no longer necessarily a
sequence.

And we've made the following changes to the following changes to
`make_filterer()`:

```
def filtering(predicate):
    def make_filtering_reducer(reducer):
        def filter_reducer(result, item):
            return reducer(result, item) if predicate(item) else result
        return filter_reducer
    return make_filtering_reducer
```
1. Renamed `make_filterer()` to`filtering()`
2. Added a new inner function *around*`filter_reducer()` called`make_filtering_reducer()` . This function accepts the combining
reducer and returns the`filter_reducer` function.
3. Adjusted the outer `filtering()` function to return the`make_filtering_reducer` function.
4. Renamed the `sequence` argument of`map_reducer` to`result` .
5. Reworked the implementation of `filter_reducer()` to be in terms of
the passed-in combining reducer instead of`append()` , thereby
breaking the dependency on sequence behaviour.
6. Used a conditional expression rather than a conditional statement.

### Finally: Transducers!

Recall that the word 'transducer' is a convenience (and contrivance) for
functions which "transform reducers". Look closely at our inner
functions `make_mapping_reducer()` and `make_filtering_reducer()`.
Both of these functions accept a *reducer* and return a *new reducer*
based on the one passed in. These inner functions are therefore our
*transducers*; let's rename them as such to make this completely clear:

```
def mapping(transform):
    def mapping_transducer(reducer):
        def map_reducer(result, item):
            return reducer(result, transform(item))
        return map_reducer
    return mapping_transducer
def filtering(predicate):
    def filtering_transducer(reducer):
        def filter_reducer(result, item):
            return reducer(result, item) if predicate(item) else result
        return filter_reducer
    return filtering_transducer
```
The transducers identified, the outermost functions `mapping()` and
`filtering()` are exposed for what they are: transducer factories.

The wonderful thing about such transducers is that they can be
*composed* in such a way that the operations proceed element by element,
without the need for intermediate sequences. This is achievable because
the combining reducer passed to a transducer needn't be our
`appender()` it could be *any* reducing function, with `appender()`
only being used at the "bottom" of the composite reducer to put place
final results into an actual data structure, such as a list. Transducer
composition allows us to chain multiple reducing operations together
into a single composite reducer.

Let's give our transducers a whirl, using named function arguments for clarity:

```
>>> from t import \*
>>> reduce(filtering(predicate=is_prime)(reducer=appender), range(20), [])
[2, 3, 5, 7, 11, 13, 17, 19]
```
Note that we pass the last reduction operation we want to perform on
each item (appending) as a reducer *to* the first reduction operation we
want to perform (primality testing). This is the *opposite* of normal
function composition.

Instead of `appender()` we can pass a different reducer - in this case
one produced by transducer made by the `mapping()` transducer factory:

```
>>> reduce(filtering(
... predicate=is_prime)(
... reducer=mapping(
... transform=square)(
... reducer=appender)),
... range(20),
... [])
[4, 9, 25, 49, 121, 169, 289, 361]
```
What's important to recognise here, is that we have composed the
filtering and mapping operations to produce the effects of both using a
single pass over the input sequence with a single application of
`reduce()`.

In the next article in this series, will look at improving on this effective, but somewhat clunky way of composing transducers.

| [1] | Transducers are Coming: The blog post introducing transducers to the world. | 

| [2] | The fate of reduce() in Python 3000, the Artima Developer posting in which Guido van Rossum explains his "reasons for dropping `lambda` ,`map()` and`filter()` " from what was to become Python 3. He goes on "I expect tons of disagreement in the feedback, all from ex-Lisp-or-Scheme folks". Ultimately though, the proposal was watered down. | 

| [3] | Python documentation for Counter |


## Article 2 of 8 — Improving Transducer Composition

*Mon 19 January 2015 — https://sixty-north.com/blog/improving-transducer-composition.html*

In the previous article in this series we
derived a Python implementation of transducers from first principles. We
finished by showing how transducers can be composed together using regular
function call application to give us a single composite reducer which can
perform many operations with a single pass of `reduce()`. Specifically, we
showed how to filter a range of integers using a primality testing predicate,
and then mapped a squaring function over the primes, to give prime squares:

```
>>> reduce(filtering(
...     predicate=is_prime)(
...         reducer=mapping(
...             transform=square)(
...                 reducer=appender)),
...     range(20),
...     [])
[4, 9, 25, 49, 121, 169, 289, 361]
```
Although this clearly works, composing transducers this way quickly becomes ungainly and the code certainly has a Lisp-ish flavour. Keeping track of the parentheses in Python, when we have function calls which return functions which we immediately call, is somewhat awkward. Most functional programming languages include a function called "compose" to help with composing functions; many imperative programming languages do not, including Python, so we'll have to write one:

```
def compose(f, *fs):
    """Compose functions right to left.
    compose(f, g, h)(x) -> f(g(h(x)))
    Args:
    f, *fs: The head and rest of a sequence of callables. The
        rightmost function passed can accept any arguments and
        the returned function will have the same signature as
        this last provided function. All preceding functions
        must be unary.
    Returns:
        The composition of the argument functions. The returned
        function will accept the same arguments as the rightmost
        passed in function.
    """
    rfs = list(chain([f], fs))
    rfs.reverse()
    def composed(*args, **kwargs):
        return reduce(
            lambda result, fn: fn(result),
            rfs[1:],
            rfs[0](*args, **kwargs))
    return composed
```
The signature of `compose()` forces us to accept at least one
argument. The first part of the function reassembles the supplied
arguments into a single list and reverses it to put it in first-to-last
application order. We then define the inner `composed()` function
which uses `reduce()` over the list of functions to apply each in
turn, carrying the intermediate results forward. Any arguments to the
`composed()` function are forwarded to the first function in the
chain.

With `compose()` in our armoury, it becomes somewhat easier to specify
multi-step composite reductions using transducers:

```
>>> reduce(compose(filtering(is_prime), mapping(square))(appender), range(20), [])
[4, 9, 25, 49, 121, 169, 289, 361]
```
This becomes even clearer if we put in some line breaks:

```
>>> reduce(compose(filtering(is_prime), mapping(square)) (appender), range(20), [])
[4, 9, 25, 49, 121, 169, 289, 361]
```
Now it's hopefully easier to see that the call to `compose()` produces
a new transducer to which we pass the `appender` reducer to get a
composite reducer which performs filtering, mapping and appending in one
step. It is this composite reducer we pass to `reduce()` along with
the input range and the empty list seed value.

In the next article in our series on transducers, we'll look at fixing some of the not-so-obvious shortcomings of the previous snippet and bringing the capabilities of our Python transducers more in line with those of the Clojure originals.


## Article 3 of 8 — Enriching the Transducer Protocol

*Mon 19 January 2015 — https://sixty-north.com/blog/enriching-the-transducer-protocol.html*

In the previous article
in the series we looked at improving the experience of composing
transducers together in Python, by introducing a `compose()`
function. We finished by showing this snippet, which composes a
filtering transducer with a mapping transducer to produce a
prime-squaring transducer. Recalling that transducers are used to
*trans*form-re*ducers*, we pass an appending reducer to the
prime-squaring transducer to get a prime-squaring-appending *reducer*.
This is passed in turn to `reduce()`, along with an input range and
an empty seed `list` for the result:

```
>>> reduce(compose(filtering(is_prime),
...                mapping(square))
...        (appender), # appender assumes a MUTABLE-sequence
...        range(20),
...        []) # list is a MUTABLE sequence
[4, 9, 25, 49, 121, 169, 289, 361]
```
And therein lies the rub. There's a fairly well disguised implicit
dependency here, between the empty list we've passed as the initial
value for the reduction and our passing of `appender()` as the
ultimate reducer. We can illustrate this by passing an *immutable*
sequence type, which doesn't support `append()`, rather than a mutable
sequence type, which does. Look what happens if we pass in an empty
`tuple` instead of an empty list:

```
>>> reduce(compose(filtering(is_prime),
...                mapping(square))
...        (appender), # appender assumes a MUTABLE-sequence
...        range(20),
...        tuple()) # tuple is an IMMUTABLE sequence
Traceback (most recent call last):
  File "", line 1, in
  File "", line 4, in filter_reducer
  File "", line 4, in map_reducer
  File "", line 2, in appender
AttributeError: 'tuple' object has no attribute 'append'
```
We can "fix" this by passing another reducer, rather than `appender`,
called `conjoiner` [1]:

```
def conjoiner(result, item):
    return result + type(result)((item,))
```
which we can use like this:

```
>>> reduce(compose(filtering(is_prime),
...                mapping(square))
...        (conjoiner), # conjoiner assumes an IMMUTABLE-sequence
...        range(20),
...        tuple()) # tuple is an IMMUTABLE sequence
(4, 9, 25, 49, 121, 169, 289, 361)
```
Notice that the result is now enclosed in parentheses rather than square
brackets, indicating that it is a `tuple`.

In order to address the fact that the ultimate reducer and the seed value will often need to change in sympathy, meaning that if one changes we need to remember to change the other, we'll to enrich the transducer interface. It will got from being a simple function call, to something that is at once more complex and more capable. To understand what those complexities are, we'll refer back to the Clojure archetype.

## Examining the Clojure original

Our code has a very general form, but it is lacking features of the
Clojure original, such as early termination of the reduction process.
Let's look at the Clojure original for `map` [2] when called with a single argument:

```
(defn map
  ([f]
  (fn [rf]
    (fn
      ([] (rf))
      ([result] (rf result))
      ([result input]
        (rf result (f input)))
      ([result input & inputs]
        (rf result (apply f input inputs))))))
```
This may not be very clear - even if you can read Clojure! - so let's annotate it with some comments:

```
(defn map ;; The tranducer factory...
  ([f] ;; ...accepts a single argument 'f', the
  ;; transforming function
  (fn [rf] ;; The transducer function accepts a
    ;; reducing function 'rf'
    (fn ;; This is the reducing function returned
      ;; by the transducer
      ([] (rf)) ;; 0-arity : Forward to the zero-arity
      ;; reducing function 'rf'
      ([result] (rf result)) ;; 1-arity : Forward to the one-arity
      ;; reducing function 'rf'
      ([result input] ;; 2-arity : Perform the reduction with
        ;; one arg to 'f'
        (rf result (f input)))
      ([result input & inputs] ;; n-arity : Perform the reduction with
        ;; multiple args to 'f'
        (rf result (apply f input inputs))))))
```
Here's our reducing function in Python, which only implements the equivalent of the 2-arity version which performs the actual reduction:

```
def map_reducer(result, item):
    return reducer(result, transform(item))
```
The Clojure definitions of the zero- and one-arity reduction functions don't provide much clue as to what they are for - they're just contract preserving functions which forward from the 'new' reducer to the underlying reducer which has been wrapped by it.

In fact, the zero-arity function is called to produce the initial seed value
when one isn't provided. For example, for addition the seed needs to be zero
[3], for multiplication the seed needs to be one [4] , and in our Python
examples for appending the seed should be the empty list, and for conjoining the
seed should be an empty tuple. The `map` reducer simply delegates this to
the underlying reducer, since it can't know – and indeed shouldn't know – upon
which kind of data structure it is operating.

The one-arity function, which accepts only the intermediate result and no further input is used to perform transducer chain clean-up or reduction to a final result when processing of the sequence is complete or terminated early. This is useful for certain stateful transducers which need to deal with any left-over state. We'll look at some examples later.

So to document our improved understanding in comments:

```
(defn map ;; The tranducer factory...
  ([f] ;; ...accepts a single argument 'f', the
  ;; transforming function
  (fn [rf] ;; The transducer function accepts a
    ;; reducing function 'rf'
    (fn ;; This is the reducing function returned
      ;; by the transducer
      ([] (rf)) ;; 0-arity : Return a 'seed' value
      ;; obtained from 'rf'
      ([result] (rf result)) ;; 1-arity : Obtain final result from 'rf'
      ;; and clean-up
      ([result input] ;; 2-arity : Perform the reduction with
        ;; one arg to 'f'
        (rf result (f input)))
      ([result input & inputs] ;; n-arity : Perform the reduction with
        ;; multiple args to 'f'
        (rf result (apply f input inputs))))))
```
In fact, to fully implement the concepts inherent in Clojure reducers and transducers we need to do more work in our Python version to support:

1. Explicit (although optional) association of the seed value with the reduction operation
2. Early termination of reduction processes. For example, a search can terminate early without needing to reducing a whole series
3. Reduction to a final value and opportunity to clean-up left-over state

Clojure supports these distinct behaviours through different arity versions of the same anonymous reducing function. Python doesn't support overloading on arity, and in any case, overloading on arity in order to support different operations can seem obtuse. [5] We have a perfectly good tool for bundling related named functions together in Python, and that tool is the class.

In the next phase, we'll convert our reducing functions into classes and
necessarily replace our use of the Python Standard Library `reduce()`
function with something a little more sophisticated which can support
our new class-based reducers.

In Python, the conceptual interface to a reducer, will look like this:

```
class Reducer:
    def __init__(self, reducer): # Construct from reducing function
        pass
    def initial(self): # Return the initial seed value
        pass # 0-arity
    def step(self, result, item): # Next step in the reduction
        pass # 2-arity
    def complete(self, result): # Produce a final result and clean up
        pass # 1-arity
```
Notice that the `__init__()` function – and therefore the class – is a
*transducer*. It accepts a reducer and returns a reducer!

```
new_reducer = Reducer(reducer)
```
It takes a particularly clear-minded and highly-caffeinated state to appreciate that the class is a transducer but instances of the class are reducers! In fact, we've found it so confusing, that we generally wrap the constructor call in another function with a more appropriate name:

```
def transducer(reducer):
    return Reducer(reducer)
```
More concretely, here is our `mapping()` transducer factory, the
transducing function and the reducer it creates:

```
def mapping(transform):
    def mapping_transducer(reducer):
        return Mapping(reducer, transform)
    return mapping_transducer
```
Let's implement our `Mapping` reducer cum transducer class:

```
class Mapping:
    def __init__(self, reducer, transform):
        self._reducer = reducer
        self._transform = transform
    def initial(self):
        return self._reducer.initial()
    def step(self, result, item):
        return self._reducer.step(result, self._transform(item))
    def complete(self, result):
        return self._reducer.complete(result)
```
In the absence of any necessary behaviours specific to a particular
reduction algorithm, the `initial()`, `step()` and `complete()`
methods simply forward to the next reducer in the chain
(`self._reducer`). The only behaviour here specialised for `Mapping`
is to apply `self._transform()` to the item before passing the result
down the chain.

And here's our filtering transducer-factory together with the
`Filtering` reducer cum transducer:

```
class Filtering:
    def __init__(self, reducer, predicate):
        self._reducer = reducer
        self._predicate = predicate
    def initial(self):
        return self._reducer.initial()
    def step(self, result, item):
        return self._reducer.step(result, item) if self._predicate(item)
        else result
    def complete(self, result):
        return self._reducer.complete(result)
def filtering(predicate):
    def filtering_transducer(reducer):
        return Filtering(reducer, predicate)
    return filtering_transducer
```
To allow the chain of reducers produced by our transducers to terminate
in a regular reducer, such as `appending`, we'll replace our
`appending` and `conjoining` reducing functions with classes which
sport the same interface as our other reducers:

```
class Appending:
    def initial(self):
        return []
    def step(self, result, item):
        result.append(item)
        return result
    def complete(self, result):
        return result
class Conjoining:
    def initial(self):
        return tuple()
    def step(self, result, item):
        return result + type(result)((item,))
    def complete(self, result):
        return result
```
These two reducing classes have no internal state, and hence no need for
initialisation functions, but crucially, we use the ability afforded by
the `initial()` method to associate a seed value with the reducing
operation. [[[Being stateless, we could have decorated the methods of
these reducers with `@staticmethod`; we haven't done so though, to
avoid detracting from the important similarity between our reducer and
transducer classes.]]]

To make use of our class-based transducers, we need an alternative to
`reduce()` which understands our new transducer/reducer protocol.
Following Clojure's lead, we will call it `transduce()`:

```
UNSET = object()
def transduce(transducer, reducer, iterable, init=UNSET):
    r = transducer(reducer)
    accumulator = init if (init is not UNSET) else r.initial()
    for item in iterable:
        accumulator = r.step(accumulator, item)
    return r.complete(accumulator)
```
We supply the `reducer` separately, rather than bundling it up inside
the `transducer` object, because it contains the knowledge of how to
accumulate the final result. Excluding that from our transducer
definition, allows us to keep our transducer more general and reusable
without committing to a particular result representation. For example,
we might compose a complex transducer and want to keep that separate
from whether the final result is accumulated in a `list` or in a
`tuple`.

Let's try to use our new `transduce()` function to apply a transducer
to a list of numbers. We'll do this step-by-step to keep things clear.
First we'll compose the transducer from a filtering and and mapping:

```
>>> square_primes_transducer = compose(
...     filtering(is_prime),
...     mapping(square))
```
Then we'll construct the reducer which will accumulate the final result.
We want a `list`, so we'll use `Appending`:

```
>>> appending_reducer = Appending()
```
Now we'll pass these to `transduce()`:

```
>>> transduce(square_primes_transducer, appending_reducer, range(100))
[4, 9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369, 1681, 1849,
2209, 2809, 3481, 3721, 4489, 5041, 5329, 6241, 6889, 7921, 9409]
```
*Et voila!*

By using `transduce()` and enriching our notion of what a reducer
looks like, we no longer need to separately specify the seed value. If
we want a `tuple`, we can use a different reducer:

```
>>> conjoining_reducer = Conjoining()
>>> transduce(square_primes_transducer, conjoining_reducer, range(100))
(4, 9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369, 1681, 1849,
2209, 2809, 3481, 3721, 4489, 5041, 5329, 6241, 6889, 7921, 9409)
```
This decoupling of the transducer processing pipeline from the result type may not seem important in this example, but as we see later, it buys us a great deal of flexibility and re-use potential.

In the next article, we'll look at stateful transducers, and how having our transducers implemented as classes makes this particularly straightforward.

| [1] | conjoin *verb* To join together. There is also an equivalent Clojure function`conj` , and Clojure/conj is a Clojure programming conference. | 

| [2] | The definition of the mapping transducer factory source code on Github. | 

| [3] | We say the additive identity is zero. | 

| [4] | We say the multiplicative identity is one. | 

| [5] | It seems I'm not the only person who found Clojure's use of overloading by arity an impediment to understanding transducers. In fact, overloading by arity is incidental to the concept of transducers, and a curiosity of the Clojure archetype. |


## Article 4 of 8 — Stateful Transducers

*Mon 19 January 2015 — https://sixty-north.com/blog/stateful-transducers.html*

In the previous article in
this series on transducers we saw how we can develop the notion of the
transducer from a single function which literally *transforms reducers* to a
more capable protocol which supports two further capabilities: First of all, the
association of initial 'seed' values with a reduction operation, and secondly
the opportunity for cleanup for stateful transducers. So far, we've exercised
the first capability, but not the second. To demonstrate clean-up, we need to
introduce stateful transducers.

The mapping and filtering transducers we have seen so far are stateless.
What this means is that the result for the current item being processed
depends only on the values of the result accumulated so far and the new
item. We can, however, make stateful transducers, and the fact that our
Python transducers are classes makes this particularly easy, because it
gives us an obvious place to store the state, in instances of those
classes. Perhaps the simplest example is an *enumerating* transducer
which keeps track of item indexes and accumulates (index, item)
`tuple` pairs into the result:

```
class Enumerating:
    def __init__(self, reducer, start):
        self._reducer = reducer
        self._counter = start
    def initial(self):
        return self._reducer.initial()
    def step(self, result, item):
        index = self._counter
        self._counter += 1
        return self._reducer.step(result, (index, item))
    def complete(self, result):
        return self._reducer.complete(result)
    def enumerating(start=0):
        """Create a transducer which enumerates items."""
        def enumerating_transducer(reducer):
            return Enumerating(reducer, start)
        return enumerating_transducer
```
We'll use this by composing it onto the end of our existing transducer chain:

```
>>> square_primes_transducer = compose(
...     filtering(is_prime),
...     mapping(square))
>>>
>>> enumerated_square_primes_transducer = compose(
...     square_primes_transducer,
...     enumerating())
>>>
>>> appending_reducer = Appending()
>>>
>>> transduce(enumerated_square_primes_transducer,
...     appending_reducer,
...     range(100))
[(0, 4), (1, 9), (2, 25), (3, 49), (4, 121), (5, 169), (6, 289),
(7, 361), (8, 529), (9, 841), (10, 961), (11, 1369), (12, 1681),
(13, 1849), (14, 2209), (15, 2809), (16, 3481), (17, 3721),
(18, 4489), (19, 5041), (20, 5329), (21, 6241), (22, 6889),
(23, 7921), (24, 9409)]
```
## Cleaning up left-over state

So far, the implementations of the `complete()` method in our
transducers haven't been very interesting. They've simply delegated the
call to next reducer in the chain. At the end of the chain, the
`complete()` implementations of the `Appending` or `Conjoining`
reducers simply return whatever was passed to them.

Sometimes, the state accumulated within the transducer needs to be
returned as part of the final result. For example, consider a *batching*
transducer which collects successive items together into non-overlapping
groups of a specified size. The transducer maintains a pending batch as
internal state, and when the batch has grown to the requisite size,
accumulates it into the result. When we reach the end of the input data,
there may be a partial batch. If our design calls for returning the
partial batch, we need a way to detect the end of processing and deal
with any internal state. This is where the `complete()` method comes
into play. Here's our batching transducer and its corresponding
transducer factory:

```
class Batching:
    def __init__(self, reducer, size):
        self._reducer = reducer
        self._size = size
        self._pending = []
    def initial(self):
        return self._reducer.initial()
    def step(self, result, item):
        self._pending.append(item)
        if len(self._pending) == self._size:
            batch = self._pending
            self._pending = []
            return self._reducer.step(result, batch)
        return result
def complete(self, result):
    r = self._reducer.step(result, self._pending) if len(self._pending) > 0 else result
    return self._reducer.complete(r)
def batching(size):
    """Create a transducer which produces non-overlapping batches."""
    if size < 1:
        raise ValueError("batching() size must be at least 1")
    def batching_transducer(reducer):
        return Batching(reducer, size)
    return batching_transducer
```
Here we see that the complete method, calls `step()` on the underlying
reducer one more time to pass on the partial batch. Here it is in
action:

```
>>> batched_primes_transducer = compose(filtering(is_prime), batching(3))
>>> transduce(batched_primes_transducer, Appending(), range(100))
[[2, 3, 5], [7, 11, 13], [17, 19, 23], [29, 31, 37], [41, 43, 47],
[53, 59, 61], [67, 71, 73], [79, 83, 89], [97]]
```
Notice in particular the partial batch included at the end.

With stateful transducers and special handling of result completion and clean-up in place, in the next article we'll look at how to signal and detect early termination of a reduction operation, such as occurs when searching for and finding an item in a data series.


## Article 5 of 8 — Terminating Transducers

*Mon 19 January 2015 — https://sixty-north.com/blog/terminating-transducers.html*

In the previous article in this series
on transducers, we showed how to implement stateful transducers, and how to deal
with any left-over state or other clean-up operations when the reduction
operation is complete. Sometimes, however, there is no need to process a whole
series of items in order to produce the final result. For example, if we just
want the first item from a series, we only need to process the first item. Our
existing implementation of `transduce()` looks like this:

```
UNSET = object()
def transduce(transducer, reducer, iterable, init=UNSET):
    r = transducer(reducer)
    accumulator = init if (init is not UNSET) else reducer.initial()
    for item in iterable:
        accumulator = r.step(accumulator, item)
    return r.complete(accumulator)
```
It uses a for-loop to iteratively apply the reducing function, but there is no way of exiting the loop early.

To accommodate early termination, we'll allow our `step()` method to
return a sentinel value indicating that reduction is complete. This
sentinel will actually be a 'box' called called `Reduced` which we can
detect by type, and which will contain the final value:

```
class Reduced:
    """A sentinel 'box' used to return the final value of a reduction."""
    def __init__(self, value):
        self._value = value
    @property
    def value(self):
        return self._value
```
It has only an initialiser which accepts a single value, and a property to provide read-only access to that value.

Our updated `Reduced` detecting `transduce()` function looks like
this:

```
def transduce(transducer, reducer, iterable, init=UNSET):
    r = transducer(reducer)
    accumulator = init if (init is not UNSET) else reducer.initial()
    for item in iterable:
        accumulator = r.step(accumulator, item)
        if isinstance(accumulator, Reduced):
            accumulator = accumulator.value
            break
    return r.complete(accumulator)
```
When we detect `Reduced` we simply un-box its value and then break
from the loop.

Our `First` transducer, which accepts an optional predicate looks like
this:

```
class First:
    def __init__(self, reducer, predicate):
        self._reducer = reducer
        self._predicate = predicate
    def initial(self):
        return self._reducer.initial()
    def step(self, result, item):
        return Reduced(self._reducer.step(result, item)) if self._predicate(item) else result
    def complete(self, result):
        return self._reducer.complete(result)
    def first(predicate=None):
        predicate = true if predicate is None else predicate
        def first_transducer(reducer):
            return First(reducer, predicate)
        return first_transducer
```
Notice that `true` here can be passed to the transducer constructor in
lieu of the predicate function being supplied; this isn't the Python
constant `True` but a function which always returns `True`,
irrespective of what it is passed. We need to define ourselves:

```
def true(*args, **kwargs):
    return True
```
Putting it all together, we get:

```
>>> transduce(
...   compose(
...     filtering(is_prime),
...     mapping(square),
...     first(lambda x: x > 1000)),
...   Appending(),
...   range(1000))
[1369]
```
Notice that single result is returned as the only element of a list. This is
because we used `Appending` as our reducer. If we'd prefer a scalar value to
be returned, we can simply define a new reducer called `ExpectingSingle` that
only expects exactly one `step()` operation to be performed:

```
class ExpectingSingle:
    def __init__(self):
        self._num_steps = 0
    def initial(self):
        return None
    def step(self, result, item):
        assert result is None
        self._num_steps += 1
        if self._num_steps > 1:
            raise RuntimeError("Too many steps!")
        return item
    def complete(self, result):
        if self._num_steps < 1:
            raise RuntimeError("Too few steps!")
        return result
```
Reattempting our example, we now get a scalar value:

```
>>> transduce(
...   compose(
...     filtering(is_prime),
...     mapping(square),
...     first(lambda x: x > 1000)),
...   ExpectingSingle(),
...   range(1000))
1369
```
We've now exercised all the parts of the transducer protocol:

- Association of the initial value through `initial()`
- Incremental reduction through `step()`
- Completion and clean-up of state through `complete()`
- Signalling early completion with `Reduced()`

In the next article, we'll show how this protocol allows transducers to produce more items than they consume, which may be obvious, be is an important case to be handled when we implement lazy transduction in a future article.


## Article 6 of 8 — Item Injecting Transducers

*Mon 19 January 2015 — https://sixty-north.com/blog/item-injecting-transducers.html*

In the previous article in our series on understanding transducers through Python we showed how to support early termination of a reduction operation. This time, we'll demonstrate how transducers can produce more items than they consume. Although this may seem obvious, it leads to some important consequences for implementing lazy evaluation of transducers, which is what we'll look at next time.

Consider a transducer `Repeating` which repeats each source item
multiple times into the output:

```
class Repeating:
    def __init__(self, reducer, num_times):
        self._reducer = reducer
        self._num_times = num_times
    def initial(self):
        return self._reducer.initial()
    def step(self, result, item):
        for _ in range(self._num_times):
            result = self._reducer.step(result, item)
        return result
    def complete(self, result):
        return self._reducer.complete(result)
    def repeating(num_times):
        if num_times < 0:
            raise ValueError("num_times cannot be negative")
        def repeating_transducer(reducer):
            return Repeating(reducer, num_times)
        return repeating_transducer
```
The key point to notice here, is that each call to `Repeating.step()`
results in multiple calls to the underlying reducer's
`self._reducer.step()`, thereby injecting more items into the output
series than are received in the input series.

By composing it with our filtering primality checking predicate, we can use it to repeat each prime number three times:

```
>>> primes_repeating = compose(filtering(is_prime), repeating(3))
>>> transduce(primes_repeating, Appending(), range(100))
[2, 2, 2, 3, 3, 3, 5, 5, 5, 7, 7, 7, 11, 11, 11, 13, 13, 13, 17, 17,
 17, 19, 19, 19, 23, 23, 23, 29, 29, 29, 31, 31, 31, 37, 37, 37, 41,
 41, 41, 43, 43, 43, 47, 47, 47, 53, 53, 53, 59, 59, 59, 61, 61, 61,
 67, 67, 67, 71, 71, 71, 73, 73, 73, 79, 79, 79, 83, 83, 83, 89, 89,
 89, 97, 97, 97]
```
In the next article, we'll see
that although seemingly fairly innocuous, support for item injecting transducers
such as `Repeating` complicates lazy evaluation quite a bit!


## Article 7 of 8 — Lazy Transducer Evaluation

*Tue 20 January 2015 — https://sixty-north.com/blog/lazy-transducer-evaluation.html*

In the previous article in this series on transducers we looked at transducers which push more items downstream through the reducer chain than they receive from upstream. We promised that this would make lazy evaluation of transducer chains quite interesting.

When used with our `transduce()` function, our mapping and filtering
transducers are in some ways less flexible than the `map()` and
`filter()` functions built into Python 3 because our `transduce()`
eagerly evaluates the reduction operation, whereas the built-in
`map()` and `filter()` are lazy. [1]

The eagerness of our mapping and filtering transducers is not inherent
in their implementation though. The eagerness is a result of the
for-loop in `transduce()` which must run to completion before
returning. Thankfully, due to the clear separation of concerns between
the reduction algorithm embodied in the transducers and the transducer
"driver", we can design an alternative transducible process which is
lazy.

Here's a reminder of our non-lazy `transduce()` function:

```
UNSET = object()
def transduce(transducer, reducer, iterable, init=UNSET):
    r = transducer(reducer)
    accumulator = init if (init is not UNSET) else reducer.initial()
    for item in iterable:
        accumulator = r.step(accumulator, item)
        if isinstance(accumulator, Reduced):
            accumulator = accumulator.value
            break
    return r.complete(accumulator)
```
Recall that our non-lazy `transduce()` function accepts, in addition
to the transducer, a separate `reducer` argument which is used to
collect the results of applying the transducer into, say, a `list`.
Our lazy transduction function will be implemented as a Python generator
function which *yields* each result as it becomes available, returning
control to the caller, and then resumes execution when the next value is
requested.

In order to handle early terminating transducers such as `First`,
stateful transducers which emit left-over state such as `Batching`,
and transducers which emit more elements than they consume such as
`Repeating`, the `lazy_transduce()` function is necessarily quite
complex:

```
from collections import deque
def lazy_transduce(transducer, iterable):
    """Lazy application of a transducer to an iterable."""
    r = transducer(Appending())
    accumulator = deque()
    reduced = False
    for item in iterable:
        accumulator = r.step(accumulator, item)
        if isinstance(accumulator, Reduced):
            accumulator = accumulator.value
            reduced = True
        yield from all_pending_items_in(accumulator)
        if reduced:
            break
    left_overs = r.complete(accumulator)
    assert left_overs is accumulator
    yield from all_pending_item_in(left_overs)
def all_pending_items_in(queue):
    while queue:
        yield queue.popleft()
```
Our function accepts only a transducer and the iterable series of source items.
There's no need to provide a reducer, because this function hardwires it's own
on the first line, where we provide an `Appending` reducer. Notice that unlike
the eager `transduce()` we never call the `Appending.initial()` method to
retrieve the seed value for the reduction, so we must provide a legitimate
mutable sequence type. For reasons that will become clear shortly, we provide a
`deque` from the Python Standard Library `collections` module [2] - a
double-ended queue, which supports `append()` to push items into the
right-hand end.

We also set a flag `reduced` so we know when we're finished.

The first part of the body of the for-loop is the same as for eager
`transduce()`: we step the transducer, accumulating each item, looking
for the sentinel `Reduced` value as we go. If we encounter `Reduced`
we un-box its contents and set the `reduced` flag to signal that we're
(nearly) done.

The next part of the for-loop body is where things really diverge from
the eager `transduce()` version. Bearing in mind that the call to
`step()` may have appended multiple items to the accumulator, we now
need to yield them one by-one to the client. We do this using the
`yield from` statement which delegates to another generator function
`all_items_pending_in()` which simply keeps yielding items from the
queue until it is empty.

At the end of the for-loop, we check the `reduced` flag, and `break`
out of the loop if we're done.

After the loop, with all the input items dealt with, we make the
necessary call to `complete()`, bearing in mind that this may append
further results to the accumulator queue. After a sanity check that the
return value from `complete()` is indeed the queue (which we know it
should be, because `Appending.complete()` simply returns its argument)
we use the `yield from all_pending_items_in(left_overs)` statement one
last time to yield any lingering results to the client.

In order to demonstrate the laziness in action, we'll create a little
wrapper around the built-in `range()` sequence that logs the yielded
integers to the console:

```
def logging_range(n):
    for i in range(n):
        print("i =", i)
        yield i
```
Here it in in action, demonstrating it's laziness:

```
>>> primes_repeating = compose(filtering(is_prime), repeating(3))
>>> repeated_primes = lazy_transduce(primes_repeating, logging_range(100))
>>> repeated_primes
>>> next(repeated_primes)
i = 0
i = 1
i = 2
2
>>> next(repeated_primes)
2
>>> next(repeated_primes)
2
>>> next(repeated_primes)
i = 3
3
>>> next(repeated_primes)
3
>>> next(repeated_primes)
3
>>> next(repeated_primes)
i = 4
i = 5
5
>>> next(repeated_primes)
5
>>> next(repeated_primes)
5
>>> next(repeated_primes)
i = 6
i = 7
7
>>> next(repeated_primes)
7
>>> next(repeated_primes)
7
>>> next(repeated_primes)
i = 8
i = 9
i = 10
i = 11
11
>>> next(repeated_primes)
11
>>> next(repeated_primes)
11
>>> next(repeated_primes)
i = 12
i = 13
13
```
So we see that transducers allow orthogonal specification of the reducing operation, the result collection and whether to evaluate eagerly or lazily. Neat!

In a future article we'll look at using transducers to process 'push' events modelled by Python coroutines.

| [1] | Back in Python 2 `map()` and`filter()` were eager. | 

| [2] | The documentation for the Python collections.deque double-ended queue. |


## Article 8 of 8 — Event Processing with Transducers

*Fri 27 March 2015 — https://sixty-north.com/blog/event-processing-with-transducers.html*

In the previous article in this
series on transducers we looked at lazily evaluating transducers. This time
we'll look not at *pulling* output through a transducer chain from downstream,
but at *pushing* input items into the chain from upstream.

All of the uses of transducers we've demonstrated in Python so far are probably better handled by existing and well established Python programming techniques, such as generator expressions and generator functions. At this point in the series, we move definitely beyond that into new territory where transducers bring completely new capabilities to Python.

One the key selling points of transducers is that they abstract the essence of a transformation away from the details of the data series that is being transformed. We'll show this in Python by using transducers to transform a series of events modelled using Python coroutines.

## Coroutines in Python

Coroutines in Python are little-used, and their workings are not widely known, so their implementation bears repeating here. If you're familiar with the notion of coroutines in general, and the specifics of how they're implemented in Python, you can skim over this section.

Coroutines are like generator functions insofar as they are *resumable*
functions. In fact, coroutines in Python *are* generator functions which
use `yield` as an expression rather than a statement. What this means
in practice is that generator function objects sport a `send()` method
which allows the client of the generator function to transmit
information to a running generator and for the generator to receive this
data as the value of the `yield` expression. As usual, an example will
serve to make things clearer.

We'll start by defining a generator function which enters an infinite
loop, waits at the `yield` expression for a value to be received, and
then prints this value to the console.

```
>>> def event_receiver():
...     while True:
...         message = (yield)
...         print("Message:", message)
...
>>>
```
We create a generator object just the same as we would with any other generator:

```
>>> r = event_receiver()
>>> r
```
Now we'll try to send it a message, using the `send()` method of the
generator object:

```
>>> r.send("message")
Traceback (most recent call last):
  File "", line 1, in
TypeError: can't send non-None value to a just-started generator
>>>
```
This actually fails, because the generator code has not yet been
executed at all. We need to prime the pump, so to speak, by advancing
execution to the first occurrence of `yield`. We can do this by
passing the generator to the `next()` built-in:

```
>>> next(r)
```
We'll fix this pump-priming annoyance of generator based coroutines shortly.

Now we can send messages:

```
>>> r.send("message")
Message: message
>>> r.send("another message")
Message: another message
```
When we're done, we terminate the coroutine by calling the `close()`
method. (This actually raises a `GeneratorExit` exception at the site
of the `yield` expression, which allows control flow to exit the
otherwise infinite loop; this special exception is intercepted by the
Python runtime system, so it isn't seen by us at the console).

```
>>> r.close()
>>>
```
Any further attempts to `send()` messages into the generator function
cause `StopIteration` to be raised. This, of course, is the normal
means of indicating that a generator is exhausted:

```
>>> r.send("message")
Traceback (most recent call last):
  File "", line 1, in
StopIteration
>>>
```
### Priming generator-based coroutines

Now to address the awkwardness of having to prime coroutine generator
functions by initially passing them to `next()`. We can improve this
with a function decorator which creates the generator object and calls
next on our behalf. We'll call the decorator `@coroutine`:

```
def coroutine(func):
    def start(*args, **kwargs):
        g = func(*args, **kwargs)
        next(g)
        return g
    return start
```
We'll use our new decorator to assist in defining a slightly more
sophisticated coroutine for printing, called `rprint()`:

```
import sys
@coroutine
def rprint(sep='\n', end=''):
    """A coroutine sink which prints received items to stdout
    Args:
      sep: Optional separator to be printed between received items.
      end: Optional terminator to be printed after the last item.
    """
    try:
        first_item = (yield)
        sys.stdout.write(str(first_item))
        sys.stdout.flush()
        while True:
            item = (yield)
            sys.stdout.write(sep)
            sys.stdout.write(str((item)))
            sys.stdout.flush()
    except GeneratorExit:
        sys.stdout.write(end)
        sys.stdout.flush()
```
In this implementation, we intercept `GeneratorExit` explicitly to
give us the opportunity to print a terminator. We also regularly flush
the stream so we get immediate feedback for our following experiments.

### Event sources

The opposite of a sink is a source. Until now, we've been sourcing
'events' ourself by sending them from the REPL, but to make this a
little more interesting, we'll cook up a function – just a plain old
function, not a generator – which takes values from an iterable series
and intermittently sends them, after a delay, to anything with a
`send()` method such as our coroutine generators. For fun, the random
delay has a so-called Poisson distribution which mimics a radioactive
source; imagine a device with a geiger counter which sends the next item
from an iterable series each time an atom decays:

```
def poisson_source(rate, iterable, target):
    """Send events at random times with uniform probability.
    Args:
      rate: The average number of events to send per second.
      iterable: A series of items which will be sent to the target
          one by one.
      target: The target coroutine or sink.
    Returns:
        The completed value, or None if iterable was exhausted
        and the target was closed.
    """
    for item in iterable:
        duration = random.expovariate(rate)
        sleep(duration)
        try:
            target.send(item)
        except StopIteration as e:
            return e.value
    target.close()
    return None
```
When either the iterable series is exhausted or the target signals it
has terminated (by raising `StopIteration`) we call `close()` on the
target. Note that by supplying an infinite iterable series we could make
the source send events forever.

Let's hook our source and sink together at the REPL:

```
>>> printer = rprint(sep=', ', end='\nDONE!\n')
>>> count_to_nine = range(10)
>>> poisson_source(rate=0.5, iterable=count_to_nine, target=printer)
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
DONE!
>>>
```
### Combined event sources and event sinks

Of course, we can build functions which act as both sinks *and* sources,
transforming the messages they receive in some way and forwarding the
processed results onwards to another sink. Here's a combined source and
sink function, which simply doubles the values it receives:

```
@coroutine
def doubler(target):
    while True:
        item = (yield)
        doubled_item = item * 2
        try:
            target.send(doubled_item)
        except StopIteration as e:
            return e.value
```
To use, `doubler()` we chain the components of the pipeline together

```
>>> printer = rprint(sep=', ', end='\nDONE!\n')
>>> count_to_nine = range(10)
>>> poisson_source(rate=0.5,
...                iterable=count_to_nine,
...                target=doubler(target=printer))
0, 2, 4, 6, 8, 10, 12, 14, 16, 18
DONE!
```
From `doubler()` it's but a short hop to a more general `mapper()`
which accepts an arbitrary transforming function:

```
@coroutine
def mapper(transform, target):
    while True:
        item = (yield)
        transformed_item = transform(item)
        try:
            target.send(transformed_item)
        except StopIteration as e:
            return e.value
```
Used like so,

```
>>> printer = rprint(sep=', ', end='\nDONE!\n')
>>> count_to_nine = range(10)
>>> poisson_source(rate=0.5, iterable=count_to_nine, target=mapper(transform=square, target=printer))
0, 1, 4, 9, 16, 25, 36, 49, 64, 81
DONE!
```
From here, you can see how we could also implement equivalents of
`filter()`, `reduce()` and so on to operate on the 'push' event
stream modelled by Python coroutines.

The point here is that we can't just re-use any existing functions which
process 'pull' data series – such as the functions in `itertools` –
with 'push' data series. Each and every function needs to be
reimplemented to accept values pushed from upstream, and send processed
results downstream.

### Transducing events

Transducers provide a way out of this quandary. We've demonstrated
earlier in this series that 'reduce' is a fundamental operation, and by
reimagining `reduce()` into a more general `transduce()` we were
able to use the same transducers to operate on both eager and lazy data
series. We can do the same with coroutine-based push events, by
implementing a version of `transduce()` which allows us to use any
transducer to process a stream of such events.

Our `reactive_transduce()` is a coroutine which accepts two arguments:
a transducer and a target sink to which the transduced results will be
sent:

```
@coroutine
def reactive_transduce(transducer, target=None):
    reducer = transducer(sending())
    accumulator = target if (target is not None) else reducer.initial()
    try:
        while True:
            item = (yield)
            accumulator = reducer.step(accumulator, item)
            if isinstance(accumulator, Reduced):
                accumulator = accumulator.value
                break
    except GeneratorExit:
        pass
    return reducer.complete(accumulator)
```
The `reactive_transduce()` function connects to the upstream end of a
transducer chain, adapting from the coroutine protocol to the reducer
interface. At the downstream end of the transducer chain, we need to
adapt the other way, from the reducer interface to the coroutine
protocol. To do this we use a reducer called `Sending`, which we
hard-wire as the 'bottom' reducer on the first line of
`reactive_transduce()`. The `Sending` reducer looks like this:

```
class Sending:
    def initial(self):
        return null_sink()
    def step(self, result, item):
        try:
            result.send(item)
        except StopIteration:
            return Reduced(result)
        else:
            return result
    def complete(result):
        result.close()
        return result
```
The `step()` method literally sends the next item to the `result` –
which must therefore be a legitimate event sink. Should the sink
indicate that it can't accept a further item, by raising
`StopIteration` we return the result wrapped in the `Reduced`
sentinel. The `initial()` method provides a legitimate sink – just a
simple do-nothing sink defined as:

```
@coroutine
def null_sink():
    while True:
        _ = (yield)
```
Going back to `reactive_transduce()` the main loop continues to
iterate, receiving new values via a yield expression, until such time as
`GeneratorExit` is signalled by the client or the reducer signals
termination by returning `Reduced`.

When the main loop is exited by whatever means, we give the reducer
opportunity to `complete()`, and the `Sending.complete()` method
ensures that `close()` is called on the target.

With these pieces in place, let's look at how to use
`reactive_transduce()`. We'll reproduce our previous example where we
squared the output from `poisson_source()`, but this time using the
`mapping()` transducer to do the work:

```
>>> poisson_source(rate=0.5,
...                iterable=range(10),
...                target=transduce(transducer=mapping(square),
...                target=printer))
...
0, 1, 4, 9, 16, 25, 36, 49, 64, 81
DONE!
```
The key point here is that we can now take an *arbitrary transducer* and
reuse it with eager collections, lazy iterables, and push-events! In
fact, simply by devising an appropriate transduce function we can use
re-use our transducers in an arbitrary data-series processing context.

This is the true power of transducers: Data processing components completely abstracted away from how the input data arrives, or to where the output results are sent.
