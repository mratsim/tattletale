---
title: "Transducers are Coming"
author: Rich Hickey
source: cognitect.com blog
url: https://cognitect.com/blog/2014/8/6/transducers-are-coming
date: "2014-08-06"
copyright: "© Rich Hickey / Cognitect — web-published copy reproduced for private research/study with attribution and link; all rights remain with the author."
context-note: "The announcement post introducing transducers (Clojure 1.7, core.async); companion to the earlier reducers post. Read alongside the Sixty North series deriving the same concepts from first principles in Python."
---

Transducers are a powerful and composable way to build algorithmic transformations that you can reuse in many contexts, and they're coming to Clojure core and core.async.

Two years ago, in a blog post describing how reducers work, I described the *reducing function transformers* on which they were based, and provided explicit examples like '**mapping**', '**filtering**' and '**mapcatting**'. Because the reducers library intends to deliver an API with the same 'shape' as existing sequence function APIs, these transformers were never exposed a la carte, instead being encapsulated by the macrology of reducers.

In working recently on providing algorithmic combinators for core.async, I became more and more convinced of the superiority of reducing function transformers over channel->channel functions for algorithmic transformation. In fact, I think they are a better way to do many things for which we normally create bespoke replicas of map, filter etc.

So, reducing function transformers are getting a name - '**transducers**', and first-class support in Clojure core and core.async.

To recap that earlier post:

A reducing function is just the kind of function you'd pass to **reduce** - it takes a result so far and a new input and returns the next result-so-far. In the context of transducers it's best to think about this most generally:

;;reducing function signature
whatever, input -> whatever

and a transducer is a function that takes one reducing function and returns another:

;;transducer signature
(whatever, input -> whatever) -> (whatever, input -> whatever)

The primary power of transducers comes from their fundamental decoupling - they don't care (or know about):

The other source of power comes from the fact that transducers compose using *ordinary function composition*.

The reducers library leverages transducers' decoupling from the job, the representation, and the source of inputs to accomplish parallel reduction. But transducers can also be used for:

All of this is coming to Clojure core and core.async.

Concretely, most of the core sequence functions are gaining a new arity, one shorter than their current shortest, which elides the final collection source argument. This arity will return a transducer that represents the same logic, independent of lazy sequence processing.

Thus:

;;look Ma, no collection!
(map f)

returns a 'mapping' transducer. filter et al get similar support.

You can build a 'stack' of transducers using ordinary function composition (comp):

(def xform (comp (map inc) (filter even?)))

You might notice the similarity between the above comp and a call to ->>:

(->> aseq (map inc) (filter even?))

One way to think of transducers is like ->> but independent of the job (lazy sequence creation) and the source of inputs (aseq).

Once you've got a transducer, what can you do with it? *An open set of things.*

For instance, given the above transducer and some data in a vector, you can:

(sequence xform data)

(transduce xform + 0 data)

(into [] xform data)

(iteration xform data)

(chan 1 xform)

The latter demonstrates the corresponding new capability of core.async channels - they can take transducers.

This post is just to serve as a heads up on what the ongoing work is about. There will be more explanations, tutorials and derivations to follow, here and elsewhere.

I'm excited about transducers and the power they bring, and I hope you are too!

Rich
