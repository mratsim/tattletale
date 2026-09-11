---
title: Pipe Dreams, part 3.1
author: Marc Gravell
url: https://blog.marcgravell.com/2018/07/pipe-dreams-part-31.html
hostname: marcgravell.com
description: Pipelines - a guided tour of the new IO API in .NET, part 3.1 ( part 1 , part 2 , part 3 ) After part 3, I got some great feedback - mostl...
sitename: blog.marcgravell.com
date: "2018-07-30"
---
# Pipelines - a guided tour of the new IO API in .NET, part 3.1

After part 3, I got some great feedback - mostly requests to clarify things that I touched on, but could do with further explanation. Rather than make part 3 *even longer*, I want to address those here! Yay, more words!

### Isn't `ArrayPoolOwner<T>` doing the same thing as `MemoryPool<T>`? Why don't you just use `MemoryPool<T>` ?

Great question! I didn't actually mention `MemoryPool<T>`, so I'd better introduce it.

`MemoryPool<T>` is an `abstract` base type that offers an API of the form:

```
public abstract class MemoryPool<T> : IDisposable
{
    public abstract IMemoryOwner<T> Rent(int minBufferSize = -1);
    // not shown: a few unrelated details
}
```
As you can see, this `Rent()` method looks exactly like what we were looking for before - it takes a size and returns an `IMemoryOwner<T>` (to provide a `Memory<T>`), with it being returned from whence it came upon disposal.

`MemoryPool<T>` also has a default implementation (`public static MemoryPool<T> Shared { get; }`), which returns a `MemoryPool<T>` that is based on the `ArrayPool<T>` (i.e. `ArrayMemoryPool<T>`). The `Rent()` method returns an `ArrayMemoryPoolBuffer`, which looks *remarkably like* the thing that I called `ArrayPoolOwner<T>`.

So: a very valid question would be: "Marc, didn't you just re-invent the default memory pool?". The answer is "no", but it is for a *very* subtle reason that I probably should have expounded upon at the time.

The problem is in the name `minBufferSize`; well... not really the *name*, but the *consequence*. What this means is: when you `Rent()` from the default `MemoryPool<T>.Shared`, the `.Memory` that you get back will be *over-sized*. Often this isn't a problem, but in our case we *really* want the `.Memory` to represent the actual number of bytes that were sent (*even if* we are, behind the scenes, using a larger array from the pool to contain it).

We *could* use an extension method on arbitrary memory pools to *wrap* potentially oversized memory, i.e.

```
public static IMemoryOwner<T> RentRightSized<T>(
    this MemoryPool<T> pool, int size)
{
    var leased = pool.Rent(size);
    if (leased.Memory.Length == size)
        return leased; // already OK
    return new RightSizeWrapper<T>(leased, size);
}
class RightSizeWrapper<T> : IMemoryOwner<T>
{
    public RightSizeWrapper(
        IMemoryOwner<T> inner, int length)
    {
        _inner = inner;
        _length = length;
    } 
    IMemoryOwner<T> _inner;
    int _length;
    public void Dispose() => _inner.Dispose();
    public Memory<T> Memory
        => _inner.Memory.Slice(0, _length);
}
```
but... this would mean allocating *two* objects for most leases - one for the *actual* lease, and one for the thing that fixes the length. So, since we only *really* care about the array-pool here, it is preferable IMO to cut out the extra layer, and write our own right-sized implementation from scratch.

So: that's the difference in the reasoning and implementation. As a side note, though: it prompts the question as to whether I should refactor my API to actually implement the `MemoryPool<T>` API.

### You *might* not want to complete with success if the cancellation token is cancelled

This is in relation to the `while` in the read loop:

```
while (!cancellationToken.IsCancellationRequested)
{...}
```
The *more typical* expectation for cancellation is for it to throw with a cancellation exception of some kind; therefore, if it *is* cancelled, I might want to reflect that.

This is very valid feedback! Perhaps the most practical fix here is simply to use `while (true)` and let the subsequent `await reader.ReadAsync(cancellationToken)` worry about what cancellation should look like.

### You should clarify about testing the result in async "sync path" scenarios

In my aside about `async` uglification (optimizing when we *expect* it to be synchronous in most cases), I ommitted to talk about
getting results from the pseudo-awaited operation. Usually, this comes down to calling `.Result` on an awaitable (something like
a `Task<T>`, `ValueTask<T>`, or `.GetResult()` on an awaiter (the thing you get from `.GetAwaiter()`). I haven't done it *in the
example* because in `async` terms this would simply have been an `await theThing;` usage, not a `var local = await theThing;` usage; but
you can if you need that.

I must, however, clarify a few points that perhaps weren't clear:

- you **should not** (usually) try to access the`.Result` of a task**unless** you know that it
has already completed
- knowing that it has completed *isn't enough* to know that it has completed*successfully* ; if you only test "is completed", you can use`.GetResult()` on the awaiter to*check for exceptions* while also fetching the result (which you can then discard if you like)
- in my case, I'm taking a shortcut by checking `IsCompletedSuccessfully` ; this exists on`ValueTask[<T>]` (and on`Task[<T>]` in .NET Core 2.0, else you can check`.Status == TaskStatus.RanToCompletion` ) - which is only`true` in the "completed without an exception" case
- because of expectations around how exceptions on `async` operations are wrapped and surfaced, it is almost always preferable to just switch into the`async` flow if you know a task has faulted, and just`await` it; the compiler knows how to get the exception out in the most suitable way, so: let it do the hard work

### You should explain more about `ValueTask[<T>]` vs `Task[<T>]` - not many people understand them well

OK! Many moons ago, `Task<T>` became a thing, and all was well. `Task<T>` actually happened *long before* C# had any kind of support
for `async`/`await`, and the main scenarios it was concerned about were *genuinely asynchronous* - it was *expected* that the answer
*would not be immediately available*. So, the ovehead of allocating a placeholder object was fine and dandy, dandy and fine.

As the usage of `Task<T>` grew, and the language support came into effect, it started to become clear that there were many cases where:

- the operation would often be available immediately (think: caches, buffered data, uncontested locking primitives, etc)
- it was being used *inside a tight loop* , or just at high frequency - i.e. something that happens thousands of times a second (file IO, network IO, synchronization over a collection, etc)

When you put those two things together, you find yourself allocating *large numbers* of objects for something that *was only rarely actually asynchronous* (so: when there *wasn't* data available in the socket, or the file buffer was empty, or the lock was contested). For some scenarios, there are pre-completed reusable task instances available (such as `Task.CompletedTask`, and inbuilt handling for some low integers), but this doesn't help if the return value is outside this very limited set. To help avoid the allocations in the general case, `ValueTask[<T>]` was born. A `ValueTask[<T>]` is a `struct` that implements the "awaitable" pattern (a duck-typed pattern, like `foreach`, but that's a story for another day), that essentially contains two fields:

- a `T` if the value was known immediately (obviously not needed for the untyped`ValueTask` case)
- a `Task<T>` if the value is pending and the answer depends on the result of the incomplete operation

That means that *if the value is known now*,  no `Task[<T>]` (and no corresponding `TaskCompletionSource<T>`) *ever needs to be allocated* - we just throw back the `struct`, it gets unwrapped by the `async` pattern, and life is good. Only in the case where the operation is *actually asynchronous* does an object need to be allocated.

Now, there are three common views on what we should do with this:

1. always expose `Task[<T>]` , regardless of whether it is likely to be synchronous
2. expose `Task[<T>]` if we know it will be async, expose`ValueTask[<T>]` if we think it*may* be synchronous
3. always expose `ValueTask[<T>]`

Frankly, the only valid reason to use `1` is because your API surface was baked and fixed back before `ValueTask[<T>]` existed.

The choice between `2` and `3` is interesting; what we're actually talking about there is an implementation detail, so a good case *could be argued* for `3`, allowing you to amaze yourself later if you find a way of doing something synchronously (where it was previously asynchronous), without breaking the API. I went for `2` in the code shown, but it would be something I'd be willing to change without much prodding.

You should also note that there is actually a fourth option: **use custom awaitables** (meaning: a custom type that implements the "awaitable" duck-typed pattern). This is an advanced topic, and needs *very* careful consideration. I'm not even going to give examples of *how* to do that, but it is worth noting that `ReadAsync` and `FlushAsync` ("pipelines" methods that we've used extensively here) *do return* custom awaitables. You'd need to *really, really* understand your reasons before going down that path, though.

### I spotted a bug in your "next message number" code

Yes, the code shown in the post can generate two messages with id `1`, after 4-billion-something messages:

```
messageId = ++_nextMessageId;
if (messageId == 0) messageId = 1;
```
Note that I didn't increment `_nextMessageId` when I dodged the sentinel (zero). There's also a *very* small chance that a previous message from 4-billion-ago *still hasn't been replied to*. Both of these are fixed in the "real" code.

### You might be leaking your lease around the `TrySetResult`

In the original blog code, I had

`tcs?.TrySetResult(payload.Lease());`
If `tcs` is not `null` (via the "Elvis operator"), this allocates a lease and then invokes `TrySetResult`. However, `TrySetResult` *can return `false`* - meaning: it couldn't do that, because the underlying task was already completed in some other way (perhaps we added timeout code). The only time we should consider that we have *successfully* transferred ownership of the lease to the task is if it returns `true`. The real code fixes this, ensuring that it is disposed in all cases *except* where `TrySetResult` returns `true`.

### What about incremental frame parsers?

In my discussion of handling the frame, I was using an approach that processed a frame either in it's entirety, or not at all. This is not the only option, and you can consume *any amount of the frame that you want*, as long as you write code to track the internal state. For example, if you are parsing http, you could parse the http headers into some container as long as you have at least one entire http header name/value pair (without requiring *all* the headers to start parsing). Similarly, you could consume *some* of the payload (perhaps writing what you have so far to disk). In both cases, you would simply need to `Advance` past the bit that you consider consumed, and update your own state object.

So yes, that is absolutely possible - even highly desirable in some cases. In some cases it is highly desirable *not to start* until you've got everything. Remember that parsing often means taking the data from a *streamed* representation, and pushing it into a *model* representation - you might actually need *more* memory for the *model* representation (especially if the source data is compressed or simply stored in a dense binary format). An *advantage* of incremental parsing is that when the last few bytes dribble in, you might already have done *most* of the parsing work - allowing you to overlap pre-processing the data with data receive - rather than "buffer, buffer, buffer; right - now *start* parsing it".

However, in the case I was discussing: the header was 8 bytes, so there's not much point trying to over-optimize; if we don't have an entire header *now*, we'll mostly likely have a complete header when the next packet arrives, or we'll *never* have an entire header. Likewise, because we want to hand the entire payload to the consumer as a single chunk, we need all the data. We *could* actually lease the target array as soon as we know the size, and start copying data into *that* buffer and releasing the source buffers. We're not actually gaining much by this - we're simply exchanging data in one buffer for the same amount of data in another buffer; but we're actually exposing ourselves to an attack vector: a malicious (or buggy) client can sent a message-header that claims to be sending a large amount of data (maybe 1GiB), then just ... keeps the socket open and doesn't send anything more. In this scenario, the client has sent *almost nothing* (maybe just 8 bytes!), but they've chewed up a **lot** of server memory. Now imagine they do this from 10, 100, or 1000 parallel connections - and you can see how they've achieved *disproportionate* harm to our server, for almost zero cost at the client. There are two pragmatic fixes for this:

- Put an upper limit on the message size, and put an upper limit on the connections from a single endpoint
- Make the client pay their dues: if they claim to be sending a large message (which may indeed have legitimate uses): don't lease any expensive resources until they've actually sent that much (which is what the code as-implemented achieves)

Emphasis: your choice of frame parsing strategy is entirely contextual, and you can play with other implementations.

So; that's the amendments. I hope they are useful. A huge "thanks" to the people who are keeping me honest here, including Shane Grüling, David Fowler, and Nick Craver.
