---
title: Pipe Dreams, part 2
author: Marc Gravell
url: https://blog.marcgravell.com/2018/07/pipe-dreams-part-2.html
hostname: marcgravell.com
description: Pipelines - a guided tour of the new IO API in .NET, part 2 In part 1 , we discussed some of the problems that exist in the familiar Stream...
sitename: blog.marcgravell.com
date: "2018-07-03"
---
# Pipelines - a guided tour of the new IO API in .NET, part 2

In part 1, we discussed some of the problems that exist in the familiar `Stream` API, and we had an introduction to the `Pipe`, `PipeWriter` and `PipeReader` APIs, looking at how to write to a single `Pipe` and then consume the data from that `Pipe`; we also discussed how `FlushAsync()` and `ReadAsync()` work together to keep both sides of the machinery working, dealing with "empty" and "full" scenarios - suspending the reader when there is nothing to do, and resuming it when data arrives; and suspending the writer when it is out-pacing the reader (over-filling the pipe), and resuming when the reader has caught up; and we discussed what it *means* to "resume" here, in terms of the threading model.

In this part, we're going to discuss the *memory model* of pipelines: where does the data actually *live?*. We'll also start looking at how we can use pipelines in realistic scenarios to fulfil real needs.

## The memory model: where are all my datas?

In part 1, we spoke about how the pipe owns all the buffers, allowing the writer to request a buffer via `GetMemory()` and `GetSpan()`, with the committed data later being exposed to the reader via the `.Buffer` on `ReadAsync()` - which is a `ReadOnlySequence<byte>`, i.e some number of *segments* of data.

So what *actually happens*?

Each `Pipe` instance has a reference to a `MemoryPool<byte>` - a new device in `System.Memory` for, unsurprisingly, creating a memory pool. You can specify a specific `MemoryPool<byte>` in the options when creating a `Pipe`, but by default (and, I imagine, almost always) - a shared application-wide pool (`MemoryPool<byte>.Shared`) is used.

The `MemoryPool<byte>` concept is very open-ended. The *default* implementation simply makes use of `ArrayPool<byte>.Shared` (the application wide array-pool), renting arrays as needed, and returning them when done. This `ArrayPool<T>` is implemented using `WeakReference`, so pooled arrays are collectible if memory pressure demands it. However, when you ask `GetMemory(someSize)` or `GetSpan(someSize)`, it doesn't simply ask the memory pool for *that amount*; instead, it tracks a "segment" internally. A new "segment" will be (by default, configurable) the larger of `someSize` or 2048 bytes. Requesting a non-trivial amount of memory means that we aren't filling the system with tiny arrays, which would significantly impact garbage collection. When you `Advance(bytesWritten)` in the writer, it:

- moves an internal counter that is how much of the current segment has been used
- updates the end of the "available to be read" chain for the reader; if we've just written the first bytes of an empty segment, this will mean adding a new segment to the chain, otherwise it'll mean increasing the end marker of the final segment of the existing chain

It is this "available to be read" chain that we fetch in `ReadAsync()`; and as we `AdvanceTo` in the reader - when entire segments are consumed, the pipe hands those segments back to the memory pool. From there, they can be reused many times. And as a direct consequence of the two points above, we can see that *most of the time*, even with multiple calls to `Advance` in the writer, we may end up with a single segment in the reader, with multiple segments happening either at segment boundaries, or where the reader is falling behind the writer, and data is starting to accumulate.

What this achieves *just using the default pool* is:

- we don't need to keep allocating every time we call `GetMemory()` /`GetSpan()`
- we don't need a separate array per `GetMemory()` /`GetSpan()` - we'll often just get a different range of the same "segment"
- a relatively small number of non-trivial buffer arrays are used
- they are automatically recycled without needing lots of library code
- when not being used, they are available for garbage collection

This also explains why the approach of requesting a very small amount in `GetMemory()` / `GetSpan()` and then checking the size can be so successful: we have access to the *rest of the unused part of the current segment*. Meaning: with a segment size of 2048, of which 200 bytes were already used by previous writes - even if we only ask for 5 bytes, we'll probably find we have 1848 bytes available to play with. Or possibly more - remember that obtaining an array from `ArrayPool<T>.Shared` is *also* an "at least this big" operation.

## Zero copy buffers

Something else to notice in this setup is that we get data buffering without *any* data copying. The writer asked for a buffer, and wrote the data to *where it needed to be* the first time, on the way in. This then acted as a buffer between the writer and the reader without any need to copy data around. And if the reader couldn't process all the data yet, it was able to push data back into the pipe simply by saying explicitly what it *did* consume. There was no need to maintain a separate backlog of data for the reader, something that is *very* common in protocol processing code using `Stream`.

It is this combination of features that makes the *memory* aspect of pipeline code so friendly. You could *do* all of this with `Stream`, but it is an excruciating amount of error-prone code to do it, and even more if you want to do it *well* - and you'd pretty much have to implement it separately for each scenario. Pipelines makes good memory handling the default simple path - the pit of success.

## More exotic memory pools

You aren't limited to the memory model discussed; you can implement your own custom memory pool! The advantage of the default pool is that it is simple. In particular, it *doesn't really matter* if we aren't 100% perfect about returning every segment - if we somehow drop a pipe on the floor, the worst that can happen is that the garbage collector collects the abandoned segments at some point. They won't go back into the pool, but that's fine.

You *can*, however, do much more interesting things. Imagine, for example, a `MemoryPool<byte>` that takes huge *slabs* of memory - either managed memory via a number of very large arrays, or unmanaged memory via `Marshal.AllocHGlobal` (note that `Memory<T>` and `Span<T>` *are not limited to arrays* - all they require is some kind of contiguous memory), leasing blocks of this larger chunk as required. This has great potential, but it becomes increasingly important to ensure that segments are reliably returned. Most systems shouldn't need this, but it is good that the flexibility is offered.

# Useful pipes in real systems

The example that we used in part 1 was of a single `Pipe` that was written and read by the same code. That's clearly not a realistic scenario (unless we're trying to mock an "echo" server), so what can we do for more realistic scenarios? First, we need to connect our pipelines to something. We don't usually want a `Pipe` in isolation; we want a pipe that *integrates with a common system or API*. So; let's start by seeng what this would look like.

Here we need a bit of caveat and disclaimer: the pipelines released in .NET Core 2.1 *do not include any endpoint implementations*. Meaning: the `Pipe` machinery is there, but nothing is shipped *inside the box* that actually connects pipes with any other existing systems - like shipping the abstract `Stream` base-type, but without shipping `FileStream`, `NetworkStream`, etc. Yes, that sounds frustrating, but it was a pragmatic reality of time constraints. Don't panic! There are... "lively" conversations going on right now about which bindings to implement with which priority; and there are few community offerings to bridge the most obvious gaps for today.

Since we find ourselves in that position, we might naturally ask: "what does it take to connect pipelines to another data backend?".

Perhaps a good place to start would be connecting a pipe to a `Stream`. I know what you're thinking: "Marc, but in part 1 you went out of your way to say how terrible `Stream` is!". I haven't changed my mind; it isn't necessarily *ideal* - for any scenario-specific `Stream` implementation (such as `NetworkStream` or `FileStream`) we *could* have a dedicated pipelines-based endpoint that talked *directly* to that service with minimal indirection; but it is a *useful* first step:

- it gives us immediate access to a *huge* range of API surfaces - anything that can expose data via`Stream` , and anything that can act as a middle-layer via wrapped streams (encryption, compression, etc)
- it hides all the wrinkly bits of the `Stream` API behind a clear unambiguous surface
- it gives us *almost all* of the advantages that we have mentioned so far

So, let's get started! The first thing we need to think about is: what is the *direction* here? As previously mentioned, a `Stream` is ambiguous - and could be read-only, write-only, or read-write. Let's assume we want to deal with the most general case: a read-write stream that acts in a duplex manner - this will give us access to things like sockets (via `NetworkStream`). This means we're actually going to want *two* pipes - one for the input, one for the output. Pipelines helps clear this up for us, by declaring an interface expressly for this: `IDuplexPipe`. This is a very simple interface, and being handed an `IDuplexPipe` is analogous to being handed the ends of two pipes - one marked "in", one marked "out":

```
interface IDuplexPipe
{
    PipeReader Input { get; }
    PipeWriter Output { get; }
}
```
What we want to do, then, is create a type that *implements* `IDuplexPipe`, but using 2 `Pipe` instances internally:

- one `Pipe` will be the output buffer (from the consumer's perspective), which will be filled by caller-code writing to`Output` - and we'll have a loop that consumes this`Pipe` and pushes the data into the underlying`Stream` (to be written to the network, or whatever the stream does)
- one `Pipe` will be the input buffer (from the consumer's perspective); we'll have a loop that*reads* data from the underlying`Stream` (from the network, etc) and pushes it into the`Pipe` , where it will be drained by caller-code reading from`Input`

This approach immediately solves a *wide range* of problems that commonly affect people using `Stream`:

- we now have input/output buffers that decouple stream access from the read/write caller-code, without having to add `BufferedStream` or similar to prevent packet fragmentation (for the writing code), and to make it very easy to continue receiving more data while we process it (for the reading code especially, so we don't have to keep pausing while we ask for more data)
- if the caller-code is writing faster than the stream `Write` can process, the back-pressure feature will kick in, throttling the caller-code so we don't end up with a huge buffer of unsent data
- if the stream `Read` is out-pacing the caller-code that is*consuming* the data, the back-pressure will kick in here too, throttling our stream read loop so we don't end up with a huge buffer of unprocessed data
- both the read and write implementations benefit from all the memory pool goodness that we discussed above
- the caller-code doesn't ever need to worry about backlog of data (incomplete frames), etc - the pipe deals with it

# So what might that look like?

Essentially, all we need to do, is something like:

```
class StreamDuplexPipe : IDuplexPipe
{
    Stream _stream;
    Pipe _readPipe, _writePipe;
    public PipeReader Input => _readPipe.Reader;
    public PipeWriter Output => _writePipe.Writer;
    
    // ... more here
}
```
Note that we have two different pipes; the caller gets one end of each pipe - and our code will act on the *other* end of each pipe.

## Pumping the pipe

So what does the code look like to interact with the stream? We need two methods, as disccused above. The first - and simplest - has a loop that reads data from the `_stream` and pushes it to `_readPipe`, to be consumed by the calling code; the core of this method could be *something like*

```
while (true)
{
    // note we'll usually get *much* more than we ask for
    var buffer = _readPipe.Writer.GetMemory(1);
    int bytes = await _stream.ReadAsync(buffer);
    _readPipe.Writer.Advance(bytes);
    if (bytes == 0) break; // source EOF
    var flush = await _readPipe.Writer.FlushAsync();
    if (flush.IsCompleted || flush.IsCanceled) break;
}
```
This loop asks the pipe for a buffer, then uses the new `netcoreapp2.1` overload of `Stream.ReadAsync` that accepts a `Memory<byte>` to populate that buffer - we'll discuss what to do if you don't have an API that takes `Memory<byte>` shortly. When the read is complete, it commits that-many bytes to the pipe using `Advance`, then it invokes `FlushAsync()` on the *pipe* to (if needed) awaken the reader, or pause the write loop while the back-pressure eases. Note we should also check the outcome of the `Pipe`'s `FlushAsync()` - it could tell us that the pipe's *consumer* has signalled that they've finished reading the data they want (`IsCompleted`), or that the pipe itself was shut down (`IsCanceled`).

Note that in both cases, we want to ensure that we tell the pipe when this loop has exited - *however it exits* - so that we don't end up with the calling code awaiting forever on data that will never come. Accidents happen, and sometimes the call to `_stream.ReadAsync` (or any other method) might throw an exception, so a good way to do this is with a `try`/`finally` block:

```
Exception error = null;
try
{
    // our loop from the previous sample
}
catch(Exception ex) { error = ex; }
finally { _readPipe.Writer.Complete(error); }
```
If you prefer, you could also use two calls to `Complete` - one at the end of the `try` (for success) and one inside the `catch` (for failure).

The second method we need is a bit more complex; we need a loop that consumes data from `_writePipe` and pushes it to `_stream`. The core of this could be something like:

```
while (true)
{
    var read = await _writePipe.Reader.ReadAsync();
    var buffer = read.Buffer;
    if (buffer.IsCanceled) break;
    if (buffer.IsEmpty && read.IsCompleted) break;
    // write everything we got to the stream
    foreach (var segment in buffer)
    {
        await _stream.WriteAsync(segment);
    }
    _writePipe.AdvanceTo(buffer.End);
    await _stream.FlushAsync();    
}
```
This awaits *some* data (which could be in multiple buffers), and checks some exit conditions; as before, we can give up if `IsCanceled`, but the next check is more subtle: we don't want to stop writing just because the *producer* indicated that they've written everything they wanted to (`IsCompleted`), or we might not write the last few segments of their data - we need to continue until *we've written all their data*, so `buffer.IsEmpty`. This is simplified in this case because we're always writing everything - we'll see a more complex example shortly. Once we have data, we write each of the non-contiguous buffers to the stream sequentially - because `Stream` can only write one buffer at a time (again, I'm using the `netcoreapp2.1` overload here that accepts `ReadOnlyMemory<byte>`, but we aren't restricted to this). Once it has written the buffers, it tells the pipe that we have consumed the data, and flushes the underlying `Stream`.

In "real" code we *might* want to be a bit more aggressive about optimizing to reduce flushing the underlying stream until we know there is no more data readily available, perhaps using the `_writePipe.Reader.TryRead(...)` method in addition to `_writePipe.Reader.ReadAsync()` method; this method works similarly to `ReadAsync()` but is guaranteed to always return synchronously - useful for testing "did the writer append something while I was busy?". But the above illustrates the point.

Additionally, like before we would want to add a `try`/`finally`, so that we always call `_writePipe.Reader.Complete();` when we exit.

We can use the `PipeScheduler` to start these two pumps, which will ensure that they run in the intended context, and our loops start pumping data. We'd have a *little* more house-keeping to add (we'd probably want a mechanism to `Close()`/`Dispose()` the underlying stream, etc) - but as you can see, it doesn't have to be a *huge* task to connect an `IDuplexPipe` to a source that wasn't designed with pipelines in mind.

# Here's one I made earlier...

I've simplified the above a little (not too much, honest) to make it consise for discussion, but you still probably don't want to start copying/pasting chunks from here to try and get it to work. I'm not claiming they are the perfect solution for all situations, but as part of the 2.0 work for `StackExchange.Redis`, we have implemented a range of bindings for pipelines that we are making available on nuget - unimaginatively titled `Pipelines.Sockets.Unofficial` (nuget, github); this includes:

- converting a duplex `Stream` to an`IDuplexPipe` (like the above)
- converting a read-only `Stream` to a`PipeReader`
- converting a write-only `Stream` to a`PipeWriter`
- converting an `IDuplexPipe` to a duplex`Stream`
- converting a `PipeReader` to a read-only`Stream`
- converting a `PipeWriter` to a writer-only`Stream`
- converting a `Socket` to an`IDuplexPipe` directly (without going via`NetworkStream` )

The first six are all available via static methods on `StreamConnection`; the last is available via `SocketConnection`.

`StackExchange.Redis` is very involved in `Socket` work, so we are very interested in how to connect pipelines to sockets; for redis connections without TLS, we can connect our `Socket` direct to the pipeline:

- `Socket` ⇔`SocketConnection`

For redis connections *with* TLS (in particular: cloud redis providers), we can connect the pieces thusly:

- `Socket` ⇔`NetworkStream` ⇔`SslStream` ⇔`StreamConnection`

Both of these configurations give us a `Socket` at one end, and an `IDuplexPipe` at the other, and it begins to show how we can orchcestrate pipelines as part of a more complex system. Perhaps more importantly, it gives us room in the future to *change* the implementation. As examples of future possibilities:

- Tim Seaward has been working on `Leto` , which provides TLS capability as an`IDuplexPipe` directly, without requiring`SslStream` (and thus: no stream inverters)
- between Tim Seaward, David Fowler and Ben Adams, there are a *range* of experimental or in-progress network layers directly implementing pipelines without using managed sockets, including "libuv", "RIO" (Registered IO), and most recently, "magma" - which pushes the entire TCP stack into user code to reduce syscalls.

It'll be interesting to see how this space develops!

## But my existing API doesn't talk in `Span<byte>` or `Memory<byte>`!

When writing code to pump data from a pipe to another system (such as a `Socket`), it is very likely you'll bump into APIs that don't take `Memory<byte>` or `Span<byte>`. Don't panic, all is not lost! You still have multiple ways of breaking out of that world into something more ... traditional.

The first trick, for when you have a `Memory<T>` or `ReadOnlyMemory<T>`, is `MemoryMarshal.TryGetArray(...)`. This takes in a *memory* and attempts to get an `ArraySegment<T>` that describes the same data in terms of a `T[]` vector and an `int` offset/count pair. Obviously this can only work if the memory *was based on* a vector, which is not always the case. So this can fail *on exotic memory pools*. Our second escape hatch is `MemoryMarshal.GetReference(...)`. This takes in a *span* and returns a reference (actually a "managed pointer", aka `ref T`) to the start of the data. Once we have a `ref T`, we can use `unsafe` C# to get an unmanaged pointer to the data, useful for APIs that talk in such:

```
Span<byte> span = ...
fixed(byte* ptr = &MemoryMarshal.GetReference(span))
{
    // ...
}
```
It can still do this if the length of the span is zero, returning a reference to where the zeroth item *would have been*, and it *even works* for a `default` span where there never was any backing memory. This last one requires a slight word of caution because a `ref T` is *not usually expected to be null*, but that's exactly what you get here. Essentially, as long as you don't ever try to dereference this kind of null reference: you'll be fine. If you use `fixed` to convert it to an unmanaged pointer, you get back a null (zero) pointer, which *is* more expected (and can be useful in some P/Invoke scenarios). `MemoryMarshal` is *essentially* synonymous with `unsafe` code, even if the method you're calling doesn't require the `unsafe` keyword. It is perfectly valid to use it, but if you use it incorrectly, it reserves the right to hurt you - so just be careful.

# What about the app-code end of the pipe?

OK, we've got our `IDuplexPipe`, and we've seen how to connect the "business end" of both pipes to your backend data service of choice. Now; how do we use it in our app code?

As in our example from part 1, we're going to hand the `PipeWriter` from `IDuplexPipe.Output` to our outbound code, and the `PipeReader` from `IDuplexPipe.Input` to our inbound code.

The *outbound* code is typically very simple, and is usually a very direct port to get from `Stream`-based code to `PipeWriter`-based. The key difference, once again, is that *you don't control the buffers*. A typical implementation might look something like:

```
ValueTask<bool> Write(SomeMessageType message, PipeWriter writer)
{
    // (this may be multiple GetSpan/Advance calls, or a loop,
    // depending on what makes sense for the message/protocol)
    var span = writer.GetSpan(...);
    // TODO: ... actually write the message
    int bytesWritten = ... // from writing
    writer.Advance(bytesWritten);
    return FlushAsync(writer);
}
private static async ValueTask<bool> FlushAsync(PipeWriter writer)
{
    // apply back-pressure etc
    var flush = await writer.FlushAsync();
    // tell the calling code whether any more messages
    // should be written
    return !(flush.IsCanceled || flush.IsCompleted);
}
```
The first part of `Write` is our business code - we do whatever we need to write the data to the buffers from `writer`; typically this will include multiple calls to `GetSpan(...)` and `Advance()`. When we've written our message, we can flush it to ensure the pump is active, and apply back-pressure. For very large messages we *could* also flush at intermediate points, but for most simple scenarios: flushing once per message is fine.

If you're wondering why I split the `FlushAsync` code into a separate method: that's because I want to `await` the result of `FlushAsync` to check the exit conditions, so it needs to be in an `async` method. The most efficient way to access memory here is via the `Span<byte>` API, and `Span<byte>` is a `ref struct` type; as a consequence we **cannot** use a `Span<byte>` local variable in an `async` method. A pragmatic solution is to simply split the methods, so one method deals with the `Span<byte>` work, and another method deals with the `async` aspect.

## Random aside: async code, hot synchronous paths, and async machinery overhead

The machinery involved in `async` / `await` is pretty good, but it can still be a surprising amount stack work - you can see this on sharplab.io - take a look at the generated machinery for the `OurCode.FlushAsync` method - and the entirety of `struct <FlushAsync>d__0`. Now, this code is *not terrible* - it tries hard to avoid allocations in the synchronous path - but it is *unnecessary*. There are two ways to signficantly improve this; one is to not `await` *at all*, which is often possible if the `await` is the last line in a method **and we don't need to process the results**: don't `await` - just remove the `async` and `return` the task - complete or incomplete. We can't do that here, because we need to check the state of the result, but we can optimize for success by checking whether the task *is already complete* (via `.IsCompletedSuccessfully` - if it has completed but faulted, we still want to use the `await` to make sure the exception behaves correctly). If it *is* successfully completed, we're allowed to access the `.Result`; so we could *also* write our `FlushAsync` method as:

```
private static ValueTask<bool> Flush(PipeWriter writer)
{
    bool GetResult(FlushResult flush)
        // tell the calling code whether any more messages
        // should be written
        => !(flush.IsCanceled || flush.IsCompleted);
    async ValueTask<bool> Awaited(ValueTask<FlushResult> incomplete)
        => GetResult(await incomplete);
    // apply back-pressure etc
    var flushTask = writer.FlushAsync();
    return flushTask.IsCompletedSuccessfully
        ? new ValueTask<bool>(GetResult(flushTask.Result))
        : Awaited(flushTask);
}
```
This *completely avoids* the `async`/`await` machinery in the most common case: synchronous completion - as we can see again on sharplab.io. I should emphasize: there's absolutely no point doing this if the code is usually (or exclusively) going to *actually be asynchronous*; it *only* helps when the result is usually (or exclusively) going to be available synchronously.

## And what about the reader?

As we've seen many times, the reader is often slightly more complicated - we can't know that a single "read" operation will contain exactly one inbound message. We may need to loop until we have all the data we need, and we may have *additional* data that we need to push back. So let's assume we want to consume a single message of some kind:

```
async ValueTask<SomeMessageType> GetNextMessage(
    PipeReader reader,
    CancellationToken cancellationToken = default)
{
    while (true)
    {
        var read = await reader.ReadAsync(cancellationToken);
        if (read.IsCanceled) ThrowCanceled();
        // can we find a complete frame?
        var buffer = read.Buffer;
        if (TryParseFrame(
            buffer,
            out SomeMessageType nextMessage,
            out SequencePosition consumedTo))
        {
            reader.AdvanceTo(consumedTo);
            return nextMessage;
        }
        reader.AdvanceTo(buffer.Start, buffer.End);
        if (read.IsCompleted) ThrowEOF();        
    }
}
```
Here we obtain *some* data from the pipe, checking exit conditions like cancelation. Next, we *try to find a message*; what this means depends on your exact code - this could mean:

- looking through the buffer for some sentinel value such as an ASCII line-ending, then treating everything up to that point as a message (discarding the line ending)
- parsing a well-defined binary frame header, obtaining the payload length, checking that we have that much data, and processing it
- or anything else you want!

If we *do* manage to find a message, we can tell the pipe to discard the data that we've consumed - by `AdvanceTo(consumedTo)`, which uses whatever our own frame-parsing code told us that we consumed. If we *don't* manage to find a message, the first thing to do is tell the pipe that we consumed nothing despite trying to read everything - by `reader.AdvanceTo(buffer.Start, buffer.End)`. At this point, there are two possibilities:

- we haven't got enough data *yet*
- the pipe is dead and there will *never* be enough data

Our check on `read.IsCompleted` tests this, reporting failure in the latter case; otherwise we continue the loop, and await more data. What is left, then, is our frame parsing - we've reduced complex IO management down to simple operations; for example, if our messages are separated by line-feed sentinels:

```
private static bool TryParseFrame(
    ReadOnlySequence<byte> buffer,
    out SomeMessageType nextMessage,
    out SequencePosition consumedTo)
{
    // find the end-of-line marker
    var eol = buffer.PositionOf((byte)'\n');
    if (eol == null)
    {
        nextMessage = default;
        consumedTo = default;
        return false;
    }
    // read past the line-ending
    consumedTo = buffer.GetPosition(1, eol.Value);
    // consume the data
    var payload = buffer.Slice(0, eol.Value);
    nextMessage = ReadSomeMessageType(payload);
    return true;
}
```
Here `PositionOf` tries to find the first location of a line-feed. If it can't find one, we give up. Otherwise, we set `consumedTo` to be "the line-feed plus one" (so we consume the line-feed), and we slice our buffer to create a sub-range that represents the payload *without* the line-feed, which we can then parse (however). Finally, we report success, and can rejoice at the simplicity of parsing linux-style line-endings.

## What's the point here?

With minimal code that is *very similar to the most naïve and simple `Stream` version* (without any nice features) our app code now has a reader and writer chain that *automatically* exploits a wide range of capabilities to ensure efficient and effective processing. Again, you *can do all these things* with `Stream`, but it is *really, really hard* to do well and reliably. By pushing all theses features into the framework, multiple code-bases can benefit from a single implementation. It also gives future scope for interesting custom pipeline endpoints and decorators that work directly on the pipeline API.

# Summary

In this section, we looked at the memory model used by pipelines, and how it helps us avoid allocations. Then we looked at how we might integrate pipelines into existing APIs and systems such a `Stream` - and we introduced `Pipelines.Sockets.Unofficial` as an available utility library. We looked at the options available for integrating span/memory code with APIs that don't offer those options, and finally we looked at what the *actual calling code* might look like when talking to pipelines (taking a brief side step into how to optimize `async` code that is usually synchronous) - showing what our *application* code might look like. In the third and final part, we'll look at how we combine all these learning points when looking at a real-world library such at `StackExchange.Redis` - discussing what complications the code needed to solve, and how pipelines made it simple to do so.
