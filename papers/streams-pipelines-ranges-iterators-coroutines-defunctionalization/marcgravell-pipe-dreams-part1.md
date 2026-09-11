---
title: Pipe Dreams, part 1
author: Marc Gravell
url: https://blog.marcgravell.com/2018/07/pipe-dreams-part-1.html
hostname: marcgravell.com
description: Pipelines - a guided tour of the new IO API in .NET, part 1 (part 2 here) About two years ago I blogged about an upcoming experimental IO ...
sitename: blog.marcgravell.com
date: "2018-07-02"
---
# Pipelines - a guided tour of the new IO API in .NET, part 1

About two years ago I blogged about an upcoming experimental IO API in the .NET world - at the time provisionally called "Channels"; at the end of May 2018, this finally shipped - under the name `System.IO.Pipelines`. I am hugely interested in the API, and over the last few weeks I'm been *consumed* with converting `StackExchange.Redis` to use "pipelines", as part of our 2.0 library update.

My hope in this series, then, is to discuss:

- what "pipelines" *are*
- how to use them in terms of code
- *when* you might want to use them

To help put this in concrete terms, after introducing "pipelines" I intend to draw heavily on the `StackExchange.Redis` conversion - and in particular by discussing which problems it solves for us in each scenario. Spoiler: in virtually all cases, the answer can be summarized as:

It perfectly fits a complex but common stumbling point in IO code; allowing us to replace an ugly kludge, workaround or compromise in *our* code - with a purpose-designed elegant solution that is in framework code.


I'm pretty sure that the pain points I'm going to cover below will be familiar to anyone who works at "data protocol" levels, and I'm equally sure that the hacks and messes that we'll be replacing with pipelines will be duplicated in a lot of code-bases.

## What do pipelines replace / complement?

The starting point here has to be: *what is the closest analogue in existing framework code?* And that is simple: `Stream`. The `Stream` API will be familiar to anyone who has worked with  serialization or data protocols. As an aside: `Stream` is actually a very ambiguous API - it works *very* differently in different scenarios:

- some streams are read-only, some are write-only, some are read-write
- the same *concrete type* can sometimes be read-only, and sometimes write-only (`DeflateStream` , for example)
- when a stream is read-write, sometimes it works like a cassette tape, where read and write are operating on the same underlying data (`FileStream` ,`MemoryStream` ); and sometimes it works like two separate streams, where read and write are essentially completely separate streams (`NetworkStream` ,`SslStream` ) - a*duplex stream*
- in many of the duplex cases, it is hard or impossible to express "no more data will be arriving, but you should continue to read the data to the end" - there's just `Close()` , which usually kills both halves of the duplex
- sometimes streams are seekable and support concepts like `Position` and`Length` ; often they're not
- because of the progression of APIs over time, there are often multiple ways of performing the same operation - for example, we could use `Read` (synchronous),`BeginRead` /`EndRead` (asynchronous using the`IAsyncResult` pattern), or`ReadAsync` (asynchronous using the`async` /`await` pattern); calling code has no way*in the general case* of knowing which of these is the "intended" (optimal) API
- if you use either of the asynchronous APIs, it is often unclear what the threading model is; will it always actually be synchronous? if not, what thread will be calling me back? does it use sync-context? thread-pool? IO completion-port threads?
- and more recently, there are also extensions to allow `Span<byte>` /`Memory<byte>` to be used in place of`byte[]` - again, the caller has no way of knowing which is the "preferred" API
- the nature of the API *encourages* copying data; need a buffer? that's a block-copy into another chunk of memory; need a backlog of data you haven't processed yet? block-copy into another chunk of memory; etc

So even before we start talking about real-world `Stream` examples and the problems that happen *when using it*, it is clear that there are a *lot* of problems in the `Stream` API *itself*. The first unsurprising news, then, is that pipelines sorts this mess out!

## What are pipelines?

By "pipelines", I mean a set of 4 key APIs that between them implement decoupled and overlapped reader/writer access to a binary stream (not `Stream`), including buffer management (pooling, recycling), threading awareness, rich backlog control, and over-fill protection via back-pressure - all based around an API designed around non-contiguous memory. That's a *heck* of a word salad - but don't worry, I'll be talking about each element to explain what I mean.

## Starting out simple: writing to, and reading from, a single pipe

Let's start with a `Stream` analogue, and write sometthing simple to a stream, and read it back - sticking to just the `Stream` API. We'll use ASCII text so we don't need to worry about any complex encoding concerns, and our read/write code shouldn't assume anything about the underlying stream. We'll just write the data, and then read to the end of the stream to consume it.

We'll do this with `Stream` first - familiar territory. Then we'll re-implement it with pipelines, to see where the similarities and differences lie. After that, we'll investigate what is actually happening under the hood, so we understand *why* this is interesting to us!

Also, before you say it: yes, I'm aware of `TextReader`/`TextWriter`; I'm not using them intentionally - because I'm trying to talk about the `Stream` API here, so that the example extends to a wide range of data protocols and scenarios.

```
using (MemoryStream ms = new MemoryStream())
{
    // write something
    WriteSomeData(ms);
    // rewind - MemoryStream works like a tape
    ms.Position = 0;
    // consume it
    ReadSomeData(ms);
}
```
Now, to write to a `Stream` the caller needs to obtain and populate a buffer which they then pass to the `Stream`. We'll keep it simple for now by using the synchronous API and simply allocating a `byte[]`:

```
void WriteSomeData(Stream stream)
{
    byte[] bytes = Encoding.ASCII.GetBytes("hello, world!");
    stream.Write(bytes, 0, bytes.Length);
    stream.Flush();
}
```
Note: there are *tons* of things in the above I could do for efficiency; but that isn't the point yet. So if you're familiar with this type of code and are twitching at the above... don't panic; we'll make it uglier - er, I mean *more efficient* - later.

The *reading* code is typically more complex than the writing code, because the reading code can't assume that it will get everything in a single call to `Read`. A read operation on a `Stream` can return nothing (which indicates the end of the data), or it could fill our buffer, or it could return a single byte despite being offered a huge buffer. So read code on a `Stream` is *almost always* a loop:

```
void ReadSomeData(Stream stream)
{
    int bytesRead;
    // note that the caller usually can't know much about
    // the size; .Length is not usually usable
    byte[] buffer = new byte[256];
    do
    {
        bytesRead = stream.Read(buffer, 0, buffer.Length);
        if (bytesRead > 0)
        {   // note this only works for single-byte encodings
            string s = Encoding.ASCII.GetString(
                buffer, 0, bytesRead);
            Console.Write(s);
        }
    } while (bytesRead > 0);
}
```
Now let's translate that to pipelines. A `Pipe` is broadly comparable to a `MemoryStream`, except instead of being able to rewind it many times, the data is more simply a "first in first out" queue. We have a *writer* API that can push data in at one end, and a *reader* API that can pull the data out at the other. The `Pipe` is the buffer that sits between the two. Let's reproduce our previous scenario, but using a single `Pipe` instead of the `MemoryStream` (again not something we'd usually do in practice, but it is simple to illustrate):

```
Pipe pipe = new Pipe();
// write something
await WriteSomeDataAsync(pipe.Writer);
// signal that there won't be anything else written
pipe.Writer.Complete();
// consume it
await ReadSomeDataAsync(pipe.Reader);
```
First we create a pipe using the default options, then we write to it. Note that IO operations on pipes are usually asynchronous, so we'll need to `await` our two helper methods. Note also that we don't pass the `Pipe` to them - unlike `Stream`, pipelines have separate API surfaces for read and write operations, so we pass a `PipeWriter` to the helper method that does our writing, and a `PipeReader` to the helper method that does our reading. After writing the data, we call `Complete()` on the `PipeWriter`. We didn't have to do this with the `MemoryStream` because it automatically EOFs when it reaches the end of the buffered data - but on some other `Stream` implementations - especially one-way streams - we might have had to call `Close` after writing the data.

OK, so what does `WriteSomeDataAsync` look like? Note, I've deliberately over-annotated here:

```
async ValueTask WriteSomeDataAsync(PipeWriter writer)
{
    // use an oversized size guess
    Memory<byte> workspace = writer.GetMemory(20);
    // write the data to the workspace
    int bytes = Encoding.ASCII.GetBytes(
        "hello, world!", workspace.Span);
    // tell the pipe how much of the workspace
    // we actually want to commit
    writer.Advance(bytes);
    // this is **not** the same as Stream.Flush!
    await writer.FlushAsync();
}
```
The first thing to note is that when dealing with pipelines: *you don't control the buffers*: the `Pipe` does. Recall how in our `Stream` code, both the read and write code created a local `byte[]`, but we don't have that here. Instead, we ask the `Pipe` for a buffer (`workspace`), via the `GetMemory` method (or it's twin - `GetSpan`). As you might expect from the name, this gives us either a `Memory<byte>` or a `Span<byte>` - of size *at least* twenty bytes.

Having obtained this buffer, we encode our `string` into it. This means that we're writing directly into the pipe's memory, and keep track of how many bytes we *actually used*, so we can tell it in `Advance`. We are under no obligation to use the twenty that we asked for: we could write zero, one, twenty, or even fifty bytes. The last one may seem surprising, but it is actually actively encouraged! The emphasis previously was on "*at least*" - the writer can actually give us a *much bigger* buffer than we ask for. When dealing with larger data, it is common to make modest requests but expect greatness: ask for the *minumum we can usefully utilize*, but then check the size of the memory/span that it gives us before deciding how much to actually write.

The call to `Advance` is important; this completes a single write operation, making the data available in the pipe to be consumed by a reader. The call to `FlushAsync` is *equally important*, but much more nuanced. However, before we can adequately describe what it does, we need to take a look at the reader. So; here's our `ReadSomeDataAsync` method:

```
async ValueTask ReadSomeDataAsync(PipeReader reader)
{
    while (true)
    {
        // await some data being available
        ReadResult read = await reader.ReadAsync();
        ReadOnlySequence<byte> buffer = read.Buffer;
        // check whether we've reached the end
        // and processed everything
        if (buffer.IsEmpty && read.IsCompleted)
            break; // exit loop
        // process what we received
        foreach (Memory<byte> segment in buffer)
        {
            string s = Encoding.ASCII.GetString(
                segment.Span);
            Console.Write(s);
        }
        // tell the pipe that we used everything
        reader.AdvanceTo(buffer.End);
    }
}
```
Just like with the `Stream` example, we have a loop that continues until we've reached the end of the data. With `Stream`, that is defined as being when `Read` returns a non-positive result, but with pipelines there are two things to check:

- `read.IsCompleted` tells us whether the write pipe has been signalled as completed and therefore no more data will be written (`pipe.Writer.Complete();` in our earlier code did this)
- `buffer.IsEmpty` tells us whether there is any data left to proces*in this iteration*

If there's nothing in the pipe now *and* the writer has been completed, then there will **never** be anything in the pipe, and we can exit.

If we *do* have data, then we can look at `buffer`. So first - let's talk about `buffer`; in the code it is a `ReadOnlySequence<byte>`, which is a new type - this concept combines a few roles:

- describing non-contiguous memory, speficially a sequence of zero, one or many `ReadOnlyMemory<byte>` chunks
- describing a logical position (`SequencePosition` ) in such a data-stream - in particular via`buffer.Start` and`buffer.End`

The *non-contiguous* is very important here. We'll look at where the data is actually going shortly, but in terms of reading: we need to be prepared to handle data that *could* be spread accross multiple segments. In this case, we do this by a simple `foreach` over the `buffer`, decoding each segment in turn. Note that even though the API is designed to be able to describe multiple non-contiguous buffers, it is frequently the case that the data received is contiguous in a single buffer; and in that case, it is often possible to write an optimized implementation for a single buffer. You can do that by checking `buffer.IsSingleSegment` and accessing `buffer.First`.

Finally, we call `AdvanceTo`, which tells the pipe *how much data we actually used*.

# Key point: *you don't need to take everything you are given!*

Contrast to `Stream`: when you call `Read` on a `Stream`, it puts data into the buffer you gave it. In most real-world scenarios, it isn't always possible to consume all the data yet - maybe it only makes sense to consider "commands" as "entire text lines", and you haven't yet seen a `cr`/`lf` in the data. With `Stream`: this is tough - once you've been given the data, it is your problem; if you can't use it yet, you need to store the backlog somewhere. However, with pipelines, *you can tell it* what you've consumed. In our case, we're telling it that we consumed everything we were given, which we do by passing `buffer.End` to `AdvanceTo`. That means we'll never see that data again, just like with `Stream`. However, we could also have passed `buffer.Start`, which would mean "we didn't use anything" - and *even though we had chance to inspect the data*, it would remain in the pipe for subsequent reads. We can also get arbitrary `SequencePosition` values inside the buffer - if we read 20 bytes, for example - so we have full control over how much data is dropped from the pipe. There are two ways of getting a `SequencePosition`:

- you can `Slice(...)` a`ReadOnlySequence<byte>` in the same way that you`Slice(...)` a`Span<T>` or`Memory<T>` - and access the`.Start` or`.End` of the resulting sub-range
- you can use the `.GetPosition(...)` method of the`ReadOnlySequence<byte>` , which returns a relative position*without* actually slicing

Even more subtle: we can tell it separetely that we *consumed* some amount, but that we *inspected* a different amount. The most common example here is to express "you can drop *this much* - I'm done with that; but I looked at everything, I can't make any more progress at the moment - I need more data" - specifically:

`reader.AdvanceTo(consumedToPosition, buffer.End);`
This is where the subtle interplay of `PipeWriter.FlushAsync()` and `PipeReader.ReadAsync()` starts to come into play. I skipped over `FlushAsync` earlier, but it actually serves two different functions in one call:

- if there is a `ReadAsync` call that is outstanding because it needs data, then it*awakens* the reader, allowing the read loop to continue
- if the writer is out-pacing the reader, such that the pipe is filling up with data that isn't being cleared by the reader, it can *suspend* the writer (by not completing synchronously) - to be reactivated when there is more space in the pipe (the thresholds for writer suspend/resume can be optionally specified when creating the`Pipe` instance)

Obviously these concepts don't come into play in our example, but they are central ideas to how pipelines works. The ability to push data back into the pipe *hugely* simplifies a vast range of IO scenarios. Virtually every piece of protocol handling code I've seen before pipelines has *masses* of code related to handling the backlog of incomplete data - it is such a repeated piece of logic that I am *incredibly* happy to see it handled well in a framework library instead.

## What does "awaken" or "reactivate" mean here?

You might have observed that I didn't really define what I meant here. At the *obvious* level, I mean that: an `await` operation of `ReadAsync` or `FlushAsync` had previously returned as incomplete, so now the asynchronous continuation gets invoked, allowing our `async` method to resume execution. Yeah, OK, but that's just re-stating what `async`/`await` *mean*. It is bug-bear of mine that I care *deeply* (really, it is alarming how deep) about which threads code runs on - for reasons that I'll talk about later in this series. So saying "the asynchronous continuation gets invoked" *isn't enough for me*. I want to understand *who is invoking it*, in terms of threads. The most common answers to this are:

- it delegates via the `SynchronizationContext` (note: many systems*do not have* a`SynchronizationContext` )
- the thread that *triggered the state change* gets used, at the point of the state change, to invoke the continuation
- the global thread-pool is used to invoke the continuation

All of these can be fine in some cases, and all of these can be terrible in some cases! Sync-context is a well-established mechanism for getting from worker threads back to primary application threads (epecially: the UI thread in desktop applications). However, it isn't necessarily the case that just because we've finished one IO operation, we're ready to jump back to an application thread; and doing so can effectively push a lot of IO code and data processing code *onto* an application thread - usually the one thing we explicitly want to avoid. Additionally, it can be prone to deadlocks if the application code has used `Wait()` or `.Result` on an asynchronous call (which, to be fair, you're not meant to do). The second option (performing the callback "inline" on the thread that triggered it) can be problematic because it can steal a thread that you expected to be doing something else (and can lead to deadlocks as a consequence); and in some extreme cases it can lead to a stack-dive (and eventually a stack-overflow) when two asynchronous methods are essentially functioning as co-routines. The final option (global thread-pool) is immune to the problems of the other two - but can run into severe problems under some load conditions - something again that I'll discuss in a later part in this series.

However, the good news is that *pipelines gives you control here*. When creating the `Pipe` instance, we can supply `PipeScheduler` instances to use for the reader and writer (separately). The `PipeScheduler` is used to perform these activations. If not specified, then it defaults *first* to checking for `SynchronizationContext`, then using the global thread-pool, with "inline" continuations (i.e. intionally using the thread that caused the state change) as another option readily available. But: *you can provide your own implementation* of a `PipeScheduler`, giving you full control of the threading model.

## Summary

So: we've looked at what a `Pipe` is when considered individually, and how we can write to a pipe with a `PipeWriter`, and read from a pipe with a `PipeReader` - and how to "advance" both reader and writer. We've looked at the similarity and differences with `Stream`, and we've discussed how `ReadAsync()` and `FlushAsync()` can interact to control how the writer and reader pieces execute. We looked at how responsibility for buffers is reversed, with the pipe providing all buffers - and how the pipe can simplify backlog management. Finally, we discussed the threading model that is active for continuations in the `await` operations.

That's probably enough for step 1; next, we'll look at how the *memory model* for pipelines works - i.e. where does the data live. We'll also look at *how we can use pipelines in real scenarios* to start doing interesting things.
