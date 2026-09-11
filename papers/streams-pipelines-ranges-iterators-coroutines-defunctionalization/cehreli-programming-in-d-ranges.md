---
title: "Ranges (Programming in D, chapter)"
author: Ali Çehreli
book: "Programming in D — Tutorial and Reference"
url: https://ddili.org/ders/d.en/ranges.html
date: "2009–2019 (book edition; chapter online)"
license: "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 United States"
license-url: https://ddili.org/copyright.html
copyright: "Copyleft (ɔ) 2009-2019 Ali Çehreli — Ddili.org content by Ali Cehreli, licensed CC BY-NC-SA 3.0 US; permissions beyond the scope of this license at http://ddili.org/copyright.html"
source-note: "This is the introductory ranges chapter (empty/front/popFront protocol; the InputRange/ForwardRange/RandomAccessRange taxonomy) of the free online D book."
---

#### Ranges

Ranges are an abstraction of element access. This abstraction enables the use of great number of algorithms over great number of container types. Ranges emphasize how container elements are accessed, as opposed to how the containers are implemented.

Ranges are a very simple concept that is based on whether a type defines certain sets of member functions. We have already seen this concept in the `foreach` with Structs and Classes chapter: any type that provides the member functions `empty`, `front`, and `popFront()` can be used with the `foreach` loop. The set of those three member functions is the requirement of the range type `InputRange`.

I will start introducing ranges with `InputRange`, the simplest of all the range types. The other ranges require more member functions over `InputRange`.

Before going further, I would like to provide the definitions of containers and algorithms.

  **Container (data structure):** Container is a very useful concept that appears in almost every program. Variables are put together for a purpose and are used together as elements of a container. D's containers are its core features arrays and associative arrays, and special container types that are defined in the `std.container` module. Every container is implemented as a specific data structure. For example, associative arrays are a *hash table* implementation.

Every data structure stores its elements and provides access to them in ways that are special to that data structure. For example, in the array data structure the elements are stored side by side and accessed by an element index; in the linked list data structure the elements are stored in nodes and are accessed by going through those nodes one by one; in a sorted binary tree data structure, the nodes provide access to the preceding and successive elements through separate branches; etc.

In this chapter, I will use the terms *container* and *data structure* interchangeably.

 **Algorithm (function):** Processing of data structures for specific purposes in specific ways is called an *algorithm*. For example, *linear search* is an algorithm that searches by iterating over a container from the beginning to the end; *binary search* is an algorithm that searches for an element by eliminating half of the candidates at every step; etc.

In this chapter, I will use the terms *algorithm* and *function* interchangeably.

For most of the samples below, I will use `int` as the element type and `int[]` as the container type. In reality, ranges are more powerful when used with templated containers and algorithms. In fact, most of the containers and algorithms that ranges tie together are all templates. I will leave examples of templated ranges to the next chapter.

##### History

A very successful library that abstracts algorithms and data structures from each other is the Standard Template Library (STL), which also appears as a part of the C++ standard library. STL provides this abstraction with the *iterator* concept, which is implemented by C++'s templates.

Although they are a very useful abstraction, iterators do have some weaknesses. D's ranges were designed to overcome these weaknesses.

Andrei Alexandrescu introduces ranges in his paper On Iteration and demonstrates how they can be superior to iterators.

##### Ranges are an integral part of D

D's slices happen to be implementations of the most powerful range `RandomAccessRange`, and there are many range features in Phobos. It is essential to understand how ranges are used in Phobos.

Many Phobos algorithms return temporary range objects. For example, `filter()`, which chooses elements that are greater than 10 in the following code, actually returns a range object, not an array:

import std.stdio;
import std.algorithm;
void main() {
    int[] values = [ 1, 20, 7, 11 ];
    writeln(values.filter!(value => value > 10));
}

`writeln` uses that range object lazily and accesses the elements as it needs them:

[20, 11]

That output may suggest that `filter()` returns an `int[]` but this is not the case. We can see this from the fact that the following assignment produces a compilation error:

    int[] chosen = values.filter!(value => value > 10); // ← compilation ERROR

The error message contains the type of the range object:

```
Error: cannot implicitly convert expression (filter(values))
of type FilterResult!(__lambda2, int[]) to int[]
```
**Note:** The type may be different in the version of Phobos that you are using.

It is possible to convert that temporary object to an actual array, as we will see later in the chapter.

##### Traditional implementations of algorithms

In traditional implementations of algorithms, the algorithms know how the data structures that they operate on are implemented. For example, the following function that prints the elements of a linked list must know that the nodes of the linked list have members named `element` and `next`:

struct Node {
    int element;
    Node * next;
}
void print(const(Node) * list) {
    for ( ; list; list = list.next) {
        write(' ', list.element);
    }
}

Similarly, a function that prints the elements of an array must know that arrays have a `length` property and their elements are accessed by the `[]` operator:

void print(const int[] array) {
    for (int i = 0; i != array.length; ++i) {
        write(' ', array[i]);
    }
}

**Note:** We know that `foreach` is more useful when iterating over arrays. As a demonstration of how traditional algorithms are tied to data structures, let's assume that the use of `for` is justified.

Having algorithms tied to data structures makes it necessary to write them specially for each type. For example, the functions find(), sort(), swap(), etc. must be written separately for array, linked list, associative array, binary tree, heap, etc. As a result, N algorithms that support M data structures must be written NxM times. (Note: In reality, the count is less than NxM because not every algorithm can be applied to every data structure; for example, associative arrays cannot be sorted.)

Conversely, because ranges abstract algorithms away from data structures, implementing just N algorithms and M data structures would be sufficient. A newly implemented data structure can work with all of the existing algorithms that support the type of range that the new data structure provides, and a newly implemented algorithm can work with all of the existing data structures that support the range type that the new algorithm requires.

##### Phobos ranges

The ranges in this chapter are different from number ranges that are written in the form `begin..end`. We have seen how number ranges are used with the `foreach` loop and with slices:

    foreach (value; 3..7) {       // number range,
                                  // NOT a Phobos range
    int[] slice = array[5..10];   // number range,
                                  // NOT a Phobos range

When I write *range* in this chapter, I mean a Phobos range .

Ranges form a *range hierarchy*. At the bottom of this hierarchy is the simplest range `InputRange`. The other ranges bring more requirements on top of the range on which they are based. The following are all of the ranges with their requirements, sorted from the simplest to the more capable:

- `InputRange` : requires the`empty` ,`front` and`popFront()` member functions
- `ForwardRange` : additionally requires the`save` member function
- `BidirectionalRange` : additionally requires the`back` and`popBack()` member functions
- `RandomAccessRange` : additionally requires the`[]` operator (and another property depending on whether the range is finite or infinite)

This hierarchy can be shown as in the following graph. `RandomAccessRange` has finite and infinite versions:

```
                    InputRange
                        ↑
                   ForwardRange
                   ↗         ↖
     BidirectionalRange    RandomAccessRange (infinite)
             ↑
  RandomAccessRange (finite)
```
The graph above is in the style of class hierarchies where the lowest level type is at the top.

Those ranges are about providing element access. There is one more range, which is about element *output*:

- `OutputRange` : requires support for the`put(range, element)` operation

These five range types are sufficient to abstract algorithms from data structures.

###### Iterating by shortening the range

Normally, iterating over the elements of a container does not change the container itself. For example, iterating over a slice with `foreach` or `for` does not affect the slice:

    int[] slice = [ 10, 11, 12 ];
    for (int i = 0; i != slice.length; ++i) {
        write(' ', slice[i]);
    }
    assert(slice.length == 3);  // ← the length doesn't change

Another way of iteration requires a different way of thinking: iteration can be achieved by shortening the range from the beginning. In this method, it is always the first element that is used for element access and the first element is *popped* from the beginning in order to get to the next element:

    for ( ; slice.length; slice = slice[1..$]) {
        write(' ', slice[0]);   // ← always the first element
    }

*Iteration* is achieved by removing the first element by the `slice = slice[1..$]` expression. The slice above is completely consumed by going through the following stages:

```
[ 10, 11, 12 ]
    [ 11, 12 ]
        [ 12 ]
           [ ]
```
The iteration concept of Phobos ranges is based on this new thinking of shortening the range from the beginning. (`BidirectionalRange` and finite `RandomAccessRange` types can be shortened from the end as well.)

Please note that the code above is just to demonstrate this type of iteration; it should not be considered normal to iterate as in that example.

Since losing elements just to iterate over a range would not be desired in most cases, a surrogate range may be consumed instead. The following code uses a separate slice to preserve the elements of the original one:

    int[] slice = [ 10, 11, 12 ];
    int[] surrogate = slice;
    for ( ; surrogate.length; surrogate = surrogate[1..$]) {
        write(' ', surrogate[0]);
    }
    assert(surrogate.length == 0); // ← surrogate is consumed
    assert(slice.length == 3);     // ← slice remains the same

This is the method employed by most of the Phobos range functions: they return special range objects to be consumed in order to preserve the original containers.

#####  `InputRange`

This type of range models the type of iteration where elements are accessed in sequence as we have seen in the `print()` functions above. Most algorithms only require that elements are iterated in the forward direction without needing to look at elements that have already been iterated over. `InputRange` models the standard input streams of programs as well, where elements are removed from the stream as they are read.

For completeness, here are the three functions that `InputRange` requires:

-  `empty` : specifies whether the range is empty; it must return`true` when the range is considered to be empty, and`false` otherwise
-  `front` : provides access to the element at the beginning of the range
-  `popFront()` : shortens the range from the beginning by removing the first element

**Note:** I write `empty` and `front` without parentheses, as they can be seen as properties of the range; and `popFront()` with parentheses as it is a function with side effects.

Here is how `print()` can be implemented by using these range functions:

void print(T)(T range) {
    for ( ; !range.empty; range.popFront()) {
        write(' ', range.front);
    }
    writeln();
}

Please also note that `print()` is now a function template to avoid limiting the range type arbitrarily. `print()` can now work with any type that provides the three `InputRange` functions.

###### `InputRange` example

Let's redesign the `School` type that we have seen before, this time as an `InputRange`. We can imagine `School` as a `Student` container so when designed as a range, it can be seen as a range of `Student`s.

In order to keep the example short, let's disregard some important design aspects. Let's

- implement only the members that are related to this section
- design all types as structs
- ignore specifiers and qualifiers like `private` ,`public` , and`const`
- not take advantage of contract programming and unit testing

import std.string;
struct Student {
    string name;
    int number;
    string toString() const {
        return format("%s(%s)", name, number);
    }
}
struct School {
    Student[] students;
}
void main() {
    auto school = School( [ Student("Ebru", 1),
                            Student("Derya", 2) ,
                            Student("Damla", 3) ] );
}

To make `School` be accepted as an `InputRange`, we must define the three `InputRange` member functions.

For `empty` to return `true` when the range is empty, we can use the length of the `students` array. When the length of that array is 0, the range is considered empty:

struct School {
    // ...
    bool empty() const {
        return students.length == 0;
    }
}

For `front` to return the first element of the range, we can return the first element of the array:

struct School {
    // ...
    ref Student front() {
        return students[0];
    }
}

**Note:** I have used the `ref` keyword to be able to provide access to the actual element instead of a copy of it. Otherwise the elements would be copied because `Student` is a struct.

For `popFront()` to shorten the range from the beginning, we can shorten the `students` array from the beginning:

struct School {
    // ...
    void popFront() {
        students = students[1 .. $];
    }
}

**Note:** As I have mentioned above, it is not normal to lose the original elements from the container just to iterate over them. We will address this issue below by introducing a special range type.

These three functions are sufficient to make `School` to be used as an `InputRange`. As an example, let's add the following line at the end of `main()` above to have our new `print()` function template to use `school` as a range:

    print(school);

`print()` uses that object as an `InputRange` and prints its elements to the output:

 Ebru(1) Derya(2) Damla(3)

We have achieved our goal of defining a user type as an `InputRange`; we have sent it to an algorithm that operates on `InputRange` types. `School` is actually ready to be used with algorithms of Phobos or any other library that work with `InputRange` types. We will see examples of this below.

######  The `std.array` module to use slices as ranges

Merely importing the `std.array` module makes the most common container type conform to the most capable range type: slices can seamlessly be used as `RandomAccessRange` objects.

The `std.array` module provides the functions `empty`, `front`, `popFront()` and other range functions for slices. As a result, slices are ready to be used with any range function, for example with `print()`:

import std.array;
// ...
    print([ 1, 2, 3, 4 ]);

It is not necessary to import `std.array` if the `std.range` module has already been imported.

Since it is not possible to remove elements from fixed-length arrays, `popFront()` cannot be defined for them. For that reason, fixed-length arrays cannot be used as ranges themselves:

void print(T)(T range) {
    for ( ; !range.empty; range.popFront()) {  // ← compilation ERROR
        write(' ', range.front);
    }
    writeln();
}
void main() {
    int[4] array = [ 1, 2, 3, 4 ];
    print(array);
}

It would be better if the compilation error appeared on the line where `print()` is called. This is possible by adding a template constraint to `print()`. The following template constraint takes advantage of `isInputRange`, which we will see in the next chapter. By the help of the template constraint, now the compilation error is for the line where `print()` is called, not for a line where `print()` is defined:

void print(T)(T range)
        if (isInputRange!T) {    // template constraint
    // ...
}
// ...
    print(array);    // ← compilation ERROR

The elements of a fixed-length array can still be accessed by range functions. What needs to be done is to use a slice of the whole array, not the array itself:

    print(array[]);    // now compiles

Even though slices can be used as ranges, not every range type can be used as an array. When necessary, all of the elements can be copied one by one into an array. `std.array.array` is a helper function to simplify this task; `array()` iterates over `InputRange` ranges, copies the elements, and returns a new array:

import std.array;
// ...
    // Note: Also taking advantage of UFCS
    auto copiesOfStudents = school.array;
    writeln(copiesOfStudents);

The output:

[Ebru(1), Derya(2), Damla(3)]

Also note the use of UFCS in the code above. UFCS goes very well with range algorithms by making code naturally match the execution order of expressions.

######    Automatic decoding of strings as ranges of `dchar`

Being character arrays by definition, strings can also be used as ranges just by importing `std.array`. However, `char` and `wchar` strings cannot be used as `RandomAccessRange`.

`std.array` provides a special functionality with all types of strings: Iterating over strings becomes iterating over Unicode code points, not over UTF code units. As a result, strings appear as ranges of Unicode characters.

The following strings contain ç and é, which cannot be represented by a single `char`, and 𝔸 (mathematical double-struck capital A), which cannot be represented by a single `wchar` (note that these characters may not be displayed correctly in the environment that you are reading this chapter):

import std.array;
// ...
    print("abcçdeé𝔸"c);
    print("abcçdeé𝔸"w);
    print("abcçdeé𝔸"d);

The output of the program is what we would normally expect from a *range of letters*:

 a b c ç d e é 𝔸
 a b c ç d e é 𝔸
 a b c ç d e é 𝔸

As you can see, that output does not match what we saw in the Characters and Strings chapters. We have seen in those chapters that `string` is an alias to an array of `immutable(char)` and `wstring` is an alias to an array of `immutable(wchar)`. Accordingly, one might expect to see UTF code units in the previous output instead of the properly decoded Unicode characters.

The reason why the characters are displayed correctly is because when used as ranges, string elements are automatically decoded. As we will see below, the decoded `dchar` values are not actual elements of the strings but rvalues.

As a reminder, let's consider the following function that treats the strings as arrays of code units:

void printElements(T)(T str) {
    for (int i = 0; i != str.length; ++i) {
        write(' ', str[i]);
    }
    writeln();
}
// ...
    printElements("abcçdeé𝔸"c);
    printElements("abcçdeé𝔸"w);
    printElements("abcçdeé𝔸"d);

When the characters are accessed directly by indexing, the elements of the arrays are not decoded:

 a b c � � d e � � � � � �
 a b c ç d e é ��� ���
 a b c ç d e é 𝔸

 Automatic decoding is not always the desired behavior. For example, the following program that is trying to assign to the first element of a string cannot be compiled because the return value of `.front` is an rvalue:

import std.array;
void main() {
    char[] s = "hello".dup;
    s.front = 'H';                   // ← compilation ERROR
}

```
Error: front(s) is not an lvalue
```
When a range algorithm needs to modify the actual code units of a string (and when doing so does not invalidate the UTF encoding), then the string can be used as a range of `ubyte` elements by `std.string.represention`:

import std.array;
import std.string;
void main() {
    char[] s = "hello".dup;
    s.representation.front = 'H';    // compiles
    assert(s == "Hello");
}

`representation` presents the actual elements of `char`, `wchar`, and `dchar` strings as ranges of `ubyte`, `ushort`, and `uint`, respectively.

###### Ranges without actual elements

The elements of the `School` objects were actually stored in the `students` member slices. So, `School.front` returned a reference to an existing `Student` object.

One of the powers of ranges is the flexibility of not actually owning elements. `front` need not return an actual element of an actual container. The returned *element* can be calculated each time when `popFront()` is called, and can be used as the value that is returned by `front`.

We have already seen a range without actual elements above: Since `char` and `wchar` cannot represent all Unicode characters, the Unicode characters that appear as range elements cannot be actual elements of any `char` or `wchar` array. In the case of strings, `front` returns a `dchar` that is *constructed* from the corresponding UTF code units of arrays:

import std.array;
void main() {
    dchar letter = "é".front; // The dchar that is returned by
                              // front is constructed from the
                              // two chars that represent é
}

Although the element type of the array is `char`, the return type of `front` above is `dchar`. That `dchar` is not an element of the array but is an rvalue decoded as a Unicode character from the elements of the array.

Similarly, some ranges do not own any elements but are used for providing access to elements of other ranges. This is a solution to the problem of losing elements while iterating over `School` objects above. In order to preserve the elements of the actual `School` objects, a special `InputRange` can be used.

To see how this is done, let's define a new struct named `StudentRange` and move all of the range member functions from `School` to this new struct. Note that `School` itself is not a range anymore:

struct School {
    Student[] students;
}
struct StudentRange {
    Student[] students;
    this(School school) {
        this.students = school.students;
    }
    bool empty() const {
        return students.length == 0;
    }
    ref Student front() {
        return students[0];
    }
    void popFront() {
        students = students[1 .. $];
    }
}

The new range starts with a member slice that provides access to the students of `School` and consumes that member slice in `popFront()`. As a result, the actual slice in `School` is preserved:

    auto school = School( [ Student("Ebru", 1),
                            Student("Derya", 2) ,
                            Student("Damla", 3) ] );
    print(StudentRange(school));
    // The actual array is now preserved:
    assert(school.students.length == 3);

**Note:** Since all its work is dispatched to its member slice, `StudentRange` may not be seen as a good example of a range. In fact, assuming that `students` is an accessible member of `School`, the user code could have created a slice of `School.students` directly and could have used that slice as a range.

###### Infinite ranges

Another benefit of not storing elements as actual members is the ability to create infinite ranges.

Making an infinite range is as simple as having `empty` always return `false`. Since it is constant, `empty` need not even be a function and can be defined as an `enum` value:

    enum empty = false;                   // ← infinite range

Another option is to use an immutable `static` member:

    static immutable empty = false;       // same as above

As an example of this, let's design a range that represents the Fibonacci series. Despite having only two `int` members, the following range can be used as the infinite Fibonacci series:

struct FibonacciSeries
{
    int current = 0;
    int next = 1;
    enum empty = false;   // ← infinite range
    int front() const {
        return current;
    }
    void popFront() {
        const nextNext = current + next;
        current = next;
        next = nextNext;
    }
}

**Note:** Although it is infinite, because the members are of type `int`, the elements of this Fibonacci series would be wrong beyond `int.max`.

Since `empty` is always `false` for `FibonacciSeries` objects, the `for` loop in `print()` never terminates for them:

```
    print(FibonacciSeries());    // never terminates
```
An infinite range is useful when the range need not be consumed completely right away. We will see how to use only some of the elements of a `FibonacciSeries` below.

###### Functions that return ranges

Earlier, we have created a `StudentRange` object by explicitly writing `StudentRange(school)`.

 In most cases, a convenience function that returns the object of such a range is used instead. For example, a function with the whole purpose of returning a `StudentRange` would simplify the code:

StudentRange studentsOf(ref School school) {
    return StudentRange(school);
}
// ...
    // Note: Again, taking advantage of UFCS
    print(school.studentsOf);

This is a convenience over having to remember and spell out the names of range types explicitly, which can get quite complicated in practice.

 We can see an example of this with the simple `std.range.take` function. `take()` is a function that provides access to a specified number of elements of a range, from the beginning. In reality, this functionality is not achieved by the `take()` function itself, but by a special range object that it returns. This fact need not be explicit when using `take()`:

import std.range;
// ...
    auto school = School( [ Student("Ebru", 1),
                            Student("Derya", 2) ,
                            Student("Damla", 3) ] );
    print(school.studentsOf.take(2));

`take()` returns a temporary range object above, which provides access to the first 2 elements of `school`. In turn, `print()` uses that object and produces the following output:

 Ebru(1) Derya(2)

The operations above still don't make any changes to `school`; it still has 3 elements:

```
    print(school.studentsOf.take(2));
    assert(school.students.length == 3);
```
The specific types of the range objects that are returned by functions like `take()` are not important. These types may sometimes be exposed in error messages, or we can print them ourselves with the help of `typeof` and `stringof`:

```
    writeln(typeof(school.studentsOf.take(2)).stringof);
```
According to the output, `take()` returns an instance of a template named `Take`:

Take!(StudentRange)

######   `std.range` and `std.algorithm` modules

A great benefit of defining our types as ranges is being able to use them not only with our own functions, but with Phobos and other libraries as well.

`std.range` includes a large number of range functions, structs, and classes. `std.algorithm` includes many algorithms that are commonly found also in the standard libraries of other languages.

 To see an example of how our types can be used with standard modules, let's use `School` with the `std.algorithm.swapFront` algorithm. `swapFront()` swaps the front elements of two `InputRange` ranges. (It requires that the front elements of the two ranges are swappable. Arrays satisfy that condition.)


import std.algorithm;
// ...
    auto turkishSchool = School( [ Student("Ebru", 1),
                                   Student("Derya", 2) ,
                                   Student("Damla", 3) ] );
    auto americanSchool = School( [ Student("Mary", 10),
                                    Student("Jane", 20) ] );
    swapFront(turkishSchool.studentsOf,
              americanSchool.studentsOf);
    print(turkishSchool.studentsOf);
    print(americanSchool.studentsOf);

The first elements of the two schools are swapped:

 Mary(10) Derya(2) Damla(3)
 Ebru(1) Jane(20)

 As another example, let's now look at the `std.algorithm.filter` algorithm. `filter()` returns a special range that filters out elements that do not satisfy a specific condition (a *predicate*). The operation of filtering out the elements only affects accessing the elements; the original range is preserved.

 Predicates are expressions that must evaluate to `true` for the elements that are considered to satisfy a condition, and `false` for the elements that do not. There are a number of ways of specifying the predicate that `filter()` should use. As we have seen in earlier examples, one way is to use a lambda expression. The parameter `a` below represents each student:

    school.studentsOf.filter!(a => a.number % 2)

The predicate above selects the elements of the range `school.studentsOf` that have odd numbers.

Like `take()`, `filter()` returns a special range object as well. That range object in turn can be passed to other range functions. For example, it can be passed to `print()`:

    print(school.studentsOf.filter!(a => a.number % 2));

That expression can be explained as *start with the range `school.studentsOf`, construct a range object that will filter out the elements of that initial range, and pass the new range object to `print()`*.

The output consists of students with odd numbers:

 Ebru(1) Damla(3)

As long as it returns `true` for the elements that satisfy the condition, the predicate can also be specified as a function:

import std.array;
// ...
    bool startsWithD(Student student) {
        return student.name.front == 'D';
    }
    print(school.studentsOf.filter!startsWithD);

The predicate function above returns `true` for students having names starting with the letter D, and `false` for the others.

**Note:** Using `student.name[0]` would have meant the first UTF-8 code unit, not the first letter. As I have mentioned above, `front` uses `name` as a range and always returns the first Unicode character.

This time the students whose names start with D are selected and printed:

 Derya(2) Damla(3)

 `generate()`, a convenience function template of the `std.range` module, makes it easy to present values returned from a function as the elements of an `InputRange`. It takes any callable entity (function pointer, delegate, etc.) and returns an `InputRange` object conceptually consisting of the values that are returned from that callable entity.

The returned range object is infinite. Every time the `front` property of that range object is accessed, the original callable entity is called to get a new *element* from it. The `popFront()` function of the range object does not perform any work.

For example, the following range object `diceThrower` can be used as an infinite range:

import std.stdio;
import std.range;
import std.random;
void main() {
    auto diceThrower = generate!(() => uniform(0, 6));
    writeln(diceThrower.take(10));
}

[1, 0, 3, 5, 5, 1, 5, 1, 0, 4]

###### Laziness

Another benefit of functions' returning range objects is that, those objects can be used lazily. Lazy ranges produce their elements one at a time and only when needed. This may be essential for execution speed and memory consumption. Indeed, the fact that infinite ranges can even exist is made possible by ranges being lazy.

Lazy ranges produce their elements one at a time and only when needed. We see an example of this with the `FibonacciSeries` range: The elements are calculated by `popFront()` only as they are needed. If `FibonacciSeries` were an eager range and tried to produce all of the elements up front, it could never end or find room for the elements that it produced.

Another problem of eager ranges is the fact that they would have to spend time and space for elements that would perhaps never going to be used.

Like most of the algorithms in Phobos, `take()` and `filter()` benefit from laziness. For example, we can pass `FibonacciSeries` to `take()` and have it generate a finite number of elements:

    print(FibonacciSeries().take(10));

Although `FibonacciSeries` is infinite, the output contains only the first 10 numbers:

 0 1 1 2 3 5 8 13 21 34

#####  `ForwardRange`

`InputRange` models a range where elements are taken out of the range as they are iterated over.

Some ranges are capable of saving their states, as well as operating as an `InputRange`. For example, `FibonacciSeries` objects can save their states because these objects can freely be copied and the two copies continue their lives independently from each other.

 `ForwardRange` provides the `save` member function, which is expected to return a copy of the range. The copy that `save` returns must operate independently from the range object that it was copied from: iterating over one copy must not affect the other copy.

Importing `std.array` automatically makes slices become `ForwardRange` ranges.

In order to implement `save` for `FibonacciSeries`, we can simply return a copy of the object:

struct FibonacciSeries {
// ...
    FibonacciSeries save() const {
        return this;
    }
}

The returned copy is a separate range that would continue from the point where it was copied from.

 We can demonstrate that the copied object is independent from the actual range with the following program. The algorithm `std.range.popFrontN()` in the following code removes a specified number of elements from the specified range:

import std.range;
// ...
void report(T)(const dchar[] title, const ref T range) {
    writefln("%40s: %s", title, range.take(5));
}
void main() {
    auto range = FibonacciSeries();
    report("Original range", range);
    range.popFrontN(2);
    report("After removing two elements", range);
    auto theCopy = range.save;
    report("The copy", theCopy);
    range.popFrontN(3);
    report("After removing three more elements", range);
    report("The copy", theCopy);
}

The output of the program shows that removing elements from the range does not affect its saved copy:

```
                          Original range: [0, 1, 1, 2, 3]
             After removing two elements: [1, 2, 3, 5, 8]
                                The copy: [1, 2, 3, 5, 8]
      After removing three more elements: [5, 8, 13, 21, 34]
                                The copy: [1, 2, 3, 5, 8]
```
Also note that the range is passed directly to `writefln` in `report()`. Like our `print()` function, the output functions of the `stdio` module can take `InputRange` objects. I will use `stdio`'s output functions from now on.

 An algorithm that works with `ForwardRange` is `std.range.cycle`. `cycle()` iterates over the elements of a range repeatedly from the beginning to the end. In order to be able to start over from the beginning it must be able to save a copy of the initial state of the range, so it requires a `ForwardRange`.

Since `FibonacciSeries` is now a `ForwardRange`, we can try `cycle()` with a `FibonacciSeries` object; but in order to avoid having `cycle()` iterate over an infinite range, and as a result never find the end of it, we must first make a finite range by passing `FibonacciSeries` through `take()`:

    writeln(FibonacciSeries().take(5).cycle.take(20));

In order to make the resultant range finite as well, the range that is returned by `cycle` is also passed through `take()`. The output consists of *the first twenty elements of cycling through the first five elements of `FibonacciSeries`*:

[0, 1, 1, 2, 3, 0, 1, 1, 2, 3, 0, 1, 1, 2, 3, 0, 1, 1, 2, 3]

We could have defined intermediate variables as well. The following is an equivalent of the single-line code above:

    auto series                   = FibonacciSeries();
    auto firstPart                = series.take(5);
    auto cycledThrough            = firstPart.cycle;
    auto firstPartOfCycledThrough = cycledThrough.take(20);
    writeln(firstPartOfCycledThrough);

I would like to point out the importance of laziness one more time: The first four lines above merely construct range objects that will eventually produce the elements. The numbers that are part of the result are calculated by `FibonacciSeries.popFront()` as needed.

**Note:** Although we have started with `FibonacciSeries` as a `ForwardRange`, we have actually passed the result of `FibonacciSeries().take(5)` to `cycle()`. `take()` is adaptive: the range that it returns is a `ForwardRange` if its parameter is a `ForwardRange`. We will see how this is accomplished with `isForwardRange` in the next chapter.

#####  `BidirectionalRange`

  `BidirectionalRange` provides two member functions over the member functions of `ForwardRange`. `back` is similar to `front`: it provides access to the last element of the range. `popBack()` is similar to `popFront()`: it removes the last element from the range.

Importing `std.array` automatically makes slices become `BidirectionalRange` ranges.

 A good `BidirectionalRange` example is the `std.range.retro` function. `retro()` takes a `BidirectionalRange` and ties its `front` to `back`, and `popFront()` to `popBack()`. As a result, the original range is iterated over in reverse order:

    writeln([ 1, 2, 3 ].retro);

The output:

[3, 2, 1]

Let's define a range that behaves similarly to the special range that `retro()` returns. Although the following range has limited functionality, it shows how powerful ranges are:

import std.array;
import std.stdio;
struct Reversed {
    int[] range;
    this(int[] range) {
        this.range = range;
    }
    bool empty() const {
        return range.empty;
    }
    int front() const {
        return range.back;  // ← reverse
    }
    int back() const {
        return range.front; // ← reverse
    }
    void popFront() {
        range.popBack();    // ← reverse
    }
    void popBack() {
        range.popFront();   // ← reverse
    }
}
void main() {
    writeln(Reversed([ 1, 2, 3]));
}

The output is the same as `retro()`:

[3, 2, 1]

#####  `RandomAccessRange`

 `RandomAccessRange` represents ranges that allow accessing elements by the `[]` operator. As we have seen in the Operator Overloading chapter, `[]` operator is defined by the `opIndex()` member function.

Importing `std.array` module makes slices become `RandomAccessRange` ranges only if possible. For example, since UTF-8 and UTF-16 encodings do not allow accessing Unicode characters by an index, `char` and `wchar` arrays cannot be used as `RandomAccessRange` ranges of Unicode characters. On the other hand, since the codes of the UTF-32 encoding correspond one-to-one to Unicode character codes, `dchar` arrays can be used as `RandomAccessRange` ranges of Unicode characters.

 It is natural that every type would define the `opIndex()` member function according to its functionality. However, computer science has an expectation on its algorithmic complexity: random access must take *constant time*. Constant time access means that the time spent when accessing an element is independent of the number of elements in the container. Therefore, no matter how large the range is, element access should not depend on the length of the range.

In order to be considered a `RandomAccessRange`, *one* of the following conditions must also be satisfied:

- to be an infinite `ForwardRange`

or

Depending on the condition that is satisfied, the range is either infinite or finite.

###### Infinite `RandomAccessRange`

The following are all of the requirements of a `RandomAccessRange` that is based on an *infinite `ForwardRange`*:

- `empty` ,`front` and`popFront()` that`InputRange` requires
- `save` that`ForwardRange` requires
- `opIndex()` that`RandomAccessRange` requires
- the value of `empty` to be known at compile time as`false`

We were able to define `FibonacciSeries` as a `ForwardRange`. However, `opIndex()` cannot be implemented to operate at constant time for `FibonacciSeries` because accessing an element requires accessing all of the previous elements first.

As an example where `opIndex()` can operate at constant time, let's define an infinite range that consists of squares of integers. Although the following range is infinite, accessing any one of its elements can happen at constant time:

class SquaresRange {
    int first;
    this(int first = 0) {
        this.first = first;
    }
    enum empty = false;
    int front() const {
        return opIndex(0);
    }
    void popFront() {
        ++first;
    }
    SquaresRange save() const {
        return new SquaresRange(first);
    }
    int opIndex(size_t index) const {
         /* This function operates at constant time */
        immutable integerValue = first + cast(int)index;
        return integerValue * integerValue;
    }
}

**Note:** It would make more sense to define `SquaresRange` as a `struct`.

Although no space has been allocated for the elements of this range, the elements can be accessed by the `[]` operator:

    auto squares = new SquaresRange();
    writeln(squares[5]);
    writeln(squares[10]);

The output contains the elements at indexes 5 and 10:

25
100

The element with index 0 should always represent the first element of the range. We can take advantage of `popFrontN()` when testing whether this really is the case:

```
    squares.popFrontN(5);
    writeln(squares[0]);
```
The first 5 elements of the range are 0, 1, 4, 9 and 16; the squares of 0, 1, 2, 3 and 4. After removing those, the square of the next value becomes the first element of the range:

25

Being a `RandomAccessRange` (the most functional range), `SquaresRange` can also be used as other types of ranges. For example, as an `InputRange` when passing to `filter()`:

    bool are_lastTwoDigitsSame(int value) {
        /* Must have at least two digits */
        if (value < 10) {
            return false;
        }
        /* Last two digits must be divisible by 11 */
        immutable lastTwoDigits = value % 100;
        return (lastTwoDigits % 11) == 0;
    }
    writeln(squares.take(50).filter!are_lastTwoDigitsSame);

The output consists of elements among the first 50, where last two digits are the same:

[100, 144, 400, 900, 1444, 1600]

###### Finite `RandomAccessRange`

The following are all of the requirements of a `RandomAccessRange` that is based on a *finite `BidirectionalRange`*:

- `empty` ,`front` and`popFront()` that`InputRange` requires
- `save` that`ForwardRange` requires
- `back` and`popBack()` that`BidirectionalRange` requires
- `opIndex()` that`RandomAccessRange` requires
- `length` , which provides the length of the range

 As an example of a finite `RandomAccessRange`, let's define a range that works similarly to `std.range.chain`. `chain()` presents the elements of a number of separate ranges as if they are elements of a single larger range. Although `chain()` works with any type of element and any type of range, to keep the example short, let's implement a range that works only with `int` slices.

Let's name this range `Together` and expect the following behavior from it:

```
    auto range = Together([ 1, 2, 3 ], [ 101, 102, 103]);
    writeln(range[4]);
```
When constructed with the two separate arrays above, `range` should present all of those elements as a single range. For example, although neither array has an element at index 4, the element 102 should be the element that corresponds to index 4 of the collective range:

102

As expected, printing the entire range should contain all of the elements:

    writeln(range);

The output:

[1, 2, 3, 101, 102, 103]

`Together` will operate lazily: the elements will not be copied to a new larger array; they will be accessed from the original slices.

We can take advantage of *variadic functions*, which were introduced in the Variable Number of Parameters chapter, to initialize the range by any number of original slices:

struct Together {
    const(int)[][] slices;
    this(const(int)[][] slices...) {
        this.slices = slices.dup;
        clearFront();
        clearBack();
    }
// ...
}

Note that the element type is `const(int)`, indicating that this `struct` will not modify the elements of the ranges. However, the slices will necessarily be modified by `popFront()` to implement iteration.

The `clearFront()` and `clearBack()` calls that the constructor makes are to remove empty slices from the beginning and the end of the original slices. Such empty slices do not change the behavior of `Together` and removing them up front will simplify the implementation:

struct Together {
// ...
    private void clearFront() {
        while (!slices.empty && slices.front.empty) {
            slices.popFront();
        }
    }
    private void clearBack() {
        while (!slices.empty && slices.back.empty) {
            slices.popBack();
        }
    }
}

We will call those functions later from `popFront()` and `popBack()` as well.

Since `clearFront()` and `clearBack()` remove all of the empty slices from the beginning and the end, still having a slice would mean that the collective range is not yet empty. In other words, the range should be considered empty only if there is no slice left:

struct Together {
// ...
    bool empty() const {
        return slices.empty;
    }
}

The first element of the first slice is the first element of this `Together` range:

struct Together {
// ...
    int front() const {
        return slices.front.front;
    }
}

Removing the first element of the first slice removes the first element of this range as well. Since this operation may leave the first slice empty, we must call `clearFront()` to remove that empty slice and the ones that are after that one:

struct Together {
// ...
    void popFront() {
        slices.front.popFront();
        clearFront();
    }
}

A copy of this range can be constructed from a copy of the `slices` member:

struct Together {
// ...
    Together save() const {
        return Together(slices.dup);
    }
}

*Please note that `.dup` copies only `slices` in this case, not the slice elements that it contains.*

The operations at the end of the range are similar to the ones at the beginning:

struct Together {
// ...
    int back() const {
        return slices.back.back;
    }
    void popBack() {
        slices.back.popBack();
        clearBack();
    }
}

The length of the range can be calculated as the sum of the lengths of the slices:

struct Together {
// ...
    size_t length() const {
        size_t totalLength = 0;
        foreach (slice; slices) {
            totalLength += slice.length;
        }
        return totalLength;
    }
}

 Alternatively, the length may be calculated with less code by taking advantage of `std.algorithm.fold`. `fold()` takes an operation as its template parameter and applies that operation to all elements of a range:

import std.algorithm;
// ...
    size_t length() const {
        return slices.fold!((a, b) => a + b.length)(size_t.init);
    }

The `a` in the template parameter represents the current result (*the sum* in this case) and `b` represents the current element. The first function parameter is the range that contains the elements and the second function parameter is the initial value of the result (`size_t.init` is 0). (Note how `slices` is written before `fold` by taking advantage of UFCS.)

**Note:** Further, instead of calculating the length every time when `length` is called, it may be measurably faster to maintain a member variable perhaps named `length_`, which always equals the correct length of the collective range. That member may be calculated once in the constructor and adjusted accordingly as elements are removed by `popFront()` and `popBack()`.

One way of returning the element that corresponds to a specific index is to look at every slice to determine whether the element would be among the elements of that slice:

struct Together {
// ...
    int opIndex(size_t index) const {
        /* Save the index for the error message */
        immutable originalIndex = index;
        foreach (slice; slices) {
            if (slice.length > index) {
                return slice[index];
            } else {
                index -= slice.length;
            }
        }
        throw new Exception(
            format("Invalid index: %s (length: %s)",
                   originalIndex, this.length));
    }
}

**Note:** This `opIndex()` does not satisfy the constant time requirement that has been mentioned above. For this implementation to be acceptably fast, the `slices` member must not be too long.

This new range is now ready to be used with any number of `int` slices. With the help of `take()` and `array()`, we can even include the range types that we have defined earlier in this chapter:

    auto range = Together(FibonacciSeries().take(10).array,
                          [ 777, 888 ],
                          (new SquaresRange()).take(5).array);
    writeln(range.save);

The elements of the three slices are accessed as if they were elements of a single large array:

[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 777, 888, 0, 1, 4, 9, 16]

We can pass this range to other range algorithms. For example, to `retro()`, which requires a `BidirectionalRange`:

    writeln(range.save.retro);

The output:

[16, 9, 4, 1, 0, 888, 777, 34, 21, 13, 8, 5, 3, 2, 1, 1, 0]

Of course you should use the more functional `std.range.chain` instead of `Together` in your programs.

#####  `OutputRange`

All of the range types that we have seen so far are about element access. `OutputRange` represents streamed element output, similar to sending characters to `stdout`.

 I have mentioned earlier that `OutputRange` requires support for the `put(range, element)` operation. `put()` is a function defined in the `std.range` module. It determines the capabilities of the range and the element at compile time and uses the most appropriate method to *output* the element.

`put()` considers the following cases in the order that they are listed below, and applies the method for the first matching case. `R` represents the type of the range; `range`, a range object; `E`, the type of the element; and `e` an element of the range:

| Case Considered | Method Applied | 
|---|---|
| `R` has a member function named`put`and `put` can take an`E` as argument | `range.put(e);` | 
| `R` has a member function named`put`and `put` can take an`E[]` as argument | `range.put([ e ]);` | 
| `R` is an`InputRange`and `e` can be assigned to`range.front` | `range.front = e;``range.popFront();` | 
| `E` is an`InputRange`and can be copied to `R` | `for (; !e.empty; e.popFront())``put(range, e.front);` | 
| `R` can take`E` as argument(e.g. `R` could be a delegate) | `range(e);` | 
| `R` can take`E[]` as argument(e.g. `R` could be a delegate) | `range([ e ]);` | 

Let's define a range that matches the first case: The range will have a member function named `put()`, which takes a parameter that matches the type of the output range.

This output range will be used for outputting elements to multiple files, including `stdout`. When elements are outputted with `put()`, they will all be written to all of those files. As an additional functionality, let's add the ability to specify a delimiter to be written after each element.

struct MultiFile {
    string delimiter;
    File[] files;
    this(string delimiter, string[] fileNames...) {
        this.delimiter = delimiter;
        /* stdout is always included */
        this.files ~= stdout;
        /* A File object for each file name */
        foreach (fileName; fileNames) {
            this.files ~= File(fileName, "w");
        }
    }
    // This is the version that takes arrays (but not strings)
    void put(T)(T slice)
            if (isArray!T && !isSomeString!T) {
        foreach (element; slice) {
            // Note that this is a call to the other version
            // of put() below
            put(element);
        }
    }
    // This is the version that takes non-arrays and strings
    void put(T)(T value)
            if (!isArray!T || isSomeString!T) {
        foreach (file; files) {
            file.write(value, delimiter);
        }
    }
}

In order to be used as an output range of any type of elements, `put()` is also templatized on the element type.

 An algorithm in Phobos that uses `OutputRange` is `std.algorithm.copy`. `copy()` is a very simple algorithm, which copies the elements of an `InputRange` to an `OutputRange`.

import std.traits;
import std.stdio;
import std.algorithm;
// ...
void main() {
    auto output = MultiFile("\n", "output_0", "output_1");
    copy([ 1, 2, 3], output);
    copy([ "red", "blue", "green" ], output);
}

That code outputs the elements of the input ranges both to `stdout` and to files named "output_0" and "output_1":

1
2
3
red
blue
green

######  Using slices as `OutputRange`

The `std.range` module makes slices `OutputRange` objects as well. (By contrast, `std.array` makes them only input ranges.) Unfortunately, using slices as `OutputRange` objects has a confusing effect: slices lose an element for each `put()` operation on them; and that element is the element that has just been outputted!

The reason for this behavior is a consequence of slices' not having a `put()` member function. As a result, the third case of the previous table is matched for slices and the following method is applied:

```
    range.front = e;
    range.popFront();
```
As the code above is executed for each `put()`, the front element of the slice is assigned to the value of the *outputted* element, to be subsequently removed from the slice with `popFront()`:

import std.stdio;
import std.range;
void main() {
    int[] slice = [ 1, 2, 3 ];
    put(slice, 100);
    writeln(slice);
}

As a result, although the slice is used as an `OutputRange`, it surprisingly *loses* elements:

[2, 3]

To avoid this, a separate slice must be used as an `OutputRange` instead:

import std.stdio;
import std.range;
void main() {
    int[] slice = [ 1, 2, 3 ];
    int[] slice2 = slice;
    put(slice2, 100);
    writeln(slice2);
    writeln(slice);
}

This time the second slice is consumed and the original slice has the expected elements:

```
[2, 3]
[100, 2, 3]    ← expected result
```
Another important fact is that the length of the slice does not grow when used as an `OutputRange`. It is the programmer's responsibility to ensure that there is enough room in the slice:

    int[] slice = [ 1, 2, 3 ];
    int[] slice2 = slice;
    foreach (i; 0 .. 4) {    // ← no room for 4 elements
        put(slice2, i * 100);
    }

When the slice becomes completely empty because of the indirect `popFront()` calls, the program terminates with an exception:

```
core.exception.AssertError@...: Attempting to fetch the front
of an empty array of int
```
 `std.array.Appender` and its convenience function `appender` allows using slices as *an `OutputRange` where the elements are appended*. The `put()` function of the special range object that `appender()` returns actually appends the elements to the original slice:

import std.array;
// ...
    auto a = appender([ 1, 2, 3 ]);
    foreach (i; 0 .. 4) {
        a.put(i * 100);
    }

In the code above, `appender` is called with an array and returns a special range object. That range object is in turn used as an `OutputRange` by calling its `put()` member function. The resultant elements are accessed by its `.data` property:

    writeln(a.data);

The output:

[1, 2, 3, 0, 100, 200, 300]

`Appender` supports the `~=` operator as well:

```
    a ~= 1000;
    writeln(a.data);
```
The output:

[1, 2, 3, 0, 100, 200, 300, 1000]

######  `toString()` with an `OutputRange` parameter

Similar to how `toString` member functions can be defined as taking a `delegate` parameter, it is possible to define one that takes an `OutputRange`. Functions like `format`, `writefln`, and `writeln` operate more efficiently by placing the output characters right inside the output buffer of the output range.

To be able to work with any `OutputRange` type, such `toString` definitions need to be function templates, optionally with template constraints:

import std.stdio;
import std.range;
struct S {
    void toString(O)(ref O o) const
            if (isOutputRange!(O, char)) {
        put(o, "hello");
    }
}
void main() {
    auto s = S();
    writeln(s);
}

Note that the code inside `main()` does not define an `OutputRange` object. That object is defined by `writeln` to store the characters before printing them:

hello

##### Range templates

Although we have used mostly `int` ranges in this chapter, ranges and range algorithms are much more useful when defined as templates.

The `std.range` module includes many range templates. We will see these templates in the next chapter.

##### Summary

- Ranges abstract data structures from algorithms and allow them to be used with algorithms seamlessly.
- Ranges are a D concept and are the basis for many features of Phobos.
- Many Phobos algorithms return lazy range objects to accomplish their special tasks.
- UFCS works well with range algorithms.
- When used as `InputRange` objects, the elements of strings are Unicode characters.
- `InputRange` requires`empty` ,`front` and`popFront()` .
- `ForwardRange` additionally requires`save` .
- `BidirectionalRange` additionally requires`back` and`popBack()` .
- Infinite `RandomAccessRange` requires`opIndex()` over`ForwardRange` .
- Finite `RandomAccessRange` requires`opIndex()` and`length` over`BidirectionalRange` .
- `std.array.appender` returns an`OutputRange` that appends to slices.
- Slices are ranges of finite `RandomAccessRange`
- Fixed-length arrays are not ranges.
