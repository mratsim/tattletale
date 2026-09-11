# Papers and Talks

This folder contains the research papers and conference talks backing the engine's
design areas
- streams, pipelines, ranges, iterators and coroutines (the chat-template pipeline design)
- tile programming models for AI kernels (GPU)
- CuTe layout representation and algebra (NVIDIA CUTLASS)
- CPU matrix multiplication (the GEMM anatomy lineage)
- rank-balanced trees (the rank/WAVL structure used for longest-prefix-match)

The papers are grouped by topic in the following way:
- Tile Programming: GPU kernel programming models, statically tied to the tensor/tile layer
- CuTe Layout Algebra: the layout algebra CUTLASS kernels are built on
- CPU Matrix Multiplication: the Goto/BLIS layering, shared by every BLAS
- Streams/Pipelines: pull-based pipelines, transducers, coroutines and the
  defunctionalization that connects them
- Data Structures: trees backing the prefix-match structures

where the same defunctionalization references (Koppel, the continuation material)
serve both the streams collection and any direct-style-to-callback port.

Web-published copies are reproduced for private research/study with attribution
and links; all rights remain with their authors.

## Streams, pipelines, ranges, iterators, coroutines

### Streams and pipelines

- Pipe Dreams, Part 1: Pull, push and the pull-pull pipeline
  Marc Gravell, 2018
  https://blog.marcgravell.com/2018/07/pipe-dreams-part-1.html

- Pipe Dreams, Part 2: Push and the pull-push pipeline
  Marc Gravell, 2018
  https://blog.marcgravell.com/2018/07/pipe-dreams-part-2.html

- Pipe Dreams, Part 3: The pull-push pipeline and backpressure
  Marc Gravell, 2018
  https://blog.marcgravell.com/2018/07/pipe-dreams-part-3.html

- Pipe Dreams, Part 3.1: IO and the async pipeline
  Marc Gravell, 2018
  https://blog.marcgravell.com/2018/07/pipe-dreams-part-31.html

### Ranges (D)

- Programming in D — Ranges
  Ali Çehreli, 2017
  https://ddili.org/ders/d.en/ranges.html

- Iterators Must Go!
  Andrei Alexandrescu, BoostCon 2009

- On Iteration
  Andrei Alexandrescu, 2009
  https://www.informit.com/articles/article.aspx?p=1407357

### Transducers

- Transducers are coming
  Rich Hickey, 2014
  https://cognitect.com/blog/2014/8/6/transducers-are-coming

- Understanding Transducers Through Python
  Rob Smallshire, Sixty North, 2016

### Defunctionalization and continuations

- The Best Refactoring You've Never Heard Of
  James Koppel, Compose Conference 2019
  https://www.pathsensitive.com/2019/07/the-best-refactoring-youve-never-heard.html

### Coroutines

- C++ Coroutines
  Gor Nishanov, CppCon 2015
  https://github.com/CppCon/CppCon2015/blob/master/Presentations/C%2B%2B%20Coroutines/C%2B%2B%20Coroutines%20-%20Gor%20Nishanov%20-%20CppCon%202015.pdf

- NanoCoroutines
  Gor Nishanov, CppCon 2018
  https://github.com/GorNishanov/await/blob/master/2018_CppCon/NanoCoroutines%20-%20Gor%20Nishanov%20-%20CppCon%202018.pdf

- Exploiting Coroutines to Attack the Killer "Nanoseconds"
  Christopher Jonathan, Umar Farooq Minhas, James Hunter, Justin Levandoski, Gor Nishanov, PVLDB 11, 2018
  https://www.vldb.org/pvldb/vol11/p1702-jonathan.pdf

## Tile programming (GPU)

- ThunderKittens: Simple, Fast, and Adorable AI Kernels
  Benjamin F. Spector, Simran Arora, Aaryan Singhal, Daniel Y. Fu, Christopher Ré, 2024
  https://arxiv.org/abs/2410.20399

- TileLang: A Composable Tiled Programming Model for AI Systems
  Lei Wang, Yu Cheng, Yining Shi, Zhengju Tang, Zhiwen Mo, Wenhao Xie, Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, Zhi Yang, 2025
  https://arxiv.org/abs/2504.17577

- Fearless Concurrency on the GPU
  Melih Elibol, Jared Roesch, Isaac Gelado, Eric Buehler, Michael Garland, 2026
  https://arxiv.org/abs/2606.15991

## CuTe layout algebra

- CuTe Layout
  NVIDIA CUTLASS
  https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md

- CuTe Layout Algebra
  NVIDIA CUTLASS
  https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md

- CuTe Layout Representation and Algebra
  Cris Cecka, 2026
  https://arxiv.org/abs/2603.02298

- Categorical Foundations for CuTe Layouts
  Colfax Research, September 2025
  https://research.colfax-intl.com/

- A Note on the Algebra of CuTe Layouts
  Jay Shah, Colfax Research, January 2024
  https://research.colfax-intl.com/

## CPU matrix multiplication

- Anatomy of High-Performance Matrix Multiplication
  Kazushige Goto, Robert A. van de Geijn, ACM TOMS 34(3), 2008
  https://dl.acm.org/doi/10.1145/1356052.1356053

- Anatomy of High-Performance Many-Threaded Matrix Multiplication
  Tyler M. Smith, Robert A. van de Geijn, Mikhail Smelyanskiy, Jeff R. Hammond, Field G. Van Zee, IPDPS 2014

- Automating the Last-Mile for High Performance Dense Linear Algebra
  Richard Michael Veras, Tze Meng Low, Tyler M. Smith, Robert van de Geijn, Franz Franchetti, 2016
  https://arxiv.org/abs/1611.08035

## Data structures

- Rank-Balanced Trees
  Bernhard Haeupler, Siddhartha Sen, Robert E. Tarjan, ACM TALG 11(4), Article 30, May 2015
