Much of modern code suffers from common pitfalls: bugs, security vulnerabilities, and __performance bottlenecks__.
University curricula often teach outdated concepts, while bootcamps oversimplify crucial software development principles.
This repository provides practical examples of writing efficient C and C++ code.

![Less Slow C++](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/less_slow.cpp.jpg?raw=true)

Even when an example seems over-engineered, it doesn't make it less relevant or impractical.
The patterns discussed here often appear implicitly in large-scale software, even if most developers don't consciously recognize them.

This is why some developers gravitate toward costly abstractions like multiple inheritance with dynamic polymorphism (e.g., `virtual` functions in C++) or using dynamic memory allocation inside loops.
They rarely design benchmarks representing real-world projects with 100K+ lines of code.
They rarely scale workloads across hundreds of cores, as required in modern cloud environments.
They rarely interface with specialized hardware accelerators that have distinct address spaces.

But we're not here to be average — we're here to be better.
We want to know the cost of unaligned memory accesses, branch prediction, CPU cache misses and the latency of different cache levels, the frequency scaling policy levels, the cost of polymorphism and asynchronous programming, and the trade-offs between accuracy and efficiency in numerical computations.
Let's dig deeper into writing __less slow__, more efficient software.

## Contents

All of material is organized into a single readable `.cpp` source code file with multiple sections.

- Can random input generation be 100x more expensive than the algorithm itself?
- Is it better to use a recursive or iterative algorithm?
- How expensive is STL math and how to avoid it?
- After `-O3`, which compilation flags can give you another 2x speedup?
- How and where to write SIMD assembly and where the compiler does it better?
- What's the cost of mis-aligned memory accesses and how to avoid them?
- How expensive are coroutines and asynchronous programming?
- How do the compare to ranges, callbacks, and lambdas?
- What extra features do modern benchmarking tools provide?

Highlights include:

- 4x faster logic with `std::ranges`, compared to `std::function`.
- 40x faster computing of $sine$ compared to `std::sin`.
- 100x cheaper random inputs?!

To continue reading, jump to `less_slow.cpp` and start reading the code and comments.

## Reproducing the Benchmarks

If you are familiar with C++ and want to go through code and measurements as you read, you can clone the repository and execute the following commands.

```sh
git clone https://github.com/ashvardanian/LessSlow.cpp.git  # Clone the repository
cd LessSlow.cpp                                             # Change the directory
cmake -B build_release -D CMAKE_BUILD_TYPE=Release          # Generate the build files
cmake --build build_release --config Release                # Build the project
build_release/less_slow                                     # Run the benchmarks
```

For brevity, the tutorial is intended for GCC and Clang compilers on Linux, but should be compatible with MacOS and Windows.
To control the output or run specific benchmarks, use the following flags:

```sh
build_release/less_slow --benchmark_format=json             # Output in JSON format
build_release/less_slow --benchmark_out=results.json        # Save the results to a file, instead of `stdout`
build_release/less_slow --benchmark_filter=std_sort         # Run only benchmarks containing `std_sort` in their name
```

> The builds will [Google Benchmark](https://github.com/google/benchmark) and [Intel's oneTBB](https://github.com/uxlfoundation/oneTBB) for the Parallel STL implementation.

To enhance stability and reproducibility, use the `--benchmark_enable_random_interleaving=true` flag which shuffles and interleaves benchmarks as described [here](https://github.com/google/benchmark/blob/main/docs/random_interleaving.md).

```sh
build_release/less_slow --benchmark_enable_random_interleaving=true
```

Google Benchmark supports [User-Requested Performance Counters](https://github.com/google/benchmark/blob/main/docs/perf_counters.md) through `libpmf`.
Note that collecting these may require `sudo` privileges.

```sh
sudo build_release/less_slow --benchmark_enable_random_interleaving=true --benchmark_format=json --benchmark_perf_counters="CYCLES,INSTRUCTIONS"
```

Alternatively, use the Linux `perf` tool for performance counter collection:

```sh
sudo perf stat taskset 0xEFFFEFFFEFFFEFFFEFFFEFFFEFFFEFFF build_release/less_slow --benchmark_enable_random_interleaving=true --benchmark_filter=super_sort
```

## Further Reading

Many of the examples here are condensed versions of the articles on the ["Less Slow" blog](https://ashvardanian.com/tags/less-slow/).
For advanced parallel algorithm benchmarks, see [ashvardanian/ParallelReductionsBenchmark](https://github.com/ashvardanian/ParallelReductionsBenchmark).
For SIMD algorithms, check the production code at [ashvardanian/SimSIMD](https://github.com/ashvardanian/SimSIMD) and [ashvardanian/StringZilla](https://github.com/asvardanian/StringZilla), or individual articles:

- [Optimizing C++ & CUDA for High-Speed Parallel Reductions](https://ashvardanian.com/posts/cuda-parallel-reductions/)
- [Challenges in Maximizing DDR4 Bandwidth](https://ashvardanian.com/posts/ddr4-bandwidth/)
- [Comparing GCC Compiler and Manual Assembly Performance](https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/)
- [Enhancing SciPy Performance with AVX-512 & SVE](https://ashvardanian.com/posts/simsimd-faster-scipy/).

