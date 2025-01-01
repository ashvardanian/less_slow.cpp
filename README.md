# Learning to Write _Less Slow_ C, C++, and Assembly Code

> The benchmarks in this repository don't aim to cover every topic fully, but they help form a mindset and intuition for performance-oriented software design.
> For higher-level abstractions and languages, check out [`less_slow.rs`](https://github.com/ashvardanian/less_slow.rs) and [`less_slow.py`](https://github.com/ashvardanian/less_slow.py).

Much modern code suffers from common pitfalls, such as bugs, security vulnerabilities, and __performance bottlenecks__.
University curricula often teach outdated concepts, while bootcamps oversimplify crucial software development principles.

![Less Slow C++](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/less_slow.cpp.jpg?raw=true)

This repository offers practical examples of writing efficient C and C++ code.
It leverages C++20 features and is designed primarily for GCC and Clang compilers on Linux, though it may work on other platforms.
The topics range from basic micro-kernels executing in a few nanoseconds to more complex constructs involving parallel algorithms, coroutines, and polymorphism.
Some of the highlights include:

- __100x cheaper random inputs?!__ Discover how input generation sometimes costs more than the algorithm.
- __40x faster trigonometry:__ Speed-up standard library functions like [`std::sin`](https://en.cppreference.com/w/cpp/numeric/math/sin) in just 3 lines of code.
- __4x faster logic with [`std::ranges`](https://en.cppreference.com/w/cpp/ranges):__ Reduce stack usage and reuse registers more efficiently.
- __Compiler optimizations beyond `-O3`:__ Learn about less obvious flags and techniques for another 2x speedup.
- __Need matrix multiplications?__ Check how a 3x3x3 GEMM can be 60% slower than 4x4x4, despite 60% fewer ops.
- __How many if conditions are too many?__ Test your CPU's branch predictor with just 10 lines of code.
- __Iterative vs. recursive algorithms:__ Avoid pitfalls that could cause a `SEGFAULT` or slow your program.
- __How not to build state machines:__ Compare `std::variant`, `virtual` functions, and C++20 coroutines.
- __Scaling to many cores?__ Learn how to use OpenMP, Intel's oneTBB, or your custom thread pool.

To read, jump to the [`less_slow.cpp` source file](https://github.com/ashvardanian/less_slow.cpp/blob/main/less_slow.cpp) and read the code snippets and comments.
Follow the instructions below to run the code in your environment and compare it to the comments as you read through the source.

## Running the Benchmarks

If you are on Windows, it's recommended that you set up a Linux environment using [WSL](https://docs.microsoft.com/en-us/windows/wsl/install).
- If you are on MacOS, consider using the non-native distribution of Clang from [Homebrew](https://brew.sh) or [MacPorts](https://www.macports.org).
- If you are on Linux, make sure to install CMake and a recent version of GCC or Clang compilers to support C++20 features.

If you are familiar with C++ and want to review code and measurements as you read, you can clone the repository and execute the following commands.

```sh
git clone https://github.com/ashvardanian/less_slow.cpp.git # Clone the repository
cd less_slow.cpp                                            # Change the directory
cmake -B build_release -D CMAKE_BUILD_TYPE=Release          # Generate the build files
cmake --build build_release --config Release                # Build the project
build_release/less_slow                                     # Run the benchmarks
```

> The build will pull and compile several third-party dependencies. 
> Google [Benchmark](https://github.com/google/benchmark) is used for profiling.
> Intel's [oneTBB](https://github.com/uxlfoundation/oneTBB) is used as the Parallel STL backend.
> Victor Zverovich's [fmt](https://github.com/fmtlib/fmt) replaces `std::format` for logging.

To control the output or run specific benchmarks, use the following flags:

```sh
build_release/less_slow --benchmark_format=json             # Output in JSON format
build_release/less_slow --benchmark_out=results.json        # Save the results to a file, instead of `stdout`
build_release/less_slow --benchmark_filter=std_sort         # Run only benchmarks containing `std_sort` in their name
```

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
