# _Less Slow_ C++

Much of modern code suffers from common pitfalls: bugs, security vulnerabilities, and __performance bottlenecks__.
University curricula often teach outdated concepts, while bootcamps oversimplify crucial software development principles.

![Less Slow C++](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/less_slow.cpp.jpg?raw=true)

This repository offers practical examples of writing efficient C and C++ code.
It leverages C++20 features and is designed primarily for GCC and Clang compilers on Linux, though it may work on other platforms.
The topics range from basic micro-kernels executing in a few nanoseconds to more complex constructs involving parallel algorithms, coroutines, and polymorphism.
Some of the highlights include:

- __100x cheaper random inputs?!__ Discover how input generation sometimes costs more than the algorithm.
- __40x faster trigonometric calculations:__ Achieve significant speed-ups over standard library functions like `std::sin`.
- __4x faster logic with `std::ranges`:__ See how modern C++ abstractions can be surprisingly efficient when used correctly.
- __Trade-offs between accuracy and efficiency:__ Explore how to balance precision and performance in numerical computations.
- __Compiler optimizations beyond `-O3`:__ Learn about less obvious flags and techniques to deliver another 2x speedup.
- __Optimizing matrix multiplications?__ Learn how a 3x3x3 GEMM can be 60% slower than 4x4x4, despite performing 60% fewer math operations.
- __How many if conditions are too many?__ Test your CPU's branch predictor with just 10 lines of code.
- __Iterative vs. recursive algorithms:__ Avoid pitfalls that could cause a `SEGFAULT` or slow your program.
- __How not to build state machines:__ Compare `std::variant`, `virtual` functions, and C++20 coroutines.

To read, jump to the `less_slow.cpp` source file and read the code snippets and comments.

## Reproducing the Benchmarks

If you are familiar with C++ and want to review code and measurements as you read, you can clone the repository and execute the following commands.

```sh
git clone https://github.com/ashvardanian/less_slow.cpp.git # Clone the repository
cd less_slow.cpp                                            # Change the directory
cmake -B build_release -D CMAKE_BUILD_TYPE=Release          # Generate the build files
cmake --build build_release --config Release                # Build the project
build_release/less_slow                                     # Run the benchmarks
```

For brevity, the tutorial is __intended for GCC and Clang compilers on Linux__.
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

Many of the examples here are condensed versions of the articles on my ["Less Slow" blog](https://ashvardanian.com/tags/less-slow/) and many related repositories on my [GitHub profile](https://github.com/ashvardanian).
If you are also practicing Rust, you may find the ["Less Slow Rust"](https://github.com/ashvardanian/less_slow.rs) repository interesting.
