# Playing Around _Less Slow_ Coding Practices for C++, CUDA, and Assembly Code

> The benchmarks in this repository don't aim to cover every topic entirely, but they help form a mindset and intuition for performance-oriented software design.
> It also provides an example of using some non-[STL](https://en.wikipedia.org/wiki/Standard_Template_Library) but de facto standard libraries in C++, importing them via CMake and compiling from source.
> For higher-level abstractions and languages, check out [`less_slow.rs`](https://github.com/ashvardanian/less_slow.rs) and [`less_slow.py`](https://github.com/ashvardanian/less_slow.py).
> I needed many of these measurements to reconsider my own coding habits, but hopefully they're helpful to others as well.
> Most of the code is organized in very long, ordered, and nested `#pragma` sections — not necessarily the preferred form for everyone.

Much of modern code suffers from common pitfalls — bugs, security vulnerabilities, and __performance bottlenecks__.
University curricula and coding bootcamps tend to stick to traditional coding styles and standard features, rarely exposing the more fun, unusual, and potentially efficient design opportunities.
This repository explores just that.

![Less Slow C++](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/less_slow.cpp.jpg?raw=true)

The code leverages C++20 and CUDA features and is designed primarily for GCC, Clang, and NVCC compilers on Linux, though it may work on other platforms.
The topics range from basic micro-kernels executing in a few nanoseconds to more complex constructs involving parallel algorithms, coroutines, and polymorphism.
Some of the highlights include:

- __100x cheaper random inputs?!__ Discover how input generation sometimes costs more than the algorithm.
- __1% error in trigonometry at 1/40 cost:__ Approximate STL functions like [`std::sin`](https://en.cppreference.com/w/cpp/numeric/math/sin) in just 3 lines of code.
- __4x faster lazy-logic__ with custom [`std::ranges`](https://en.cppreference.com/w/cpp/ranges) and iterators!
- __Compiler optimizations beyond `-O3`:__ Learn about less obvious flags and techniques for another 2x speedup.
- __Multiplying matrices?__ Check how a 3x3x3 GEMM can be 70% slower than 4x4x4, despite 60% fewer ops.
- __Scaling AI?__ Measure the gap between theoretical [ALU](https://en.wikipedia.org/wiki/Arithmetic_logic_unit) throughput and your [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms).
- __How many if conditions are too many?__ Test your CPU's branch predictor with just 10 lines of code.
- __Prefer recursion to iteration?__ Measure the depth at which your algorithm will [`SEGFAULT`](https://en.wikipedia.org/wiki/Segmentation_fault).
- __Why avoid exceptions?__ Take `std::error_code` or [`std::variant`](https://en.cppreference.com/w/cpp/utility/variant)-like [ADTs](https://en.wikipedia.org/wiki/Algebraic_data_type)?
- __Scaling to many cores?__ Learn how to use [OpenMP](https://en.wikipedia.org/wiki/OpenMP), Intel's oneTBB, or your custom thread pool.
- __How to handle [JSON](https://www.json.org/json-en.html) avoiding memory allocations?__ Is it easier with C++ 20 or old-school C 99 tools?
- __How to properly use STL's associative containers__ with custom keys and transparent comparators?
- __How to beat a hand-written parser__ with [`consteval`](https://en.cppreference.com/w/cpp/language/consteval) RegEx engines?
- __Is the pointer size really 64 bits__ and how to exploit [pointer-tagging](https://en.wikipedia.org/wiki/Tagged_pointer)?
- __How many packets is [UDP](https://www.cloudflare.com/learning/ddos/glossary/user-datagram-protocol-udp/) dropping__ and how to serve web requests in [`io_uring`](https://en.wikipedia.org/wiki/Io_uring) from user-space?
- __Scatter and Gather__ for 50% faster vectorized disjoint memory operations.
- __Intel's oneAPI vs Nvidia's CCCL?__ What's so special about `<thrust>` and `<cub>`?
- __CUDA C++, [PTX](https://en.wikipedia.org/wiki/Parallel_Thread_Execution) Intermediate Representations, and SASS__, and how do they differ from CPU code?
- __How to choose between intrinsics, inline `asm`, and separate `.S` files__ for your performance-critical code?
- __Tensor Cores & Memory__ differences on CPUs, and Volta, Ampere, Hopper, and Blackwell GPUs!
- __How coding FPGA differs from GPU__ and what is High-Level Synthesis, Verilog, and VHDL? 🔜 #36
- __What are Encrypted Enclaves__ and what's the latency of Intel SGX, AMD SEV, and ARM Realm? 🔜 #31

To read, jump to the [`less_slow.cpp` source file](https://github.com/ashvardanian/less_slow.cpp/blob/main/less_slow.cpp) and read the code snippets and comments.
Keep in mind, that most modern IDEs have a navigation bar to help you view and jump between `#pragma region` sections.
Follow the instructions below to run the code in your environment and compare it to the comments as you read through the source.

## Running the Benchmarks

The project aims to be compatible with GCC, Clang, and MSVC compilers on Linux, MacOS, and Windows.
That said, to cover the broadest functionality, using GCC on Linux is recommended:

- If you are on Windows, it's recommended that you set up a Linux environment using [WSL](https://docs.microsoft.com/en-us/windows/wsl/install).
- If you are on MacOS, consider using the non-native distribution of Clang from [Homebrew](https://brew.sh) or [MacPorts](https://www.macports.org).
- If you are on Linux, make sure to install CMake and a recent version of GCC or Clang compilers to support C++20 features.

If you are familiar with C++ and want to review code and measurements as you read, you can clone the repository and execute the following commands.

```sh
git clone https://github.com/ashvardanian/less_slow.cpp.git # Clone the repository
cd less_slow.cpp                                            # Change the directory

pip install cmake --upgrade                                 # PyPI has a newer version of CMake
sudo apt-get install -y build-essential g++                 # Install default build tools
sudo apt-get install -y pkg-config liburing-dev             # Install liburing for kernel-bypass
sudo apt-get install -y libopenblas-base                    # Install numerics libraries

cmake -B build_release -D CMAKE_BUILD_TYPE=Release          # Generate the build files
cmake --build build_release --config Release                # Build the project
build_release/less_slow                                     # Run the benchmarks
```

The build will pull and compile several third-party dependencies from the source:

- Google's [Benchmark](https://github.com/google/benchmark) is used for profiling.
- Intel's [oneTBB](https://github.com/uxlfoundation/oneTBB) is used as the Parallel STL backend.
- Meta's [libunifex](https://github.com/facebookexperimental/libunifex) is used for senders & executors.
- Eric Niebler's [range-v3](https://github.com/ericniebler/range-v3) replaces `std::ranges`.
- Victor Zverovich's [fmt](https://github.com/fmtlib/fmt) replaces `std::format`.
- Ash Vardanian's [StringZilla](https://github.com/ashvardanian/stringzilla) replaces `std::string`.
- Hana Dusíková's [CTRE](https://github.com/hanickadot/compile-time-regular-expressions) replaces `std::regex`.
- Niels Lohmann's [json](https://github.com/nlohmann/json) is used for JSON deserialization.
- Yaoyuan Guo's [yyjson](https://github.com/ibireme/yyjson) for faster JSON processing.
- Google's [Abseil](https://github.com/abseil/abseil-cpp) replaces STL's associative containers.
- Lewis Baker's [cppcoro](https://github.com/lewissbaker/cppcoro) implements C++20 coroutines.
- Jens Axboe's [liburing](https://github.com/axboe/liburing) to simplify Linux kernel-bypass.
- Chris Kohlhoff's [ASIO](https://github.com/chriskohlhoff/asio) as a [networking TS](https://en.cppreference.com/w/cpp/experimental/networking) extension.
- Nvidia's [CCCL](https://github.com/nvidia/cccl) for GPU-accelerated algorithms.
- Nvidia's [CUTLASS](https://github.com/nvidia/cutlass) for GPU-accelerated Linear Algebra.

To build without Parallel STL, Intel TBB, and CUDA:

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D USE_INTEL_TBB=OFF -D USE_NVIDIA_CCCL=OFF
cmake --build build_release --config Release
```

To build on MacOS, pulling key dependencies from [Homebrew](https://brew.sh):

```sh
brew install openblas
cmake -B build_release \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_C_FLAGS="-I$(brew --prefix openblas)/include" \
      -D CMAKE_CXX_FLAGS="-I$(brew --prefix openblas)/include" \
      -D CMAKE_EXE_LINKER_FLAGS="-L$(brew --prefix openblas)/lib"
cmake --build build_release --config Release
```

To control the output or run specific benchmarks, use the following flags:

```sh
build_release/less_slow --benchmark_format=json             # Output in JSON format
build_release/less_slow --benchmark_out=results.json        # Save the results to a file instead of `stdout`
build_release/less_slow --benchmark_filter=std_sort         # Run only benchmarks containing `std_sort` in their name
```

To enhance stability and reproducibility, disable Simultaneous Multi-Threading __(SMT)__ on your CPU and use the `--benchmark_enable_random_interleaving=true` flag, which shuffles and interleaves benchmarks as described [here](https://github.com/google/benchmark/blob/main/docs/random_interleaving.md).

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

## Project Structure

The primary file of this repository is clearly the `less_slow.cpp` C++ file with CPU-side code.
Several other files for different hardware-specific optimizations are created:

```sh
$ tree .
.
├── CMakeLists.txt          # Build & assembly instructions for all files
├── less_slow.cpp           # Primary CPU-side benchmarking code with the majority of examples
├── less_slow_amd64.S       # Hand-written Assembly kernels for 64-bit x86 CPUs
├── less_slow_aarch64.S     # Hand-written Assembly kernels for 64-bit Arm CPUs
├── less_slow.cu            # CUDA C++ examples for parallel algorithms for Nvidia GPUs
├── less_slow_sm70.ptx      # Hand-written PTX IR kernels for Nvidia Volta GPUs
└── less_slow_sm90a.ptx     # Hand-written PTX IR kernels for Nvidia Hopper GPUs
```

## Memes and References

Educational content without memes?!
Come on!

<table>
  <tr>
    <td><img src="https://github.com/ashvardanian/ashvardanian/blob/master/memes/ieee764-vs-gnu-compiler.jpg?raw=true" alt="IEEE 754 vs GNU Compiler"></td>
    <td><img src="https://github.com/ashvardanian/ashvardanian/blob/master/memes/no-easter-bunny-no-free-abstractions.jpg?raw=true" alt="No Easter Bunny, No Free Abstractions"></td>
  </tr>
</table>

## Google Benchmark Functionality

This benchmark suite uses most of the features provided by Google Benchmark.
If you write a lot of benchmarks and avoid going to the full [User Guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md), here is a condensed list of the most useful features:

- `->Args({x, y})` - Pass multiple arguments to parameterized benchmarks
- `BENCHMARK()` - Register a basic benchmark function
- `BENCHMARK_CAPTURE()` - Create variants of benchmarks with different captured values
- `Counter::kAvgThreads` - Specify thread-averaged counters
- `DoNotOptimize()` - Prevent compiler from optimizing away operations
- `ClobberMemory()` - Force memory synchronization
- `->Complexity(oNLogN)` - Specify and validate algorithmic complexity
- `->SetComplexityN(n)` - Set input size for complexity calculations
- `->ComputeStatistics("max", ...)` - Calculate custom statistics across runs
- `->Iterations(n)` - Control exact number of iterations
- `->MinTime(n)` - Set minimum benchmark duration
- `->MinWarmUpTime(n)` - To warm up the data caches
- `->Name("...")` - Assign custom benchmark names
- `->Range(start, end)` - Profile for a range of input sizes
- `->RangeMultiplier(n)` - Set multiplier between range values
- `->ReportAggregatesOnly()` - Show only aggregated statistics
- `state.counters["name"]` - Create custom performance counters
- `state.PauseTiming()`, `ResumeTiming()` - Control timing measurement
- `state.SetBytesProcessed(n)` - Record number of bytes processed
- `state.SkipWithError()` - Skip benchmark with error message
- `->Threads(n)` - Run benchmark with specified number of threads
- `->Unit(kMicrosecond)` - Set time unit for reporting
- `->UseRealTime()` - Measure real time instead of CPU time
- `->UseManualTime()` - To feed custom timings for GPU and IO benchmarks
