/**
 *  @brief  Low-level microbenchmarks for building a performance-first mindset.
 *  @file   less_slow.cpp
 *  @author Ash Vardanian
 *
 *  There's no Easter bunny, no tooth fairy... and no free abstractions!
 *  Every abstraction—no matter how elegant—comes with tradeoffs. Sometimes
 *  the cost is in readability, like extra layers of indirection, and other
 *  times, it's in performance, with additional instructions or memory overhead.
 *
 *  This project dives into such tradeoffs, helping engineers develop a deeper
 *  understanding of the costs associated with seemingly simple constructs.
 *  While the examples are written in C++20 and focus on GCC and Clang,
 *  targeting x86_64 and ARM64 architectures, the principles are universal.
 *
 *  The benchmarks cover the costs of numerical operations, designing micro-
 *  kernels, parallelism, computational complexity, branch prediction, compiler
 *  limitations, and @b composing those with callbacks, coroutines, and ranges
 *  into more complex systems. Same principles apply to Rust, Python, and Go,
 *  so parts of the project have been reproduced in those languages.
 *
 *  @see C++ Benchmarks: https://github.com/ashvardanian/less_slow.cpp
 *  @see Rust Benchmarks: https://github.com/ashvardanian/less_slow.rs
 *  @see Python Benchmarks: https://github.com/ashvardanian/less_slow.py
 *  @see Go Benchmarks: https://github.com/ashvardanian/less_slow.go
 *
 *  Most measurements were performed on Intel Sapphire Rapids CPUs, but the
 *  findings are relevant across hardware platforms unless explicitly noted.
 *
 *  Worth noting, that some examples may seem over-engineered, but they are
 *  no less relevant or impractical. They may be hard to recognize at first,
 *  but they universally appear in larger codebases, as a form of emergent
 *  complexity.
 *
 *  Let's benchmark them all and dive into the implementation details that
 *  make those abstractions @b less_slow!
 */
#include <benchmark/benchmark.h>

namespace bm = benchmark;

#pragma region - Basics

#pragma region How to Benchmark and Randomness

/**
 *  Using Google Benchmark is simple. You define a C++ function and then
 *  register it using the provided C macros. The suite will invoke your
 *  function, passing a `State` object, that dynamically chooses the number
 *  of loops to run based on the time it takes to execute each cycle.
 *
 *  For simplicity, let's start by benchmarking the most basic operation -
 *  the 32-bit integer addition, universally natively supported by every
 *  modern CPU, be it x86, ARM, or RISC-V, 32-bit or 64-bit, big-endian
 *  or little-endian.
 */
#include <cstdint> // `std::int32_t` and other sized integers

static void i32_addition(bm::State &state) {
    std::int32_t a = 0, b = 0, c = 0;
    for (auto _ : state) c = a + b;
    (void)c; // Silence "variable `c` set but not used" warning
}

BENCHMARK(i32_addition);

/**
 *  Trivial kernels operating on constant values are not the most
 *  straightforward candidates for benchmarking. The compiler can easily
 *  optimize them, and the CPU can predict the result... showing "0 ns" - zero
 *  nanoseconds per iteration. Unfortunately, no operation runs this fast on the
 *  computer. On a 3 GHz CPU, you would perform 3 Billion ops every second.
 *  So, each would take 0.33 ns, not 0 ns. If we change the compilation
 *  settings, discarding the @b `-O3` flag for "Release build" optimizations,
 *  we may see a non-zero value, but it won't represent real-world performance.
 *
 *  Another thing we can try - is generating random inputs on the fly with
 *  @b `std::rand()`, one of the most controversial operations in the
 *  C standard library.
 */
#include <cstdlib> // `std::rand`

static void i32_addition_random(bm::State &state) {
    std::int32_t c = 0;
    for (auto _ : state) c = std::rand() + std::rand();
    (void)c; // Silence "variable `c` set but not used" warning
}

BENCHMARK(i32_addition_random);

/**
 *  Running this will report @b 25ns or about 100 CPU cycles. Is integer
 *  addition really that expensive? It's used all the time, even when you are
 *  accessing @b `std::vector` elements and need to compute the memory address
 *  from the pointer and the index passed to the @b `operator[]` or `at()`
 *  functions. The answer is - no, it's not. The addition takes a single CPU
 *  cycle and is very fast.
 *
 *  Chances are we just benchmarked something else... the @b `std::rand()`
 *  function. What if we could ask Google Benchmark to ignore the time spent
 *  in the `std::rand()` function? There are `PauseTiming` and `ResumeTiming`
 *  functions just for that!
 */

static void i32_addition_paused(bm::State &state) {
    std::int32_t a = 0, b = 0, c = 0;
    for (auto _ : state) {
        state.PauseTiming();
        a = std::rand(), b = std::rand();
        state.ResumeTiming();
        bm::DoNotOptimize(c = a + b);
    }
}

BENCHMARK(i32_addition_paused);

/**
 *  However, the `PauseTiming` and `ResumeTiming` functions are neither free.
 *  In the current implementation, they can easily take @b ~127 ns, or around
 *  150 CPU cycles. They are useless in this case, but there is an alternative!
 *
 *  A typical pattern when implementing a benchmark is to initialize with a
 *  random value and then define a very cheap update policy that won't affect
 *  the latency much but will update the inputs. Increments, bit shifts, and bit
 *  rotations are common choices!
 *
 *  It's also a good idea to use native @b CRC32 and @b AES instructions to
 *  produce a random state, as it's often done in StringZilla. Another common
 *  approach is to use integer multiplication, usually derived from the
 *  Golden Ratio, as in the Knuth's multiplicative hash (with `2654435761`).
 *
 *  @see StringZilla: https://github.com/ashvardanian/stringzilla
 */

static void i32_addition_randomly_initialized(bm::State &state) {
    std::int32_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state) bm::DoNotOptimize(c = (++a) + (++b));
}

BENCHMARK(i32_addition_randomly_initialized);

/**
 *  On x86, the `i32_addition_randomly_initialized` benchmark performs two
 *  @b `inc` instructions and one @b `add` instruction. This should take less
 *  than @b 0.7ns on a modern CPU. The first cycle increments `a` and `b`
 *  simultaneously on different Arithmetic Logic Units (ALUs) of the same core,
 *  while the second cycle performs the final accumulation. At least @b 97% of
 *  the benchmark time was spent in the `std::rand()` function... even in a
 *  single-threaded benchmark.
 *
 *  This may seem like a trivial example, far removed from "real-world
 *  production systems" of "advanced proprietary software designed by the
 *  world's leading engineers." Sadly, issues like this persist in many
 *  benchmarks and sometimes influence multi-billion-dollar decisions 🤬
 *
 *  @see Bad I/O benchmark examples: https://www.unum.cloud/blog/2022-03-22-ucsb
 *
 *  How bad is it? Let's re-run the same two benchmarks, this time on 8 threads.
 */

BENCHMARK(i32_addition_random)->Threads(8);
BENCHMARK(i32_addition_randomly_initialized)->Threads(8);

/**
 *  The latency of the `std::rand` variant skyrocketed from @b 15ns in
 *  single-threaded mode to @b 12'000ns when running on multiple threads,
 *  while our optimized variant remained unaffected.
 *
 *  This happens because `std::rand`, like many other LibC functions, relies
 *  on a global state protected by a mutex to ensure thread-safe access.
 *  This mutex-based synchronization becomes a severe bottleneck when multiple
 *  threads contend for the same global resource.
 *
 *  Here's the relevant snippet from the GNU C Library (GlibC) implementation:
 *
 *      long int __random (void) {
 *          int32_t retval;
 *          __libc_lock_lock (lock);
 *          (void) __random_r (&unsafe_state, &retval);
 *          __libc_lock_unlock (lock);
 *          return retval;
 *      }
 *      weak_alias (__random, random)
 *
 *  This perfectly illustrates why experienced low-level engineers often avoid
 *  the "singleton" pattern, where a single shared global state introduces
 *  contention and kills performance under multi-threaded workloads.
 *
 *  @see GlibC implementation:
 *  https://code.woboq.org/userspace/glibc/stdlib/random.c.html#291
 *  @see Faster random integer generation with batching by Daniel Lemire:
 *  https://lemire.me/blog/2024/08/17/faster-random-integer-generation-with-batching/
 */

#pragma endregion // How to Benchmark and Randomness

#pragma region Parallelism and Computational Complexity

/**
 *  The most obvious way to speed up code is to parallelize it. Since 2002, CPU
 *  clock speeds have plateaued, with CPUs getting wider instead of faster,
 *  featuring more cores and hardware threads. However, not all algorithms can
 *  be parallelized easily. Some are inherently sequential, while others are
 *  simply too small to benefit from parallel execution.
 *
 *  Let's begin with @b `std::sort`, one of the best-known and best-optimized
 *  algorithms in the C++ Standard Library.
 *
 *  @see Docs for `std::sort`: https://en.cppreference.com/w/cpp/algorithm/sort
 *
 *  A straightforward benchmarking strategy could involve applying a random
 *  shuffle before each sort. However, this would introduce variability, making
 *  the benchmark less predictable. Knowing that `std::sort` uses a Quick-Sort
 *  variant, we can instead reverse the array on each iteration — a classic
 *  worst-case scenario for this family of algorithms.
 *
 *  We can also parameterize the benchmark with runtime-configurable values,
 *  such as the array size and whether to include the preprocessing step.
 *  Google Benchmark provides the @b `Args` function precisely for this purpose.
 */
#include <algorithm> // `std::sort`
#include <numeric>   // `std::iota`
#include <vector>    // `std::vector`

static void sorting(bm::State &state) {

    auto count = static_cast<std::size_t>(state.range(0));
    auto include_preprocessing = static_cast<bool>(state.range(1));

    std::vector<std::uint32_t> array(count);
    std::iota(array.begin(), array.end(), 1u);

    for (auto _ : state) {

        if (!include_preprocessing) state.PauseTiming();
        // Reverse order is the most classical worst case, but not the only one.
        std::reverse(array.begin(), array.end());
        if (!include_preprocessing) state.ResumeTiming();

        std::sort(array.begin(), array.end());
        bm::DoNotOptimize(array.size());
    }
}

BENCHMARK(sorting)->Args({3, false})->Args({3, true});
BENCHMARK(sorting)->Args({4, false})->Args({4, true});
BENCHMARK(sorting)->Args({1024, false})->Args({1024, true});
BENCHMARK(sorting)->Args({8196, false})->Args({8196, true});

/**
 *  This highlights how optimal control flow depends on input size:
 *  - On small inputs, it's faster to perform preprocessing.
 *  - On larger inputs, the overhead from preprocessing outweighs its benefits.
 *
 *  Until C++17, the standard lacked built-in parallel algorithms.
 *  The C++17 standard introduced the @b `std::execution` namespace, including
 *  the @b `std::execution::par_unseq` policy for parallel, order-independent
 *  execution.
 *
 *  To check for support, the @b `__cpp_lib_parallel_algorithm` standard
 *  feature testing macro can be used.
 *
 *  @see More feature testing macros:
 *  https://en.cppreference.com/w/cpp/utility/feature_test
 */

#if defined(__cpp_lib_parallel_algorithm)
#include <execution> // `std::execution::par_unseq`

template <typename execution_policy_>
static void sorting_with_executors( //
    bm::State &state, execution_policy_ &&policy) {

    auto count = static_cast<std::size_t>(state.range(0));
    std::vector<std::uint32_t> array(count);
    std::iota(array.begin(), array.end(), 1u);

    for (auto _ : state) {
        std::reverse(policy, array.begin(), array.end());
        std::sort(policy, array.begin(), array.end());
        bm::DoNotOptimize(array.size());
    }

    state.SetComplexityN(count);
    state.SetItemsProcessed(count * state.iterations());
    state.SetBytesProcessed(count * state.iterations() * sizeof(std::int32_t));

    // Want to report something else? Sure, go ahead:
    //
    //      state.counters["temperature_on_mars"] = bm::Counter(-95.4);
    //
    // Just please, for the love of rockets, use the metric system.
    // We've already lost one Mars climate orbiter to a tragic "feet vs. meters"
    // debate. Let's not make NASA cry again. 🚀💥
}

BENCHMARK_CAPTURE(sorting_with_executors, seq, std::execution::seq)
    ->RangeMultiplier(4)
    ->Range(1l << 20, 1l << 28)
    ->MinTime(10)
    ->Complexity(bm::oNLogN)
    ->UseRealTime();

BENCHMARK_CAPTURE(sorting_with_executors, par_unseq, std::execution::par_unseq)
    ->RangeMultiplier(4)
    ->Range(1l << 20, 1l << 28)
    ->MinTime(10)
    ->Complexity(bm::oNLogN)
    ->UseRealTime();

/**
 *  Without @b `UseRealTime()`, CPU time is measured by default.
 *  This distinction matters: if your process sleeps, it no longer
 *  accumulates CPU time.
 *
 *  The @b `Complexity` function specifies the asymptotic computational
 *  complexity of the benchmark. To estimate the scaling factor, we benchmark
 *  over a broad range of input sizes, from 2^20 (1 million)
 *  to 2^28 (256 million) entries — translating to 4 MB to 1 GB of data.
 *
 *  This approach outputs both timings and inferred complexity estimates:
 *
 *     sorting_with_executors/seq/1048576            5776408 ns      5776186 ns
 *     sorting_with_executors/seq/4194154           25323450 ns      2532153 ns
 *     sorting_with_executors/seq/16777216         109073782 ns    109071515 ns
 *     sorting_with_executors/seq/67108864         482794615 ns    482777617 ns
 *     sorting_with_executors/seq/268435456       2548725384 ns   2548695506 ns
 *     sorting_with_executors/seq_BigO                  0.34 NlgN       0.34 NlgN
 *     sorting_with_executors/seq_RMS                      8 %             8 %
 *
 *  As demonstrated, scaling isn't strictly linear, especially for tasks
 *  that aren't fully data-parallel.
 */

#endif // defined(__cpp_lib_parallel_algorithm)

#pragma endregion // Parallelism and Computational Complexity

/**
 *  The `std::sort` and the underlying Quick-Sort are perfect research subjects
 * for benchmarking and understanding how the computer works. Naively
 * implementing the Quick-Sort in C/C++ would still put us at disadvantage,
 * compared to the STL.
 *
 *  Most implementations we can find in textbooks, use recursion. Recursion is a
 * beautiful concept, but it's not always the best choice for performance. Every
 * nested call requires a new stack frame, and the stack is limited. Moreover,
 * local variables need to be constructed and destructed, and the CPU needs to
 * jump around in memory.
 *
 *  The alternative, as it often is in computing, is to use compensate runtime
 * issue with memory. We can use a stack data structure to continuously store
 * the state of the algorithm, and then process it in a loop.
 *
 *  The same ideas common appear when dealing with trees or graph algorithms.
 */
#include <utility> // `std::swap`

#pragma region Recursion

/**
 *  @brief  Quick-Sort helper function for array partitioning, reused by both
 *          recursive and iterative implementations.
 */
template <typename element_type_>
struct quick_sort_partition {
    using element_t = element_type_;

    inline std::ptrdiff_t operator()(element_t *arr, std::ptrdiff_t const low, std::ptrdiff_t const high) noexcept {
        element_t pivot = arr[high];
        std::ptrdiff_t i = low - 1;
        for (std::ptrdiff_t j = low; j <= high - 1; j++) {
            if (arr[j] > pivot) continue;
            i++;
            std::swap(arr[i], arr[j]);
        }
        std::swap(arr[i + 1], arr[high]);
        return i + 1;
    }
};

/**
 *  @brief  Quick-Sort implementation as a C++ function object, using recursion.
 *          Note, recursion and @b inlining are not compatible.
 */
template <typename element_type_>
struct quick_sort_recurse {
    using element_t = element_type_;
    using quick_sort_partition_t = quick_sort_partition<element_t>;
    using quick_sort_recurse_t = quick_sort_recurse<element_t>;

    void operator()(element_t *arr, std::ptrdiff_t low, std::ptrdiff_t high) noexcept {
        if (low >= high) return;
        auto pivot = quick_sort_partition_t {}(arr, low, high);
        quick_sort_recurse_t {}(arr, low, pivot - 1);
        quick_sort_recurse_t {}(arr, pivot + 1, high);
    }
};

/**
 *  @brief  Quick-Sort implementation as a C++ function object, with iterative
 *  deepening using a "stack" data-structure. Note, this implementation can be
 *  inlined, but can't be @b `noexcept`, due to a potential memory allocation in
 *  the `std::vector::resize` function.
 *
 *  Fun fact: The `std::vector` is actually a better choice for a "stack" than
 *  the `std::stack`, as the latter builds on top of a `std::deque`, which is
 *  normally implemented as a sequence of individually allocated fixed-size arrays,
 *  with additional bookkeeping. In our logic we never need to pop from the middle
 *  or from the front, so a `std::vector` is a better choice.
 */

template <typename element_type_>
struct quick_sort_iterate {
    using element_t = element_type_;
    using quick_sort_partition_t = quick_sort_partition<element_t>;

    std::vector<std::ptrdiff_t> stack;

    void operator()(element_t *arr, std::ptrdiff_t low, std::ptrdiff_t high) noexcept(false) {

        stack.resize((high - low + 1) * 2);
        std::ptrdiff_t top = -1;

        stack[++top] = low;
        stack[++top] = high;

        while (top >= 0) {
            high = stack[top--];
            low = stack[top--];
            auto pivot = quick_sort_partition_t {}(arr, low, high);

            // If there are elements on left side of pivot,
            // then push left side to stack
            if (low < pivot - 1) {
                stack[++top] = low;
                stack[++top] = pivot - 1;
            }

            // If there are elements on right side of pivot,
            // then push right side to stack
            if (pivot + 1 < high) {
                stack[++top] = pivot + 1;
                stack[++top] = high;
            }
        }
    }
};

template <typename sorter_type_, std::size_t length_> //
static void recursion_cost(bm::State &state) {
    using element_t = typename sorter_type_::element_t;
    sorter_type_ sorter;
    std::vector<element_t> arr(length_);
    for (auto _ : state) {
        for (std::size_t i = 0; i != length_; ++i) arr[i] = length_ - i;
        sorter(arr.data(), 0, static_cast<std::ptrdiff_t>(length_ - 1));
    }
}

using recursive_sort_i32s = quick_sort_recurse<std::int32_t>;
using iterative_sort_i32s = quick_sort_iterate<std::int32_t>;

BENCHMARK_TEMPLATE(recursion_cost, recursive_sort_i32s, 16);
BENCHMARK_TEMPLATE(recursion_cost, iterative_sort_i32s, 16);
BENCHMARK_TEMPLATE(recursion_cost, recursive_sort_i32s, 256);
BENCHMARK_TEMPLATE(recursion_cost, iterative_sort_i32s, 256);
BENCHMARK_TEMPLATE(recursion_cost, recursive_sort_i32s, 4096);
BENCHMARK_TEMPLATE(recursion_cost, iterative_sort_i32s, 4096);

/**
 *  If you try pushing the size further, the program will likely @b crash due
 *  to @b stack_overflow. The recursive version is limited by the stack size, while
 *  the iterative version can handle much larger inputs.
 */

#pragma endregion // Recursion

#pragma region Branch Prediction

/**
 *  The `if` statement and the seemingly innocent ternary operator `x ? a : b`
 *  can be surprisingly expensive in performance-critical code. This is
 *  especially noticeable when conditional execution operates at the byte level,
 *  as in text processing, parsing, searching, compression, encoding, and
 *  similar tasks.
 *
 *  Modern CPUs have sophisticated branch predictors — some of the most complex
 *  hardware components in a processor. These predictors memorize recent branch
 *  patterns, enabling "speculative execution," where the CPU starts processing
 *  the next task (`i+1`) before fully completing the current one (`i`).
 *
 *  While a single `if` in a hot-path is usually not a problem, real-world
 *  applications often involve thousands of branches. On most modern CPUs, up
 *  to @b 4096 branches can be memorized. Beyond that, branch mis-predictions
 *  occur, causing a severe slowdown due to pipeline stalls.
 *
 *  Consider this example: The same snippet can run at @b 0.7 ns per operation
 *  when branch predictions are accurate but slows down to @b 3.7 ns when
 *  predictions fail.
 */
static void branch_cost(bm::State &state) {
    auto count = static_cast<std::size_t>(state.range(0));
    std::vector<std::int32_t> random_values(count);
    std::generate_n(random_values.begin(), random_values.size(), &std::rand);
    std::int32_t variable = 0;
    std::size_t iteration = 0;

    for (auto _ : state) {
        std::int32_t random = random_values[(++iteration) & (count - 1)];
        bm::DoNotOptimize( //
            variable =     //
            (random & 1)   //
                ? (variable + random)
                : (variable * random));
    }
}

BENCHMARK(branch_cost)->RangeMultiplier(4)->Range(256, 32 * 1024);

#pragma endregion // Branch Prediction

#pragma region Cache Misses

/**
 *  Over the decades, CPU speeds have outpaced memory speeds, creating a gap
 *  mitigated by multi-level caching (L1/L2/L3). Cache misses occur when the
 *  CPU fetches data from RAM instead of the cache, incurring high latency.
 *
 *  Access patterns play a crucial role:
 *      - @b Sequential: Predictable and optimal for the CPU prefetcher.
 *      - @b Random: Unpredictable, leading to frequent cache misses.
 *
 *  Benchmarks demonstrate this gap—sequential access can outperform random access
 *  by 10x or more for data sizes exceeding cache capacity. However, the difference
 *  narrows for smaller datasets, benefiting from spatial and temporal locality.
 */

#include <random> // `std::random_device`, `std::mt19937`

enum class access_order { sequential, random };

template <access_order access_order_>
static void cache_misses_cost(bm::State &state) {
    auto count = static_cast<std::uint32_t>(state.range(0));

    // Populate with arbitrary data
    std::vector<std::int32_t> data(count);
    std::iota(data.begin(), data.end(), 0);

    // Initialize different access orders
    std::vector<std::uint32_t> indices(count);
    if constexpr (access_order_ == access_order::random) {
        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::uniform_int_distribution<std::uint32_t> uniform_distribution(0, count - 1);
        std::generate_n(indices.begin(), indices.size(), [&] { return uniform_distribution(generator); });
    }
    else { std::iota(indices.begin(), indices.end(), 0u); }

    // The actual benchmark:
    for (auto _ : state) {
        std::int64_t sum = 0;
        for (auto index : indices) bm::DoNotOptimize(sum += data[index]);
    }
}

BENCHMARK(cache_misses_cost<access_order::sequential>)
    ->MinTime(2)
    ->RangeMultiplier(8)
    ->Range(8u * 1024u, 128u * 1024u * 1024u);
BENCHMARK(cache_misses_cost<access_order::random>)
    ->MinTime(2)
    ->RangeMultiplier(8)
    ->Range(8u * 1024u, 128u * 1024u * 1024u);

/**
 *  For small arrays, the execution speed will be identical.
 *  For larger ones, the latency can differ @b 15x!
 */

#pragma endregion // Cache Misses

#pragma endregion // - Basics

#pragma region - Numerics

#pragma region Accuracy vs Efficiency of Standard Libraries

/**
 *  Numerical computing is a core subject in high-performance computing (HPC)
 *  research and graduate studies, yet its foundational concepts are more
 *  accessible than they seem. Let's start with one of the most basic operations
 *  — computing the @b sine of a number.
 */
#include <cmath> // `std::sin`

static void f64_sin(bm::State &state) {
    double argument = std::rand(), result = 0;
    for (auto _ : state) bm::DoNotOptimize(result = std::sin(argument += 1.0));
}

BENCHMARK(f64_sin);

/**
 *  Standard C library functions like `sin` and `sinf` are designed for maximum
 *  accuracy, often at the cost of performance. We can explore approximations
 *  to trade precision for speed.
 *
 *  A common approach is using the Taylor-Maclaurin @b series — a polynomial
 *  expansion of a function around a point. By limiting the expansion to a few
 *  terms, we can approximate `sin(x)` as:
 *
 *      sin(x) ≈ x - (x^3)/3! + (x^5)/5!
 *
 *  This reduces the computational cost but comes with reduced accuracy.
 *
 *  @see Taylor series: https://en.wikipedia.org/wiki/Taylor_series
 */

static void f64_sin_maclaurin(bm::State &state) {
    double argument = std::rand(), result = 0;
    for (auto _ : state) {
        argument += 1.0;
        result = argument - std::pow(argument, 3) / 6 + std::pow(argument, 5) / 120;
        bm::DoNotOptimize(result);
    }
}

BENCHMARK(f64_sin_maclaurin);

/**
 *  Result: latency reduction from @b 31 ns down to @b 21 ns.
 *
 *  The `std::pow` function is highly generic and not optimized for small,
 *  constant integer exponents. We can implement a specialized version for
 *  faster @b and slightly more accurate results.
 */

static void f64_sin_maclaurin_powless(bm::State &state) {
    double argument = std::rand(), result = 0;
    for (auto _ : state) {
        argument += 1.0;
        result = (argument) - (argument * argument * argument) / 6.0 +
                 (argument * argument * argument * argument * argument) / 120.0;
        bm::DoNotOptimize(result);
    }
}

BENCHMARK(f64_sin_maclaurin_powless);

/**
 *  Result: latency reduction to @b 2 ns - a @b 15x speedup over the standard!
 *
 *  We can force the compiler to bypass IEEE-754 compliance checks using
 *  "fast-math" attributes, enabling aggressive floating-point optimizations.
 *
 *  Different compilers support this via:
 *  - Clang and GCC: `-ffast-math`
 *  - ICC: `-fp-model=fast`
 *  - MSVC: `/fp:fast`
 *
 *  The GCC syntax can be:
 *  - Old: `__attribute__((optimize("-ffast-math")))`
 *  - New: `[[gnu::optimize("-ffast-math")]]`
 *
 *  Among other things, this may reorder floating-point operations, ignoring
 *  that floating-point arithmetic isn't strictly associative. So if you have
 *  long chains of arithmetic operations, with a arguments significantly
 *  differing in magnitude, you may get highly inaccurate results.
 *
 *  @see Beware of fast-math: https://simonbyrne.github.io/notes/fastmath/
 */
#if defined(__GNUC__) && !defined(__clang__)
#define FAST_MATH [[gnu::optimize("-ffast-math")]]
#elif defined(__clang__)
#define FAST_MATH __attribute__((target("-ffast-math")))
#else
#define FAST_MATH
#endif

FAST_MATH static void f64_sin_maclaurin_with_fast_math(bm::State &state) {
    double argument = std::rand(), result = 0;
    for (auto _ : state) {
        argument += 1.0;
        result = (argument) - (argument * argument * argument) / 6.0 +
                 (argument * argument * argument * argument * argument) / 120.0;
        bm::DoNotOptimize(result);
    }
}

BENCHMARK(f64_sin_maclaurin_with_fast_math);

/**
 *  Result: latency of @b 0.8 ns - almost @b 40x faster than the standard!
 *
 *  Advanced libraries like SimSIMD and SLEEF can achieve even better
 *  performance through SIMD-optimized implementations, sometimes trading
 *  accuracy or the breadth of the input range for speed.
 *
 *  @see SimSIMD repository: https://github.com/ashvardanian/simsimd
 *  @see SLEEF repository: https://github.com/shibatch/sleef
 */

#pragma endregion // Accuracy vs Efficiency of Standard Libraries

#pragma region Expensive Integer Operations

template <typename scalar_type_>
scalar_type_ square(scalar_type_ x) noexcept {
    return x * x;
}

/**
 *  It's common knowledge that floating-point math can be costly, but even
 *  integer operations can be surprisingly expensive. @b Division and modulo
 *  operations are notorious examples.
 */
static void integral_division(bm::State &state) {
    std::int64_t a = square<std::int64_t>(std::rand()), b = square<std::int64_t>(std::rand()), c;
    for (auto _ : state) bm::DoNotOptimize(c = (++a) / (++b));
}

BENCHMARK(integral_division);

/**
 *  Division takes around ~10 CPU cycles or @b 2.5 ns. However, if the divisor
 *  is known at compile time, the compiler can replace the division with
 *  faster shift and multiply instructions — even for large prime numbers
 *  like  `2147483647`.
 *
 *  @see More Details: https://www.sciencedirect.com/science/article/pii/S2405844021015450
 */
static void integral_division_by_constexpr(bm::State &state) {
    constexpr std::int64_t b = 2147483647;
    std::int64_t a = std::rand(), c;
    for (auto _ : state) bm::DoNotOptimize(c = (++a) / b);
}

BENCHMARK(integral_division_by_constexpr);

/**
 *  The @b `constexpr` specifier is not strictly required if the compiler can
 *  deduce that a value is constant at compile time. However, this optimization
 *  can affect benchmarking results by eliminating divisions entirely through
 *  strength reduction (replacing division with faster operations like shifts
 *  and multiplications).
 *
 *  To ensure the division is evaluated at runtime, forcing the compiler to
 *  treat the divisor as a mutable value, wrap it with `std::launder`. This
 *  prevents constant propagation and keeps the benchmark realistic.
 */
#include <new> // `std::launder`

static void integral_division_by_const(bm::State &state) {
    std::int64_t b = 2147483647;
    std::int64_t a = square<std::int64_t>(std::rand()), c = 0;
    for (auto _ : state) bm::DoNotOptimize(c = (++a) / *std::launder(&b));
}

BENCHMARK(integral_division_by_const);

/**
 *  An important optimization trick is that 32-bit integer division can be
 *  performed using 64-bit double-precision floating-point division. This
 *  technique takes advantage of the CPU's highly optimized floating-point
 *  division unit, reducing the operation's latency from approximately
 *  @b 2.5ns to @b 0.5ns.
 *
 *  Since 64-bit doubles can exactly represent all 32-bit signed integers,
 *  this method introduces @b no precision loss, making it a safe and efficient
 *  alternative when division performance is critical.
 */
static void integral_division_with_doubles(bm::State &state) {
    std::int32_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state)
        bm::DoNotOptimize(c = static_cast<std::int32_t>(static_cast<double>(++a) / static_cast<double>(++b)));
}

BENCHMARK(integral_division_with_doubles);

/**
 *  Understanding how compilation settings affect performance is crucial.
 *  The @b `-O3` optimization flag alone isn't always sufficient. Even when
 *  using compiler intrinsics like @b `__builtin_popcountll`, the actual
 *  implementation depends on the target CPU generation. If the CPU lacks
 *  a native Assembly instruction for population count, the compiler will
 *  fall back to a slower, software-emulated version.
 *
 *  To ensure optimal performance, we can use GCC attributes to specify
 *  the target CPU architecture at the function level. The only difference
 *  between the following functions is the applied target attribute,
 *  while the internal logic remains identical.
 */

#if defined(__GNUC__) && !defined(__clang__)

[[gnu::target("arch=core2")]]
int bits_popcount_emulated(std::uint64_t x) {
    return __builtin_popcountll(x);
}

[[gnu::target("arch=corei7")]]
int bits_popcount_native(std::uint64_t x) {
    return __builtin_popcountll(x);
}

static void bits_population_count_core_2(bm::State &state) {
    auto a = static_cast<std::uint64_t>(std::rand());
    for (auto _ : state) bm::DoNotOptimize(bits_popcount_emulated(++a));
}

BENCHMARK(bits_population_count_core_2);

static void bits_population_count_core_i7(bm::State &state) {
    auto a = static_cast<std::uint64_t>(std::rand());
    for (auto _ : state) bm::DoNotOptimize(bits_popcount_native(++a));
}

BENCHMARK(bits_population_count_core_i7);
#endif

/**
 *  The performance difference is substantial — a @b 3x improvement:
 *  - Core 2 variant: 2.4 ns
 *  - Core i7 variant: 0.8 ns
 *
 *  Fun fact: Only a few integer operations on select AMD CPUs can take as long
 *  as @b ~100 CPU cycles. This includes BMI2 bit-manipulation instructions such
 *  as @b `pdep` and @b `pext`, particularly on AMD Zen 1 and Zen 2 architectures.
 *
 *  @see BMI2 details: https://www.chessprogramming.org/BMI2
 */

#pragma endregion // Expensive Integer Operations

/**
 *  Understanding common algorithmic design patterns across various computational
 *  complexity levels is invaluable. So far, most tasks we've examined have
 *  featured linear complexity in both space and time. Broadly, the following
 *  distinctions are useful:
 *
 *  - Sublinear (e.g., @b O(logN) ): Often found in search workloads, typically IO-bound.
 *  - Linear and sub-quadratic (e.g., @b O(N) to @b O(N*logN) ): Most conventional "coding" tasks.
 *  - Low polynomial (e.g., @b O(N^1.5) to @b O(N^4) ): Common in matrix operations and graph algorithms.
 *  - High polynomial and exponential (e.g., O(N^5) and @b beyond): Rarely practical to solve.
 *
 *  Among low-polynomial tasks, matrix operations stand out as particularly important.
 *  Matrix multiplication forms the foundation of linear algebra and is critical in
 *  fields such as artificial intelligence, computer graphics, and physics simulations.
 *  Given their significance, many CPUs provide native instructions for small matrix
 *  multiplications (e.g., 4x4 or 8x8). Larger matrix multiplications are decomposed
 *  into smaller ones to take advantage of these optimizations.
 *
 *  Let's emulate such operations and explore the underlying principles.
 */

#pragma region Compute vs Memory Bounds with Matrix Multiplications

struct f32x4x4_t {
    float scalars[4][4];
};

f32x4x4_t f32x4x4_matmul_kernel(f32x4x4_t const &a, f32x4x4_t const &b) noexcept {
    f32x4x4_t c {};
    // This code gets auto-vectorized regardless of the loop order,
    // be it "ijk", "ikj", "jik", "jki", "kij", or "kji".
    // That's not necessarily the case for other matrix sizes:
    // https://lemire.me/blog/2024/06/13/rolling-your-own-fast-matrix-multiplication-loop-order-and-vectorization/
    for (std::size_t i = 0; i != 4; ++i)
        for (std::size_t j = 0; j != 4; ++j)
            for (std::size_t k = 0; k != 4; ++k) c.scalars[i][j] += a.scalars[i][k] * b.scalars[k][j];

    return c;
}

static void f32x4x4_matmul(bm::State &state) {
    f32x4x4_t a, b, c;
    std::iota(&a.scalars[0][0], &a.scalars[0][0] + 16, 16);
    std::iota(&b.scalars[0][0], &b.scalars[0][0] + 16, 0);

    for (auto _ : state) bm::DoNotOptimize(c = f32x4x4_matmul_kernel(a, b));

    std::size_t flops_per_cycle = 4 * 4 * 4 * 2 /* 1 addition and 1 multiplication */;
    state.SetItemsProcessed(flops_per_cycle * state.iterations());
}

BENCHMARK(f32x4x4_matmul);

/**
 *  Multiplying two NxN matrices requires up to NxNxN multiplications and NxNx(N-1)
 *  additions. The asymptotic complexity is O(N^3), with the operation count scaling
 *  cubically with the matrix side. Surprisingly, the naive kernel is fully unrolled
 *  and vectorized by the compiler, achieving @b exceptional_performance:
 *  @b ~3.1 ns for 112 arithmetic operations (64 multiplications + 48 additions).
 *
 *  Most of these operations are data-parallel (each cell is independent of others),
 *  enabling the CPU to execute them in parallel. Since the matrix size is small and
 *  known at compile time, the compiler optimizes via loop unrolling—a critical
 *  optimization technique every HPC developer should understand.
 *
 *  Every @b `for` loop is, in essence, a combination of a @b `goto` and an @b `if`.
 *  As we've seen in sections on recursion and branching, jumps and conditions introduce
 *  overhead. Knowing the loop bounds allows us to unroll them manually, expressing every
 *  operation explicitly.
 */

f32x4x4_t f32x4x4_matmul_unrolled_kernel(f32x4x4_t const &a_matrix, f32x4x4_t const &b_matrix) {
    f32x4x4_t c_matrix;
    float const(&a)[4][4] = a_matrix.scalars;
    float const(&b)[4][4] = b_matrix.scalars;
    float(&c)[4][4] = c_matrix.scalars;

    c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0] + a[0][3] * b[3][0];
    c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1] + a[0][3] * b[3][1];
    c[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2] + a[0][3] * b[3][2];
    c[0][3] = a[0][0] * b[0][3] + a[0][1] * b[1][3] + a[0][2] * b[2][3] + a[0][3] * b[3][3];

    c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0] + a[1][3] * b[3][0];
    c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1] + a[1][3] * b[3][1];
    c[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2] + a[1][3] * b[3][2];
    c[1][3] = a[1][0] * b[0][3] + a[1][1] * b[1][3] + a[1][2] * b[2][3] + a[1][3] * b[3][3];

    c[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0] + a[2][3] * b[3][0];
    c[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1] + a[2][3] * b[3][1];
    c[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2] + a[2][3] * b[3][2];
    c[2][3] = a[2][0] * b[0][3] + a[2][1] * b[1][3] + a[2][2] * b[2][3] + a[2][3] * b[3][3];

    c[3][0] = a[3][0] * b[0][0] + a[3][1] * b[1][0] + a[3][2] * b[2][0] + a[3][3] * b[3][0];
    c[3][1] = a[3][0] * b[0][1] + a[3][1] * b[1][1] + a[3][2] * b[2][1] + a[3][3] * b[3][1];
    c[3][2] = a[3][0] * b[0][2] + a[3][1] * b[1][2] + a[3][2] * b[2][2] + a[3][3] * b[3][2];
    c[3][3] = a[3][0] * b[0][3] + a[3][1] * b[1][3] + a[3][2] * b[2][3] + a[3][3] * b[3][3];

    return c_matrix;
}

static void f32x4x4_matmul_unrolled(bm::State &state) {
    f32x4x4_t a, b, c;
    std::iota(&a.scalars[0][0], &a.scalars[0][0] + 16, 16);
    std::iota(&b.scalars[0][0], &b.scalars[0][0] + 16, 0);

    for (auto _ : state) bm::DoNotOptimize(c = f32x4x4_matmul_unrolled_kernel(a, b));

    std::size_t flops_per_cycle = 4 * 4 * 4 * 2 /* 1 addition and 1 multiplication */;
    state.SetItemsProcessed(flops_per_cycle * state.iterations());
}

BENCHMARK(f32x4x4_matmul_unrolled);

/**
 *  The unrolled variant completes in @b 3.1ns, benefiting from auto-vectorization
 *  using SIMD. Modern CPUs leverage super-scalar execution, often called SIMD
 *  (Single Instruction, Multiple Data). These units operate on 128-, 256-, or
 *  512-bit words, containing multiple smaller components (e.g., 64-, 32-, 16-,
 *  or 8-bit integers or floats). While compilers handle auto-vectorization in
 *  many cases, they often fall short of manual optimizations. Achieving a 5x
 *  speedup is common with hand-tuned SIMD code, with some operations seeing
 *  10-100x improvements for certain data types.
 *
 *  @see Understanding SIMD: Infinite Complexity of Trivial Problems
 *  https://www.modular.com/blog/understanding-simd-infinite-complexity-of-trivial-problems
 *  @see GCC Compiler vs Human - 119x Faster Assembly
 *  https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/
 *
 *  To push the limits of performance further, let's switch from scalar
 *  operations in the unrolled kernel to @b SSE4.1 SIMD instructions—among the
 *  earliest SIMD instruction sets available on most x86 CPUs—and explore how
 *  close we can get to the theoretical 100x improvement.
 */

#if defined(__SSE2__)
#include <smmintrin.h> // `_mm_dp_ps` and `_MM_TRANSPOSE4_PS`

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("sse2", "sse3", "sse4.1")
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("sse2,sse3,sse4.1"))), apply_to = function)
#endif

f32x4x4_t f32x4x4_matmul_sse41_kernel(f32x4x4_t const &a, f32x4x4_t const &b) noexcept {
    f32x4x4_t c;
    // Load a continuous vector of 4x floats in a single instruction., invoked
    // by the `_mm_loadu_ps` intrinsic.
    __m128 a_row_0 = _mm_loadu_ps(&a.scalars[0][0]);
    __m128 a_row_1 = _mm_loadu_ps(&a.scalars[1][0]);
    __m128 a_row_2 = _mm_loadu_ps(&a.scalars[2][0]);
    __m128 a_row_3 = _mm_loadu_ps(&a.scalars[3][0]);

    // Load the columns of the matrix B, by loading the 4 rows and then
    // transposing with an SSE macro:
    // https://randombit.net/bitbashing/posts/integer_matrix_transpose_in_sse2.html
    __m128 b_col_0 = _mm_loadu_ps(&b.scalars[0][0]);
    __m128 b_col_1 = _mm_loadu_ps(&b.scalars[1][0]);
    __m128 b_col_2 = _mm_loadu_ps(&b.scalars[2][0]);
    __m128 b_col_3 = _mm_loadu_ps(&b.scalars[3][0]);
    _MM_TRANSPOSE4_PS(b_col_0, b_col_1, b_col_2, b_col_3);

    // Multiply A rows by B columns and store the result in C.
    // Use bitwise "OR" to aggregate dot products and store results.
    //
    // The individual dot products are calculated with the `_mm_dp_ps`
    // intrinsic, which is a dot product of two vectors, with the result stored
    // in a single float. The last argument is a mask, which specifies which
    // components of the vectors should be multiplied and added.
    __m128 c_row_0 = _mm_or_ps( //
        _mm_or_ps(_mm_dp_ps(a_row_0, b_col_0, 0xF1), _mm_dp_ps(a_row_0, b_col_1, 0xF2)),
        _mm_or_ps(_mm_dp_ps(a_row_0, b_col_2, 0xF4), _mm_dp_ps(a_row_0, b_col_3, 0xF8)));
    _mm_storeu_ps(&c.scalars[0][0], c_row_0);

    __m128 c_row_1 = _mm_or_ps( //
        _mm_or_ps(_mm_dp_ps(a_row_1, b_col_0, 0xF1), _mm_dp_ps(a_row_1, b_col_1, 0xF2)),
        _mm_or_ps(_mm_dp_ps(a_row_1, b_col_2, 0xF4), _mm_dp_ps(a_row_1, b_col_3, 0xF8)));
    _mm_storeu_ps(&c.scalars[1][0], c_row_1);

    __m128 c_row_2 = _mm_or_ps( //
        _mm_or_ps(_mm_dp_ps(a_row_2, b_col_0, 0xF1), _mm_dp_ps(a_row_2, b_col_1, 0xF2)),
        _mm_or_ps(_mm_dp_ps(a_row_2, b_col_2, 0xF4), _mm_dp_ps(a_row_2, b_col_3, 0xF8)));
    _mm_storeu_ps(&c.scalars[2][0], c_row_2);

    __m128 c_row_3 = _mm_or_ps( //
        _mm_or_ps(_mm_dp_ps(a_row_3, b_col_0, 0xF1), _mm_dp_ps(a_row_3, b_col_1, 0xF2)),
        _mm_or_ps(_mm_dp_ps(a_row_3, b_col_2, 0xF4), _mm_dp_ps(a_row_3, b_col_3, 0xF8)));
    _mm_storeu_ps(&c.scalars[3][0], c_row_3);

    return c;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#elif defined(__clang__)
#pragma clang attribute pop
#endif

static void f32x4x4_matmul_sse41(bm::State &state) {
    f32x4x4_t a, b, c;
    std::iota(&a.scalars[0][0], &a.scalars[0][0] + 16, 16);
    std::iota(&b.scalars[0][0], &b.scalars[0][0] + 16, 0);

    for (auto _ : state) bm::DoNotOptimize(c = f32x4x4_matmul_sse41_kernel(a, b));

    std::size_t flops_per_cycle = 4 * 4 * 4 * 2 /* 1 addition and 1 multiplication */;
    state.SetItemsProcessed(flops_per_cycle * state.iterations());
}

BENCHMARK(f32x4x4_matmul_sse41);
#endif // defined(__SSE2__)

/**
 *  The result is @b 19.6 ns compared to the @b 3.1 ns from the unrolled kernel.
 *  It turns out we were not as clever as we thought. Disassembling the unrolled
 *  kernel reveals that the compiler was already optimizing it better than we did.
 *  Each line compiles into a series of efficient instructions like:
 *
 *      vmovss  xmm0, dword ptr [rdi]
 *      vmovss  xmm1, dword ptr [rdi + 4]
 *      vmulss  xmm1, xmm1, dword ptr [rsi + 16]
 *      vfmadd231ss     xmm1, xmm0, dword ptr [rsi]
 *      vmovss  xmm0, dword ptr [rdi + 8]
 *      vfmadd132ss     xmm0, xmm1, dword ptr [rsi + 32]
 *      vmovss  xmm1, dword ptr [rdi + 12]
 *      vfmadd132ss     xmm1, xmm0, dword ptr [rsi + 48]
 *      vmovss  dword ptr [rdx], xmm1
 *
 *  Instructions like `vfmadd132ss` and `vfmadd231ss` operating on @b `xmm`
 *  registers show how smart the compiler was at exploiting SIMD capabilities.
 *  But the game isn't over—there's still room to optimize for larger instruction
 *  sets.
 *
 *  @see Explore the unrolled kernel assembly on GodBolt:
 *       https://godbolt.org/z/bW5nnTKs1
 *  @see Explore instruction latencies on @b uops.info:
 *       https://uops.info/table.html
 *
 *  With AVX-512 on modern CPUs, we can fit an entire matrix into a single
 *  @b `zmm` register (512 bits wide). This Instruction Set Extension is available
 *  on Intel Skylake-X, Ice Lake, and AMD Zen4 CPUs, offering powerful functionality.
 *  However, on Zen4, AVX-512 is emulated as two 256-bit operations, so the performance
 *  isn't as strong as on Intel CPUs. Native support comes with Zen5.
 */
#if defined(__AVX512F__)
#include <immintrin.h> // `_mm512_loadu_ps`
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512bw", "avx512vl", "bmi2")
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512bw,avx512vl,bmi2"))), apply_to = function)
#endif

f32x4x4_t f32x4x4_matmul_avx512_kernel(f32x4x4_t const &a, f32x4x4_t const &b) noexcept {
    __m512 a_mat = _mm512_loadu_ps(&a.scalars[0][0]);
    __m512 b_mat = _mm512_loadu_ps(&b.scalars[0][0]);

    __m512 a_vec_1 = _mm512_permute_ps(a_mat, 0x0);
    __m512 b_vec_1 = _mm512_broadcast_f32x4(_mm512_castps512_ps128(b_mat));
    __m512 c_vec = _mm512_mul_ps(a_vec_1, b_vec_1);

    __m512 a_vec_2 = _mm512_permute_ps(a_mat, 0x55);
    b_mat = _mm512_castsi512_ps(_mm512_alignr_epi64(_mm512_castps_si512(b_mat), _mm512_castps_si512(b_mat), 0x2));
    __m512 b_vec_2 = _mm512_broadcast_f32x4(_mm512_castps512_ps128(b_mat));
    c_vec = _mm512_fmadd_ps(a_vec_2, b_vec_2, c_vec);

    __m512 a_vec_3 = _mm512_permute_ps(a_mat, 0xAA);
    b_mat = _mm512_castsi512_ps(_mm512_alignr_epi64(_mm512_castps_si512(b_mat), _mm512_castps_si512(b_mat), 0x2));
    __m512 b_vec_3 = _mm512_broadcast_f32x4(_mm512_castps512_ps128(b_mat));
    c_vec = _mm512_fmadd_ps(a_vec_3, b_vec_3, c_vec);

    __m512 a_vec_4 = _mm512_permute_ps(a_mat, 0xFF);
    b_mat = _mm512_castsi512_ps(_mm512_alignr_epi64(_mm512_castps_si512(b_mat), _mm512_castps_si512(b_mat), 0x2));
    __m512 b_vec_4 = _mm512_broadcast_f32x4(_mm512_castps512_ps128(b_mat));
    c_vec = _mm512_fmadd_ps(a_vec_4, b_vec_4, c_vec);

    f32x4x4_t c;
    _mm512_storeu_ps(&c.scalars[0][0], c_vec);
    return c;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#elif defined(__clang__)
#pragma clang attribute pop
#endif

static void f32x4x4_matmul_avx512(bm::State &state) {
    f32x4x4_t a, b, c;
    std::iota(&a.scalars[0][0], &a.scalars[0][0] + 16, 16);
    std::iota(&b.scalars[0][0], &b.scalars[0][0] + 16, 0);

    for (auto _ : state) bm::DoNotOptimize(c = f32x4x4_matmul_avx512_kernel(a, b));

    std::size_t flops_per_cycle = 4 * 4 * 4 * 2 /* 1 addition and 1 multiplication */;
    state.SetItemsProcessed(flops_per_cycle * state.iterations());
}
BENCHMARK(f32x4x4_matmul_avx512);

#endif // defined(__AVX512F__)

/**
 *  The result is @b 2.8 ns on Sapphire Rapids—a modest 10% improvement. To
 *  fully leverage AVX-512, we need larger matrices where horizontal reductions
 *  don't dominate latency. For small sizes like 4x4, the wide ZMM registers
 *  aren't fully utilized.
 *
 *  As an exercise, try implementing matrix multiplication for 3x3 matrices.
 *  Despite requiring fewer operations (27 multiplications and 18 additions
 *  compared to 64 multiplications and 48 additions for 4x4), the compiler
 *  may peak at @b 5.3 ns — whopping @b 71% slower for a @b 60% smaller task!
 *
 *  AVX-512 includes advanced instructions like `_mm512_mask_compressstoreu_ps`
 *  and `_mm512_maskz_expandloadu_ps`, which could be used with a mask like
 *  @b 0b0000'0111'0111'0111 to handle 3x3 matrices. However, their high latency
 *  means the performance will still degrade—@b around 5 ns in practice.
 *
 *  Benchmark everything! Don't assume less work translates to faster execution.
 */

#pragma endregion // Compute vs Memory Bounds with Matrix Multiplications

/**
 *  When designing high-performance kernels, memory alignment is crucial.
 *  Misaligned memory accesses split data across cache lines, causing extra
 *  loads and reducing efficiency. While split loads are unavoidable for large
 *  data structures, smaller kernels can benefit significantly from careful
 *  memory management.
 *
 *  Modern CPUs typically have 64-byte cache lines, though Apple's M-series
 *  uses 128 bytes. This means assuming @b `alignas(64)` won't ensure compatibility.
 *  Instead, cache properties must be inferred at runtime for optimal performance.
 *
 *  In this benchmark, we demonstrate how `std::sort` behaves when processing
 *  cache-line-sized objects that are either correctly aligned or intentionally
 *  misaligned. This covers:
 *
 *  - Using a custom strided iterator (`strided_ptr`) to simulate offset memory access.
 *  - Generating semi-random integers using Knuth's multiplicative hash.
 *  - Reading cache properties with platform-specific APIs.
 *  - Flushing the CPU cache between runs to ensure consistent results.
 */

#pragma region Alignment of Memory Accesses

#include <cassert>  // `assert`
#include <fstream>  // `std::ifstream`
#include <iterator> // `std::random_access_iterator_tag`
#include <memory>   // `std::assume_aligned`
#include <string>   // `std::string`, `std::stoull`

/**
 *  @brief  Reads the contents of a file from the specified path into a string.
 */
std::string read_file_contents(std::string const &path) {
    std::ifstream file(path);
    std::string content;
    if (!file.is_open()) return 0;
    std::getline(file, content);
    file.close();
    return content;
}

/**
 *  @brief  Fetches the cache line size using OS-specific APIs.
 *          Supports Linux, macOS, and Windows.
 *
 *  It's easier to use the @b `std::hardware_destructive_interference_size`
 *  in C++ 17 and newer, if the `__cpp_lib_hardware_interference_size` feature
 *  macro is defined. But this will be incorrect, if the compilation platform
 *  is different from the runtime platform.
 *
 *  A more advanced approach would be to use hardware-specific instructions,
 *  like the @b `cpuid` on x86, and infer the cache line size from the returned
 *  bitmasks. That's however, not only different on Arm, but also differs
 *  between Intel and AMD!
 */
std::size_t fetch_cache_line_width() {

#if defined(__linux__)
    // On Linux, we can read the cache line size and the L2 cache size from the
    // "sysfs" virtual filesystem. It can provide the properties of each individual
    // CPU core.
    std::string file_contents = read_file_contents("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
    std::size_t cache_line_size = std::stoull(file_contents);
    return cache_line_size;

#elif defined(__APPLE__)
    // On macOS, we can use the `sysctlbyname` function to read the
    // `hw.cachelinesize` and `hw.l2cachesize` values into unsigned integers.
    // You can achieve the same by using the `sysctl -a` command-line utility.
    size_t size;
    size_t len = sizeof(size);
    if (sysctlbyname("hw.cachelinesize", &size, &len, nullptr, 0) == 0) return size;

#elif defined(_WIN32)
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
    DWORD len = sizeof(buffer);
    if (GetLogicalProcessorInformation(buffer, &len))
        for (size_t i = 0; i < len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i)
            if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == 1) return buffer[i].Cache.LineSize;
#endif

    return 0;
}

/**
 *  @brief  A minimalistic pointer with non-unit stride/step.
 */
template <typename value_type_>
class strided_ptr {
    std::byte *data_;
    std::size_t stride_;

  public:
    using value_type = value_type_;
    using pointer = value_type_ *;
    using reference = value_type_ &;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    strided_ptr(std::byte *data, std::size_t stride_bytes) : data_(data), stride_(stride_bytes) {
        assert(data_ && "Pointer must not be null, as NULL arithmetic is undefined");
    }
    reference operator*() const noexcept {
        return *std::launder(std::assume_aligned<1>(reinterpret_cast<pointer>(data_)));
    }
    reference operator[](difference_type i) const noexcept {
        return *std::launder(std::assume_aligned<1>(reinterpret_cast<pointer>(data_ + i * stride_)));
    }

    // clang-format off
    pointer operator->() const noexcept { return &operator*(); }
    strided_ptr &operator++() noexcept { data_ += stride_; return *this; }
    strided_ptr operator++(int) noexcept { strided_ptr temp = *this; ++(*this); return temp; }
    strided_ptr &operator--() noexcept { data_ -= stride_; return *this; }
    strided_ptr operator--(int) noexcept { strided_ptr temp = *this; --(*this); return temp; }
    strided_ptr &operator+=(difference_type offset) noexcept { data_ += offset * stride_; return *this; }
    strided_ptr &operator-=(difference_type offset) noexcept { data_ -= offset * stride_; return *this; }
    strided_ptr operator+(difference_type offset) noexcept { strided_ptr temp = *this; return temp += offset; }
    strided_ptr operator-(difference_type offset) noexcept { strided_ptr temp = *this; return temp -= offset; }
    friend difference_type operator-(strided_ptr const &a, strided_ptr const &b) noexcept { assert(a.stride_ == b.stride_); return (a.data_ - b.data_) / static_cast<difference_type>(a.stride_); }
    friend bool operator==(strided_ptr const &a, strided_ptr const &b) noexcept { return a.data_ == b.data_; }
    friend bool operator<(strided_ptr const &a, strided_ptr const &b) noexcept { return a.data_ < b.data_; }
    friend bool operator!=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(a == b); }
    friend bool operator>(strided_ptr const &a, strided_ptr const &b) noexcept { return b < a; }
    friend bool operator<=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(b < a); }
    friend bool operator>=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(a < b); }
    // clang-format on
};

template <bool aligned_>
static void memory_access(bm::State &state) {
    constexpr std::size_t typical_l2_size = 1024u * 1024u;
    std::size_t const cache_line_width = fetch_cache_line_width();
    assert(cache_line_width > 0 && __builtin_popcountll(cache_line_width) == 1 &&
           "The cache line width must be a power of two greater than 0");

    // We are using a fairly small L2-cache-sized buffer to show, that this is
    // not just about Big Data. Anything beyond a few megabytes with irregular
    // memory accesses may suffer from the same issues. For split-loads, pad our
    // buffer with an extra `cache_line_width` bytes of space.
    std::size_t const buffer_size = typical_l2_size + cache_line_width;
    std::unique_ptr<std::byte, decltype(&std::free)> const buffer(                        //
        reinterpret_cast<std::byte *>(std::aligned_alloc(cache_line_width, buffer_size)), //
        &std::free);
    std::byte *const buffer_ptr = buffer.get();

    // Let's initialize a strided range using out `strided_ptr` template, but
    // for `aligned_ == false` make sure that the scalar-of-interest in each
    // stride is located exactly at the boundary between two cache lines.
    std::size_t const offset_within_page = !aligned_ ? (cache_line_width - sizeof(std::uint32_t) / 2) : 0;
    strided_ptr<std::uint32_t> integers(buffer_ptr + offset_within_page, cache_line_width);

    // We will start with a random seed position and walk through the buffer.
    std::uint32_t semi_random_state = 0xFFFFFFFFu;
    std::size_t const count_pages = typical_l2_size / cache_line_width;
    for (auto _ : state) {
        // Generate some semi-random data, using Knuth's multiplicative hash
        // number derived from the golden ratio.
        std::generate_n(integers, count_pages, [&semi_random_state] { return semi_random_state *= 2654435761u; });

        // Flush all of the pages out of the cache.
        // The `__builtin___clear_cache(buffer_ptr, buffer_ptr + buffer.size())`
        // compiler intrinsic can't be used for the data cache, only the
        // instructions cache. For Arm, GCC provides a `__aarch64_sync_cache_range`
        // intrinsic, but it's not available in Clang.
        for (std::size_t i = 0; i != count_pages; ++i) _mm_clflush(&integers[i]);
        bm::ClobberMemory();

        std::sort(integers, integers + count_pages);
    }
}

static void memory_access_unaligned(bm::State &state) { memory_access<false>(state); }
static void memory_access_aligned(bm::State &state) { memory_access<true>(state); }

BENCHMARK(memory_access_unaligned)->MinTime(10);
BENCHMARK(memory_access_aligned)->MinTime(10);

/**
 *  One variant executes in 5.8 miliseconds, and the other in 5.2 miliseconds,
 *  consistently resulting a @b 10% performance difference.
 */

#pragma endregion // Alignment of Memory Accesses

#pragma region Non Uniform Memory Access

/**
 *  Takes a string like "64K" and "128M" and returns the corresponding size in
 *  bytes, expanding the multiple prefixes to the actual size, like "65536" and
 *  "134217728", respectively.
 */
std::size_t parse_size_string(std::string const &str) {
    std::size_t value = std::stoull(str);
    if (str.find("K") != std::string::npos || str.find("k") != std::string::npos) { value *= 1024; }
    else if (str.find("M") != std::string::npos || str.find("m") != std::string::npos) { value *= 1024 * 1024; }
    else if (str.find("G") != std::string::npos || str.find("g") != std::string::npos) { value *= 1024 * 1024 * 1024; }
    return value;
}

#pragma endregion // Non Uniform Memory Access

#pragma endregion // - Numerics

/**
 *  Designing efficient kernels is only the first step; composing them
 *  into full programs without losing performance is the real challenge.
 *
 *  Consider a hypothetical numeric processing pipeline:
 *
 *    1. Generate all integers in a given range (e.g., [1, 33]).
 *    2. Filter out integers that are perfect squares.
 *    3. Expand each remaining number into its prime factors.
 *    4. Sum all prime factors from the filtered numbers.
 *
 *  We'll explore four implementations of this pipeline:
 *
 *    - C++11 using `template`-based lambda functions.
 *    - C++11 using `std::function` for dynamic callbacks.
 *    - C++20 coroutines using a lightweight generator.
 *    - C++20 ranges with a lazily evaluated factorization.
 */

#pragma region - Pipelines and Abstractions

/**
 *  @brief  Checks if a number is a power of two.
 */
[[gnu::always_inline]]
inline bool is_power_of_two(std::uint64_t x) noexcept {
    return __builtin_popcountll(x) == 1;
}

/**
 *  @brief  Checks if a number is a power of three using modulo division.
 *          The largest power of three fitting in a 64-bit integer is 3^40.
 */
[[gnu::always_inline]]
inline bool is_power_of_three(std::uint64_t x) noexcept {
    constexpr std::uint64_t max_power_of_three = 12157665459056928801ull;
    return x > 0 && max_power_of_three % x == 0;
}

constexpr std::uint64_t pipe_start = 3;
constexpr std::uint64_t pipe_end = 49;

#pragma region Coroutines and Asynchronous Programming

/**
 *  @brief  Supplies the prime factors to a template-based callback.
 */
template <typename callback_type_>
[[gnu::always_inline]] inline void prime_factors_cpp11( //
    std::uint64_t input, callback_type_ &&callback) noexcept {
    // Handle factor 2 separately
    while ((input & 1) == 0) {
        callback(2);
        input >>= 1; // equivalent to `input /= 2`
    }

    // Now factor is always odd, start from 3 and go up by 2
    for (std::uint64_t factor = 3; factor * factor <= input; factor += 2) {
        while (input % factor == 0) {
            callback(factor);
            input /= factor;
        }
    }

    // If input is still greater than 1, then it's a prime factor
    if (input > 1) callback(input);
}

static void pipeline_cpp11_lambdas(bm::State &state) {
    std::uint64_t sum = 0, count = 0;
    for (auto _ : state) {
        sum = 0, count = 0;
        for (std::uint64_t value = pipe_start; value <= pipe_end; ++value) {
            if (!is_power_of_two(value) && !is_power_of_three(value))
                prime_factors_cpp11(value, [&](std::uint64_t factor) { sum += factor, count++; });
        }
        bm::DoNotOptimize(sum);
    }
    state.counters["sum"] = sum;
    state.counters["count"] = count;
}

BENCHMARK(pipeline_cpp11_lambdas);

/**
 *  A more conventional approach using @b `std::function` callbacks.
 *  While this simplifies the interface, it introduces heap allocations,
 *  and brings a lot of noisy source code into your translation unit:
 *  up to @b 27'400 lines of code, being one of the largest standard
 *  headers according to Philip Trettner's "C++ Compile Health Watchdog".
 *
 *  @see Headers length: https://artificial-mind.net/projects/compile-health/
 */
#include <functional> // `std::function`

static void for_range_stl(std::uint64_t start, std::uint64_t end, std::function<void(std::uint64_t)> const &callback) {
    for (std::uint64_t i = start; i <= end; ++i) callback(i);
}

static void filter_stl( //
    std::uint64_t value, std::function<bool(std::uint64_t)> const &predicate,
    std::function<void(std::uint64_t)> const &callback) {
    if (!predicate(value)) callback(value);
}

static void prime_factors_stl(std::uint64_t input, std::function<void(std::uint64_t)> const &callback) {
    prime_factors_cpp11(input, callback);
}

static void pipeline_cpp11_stl(bm::State &state) {
    std::uint64_t sum = 0, count = 0;
    for (auto _ : state) {
        sum = 0, count = 0;
        for_range_stl(pipe_start, pipe_end, [&](std::uint64_t value) {
            filter_stl(value, is_power_of_two, [&](std::uint64_t value) {
                filter_stl(value, is_power_of_three, [&](std::uint64_t value) {
                    prime_factors_stl(value, [&](std::uint64_t factor) { sum += factor, count++; });
                });
            });
        });
        bm::DoNotOptimize(sum);
    }
    state.counters["sum"] = sum;
    state.counters["count"] = count;
}

BENCHMARK(pipeline_cpp11_stl);

/**
 *  C++20 introduces @b coroutines in the language, but not in the library,
 *  so we need to provide a minimal implementation of a "generator" class.
 *
 *  @see "Asymmetric Transfer" blogposts on coroutines by Lewis Baker:
 *       https://lewissbaker.github.io/
 */
#include <coroutine> // `std::coroutine_handle`

template <typename integer_type_>
struct integer_generator {
    struct promise_type {
        integer_type_ value_;

        std::suspend_always yield_value(integer_type_ value) noexcept {
            value_ = value;
            return {};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        integer_generator get_return_object() noexcept {
            return integer_generator {std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { std::terminate(); }
    };

    std::coroutine_handle<promise_type> coro;

    explicit integer_generator(std::coroutine_handle<promise_type> h) noexcept : coro(h) {}
    integer_generator(integer_generator const &) = delete;
    integer_generator(integer_generator &&other) noexcept : coro(other.coro) { other.coro = nullptr; }
    ~integer_generator() noexcept {
        if (coro) coro.destroy();
    }

    struct iterator {
        std::coroutine_handle<promise_type> handle_;

        iterator &operator++() noexcept {
            handle_.resume();
            return *this;
        }
        bool operator!=(iterator const &) const noexcept { return !handle_.done(); }
        integer_type_ const &operator*() const noexcept { return handle_.promise().value_; }
    };

    iterator begin() noexcept {
        coro.resume();
        return {coro};
    }
    iterator end() noexcept { return {nullptr}; }
};

integer_generator<std::uint64_t> for_range_generator(std::uint64_t start, std::uint64_t end) noexcept {
    for (std::uint64_t value = start; value <= end; ++value) co_yield value;
}

integer_generator<std::uint64_t> filter_generator( //
    integer_generator<std::uint64_t> values, bool (*predicate)(std::uint64_t)) noexcept {
    for (auto value : values)
        if (!predicate(value)) co_yield value;
}

integer_generator<std::uint64_t> prime_factors_generator(integer_generator<std::uint64_t> values) noexcept {
    for (std::uint64_t value : values) {
        // Factor out 2 first
        while ((value & 1) == 0) {
            co_yield 2;
            value >>= 1; // Equivalent to `value /= 2`
        }

        // Only consider odd factors from here on
        for (std::uint64_t factor = 3; factor * factor <= value; factor += 2) {
            while (value % factor == 0) {
                co_yield factor;
                value /= factor;
            }
        }

        // If value is still greater than 1, it's a prime number
        if (value > 1) co_yield value;
    }
}

static void pipeline_cpp20_coroutines(bm::State &state) {
    std::uint64_t sum = 0, count = 0;
    for (auto _ : state) {
        auto range = for_range_generator(pipe_start, pipe_end);
        auto filtered = filter_generator(filter_generator(std::move(range), is_power_of_two), is_power_of_three);
        auto factors = prime_factors_generator(std::move(filtered));
        // Reduce
        sum = 0, count = 0;
        for (auto factor : factors) sum += factor, count++;
        bm::DoNotOptimize(sum);
    }
    state.counters["sum"] = sum;
    state.counters["count"] = count;
}

BENCHMARK(pipeline_cpp20_coroutines);

#pragma endregion // Coroutines and Asynchronous Programming

#pragma region Ranges and Iterators

/**
 *  C++20 ranges are a double-edged sword. They offer powerful abstractions,
 *  making complex tasks concise and expressive. But there's a price: they can
 *  be infamously hard to debug. If your compiler ever throws a range-related
 *  error, expect a wall of template gibberish that looks like a corrupted
 *  stack trace from a parallel universe.
 *
 *  Debugging them is a remarkable exercise in patience. It's highly recommended
 *  to use `-fconcepts-diagnostics-depth=10` to make meta-template errors more
 *  readable—assuming you're into that kind of suffering.
 *
 *  One especially maddening issue is handling non-homogeneous ranges, where
 *  the begin and end iterators are different types. The end iterator is often
 *  a "sentinel" like @b `std::default_sentinel_t`. But STL concepts don't always
 *  recognize such ranges as valid, causing these assertions to fail:
 *
 *      static_assert(std::ranges::view<prime_factors_view>);
 *      static_assert(std::ranges::input_range<prime_factors_view>);
 *
 *  As a result, operations like `std::views::join` may refuse to compile.
 */

#include <iterator> // `std::input_iterator_tag`
#include <ranges>   // `std::ranges`

class prime_factors_view : public std::ranges::view_interface<prime_factors_view> {
  private:
    std::uint64_t number_ = 0;

  public:
    prime_factors_view() noexcept {}
    explicit prime_factors_view(std::uint64_t n) noexcept : number_(n) {}

    class iterator {
        std::uint64_t number_ = 0;
        std::uint64_t factor_ = 0;

        inline void advance() noexcept {
            // Handle factor 2 separately
            if (factor_ == 2) {
                // Keep dividing by 2 as long as the number is even
                if ((number_ & 1) == 0) {
                    number_ >>= 1; // Equivalent to `number_ /= 2`
                    return;        // Still yielding factor = 2
                }
                else { factor_ = 3; } // No more factors of 2, move on to the next odd factor (3)
            }

            // Now handle only odd factors
            while (factor_ * factor_ <= number_) {
                if (number_ % factor_ == 0) {
                    number_ /= factor_;
                    return;
                }
                factor_ += 2; // Skip even numbers
            }

            // If we exit the loop, `number_` is either 1 or a prime:
            if (number_ > 1) { factor_ = number_, number_ = 0; } // The last entry
            else { factor_ = 0; }                                // Mark as end
        }

      public:
        using value_type = std::uint64_t;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;
        using iterator_concept = std::input_iterator_tag;

        iterator() = default;
        iterator(std::uint64_t n) noexcept : number_(n), factor_(2) { advance(); }
        std::uint64_t operator*() const noexcept { return factor_; }
        iterator &operator++() noexcept {
            advance();
            return *this;
        }
        iterator operator++(int) noexcept {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(iterator const &other) const noexcept { return factor_ == other.factor_; }
        bool operator!=(iterator const &other) const noexcept { return factor_ != other.factor_; }
    };

    iterator begin() const noexcept { return iterator(number_); }
    iterator end() const noexcept { return iterator(); }
};

static_assert(std::ranges::view<prime_factors_view>, "The range must model `std::ranges::view`");
static_assert(std::ranges::input_range<prime_factors_view>, "The range must model `std::ranges::input_range`");

/**
 *  @brief  Inverts the output of a boolean-returning function.
 *          Useful for search predicates.
 */
template <typename function_type_>
auto not_fn(function_type_ f) noexcept {
    return [f](auto &&...args) { return !f(std::forward<decltype(args)>(args)...); };
}

static void pipeline_cpp20_ranges(bm::State &state) {
    std::uint64_t sum = 0, count = 0;
    for (auto _ : state) {
        auto pipeline =                                                                    //
            std::views::iota(pipe_start, pipe_end + 1) |                                   //
            std::views::filter(not_fn(is_power_of_two)) |                                  //
            std::views::filter(not_fn(is_power_of_three)) |                                //
            std::views::transform([](std::uint64_t x) { return prime_factors_view(x); }) | //
            std::views::join;

        // Interestingly, STL still struggles with non-homogeneous ranges,
        // if the `begin` and `end` iterators are of different types:
        //
        //      std::uint64_t sum = std::accumulate(
        //          pipeline.begin(), pipeline.end(), std::uint64_t{0});
        //
        sum = 0, count = 0;
        for (std::uint64_t factor : pipeline) sum += factor, count++;
        bm::DoNotOptimize(sum);
    }
    state.counters["sum"] = sum;
    state.counters["count"] = count;
}

BENCHMARK(pipeline_cpp20_ranges);

/**
 *  The results for the input range [3, 49] are as follows:
 *
 *      - pipeline_cpp11_lambdas:      @b 295ns
 *      - pipeline_cpp11_stl:          @b 765ns
 *      - pipeline_cpp20_coroutines:   @b 712ns
 *      - pipeline_cpp20_ranges:       @b 216ns
 *
 *  Why is STL slower than C++11 lambdas? STL's `std::function` allocates memory!
 *  Why are coroutines slower than lambdas? Coroutines allocate state and have
 *  additional overhead for resuming and suspending. Those are fairly simple to grasp.
 *
 *  But why are ranges faster than lambdas? If that happens, the primary cause is
 *  the cost of moving data from registers to stack and back. Lambdas are closures,
 *  they capture their environment, and need to address it using the `rsp` register
 *  on x86. So if you see many `mov`-es and `rsp` arithmetic like:
 *
 *      mov     rax, QWORD PTR [rsp-16]
 *      xor     edx, edx
 *      add     rax, rcx
 *      mov     QWORD PTR [rsp-16], rax
 *      mov     rax, QWORD PTR [rsp-8]
 *      inc     rax
 *      mov     QWORD PTR [rsp-8], rax
 *      mov     rax, rsi
 *
 *  ... then you are probably looking at a lambda. Ranges, on the other hand, are
 *  lazy and don't need to capture anything. On the practical side, when implementing
 *  ranges, make sure to avoid branching even more than with regular code.
 *
 *  @see Standard Ranges by Eric Niebler: https://ericniebler.com/2018/12/05/standard-ranges/
 *  @see Should we stop writing functions? by Jonathan Müller
 *       https://www.think-cell.com/en/career/devblog/should-we-stop-writing-functions
 *  @see Lambdas, Nested Functions, and Blocks, oh my! by JeanHeyd Meneide:
 *       https://thephd.dev/lambdas-nested-functions-block-expressions-oh-my
 */

#pragma endregion // Ranges and Iterators

#pragma region Variants, Tuples, and State Machines

#include <tuple>   // `std::tuple`
#include <variant> // `std::variant`

#pragma endregion // Variants, Tuples, and State Machines

/**
 *  Now that we've explored how to write modern, performant C++ code,
 *  let's dive into how @b not to do it. Ironically, this style of programming
 *  is still taught in universities and used in legacy systems across the industry.
 *  If you see something like this in a codebase at a prospective job — run 🙂
 */

#pragma region Virtual Functions and Polymorphism

#include <memory> // `std::unique_ptr`

class base_virtual_class {
  public:
    virtual ~base_virtual_class() = default;
    virtual void process(std::vector<std::uint64_t> &data) const = 0;
};

class for_range_virtual : public base_virtual_class {
    std::uint64_t start_, end_;

  public:
    for_range_virtual(std::uint64_t start, std::uint64_t end) : start_(start), end_(end) {}
    void process(std::vector<std::uint64_t> &data) const override {
        data.clear();
        for (std::uint64_t value = start_; value <= end_; ++value) data.push_back(value);
    }
};

class filter_virtual : public base_virtual_class {
    bool (*predicate_)(std::uint64_t);

  public:
    filter_virtual(bool (*predicate)(std::uint64_t)) : predicate_(predicate) {}
    void process(std::vector<std::uint64_t> &data) const override {
        data.erase(std::remove_if(data.begin(), data.end(), predicate_), data.end());
    }
};

class prime_factors_virtual : public base_virtual_class {
  public:
    void process(std::vector<std::uint64_t> &data) const override {
        std::vector<std::uint64_t> expanded;
        for (auto input : data) prime_factors_cpp11(input, [&](std::uint64_t factor) { expanded.push_back(factor); });
        data.swap(expanded);
    }
};

class homogeneous_virtual_pipeline : public base_virtual_class {
    std::vector<std::unique_ptr<base_virtual_class>> stages_;

  public:
    void add_stage(std::unique_ptr<base_virtual_class> stage) { stages_.push_back(std::move(stage)); }
    void process(std::vector<std::uint64_t> &data) const override {
        for (auto const &stage : stages_) stage->process(data);
    }
};

static void pipeline_virtual_functions(bm::State &state) {
    homogeneous_virtual_pipeline pipeline;
    pipeline.add_stage(std::make_unique<for_range_virtual>(pipe_start, pipe_end));
    pipeline.add_stage(std::make_unique<filter_virtual>(is_power_of_two));
    pipeline.add_stage(std::make_unique<filter_virtual>(is_power_of_three));
    pipeline.add_stage(std::make_unique<prime_factors_virtual>());

    std::uint64_t sum = 0, count = 0;
    for (auto _ : state) {
        std::vector<std::uint64_t> data;
        pipeline.process(data);
        sum = std::accumulate(data.begin(), data.end(), std::uint64_t {0});
        count = data.size();
        bm::DoNotOptimize(sum);
    }
    state.counters["sum"] = sum;
    state.counters["count"] = count;
}

BENCHMARK(pipeline_virtual_functions);

/**
 *  Performance-wise, on this specific micro-example, the virtual functions
 *  are somewhere in the middle between C++20 ranges and C++11 STL solution.
 *
 *
 *  This code is a hazard for multiple reasons:
 *
 *  - @b Spaghetti_Code_Guaranteed: As the code evolves, additional abstraction
 *    layers will accumulate. Each layer will add new APIs and constraints until
 *    developers are forced to skip-connect grandkids to grandparents, creating
 *    a tangled mess.
 *
 *  - @b Dynamic_Allocations_Everywhere: Memory allocations occur at every stage,
 *    slowing execution and increasing fragmentation risks. In real-world scenarios,
 *    allocations frequently overlap and can overwhelm allocators, forcing them
 *    to fall back on expensive heuristics.
 *
 *  - @b Data_Packing_Overhead: This approach inevitably leads to passing data
 *    using `std::any`, arrays of strings, or even JSON objects. CPUs will spend
 *    more time chasing pointers and repacking values than performing meaningful
 *    computations.
 *
 *  - @b Unscalable_Design: Abstract Syntax Trees, interpreters, and similar
 *    constructs built this way become unsalvageable, with performance hitting a
 *    hard ceiling due to constant memory indirections and hidden costs.
 *
 *  This design is so systemically bad that after reading earlier sections, you
 *  probably don't need further convincing. We're not writing a grant proposal,
 *  so let's skip proving that bad ideas are indeed bad — and focus on the good ones.
 *
 *  Contributions are welcome — but not using `virtual` functions! 😉
 */

#pragma endregion // Virtual Functions and Polymorphism

#pragma endregion // - Abstractions

#pragma region - Structures, Tuples, ADTs, AOS, SOA

#pragma region Continuous Memory

/**
 *  It's a blessing if one can just use lazy processing, but realistically,
 *  every program needs memory and it's layout and used structures matter.
 *
 *  Algebraic Data Types (ADTs) are one of the most sought-after features in
 *  modern programming languages. They allow you to define complex data types
 *  by combining simpler ones.
 *
 *  In C, the only way to achieve that is by combining `stuct` and `union` types.
 *  In C++, the Standard Library provides `std::variant`, `std::tuple`, `std::pair`,
 *  `std::optional`, and many other types that can be used to create ADTs.
 *
 *  Common sense suggests that there should be no performance degradation from
 *  using the library types. Well, how hard can it be to implement a `std::pair`?
 *  StackOverflow has 5 answers to a similar question - all 5 are wrong.
 *
 *  @see StackOverflow discussion:
 *  https://stackoverflow.com/questions/3607352/struct-with-2-cells-vs-stdpair
 *
 *  Let's compare the performance of `std::pair` with a custom `pair_t` structure,
 *  made of the same two elements, also named boringly - `first` and `second`.
 */
#include <tuple> // `std::tuple`

static void packaging_custom_pairs(bm::State &state) {
    struct pair_t {
        int first;
        float second;
    };
    std::vector<pair_t> a, b;
    a.resize(1'000'000);
    for (auto _ : state) bm::DoNotOptimize(b = a);
}

BENCHMARK(packaging_custom_pairs)->MinTime(2);

static void packaging_stl_pair(bm::State &state) {
    std::vector<std::pair<int, float>> a, b;
    a.resize(1'000'000);
    for (auto _ : state) bm::DoNotOptimize(b = a);
}

BENCHMARK(packaging_stl_pair)->MinTime(2);

static void packaging_stl_tuple(bm::State &state) {
    std::vector<std::tuple<int, float>> a, b;
    a.resize(1'000'000);
    for (auto _ : state) bm::DoNotOptimize(b = a);
}

BENCHMARK(packaging_stl_tuple)->MinTime(2);

/**
 *  Over @b 600 microseconds for STL variants vs @b 488 microseconds for the baseline.
 *  The naive variant, avoiding STL, is faster by @b 20%.
 *
 *  Why? With the `std::pair` in its way `std::vector` can't realize that the actual
 *  contents are trivially copyable and can be moved around without any overhead.
 *
 *  @see Reddit discussion: https://www.reddit.com/r/cpp/comments/ar4ghs/stdpair_disappointing_performance/
 */
static_assert(!std::is_trivially_copyable_v<std::pair<int, float>>);
static_assert(!std::is_trivially_copyable_v<std::tuple<int, float>>);

#pragma endregion // Continuous Memory

#pragma region Trees and Graphs

#pragma endregion // Trees and Graphs

#pragma endregion // - Structures, Tuples, ADTs, AOS, SOA

#pragma region - Exceptions, Backups, Logging

/**
 *  In the real world, control-flow gets messy, as different methods will
 *  break in different places. Let's imagine a system, that:
 *
 *  - Reads an integer from a text file.
 *  - Increments it.
 *  - Saves it back to the text file.
 *
 *  As soon as we start dealing with "external devices", as opposed to the CPU itself,
 *  failures become regular. The file may not exist, the integer may not be a number,
 *  the file may be read-only, the disk may be full, the file may be locked, etc.
 */
#include <charconv>  // `std::from_chars`, `std::to_chars`
#include <stdexcept> // `std::runtime_error`, `std::out_of_range`

constexpr std::size_t fail_period_read_integer = 6;
constexpr std::size_t fail_period_convert_to_integer = 11;
constexpr std::size_t fail_period_next_string = 17;
constexpr std::size_t fail_period_write_back = 23;

static std::string read_integer_from_file_or_throw( //
    [[maybe_unused]] std::string const &filename, std::size_t iteration_index) noexcept(false) {
    if (iteration_index % fail_period_read_integer == 0) throw std::runtime_error("File read failed");
    if (iteration_index % fail_period_convert_to_integer == 0) return "abc";
    // Technically, the constructor may throw `std::bad_alloc` if the allocation fails,
    // but given the Small String Optimization, it shouldn't happen in practice.
    return "1";
}

static std::size_t string_to_integer_or_throw( //
    std::string const &value, [[maybe_unused]] std::size_t iteration_index) noexcept(false) {
    // The `std::stoull` function may throw:
    // - `std::invalid_argument` if no conversion could be performed.
    // - `std::out_of_range` if the converted value would fall out of the range of the
    //   result type or if the underlying function sets `errno` to `ERANGE`.
    // The cleaner modern way to implement this is using `std::from_chars` in C++17.
    std::size_t integral_value = 0;
    std::from_chars_result result = std::from_chars(value.data(), value.data() + value.size(), integral_value);
    if (result.ec != std::errc()) throw std::out_of_range("Conversion failed");
    return integral_value;
}

static std::string integer_to_next_string_or_throw(std::size_t value, std::size_t iteration_index) noexcept(false) {
    if (iteration_index % fail_period_next_string == 0) throw std::runtime_error("Increment failed");
    value++;
    constexpr std::size_t buffer_size = 10;
    char buffer[buffer_size] {};
    std::to_chars_result result = std::to_chars(buffer, buffer + buffer_size, value);
    if (result.ec != std::errc()) throw std::out_of_range("Conversion failed");
    return std::string(buffer, result.ptr - buffer);
}

static void write_to_file_or_throw( //
    [[maybe_unused]] std::string const &filename, [[maybe_unused]] std::string const &value,
    std::size_t iteration_index) noexcept(false) {
    if (iteration_index % fail_period_write_back == 0) throw std::runtime_error("File write failed");
}

static void exceptions_throw(bm::State &state) {
    std::string const filename = "test.txt";
    std::size_t iteration_index = 0;
    for (auto _ : state) {
        iteration_index++;
        try {
            std::string read_value = read_integer_from_file_or_throw(filename, iteration_index);
            std::size_t integer_value = string_to_integer_or_throw(read_value, iteration_index);
            std::string next_value = integer_to_next_string_or_throw(integer_value, iteration_index);
            write_to_file_or_throw(filename, next_value, iteration_index);
        }
        catch (std::exception const &) {
        }
    }
}

BENCHMARK(exceptions_throw)->MinTime(2);
BENCHMARK(exceptions_throw)->MinTime(2)->Threads(8);

/**
 *  Until C++23, we don't have a `std::expected` implementation,
 *  but it's easy to design one using `std::variant` and `std::error_code`.
 */
#include <system_error> // `std::error_code`
#include <variant>      // `std::variant`

using std::operator""s;

template <typename value_type_>
class expected {
    std::variant<value_type_, std::error_code> value_;

  public:
    expected(value_type_ const &value) noexcept : value_(value) {}
    expected(std::error_code const &error) noexcept : value_(error) {}

    explicit operator bool() const noexcept { return std::holds_alternative<value_type_>(value_); }
    value_type_ const &value() const noexcept { return std::get<value_type_>(value_); }
    std::error_code const &error() const noexcept { return std::get<std::error_code>(value_); }
    template <typename function_type_>
    auto and_then(function_type_ &&f) const noexcept {
        if (std::holds_alternative<value_type_>(value_)) return f(std::get<value_type_>(value_));
        return *this;
    }
};

static expected<std::string> read_integer_from_file_or_variants( //
    [[maybe_unused]] std::string const &filename, std::size_t iteration_index) noexcept {
    if (iteration_index % fail_period_read_integer == 0) return std::error_code {EIO, std::generic_category()};
    if (iteration_index % fail_period_convert_to_integer == 0) return "abc"s;
    // Technically, the constructor may throw `std::bad_alloc` if the allocation fails,
    // but given the Small String Optimization, it shouldn't happen in practice.
    return "1"s;
}

static expected<std::size_t> string_to_integer_or_variants( //
    std::string const &value, [[maybe_unused]] std::size_t iteration_index) noexcept {
    std::size_t integral_value = 0;
    std::from_chars_result result = std::from_chars(value.data(), value.data() + value.size(), integral_value);
    if (result.ec != std::errc()) return std::make_error_code(result.ec);
    return integral_value;
}

static expected<std::string> integer_to_next_string_or_variants(std::size_t value,
                                                                std::size_t iteration_index) noexcept {
    if (iteration_index % fail_period_next_string == 0) return std::error_code {EIO, std::generic_category()};
    value++;
    constexpr std::size_t buffer_size = 10;
    char buffer[buffer_size] {};
    std::to_chars_result result = std::to_chars(buffer, buffer + buffer_size, value);
    if (result.ec != std::errc()) return std::make_error_code(result.ec);
    return std::string(buffer, result.ptr - buffer);
}

static std::error_code write_to_file_or_variants( //
    [[maybe_unused]] std::string const &filename, [[maybe_unused]] std::string const &value,
    std::size_t iteration_index) noexcept {
    if (iteration_index % fail_period_write_back == 0) return std::error_code {EIO, std::generic_category()};
    return std::error_code {};
}

static void exceptions_variants(bm::State &state) {
    std::string const filename = "test.txt";
    std::size_t iteration_index = 0;
    for (auto _ : state) {
        iteration_index++;
        auto read_result = read_integer_from_file_or_variants(filename, iteration_index);
        if (!read_result) continue;
        auto integer_result = string_to_integer_or_variants(read_result.value(), iteration_index);
        if (!integer_result) continue;
        auto next_result = integer_to_next_string_or_variants(integer_result.value(), iteration_index);
        if (!next_result) continue;
        auto write_error = write_to_file_or_variants(filename, next_result.value(), iteration_index);
        bm::DoNotOptimize(write_error);
    }
}

BENCHMARK(exceptions_variants)->MinTime(2);
BENCHMARK(exceptions_variants)->MinTime(2)->Threads(8);

/**
 *  As practice shows, STL is almost never the performance-oriented choice!
 *  It's often good enough to get started, but every major player reimplements
 *  STL-like primitives for its own needs. In some cases, no primitives are
 *  needed at all, as the language itself provides the necessary tools.
 *
 *  Instead of using the heavy `std::variant`, that will contain an integer
 *  to distinguish between the value and the error, plus the heavy error object
 *  itself, we can use a simple `struct` with a status code.
 */
enum class status : unsigned int {
    success,
    read_failed,
    convert_failed,
    increment_failed,
    write_failed,
};

template <typename value_type_>
struct result {
    value_type_ value;
    status status_code;
};

static result<std::string> read_integer_from_file_with_status( //
    [[maybe_unused]] std::string const &filename, std::size_t iteration_index) noexcept {
    if (iteration_index % fail_period_read_integer == 0) return {{}, status::read_failed};
    if (iteration_index % fail_period_convert_to_integer == 0) return {"abc"s, status::success};
    // Technically, the constructor may throw `std::bad_alloc` if the allocation fails,
    // but given the Small String Optimization, it shouldn't happen in practice.
    return {"1"s, status::success};
}

static result<std::size_t> string_to_integer_with_status( //
    std::string const &value, [[maybe_unused]] std::size_t iteration_index) noexcept {
    std::size_t integral_value = 0;
    std::from_chars_result result = std::from_chars(value.data(), value.data() + value.size(), integral_value);
    if (result.ec != std::errc()) return {0, status::convert_failed};
    return {integral_value, status::success};
}

static result<std::string> integer_to_next_string_with_status(std::size_t value, std::size_t iteration_index) noexcept {
    if (iteration_index % fail_period_next_string == 0) return {{}, status::increment_failed};
    value++;
    constexpr std::size_t buffer_size = 10;
    char buffer[buffer_size] {};
    std::to_chars_result result = std::to_chars(buffer, buffer + buffer_size, value);
    if (result.ec != std::errc()) return {{}, status::increment_failed};
    return {std::string(buffer, result.ptr - buffer), status::success};
}

static status write_to_file_with_status( //
    [[maybe_unused]] std::string const &filename, [[maybe_unused]] std::string const &value,
    std::size_t iteration_index) noexcept {
    if (iteration_index % fail_period_write_back == 0) return status::write_failed;
    return status::success;
}

static void exceptions_with_status(bm::State &state) {
    std::string const filename = "test.txt";
    std::size_t iteration_index = 0;
    for (auto _ : state) {
        iteration_index++;
        auto [read_result, read_status] = read_integer_from_file_with_status(filename, iteration_index);
        if (read_status != status::success) continue; // In large programs, consider adding `[[unlikely]]`
        auto [integer_result, integer_status] = string_to_integer_with_status(read_result, iteration_index);
        if (integer_status != status::success) continue; // In large programs, consider adding `[[unlikely]]`
        auto [next_result, next_status] = integer_to_next_string_with_status(integer_result, iteration_index);
        if (next_status != status::success) continue; // In large programs, consider adding `[[unlikely]]`
        auto write_status = write_to_file_with_status(filename, next_result, iteration_index);
        bm::DoNotOptimize(write_status);
    }
}

BENCHMARK(exceptions_with_status)->MinTime(2);
BENCHMARK(exceptions_with_status)->MinTime(2)->Threads(8);

/**
 *  - Throwing an exception: @b 268 ns single-threaded, @b 815 ns multi-threaded.
 *  - Returning an error code: @b 7 ns single-threaded, @b 24 ns multi-threaded.
 *
 *  Similarly, logging can be done in different ways. Nice logs may look like this:
 *
 *      [time] | [filename]:[source-line] <[code]> "[message]"
 *
 *  The time should be in a ISO 8601 format, and a line for `EPERM` may look like this:
 *
 *      YYYY-MM-DDTHH:MM:SS.mmm | less_slow.cpp:2053 <1> "Operation not permitted"
 *
 *  C++ has one of the best standard libraries for time - `std::chrono`, and one of
 *  the most powerful formatting libraries - `std::format`, derived from `fmt::`.
 *  Both are more capable than most users realize!
 */
#include <chrono>          // `std::chrono`
#include <source_location> // `std::source_location`
#include <string_view>     // `std::string_view`

using std::string_view_literals::operator""sv;

static void log_with_printf(               //
    char *buffer, std::size_t buffer_size, //
    std::source_location const &location, int code, std::string_view message) noexcept {

    auto now = std::chrono::high_resolution_clock::now();
    auto time_since_epoch = now.time_since_epoch();

    // Extract seconds and milliseconds
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch) - seconds;

    // Format as ISO 8601: YYYY-MM-DDTHH:MM:SS.mmm
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto tm = std::gmtime(&time_t_now); // UTC only, no local timezone dependency

    std::snprintf( //
        buffer, buffer_size,
        "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ | "                                     // time format
        "%s:%d <%03d> "                                                              // location and code format
        "\"%.*s\"\n",                                                                // message format
        tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,                             // date
        tm->tm_hour, tm->tm_min, tm->tm_sec, static_cast<int>(milliseconds.count()), // time
        location.file_name(), location.line(), code,                                 // location and code
        static_cast<int>(message.size()), message.data()                             // message of known length
    );
}

static void logging_printf(bm::State &state) {
    struct {
        int code;
        std::string_view message;
    } errors[3] = {
        {1, "Operation not permitted"sv},
        {12, "Cannot allocate memory"sv},
        {113, "No route to host"sv},
    };
    char buffer[1024];
    std::size_t iteration_index = 0;
    for (auto _ : state) {
        log_with_printf(                     //
            buffer, sizeof(buffer),          //
            std::source_location::current(), //
            errors[iteration_index % 3].code, errors[iteration_index % 3].message);
        iteration_index++;
    }
}

#include <format> // `std::format_to_n`

static void log_with_fmt(                  //
    char *buffer, std::size_t buffer_size, //
    std::source_location const &location, int code, std::string_view message) noexcept {

    auto now = std::chrono::high_resolution_clock::now();
    auto time_since_epoch = now.time_since_epoch();

    // Extract seconds and milliseconds
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch) - seconds;

    // ISO 8601 defines the format as: YYYY-MM-DDTHH:MM:SS.mmm
    // `%F` unpacks to `%Y-%m-%d`, implementing the "YYYY-MM-DD" part
    // `%T` would expand to `%H:%M:%S`, implementing the "HH:MM:SS" part
    // To learn more about syntax, read: https://fmt.dev/11.0/syntax/
    std::format_to_n( //
        buffer, buffer_size,
        "{:%FT%R}:{:0>2}.{:0>3}Z | "                     // time format
        "{}:{} <{:0>3}> "                                // location and code format
        "\"{}\"\n",                                      // message format
        now, static_cast<unsigned int>(seconds.count()), // date and time
        static_cast<unsigned int>(milliseconds.count()), // milliseconds
        location.file_name(), location.line(), code,     // location and code
        message                                          // message of known length
    );
}

static void logging_fmt(bm::State &state) {
    struct {
        int code;
        std::string_view message;
    } errors[3] = {
        {1, "Operation not permitted"sv},
        {12, "Cannot allocate memory"sv},
        {113, "No route to host"sv},
    };
    char buffer[1024];
    std::size_t iteration_index = 0;
    for (auto _ : state) {
        log_with_fmt(                        //
            buffer, sizeof(buffer),          //
            std::source_location::current(), //
            errors[iteration_index % 3].code, errors[iteration_index % 3].message);
        iteration_index++;
    }
}

BENCHMARK(logging_printf)->MinTime(2);
BENCHMARK(logging_fmt)->MinTime(2);
BENCHMARK(logging_printf)->MinTime(2)->Threads(8);
BENCHMARK(logging_fmt)->MinTime(2)->Threads(8);

#pragma endregion // - Exceptions, Backups, Logging

/**
 *  The default variant is to invoke the `BENCHMARK_MAIN()` macro.
 *  Alternatively, we can unpack it, if we want to augment the main function
 *  with more system logging logic or pre-process some datasets before running.
 */

int main(int argc, char **argv) {

    // Let's log the CPU specs:
    std::size_t const cache_line_width = fetch_cache_line_width();
    std::printf("Cache line width: %zu bytes\n", cache_line_width);

    // Make sure the defaults are set correctly:
    char arg0_default[] = "benchmark";
    char *args_default = arg0_default;
    if (!argv) argc = 1, argv = &args_default;
    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv)) return 1;
    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}
