# Let's used CMake 3.16+ for native sanitizers support
cmake_minimum_required(VERSION 3.16 FATAL_ERROR) 

# ------------------------------------------------------------------------------
# Project Setup
# ------------------------------------------------------------------------------
project(less_slow
  VERSION 0.1.0
  LANGUAGES C CXX
  DESCRIPTION "Learning how to write Less Slow code, from numerical micro-kernels and SIMD to coroutines, ranges, and polymorphic state machines"
  HOMEPAGE_URL "https://github.com/ashvardanian/less_slow.cpp")

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Some extra logging for the user:
message(STATUS "----------------------------------------")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "----------------------------------------")

# Default to Release if no build type is set:
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set(FETCHCONTENT_QUIET OFF)
include(FetchContent)

# GTest (required by Google Benchmark)
FetchContent_Declare(
  GoogleTest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(GoogleTest)

# Google Benchmark
FetchContent_Declare(
  GoogleBenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.1
)

# Suppress building tests/docs/etc. for faster builds:
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_USE_BUNDLED_GTEST ON CACHE BOOL "" FORCE)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(BENCHMARK_ENABLE_LIBPFM OFF CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(GoogleBenchmark)

# Remove Google Benchmark's built-in debug warning in Release mode:
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_definitions(benchmark PRIVATE NDEBUG)
endif()

# Intel TBB for parallel STL algorithms
# https://github.com/oneapi-src/oneTBB/tree/onetbb_2021
FetchContent_Declare(
  IntelTBB
  GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
  GIT_TAG master
)

# Suppress TBB’s own tests:
set(TBB_TEST OFF CACHE BOOL "Do not build TBB tests" FORCE)
FetchContent_MakeAvailable(IntelTBB)

# ------------------------------------------------------------------------------
# TBB fix for -Wstringop-overflow warnings treated as errors
# ------------------------------------------------------------------------------
# The TBB library target is typically called "tbb". We can explicitly disable
# the `stringop-overflow` warning for TBB only:
if(TARGET tbb)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tbb PRIVATE -Wno-stringop-overflow)
  endif()
endif()

# FMT for logging, as `std::format` has limited functionality
FetchContent_Declare(
  VictorZverovichFMT
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.1.0
)
FetchContent_MakeAvailable(VictorZverovichFMT)

# Eric Niebler's `range-v3`, as `std::ranges` have less functionality
FetchContent_Declare(
  EricNieblerRangeV3
  GIT_REPOSITORY https://github.com/ericniebler/range-v3
  GIT_TAG master
)
FetchContent_MakeAvailable(EricNieblerRangeV3)

# Andreas Buhr's up-to-date fork of Lewis Baker's `cppcoro`
FetchContent_Declare(
  AndreasBuhrCppCoro
  GIT_REPOSITORY https://github.com/andreasbuhr/cppcoro
  GIT_TAG main
)
FetchContent_MakeAvailable(AndreasBuhrCppCoro)

# Meta's LibUnifEx - unified executors for C++
set(CXX_COROUTINES_HAVE_COROUTINES 0)
FetchContent_Declare(
  MetaLibUnifEx
  GIT_REPOSITORY https://github.com/facebookexperimental/libunifex.git
  GIT_TAG main
)
FetchContent_MakeAvailable(MetaLibUnifEx)

# StringZilla to accelerate and extend `std::string_view` functionality
FetchContent_Declare(
  AshVardanianStringZilla
  GIT_REPOSITORY https://github.com/ashvardanian/stringzilla
  GIT_TAG main
)
FetchContent_MakeAvailable(AshVardanianStringZilla)

# Hana Dusikova's CTRE for compile-time regex, replacing `std::regex`
FetchContent_Declare(
  HanaDusikovaCTRE
  GIT_REPOSITORY https://github.com/hanickadot/compile-time-regular-expressions
  GIT_TAG main
)
FetchContent_MakeAvailable(HanaDusikovaCTRE)

# Abseil for flat associative containers, before they arrive in C++26
FetchContent_Declare(
  GoogleAbseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20240722.0 # LTS version
)
FetchContent_MakeAvailable(GoogleAbseil)

# Niels Lohmann's JSON for modern JSON parsing
FetchContent_Declare(
  NielsLohmannJSON
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(NielsLohmannJSON)

# Yaoyuan Guo YYJSON for more flexible & performant C-style parsing
FetchContent_Declare(
  YaoyuanGuoYYJSON
  GIT_REPOSITORY https://github.com/ibireme/yyjson.git
  GIT_TAG 0.10.0
)
FetchContent_MakeAvailable(YaoyuanGuoYYJSON)

# ------------------------------------------------------------------------------
# Main Executable
# ------------------------------------------------------------------------------
add_executable(less_slow less_slow.cpp)
set_target_properties(less_slow PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------------------
# Compiler Flags / Options
# ------------------------------------------------------------------------------
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # Apple Clang doesn't support -march=native
  target_compile_options(less_slow PRIVATE -march=native)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # A few useful options for GCC:
  target_compile_options(less_slow PRIVATE
    -Wno-error
    -Wfatal-errors # Stop on first error
    -fconcepts-diagnostics-depth=10 # Needed to debug concepts
    -fopenmp # OpenMP support, also requires linking
  )
else()
  # For other compilers (Clang, MSVC, Intel, etc.)
  target_compile_options(less_slow PRIVATE
    -Wno-deprecated-pragma
  )
endif()

# Release vs. Debug flags via generator expressions
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(less_slow PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-Wno-unused-but-set-variable>
    $<$<CONFIG:Release>:-falign-functions=32>
    $<$<CONFIG:Debug>:-g>
  )
  set_property(TARGET less_slow PROPERTY SANITIZE_ADDRESS TRUE)
  set_property(TARGET less_slow PROPERTY SANITIZE_UNDEFINED TRUE)
endif()

# ------------------------------------------------------------------------------
# Link Libraries
# ------------------------------------------------------------------------------
target_link_libraries(less_slow
  PRIVATE
    Threads::Threads
    benchmark
    fmt::fmt
    range-v3
    cppcoro
    unifex
    stringzilla
    yyjson
    ctre
    # There is no `absl` shortcut:
    # https://github.com/abseil/abseil-cpp/blob/master/CMake/README.md#available-abseil-cmake-public-targets
    absl::flat_hash_map
    nlohmann_json::nlohmann_json
    $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:TBB::tbb>
    $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:OpenMP::OpenMP_CXX>
)