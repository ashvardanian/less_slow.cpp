# Let's used CMake 3.16+ for native sanitizers support
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# ------------------------------------------------------------------------------
# Project Setup
# ------------------------------------------------------------------------------
project(less_slow
  VERSION 0.5.3
  LANGUAGES C CXX ASM
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
include(FetchContent) # For CMake-friendly dependencies

# Fetch and build OpenBLAS
FetchContent_Declare(
  OpenBLAS
  GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
  GIT_TAG v0.3.29
)

# Set OpenBLAS build options
set(NOFORTRAN ON CACHE BOOL "Disable Fortran" FORCE)
set(BUILD_WITHOUT_LAPACK OFF CACHE BOOL "Build without LAPACK" FORCE)
set(USE_THREAD ON CACHE BOOL "Use threading" FORCE)

# Make OpenBLAS available
FetchContent_MakeAvailable(OpenBLAS)

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
  GIT_TAG master # The tags are not up-to-date
  GIT_SHALLOW TRUE
)

# Suppress TBB's own tests:
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
  GIT_TAG 11.1.2
)
FetchContent_MakeAvailable(VictorZverovichFMT)

# Eric Niebler's `range-v3`, as `std::ranges` have less functionality
FetchContent_Declare(
  EricNieblerRangeV3
  GIT_REPOSITORY https://github.com/ericniebler/range-v3
  GIT_TAG master # The tags are not up-to-date
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(EricNieblerRangeV3)

# Andreas Buhr's up-to-date fork of Lewis Baker's `cppcoro`
FetchContent_Declare(
  AndreasBuhrCppCoro
  GIT_REPOSITORY https://github.com/andreasbuhr/cppcoro
  GIT_TAG main # No tags on this fork
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(AndreasBuhrCppCoro)

# Meta's LibUnifEx - unified executors for C++
set(CXX_COROUTINES_HAVE_COROUTINES 0)
FetchContent_Declare(
  MetaLibUnifEx
  GIT_REPOSITORY https://github.com/facebookexperimental/libunifex.git
  GIT_TAG main # The tags are not up-to-date
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(MetaLibUnifEx)

# StringZilla to accelerate and extend `std::string_view` functionality
FetchContent_Declare(
  AshVardanianStringZilla
  GIT_REPOSITORY https://github.com/ashvardanian/stringzilla
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(AshVardanianStringZilla)

# Hana Dusikova's CTRE for compile-time regex, replacing `std::regex`
FetchContent_Declare(
  HanaDusikovaCTRE
  GIT_REPOSITORY https://github.com/hanickadot/compile-time-regular-expressions
  GIT_TAG main # The tags are not up-to-date
  GIT_SHALLOW TRUE
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

# Yaoyuan Guo's YYJSON for more flexible & performant C-style parsing
FetchContent_Declare(
  YaoyuanGuoYYJSON
  GIT_REPOSITORY https://github.com/ibireme/yyjson.git
  GIT_TAG 0.10.0
)
FetchContent_MakeAvailable(YaoyuanGuoYYJSON)

# Chris Karloff's ASIO standalone, avoiding Boost... integration is a bit tricky:
# https://github.com/cpm-cmake/CPM.cmake/blob/master/examples/asio-standalone/CMakeLists.txt
FetchContent_Declare(
  ChrisKohlhoffASIO
  GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
  GIT_TAG asio-1-32-0 # Latest docs are for 1.30.2 😭
)
FetchContent_MakeAvailable(ChrisKohlhoffASIO)
add_library(asio INTERFACE)
target_include_directories(asio SYSTEM INTERFACE ${chriskohlhoffasio_SOURCE_DIR}/asio/include)
target_compile_definitions(asio INTERFACE ASIO_STANDALONE ASIO_NO_DEPRECATED)

# Jens Axboe's liburing to simplify I/O operations on Linux
# https://github.com/axboe/liburing/pull/438
# https://github.com/axboe/liburing/issues/946
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBURING REQUIRED liburing)

  if(LIBURING_FOUND)
    message(STATUS "liburing found: version=${LIBURING_VERSION}")
    message(STATUS "Include dirs: ${LIBURING_INCLUDE_DIRS}")
    message(STATUS "Libraries:   ${LIBURING_LIBRARIES}")
  else()
    message(FATAL_ERROR "liburing not found. Please install liburing-dev.")
  endif()
endif()

# Eigen is one of the few libraries not using GitHub
FetchContent_Declare(
  LibEigenEigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
)
FetchContent_MakeAvailable(LibEigenEigen)

# ------------------------------------------------------------------------------
# Main Executable
# ------------------------------------------------------------------------------
add_executable(less_slow less_slow.cpp)
set_target_properties(less_slow PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Conditionally add the assembly file(s)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64|x64")
  set_source_files_properties(less_slow_amd64.S PROPERTIES LANGUAGE ASM)
  target_sources(less_slow PRIVATE less_slow_amd64.S)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
  set_source_files_properties(less_slow_aarch64.S PROPERTIES LANGUAGE ASM)
  target_sources(less_slow PRIVATE less_slow_aarch64.S)
endif()

# ------------------------------------------------------------------------------
# Compiler Flags / Options
# ------------------------------------------------------------------------------
# Check for compiler support of `-march=native`
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  target_compile_options(less_slow PRIVATE -xHost)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang" OR CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
# Apple's Clang and MSVC can't auto-detect the highest CPU features
else()
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
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_options(less_slow PRIVATE
    /MP # Build with multiple processes; equivalent to `make -j` except it spans across all cores by default
    /wd4068 # Disable the "unknown pragma" warning, as StringZilla uses many GCC and Clang pragmas
    /Zc:__cplusplus # Make `__cplusplus` macro actually match used standard
    /Zc:preprocessor # Use conformant preprocessor
  )

else()
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
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_options(less_slow PRIVATE
    $<$<CONFIG:Release>:/O2>
    $<$<CONFIG:Release>:/Ob2>
    $<$<CONFIG:Release>:/Oi>
    $<$<CONFIG:Release>:/Ot>
    $<$<CONFIG:Release>:/GL>
  )
  target_link_options(less_slow PRIVATE
    $<$<CONFIG:Release>:/LTCG:incremental>
  )
endif()

# ------------------------------------------------------------------------------
# Link Libraries
# ------------------------------------------------------------------------------
# Add OpenBLAS include directory manually
if(openblas_POPULATED)
  target_include_directories(less_slow PRIVATE ${openblas_SOURCE_DIR})

  # For config.h
  target_include_directories(less_slow PRIVATE ${openblas_BINARY_DIR})
endif()

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
  asio
  openblas

  # There is no `absl` shortcut:
  # https://github.com/abseil/abseil-cpp/blob/master/CMake/README.md#available-abseil-cmake-public-targets
  absl::flat_hash_map
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(less_slow PRIVATE TBB::tbb)
  target_link_libraries(less_slow PRIVATE OpenMP::OpenMP_CXX)

  # target_include_directories(less_slow PRIVATE ${LIBURING_INCLUDE_DIRS})
  target_link_libraries(less_slow PRIVATE ${LIBURING_LIBRARIES})
endif()
