cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Define the supported languages
project(less_slow LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Make Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Apple Clang doesn't support `-march=native` flag.
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Debugging concepts requires a higher depth
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts-diagnostics-depth=10")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra") # Forces clean code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow") # Compiling TBB

# Inherit the release flags from the default ones
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}")

# Make it fast by default :)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Less warnings
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -Wno-unused-but-set-variable")

# Stable results between recompilations
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -falign-functions=32")

# Inherit the debug flags from the default ones
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Add sanitizers
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")

# Load the dependencies only after we've configured the compilation settings
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)

# Fetch FMT for logging, as `std::format` is not feature-complete yet
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.1.0)
FetchContent_MakeAvailable(fmt)

# Fetch GTest, as its needed for GBenchmark https://stackoverflow.com/a/60417587
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2)
FetchContent_MakeAvailable(googletest)

# Suppress building tests of Google Benchmark to accelerate builds.
set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL
    OFF
    CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_DOXYGEN
    OFF
    CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS
    OFF
    CACHE BOOL "" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES
    ON
    CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(BENCHMARK_USE_BUNDLED_GTEST
    ON
    CACHE BOOL "" FORCE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(BENCHMARK_ENABLE_LIBPFM
      OFF
      CACHE BOOL "" FORCE)
endif()

# Fetch GBenchmark and suppress internal tests.
# https://github.com/google/benchmark/blob/main/docs/user_guide.md#using-register-benchmark
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.1)
FetchContent_MakeAvailable(googlebenchmark)

# Remove the Google Benchmark's "built in debug warning".
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_definitions(benchmark PRIVATE NDEBUG)
endif()

# We need TBB for Parallel CPU Algorithms in GCC.
# https://github.com/oneapi-src/oneTBB/blob/onetbb_2021/cmake/README.md
FetchContent_Declare(
  TBB
  GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
  GIT_TAG master)

# Set the TBB_TEST option to OFF before making it available
set(TBB_TEST
    OFF
    CACHE BOOL "Do not build TBB tests" FORCE)
FetchContent_MakeAvailable(TBB)

# Add dependencies
find_package(Threads REQUIRED)
find_package(OpenMP)

add_executable(less_slow less_slow.cpp)
set_target_properties(less_slow PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(less_slow pthread benchmark fmt)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(less_slow TBB::tbb)
endif()
