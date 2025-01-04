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

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Debugging concepts requires a higher depth
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts-diagnostics-depth=10")
  # Compiling TBB
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
  # Force clean code, but it's hard to support across all OSes and compilers!
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-pragma")
endif()

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
  VictorZverovichFMT
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.1.0)
FetchContent_MakeAvailable(VictorZverovichFMT)


# Fetch GTest, as its needed for GBenchmark https://stackoverflow.com/a/60417587
FetchContent_Declare(
  GoogleTest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2)
FetchContent_MakeAvailable(GoogleTest)

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
  GoogleBenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.1)
FetchContent_MakeAvailable(GoogleBenchmark)

# Remove the Google Benchmark's "built in debug warning".
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_definitions(benchmark PRIVATE NDEBUG)
endif()

# We need TBB for Parallel CPU Algorithms in GCC.
# https://github.com/oneapi-src/oneTBB/blob/onetbb_2021/cmake/README.md
FetchContent_Declare(
  IntelTBB
  GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
  GIT_TAG master)

# Set the TBB_TEST option to OFF before making it available
set(TBB_TEST
    OFF
    CACHE BOOL "Do not build TBB tests" FORCE)
FetchContent_MakeAvailable(IntelTBB)

# Fetch "range-v3" for lazy views, as `std::ranges` is not feature-complete yet
FetchContent_Declare(
  EricNieblerRangeV3
  GIT_REPOSITORY https://github.com/ericniebler/range-v3
  GIT_TAG master)
FetchContent_MakeAvailable(EricNieblerRangeV3)

# Fetch "stringzilla" to extend `std::string_view` functionality
FetchContent_Declare(
  AshVardanianStringZilla
  GIT_REPOSITORY https://github.com/ashvardanian/stringzilla
  GIT_TAG main)
FetchContent_MakeAvailable(AshVardanianStringZilla)

# Fetch "ctre" to extend `std::regex` functionality
FetchContent_Declare(
  HanaDusikovaCTRE
  GIT_REPOSITORY https://github.com/hanickadot/compile-time-regular-expressions
  GIT_TAG main)
FetchContent_MakeAvailable(HanaDusikovaCTRE)

# Add dependencies
find_package(Threads REQUIRED)
find_package(OpenMP)

add_executable(less_slow less_slow.cpp)
set_target_properties(less_slow PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(less_slow pthread benchmark::benchmark fmt::fmt)
target_link_libraries(less_slow range-v3) 
target_link_libraries(less_slow ctre) 
target_link_libraries(less_slow stringzilla) 

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(less_slow TBB::tbb)
endif()
