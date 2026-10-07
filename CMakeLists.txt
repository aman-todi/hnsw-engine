cmake_minimum_required(VERSION 3.20)
project(hnsw_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HNSW_BUILD_TESTS "Build GoogleTest suite" ON)
option(HNSW_BUILD_BENCH "Build benchmark harness and microbenchmarks" ON)
option(HNSW_BUILD_PYTHON "Build the pybind11 extension module" OFF)
option(HNSW_NATIVE "Compile everything with -march=native (benchmark builds only)" OFF)
option(HNSW_WERROR "Treat warnings as errors" ON)
set(HNSW_SANITIZE "" CACHE STRING "Sanitizers to enable: '', 'address,undefined' or 'thread'")

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Warnings / flags
# ---------------------------------------------------------------------------
add_library(hnsw_warnings INTERFACE)
target_compile_options(hnsw_warnings INTERFACE -Wall -Wextra -Wpedantic)
if(HNSW_WERROR)
  target_compile_options(hnsw_warnings INTERFACE -Werror)
endif()

if(HNSW_SANITIZE)
  add_compile_options(-fsanitize=${HNSW_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${HNSW_SANITIZE})
endif()

if(HNSW_NATIVE)
  add_compile_options(-march=native)
endif()

# ---------------------------------------------------------------------------
# Kernels: each ISA lives in its own translation unit compiled with only its
# own ISA flags; dispatch.cpp picks one at startup.
# ---------------------------------------------------------------------------
set(HNSW_ARCH_X86 OFF)
set(HNSW_ARCH_ARM64 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
  set(HNSW_ARCH_X86 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(HNSW_ARCH_ARM64 ON)
endif()
# Apple universal / cross builds report the target via CMAKE_OSX_ARCHITECTURES.
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(HNSW_ARCH_ARM64 ON)
    set(HNSW_ARCH_X86 OFF)
  endif()
endif()

set(HNSW_SOURCES
  src/index.cpp
  src/search_layer.cpp
  src/brute_force.cpp
  src/kernels_scalar.cpp
  src/dispatch.cpp
  src/io.cpp
  src/persist.cpp
  src/threadpool.cpp
  src/filter.cpp)

add_library(hnsw_engine STATIC ${HNSW_SOURCES})
add_library(hnsw::engine ALIAS hnsw_engine)
target_include_directories(hnsw_engine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hnsw_engine PUBLIC Threads::Threads PRIVATE hnsw_warnings)
set_target_properties(hnsw_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(HNSW_ARCH_X86)
  target_sources(hnsw_engine PRIVATE src/kernels_avx2.cpp src/kernels_avx512.cpp)
  set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  set_source_files_properties(src/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma")
  target_compile_definitions(hnsw_engine PRIVATE HNSW_HAVE_X86_KERNELS=1)
elseif(HNSW_ARCH_ARM64)
  target_sources(hnsw_engine PRIVATE src/kernels_neon.cpp)
  target_compile_definitions(hnsw_engine PRIVATE HNSW_HAVE_NEON_KERNELS=1)
endif()

# ---------------------------------------------------------------------------
# Dependencies via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

if(HNSW_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

if(HNSW_BUILD_BENCH)
  FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW TRUE)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)
  add_subdirectory(bench)
endif()

if(HNSW_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 2.11 CONFIG REQUIRED)
  pybind11_add_module(_core MODULE python/bindings.cpp)
  target_link_libraries(_core PRIVATE hnsw::engine hnsw_warnings)
  install(TARGETS _core LIBRARY DESTINATION hnsw_engine)
endif()
