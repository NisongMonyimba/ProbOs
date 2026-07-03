cmake_minimum_required(VERSION 3.22)
project(probos_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------
add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_options(
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
    $<$<CONFIG:Debug>:-g>
)
add_link_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)
add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<CONFIG:Release>:-march=native>
)

# -----------------------------------------------------------------------
# OpenMP
# -----------------------------------------------------------------------
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found -- parallel MC enabled")
else()
    message(WARNING "OpenMP not found -- building serial fallback")
endif()

# -----------------------------------------------------------------------
# Distribution library (Week 1)
# -----------------------------------------------------------------------
add_library(probos_distributions STATIC
    src/distributions/normal.cpp
)
target_include_directories(probos_distributions
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# -----------------------------------------------------------------------
# MC OMP library (Week 4)
# -----------------------------------------------------------------------
add_library(probos_mc_omp STATIC
    src/kernel/monte_carlo_omp.cpp
)
target_include_directories(probos_mc_omp
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(probos_mc_omp PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(probos_mc_omp PUBLIC PROBOS_OPENMP_ENABLED)
endif()

# -----------------------------------------------------------------------
# Main demo (Week 1)
# -----------------------------------------------------------------------
add_executable(probos_main src/main.cpp)
target_link_libraries(probos_main PRIVATE probos_distributions)

# -----------------------------------------------------------------------
# OpenMP benchmark (Week 4)
# -----------------------------------------------------------------------
add_executable(benchmark_omp src/kernel/benchmark_omp.cpp)
target_link_libraries(benchmark_omp PRIVATE probos_mc_omp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(benchmark_omp PRIVATE OpenMP::OpenMP_CXX)
endif()

# -----------------------------------------------------------------------
# pybind11 bindings (Month 2 Week 6)
# -----------------------------------------------------------------------
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    message(STATUS "pybind11 found -- building probos_cpp Python extension")
    pybind11_add_module(probos_cpp bindings/probos_bindings.cpp)
    target_link_libraries(probos_cpp PRIVATE probos_mc_omp)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(probos_cpp PRIVATE OpenMP::OpenMP_CXX)
    endif()
else()
    message(WARNING "pybind11 not found -- skipping probos_cpp extension. "
                     "Install with: pip install pybind11")
endif()

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "Google Test found -- building C++ tests")
    find_package(Threads REQUIRED)
    add_executable(test_normal tests/test_normal.cpp)
    target_link_libraries(test_normal PRIVATE
        probos_distributions GTest::gtest_main Threads::Threads
    )
    enable_testing()
    add_test(NAME test_normal COMMAND test_normal)
else()
    message(WARNING "Google Test not found. Install: sudo apt-get install -y libgtest-dev")
endif()
