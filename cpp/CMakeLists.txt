cmake_minimum_required(VERSION 3.22)
project(probos_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
)

add_library(probos_distributions STATIC
    src/distributions/normal.cpp
)
target_include_directories(probos_distributions
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(probos_main src/main.cpp)
target_link_libraries(probos_main PRIVATE probos_distributions)

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
