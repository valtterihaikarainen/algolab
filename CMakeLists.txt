cmake_minimum_required(VERSION 3.20)
project(vadugrad CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_COVERAGE "Build with gcov coverage flags (GCC or Clang)" OFF)

if(ENABLE_COVERAGE)
  add_compile_options(-Wall -Wextra -O0 -g)
else()
  add_compile_options(-O2 -Wall -Wextra)
endif()

add_library(vadugrad src/tensor.cpp src/dense_linear.cpp)
target_include_directories(vadugrad PUBLIC include)

if(ENABLE_COVERAGE)
  target_compile_options(vadugrad PRIVATE --coverage)
  target_link_options(vadugrad PUBLIC --coverage)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(vadugrad_cli src/main.cpp)
target_link_libraries(vadugrad_cli PRIVATE vadugrad)
set_target_properties(vadugrad_cli PROPERTIES OUTPUT_NAME vadugrad)

add_executable(tests tests/test_tensor.cpp tests/test_dense_linear.cpp)
target_link_libraries(tests PRIVATE vadugrad GTest::gtest_main)

if(ENABLE_COVERAGE)
  target_compile_options(tests PRIVATE --coverage)
  target_link_options(tests PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(tests)
