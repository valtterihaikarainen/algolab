cmake_minimum_required(VERSION 3.20)
project(vadugrad CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_COVERAGE "Build with gcov coverage flags (GCC or Clang)" OFF)
option(BUILD_TESTING "Build unit tests (downloads GoogleTest; needs network on first configure)" ON)

# Do not force -O2/-O3 here: CMAKE_BUILD_TYPE already sets Debug (-O0 -g) vs Release.
if(ENABLE_COVERAGE)
  add_compile_options(-Wall -Wextra -O0 -g)
else()
  add_compile_options(-Wall -Wextra)
endif()

add_library(vadugrad
  src/tensor.cpp
  src/tensor_ops.cpp
  src/nn_ops.cpp
  src/mnist.cpp
  src/dense_linear.cpp
  src/butterfly_linear.cpp
  src/multihead_attention.cpp
  src/layer_norm.cpp
  src/activations.cpp
  src/feed_forward.cpp
  src/embedding.cpp
  src/parameter_init.cpp
  src/transformer_block.cpp
  src/decoder_only_transformer.cpp
  src/adam.cpp
)
target_include_directories(vadugrad PUBLIC include)

if(ENABLE_COVERAGE)
  target_compile_options(vadugrad PRIVATE --coverage)
  target_link_options(vadugrad PUBLIC --coverage)
endif()

add_executable(vadugrad_cli src/main.cpp)
target_link_libraries(vadugrad_cli PRIVATE vadugrad)
set_target_properties(vadugrad_cli PROPERTIES OUTPUT_NAME vadugrad)

add_executable(train_lm apps/train_lm.cpp)
target_link_libraries(train_lm PRIVATE vadugrad)

add_executable(train_mnist apps/train_mnist.cpp)
target_link_libraries(train_mnist PRIVATE vadugrad)

if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.18.0
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(tests
    tests/test_tensor.cpp
    tests/test_dense_linear.cpp
    tests/test_butterfly_linear.cpp
    tests/test_multihead_attention.cpp
    tests/test_nn_ops.cpp
    tests/test_layer_norm.cpp
    tests/test_decoder_only_transformer.cpp
    tests/test_tiny_overfit.cpp
    tests/test_mnist_loader.cpp
  )
  target_link_libraries(tests PRIVATE vadugrad GTest::gtest_main)

  if(ENABLE_COVERAGE)
    target_compile_options(tests PRIVATE --coverage)
    target_link_options(tests PRIVATE --coverage)
  endif()

  include(GoogleTest)
  gtest_discover_tests(tests)
endif()
