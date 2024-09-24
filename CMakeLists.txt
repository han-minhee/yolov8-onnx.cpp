cmake_minimum_required(VERSION 3.15)
project(YOLO_ONNX_Cpp)
set(CMAKE_CXX_STANDARD 17)

include_directories(include)
add_subdirectory(external/onnx_cpp_lib)

# Build library from src files (excluding main.cpp)
file(GLOB_RECURSE LIB_SRC_FILES src/*.cpp src/*.cc)
list(REMOVE_ITEM LIB_SRC_FILES ${CMAKE_SOURCE_DIR}/src/main.cpp)  # Adjust the path if necessary

add_library(yolov8_lib ${LIB_SRC_FILES})
target_link_libraries(yolov8_lib onnx_cpp_lib)

# Build main executable
add_executable(yolov8 src/main.cpp)
target_link_libraries(yolov8 yolov8_lib)

# GTest
enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)

FetchContent_MakeAvailable(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

file(GLOB TEST_SOURCES "${PROJECT_SOURCE_DIR}/tests/*.cpp")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} yolov8_lib gtest gtest_main onnx_cpp_lib ${Protobuf_LIBRARIES})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
