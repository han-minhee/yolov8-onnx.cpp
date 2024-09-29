cmake_minimum_required(VERSION 3.20)
option(USE_HIP "Enable HIP support for compilation" OFF)

# Check for HIP usage and set the compiler accordingly
if(USE_HIP)
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc) # Set this before the project command
endif()

# Project declaration should come after setting the compiler
project(YOLO_ONNX_Cpp)

set(CMAKE_CXX_STANDARD 17)

# Check for HIP usage again to set relevant flags and configurations
if(USE_HIP)
    message(STATUS "USE_HIP flag is set, using HIP for compilation")
    find_package(HIP REQUIRED)
    add_compile_definitions(USE_HIP) # Define USE_HIP globally
endif()

# include
include_directories(external/stb)
include_directories(include/)


add_subdirectory(external/onnx_cpp_lib)

# Build library from src files (excluding main.cpp)
file(GLOB_RECURSE LIB_SRC_FILES src/*.cpp src/*.cc)
list(REMOVE_ITEM LIB_SRC_FILES ${CMAKE_SOURCE_DIR}/src/main.cpp) # Adjust the path if necessary
add_library(yolov8_lib ${LIB_SRC_FILES})
target_link_libraries(yolov8_lib onnx_cpp_lib)

# If HIP is enabled, link HIP libraries
if(USE_HIP)
    target_link_libraries(yolov8_lib hip::device)
    target_compile_definitions(yolov8_lib PRIVATE USE_HIP)
endif()

# Build main executable
add_executable(yolov8 src/main.cpp)
target_link_libraries(yolov8 yolov8_lib)

# GTest
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)

FetchContent_MakeAvailable(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

file(GLOB TEST_SOURCES "${PROJECT_SOURCE_DIR}/tests/*.cpp")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} yolov8_lib gtest gtest_main onnx_cpp_lib ${Protobuf_LIBRARIES})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    # If HIP is enabled, add compile definitions for tests
    if(USE_HIP)
        target_link_libraries(${TEST_NAME} hip::device)
        target_compile_definitions(${TEST_NAME} PRIVATE USE_HIP)
    endif()
endforeach()
