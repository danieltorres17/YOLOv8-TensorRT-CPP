cmake_minimum_required(VERSION 3.22)
project(yolo_tensorrt_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# CMake Options
option(ENABLE_BENCHMARKS "Benchmark the preprocessing, inference, and post processing" OFF)
if (ENABLE_BENCHMARKS)
    add_compile_definitions(ENABLE_BENCHMARKS)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

set(YOLO_INCLUDE_DIRS include)

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR /usr/)

# Build the TensorRT inference engine library
# TensorRT is found and linked by the tensorrt-cpp-api
add_subdirectory(libs/tensorrt-cpp-api)

# Build the YoloV8 library
add_library(YoloV8_TRT SHARED src/yolov8.cpp)
target_include_directories(YoloV8_TRT PUBLIC libs/tensorrt-cpp-api/src ${YOLO_INCLUDE_DIRS})
target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})

# Build and link the executables
add_executable(detect_object_image src/object_detection_image.cpp)
target_link_libraries(detect_object_image YoloV8_TRT)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark YoloV8_TRT)

add_executable(detect_object_video src/object_detection_video_stream.cpp)
target_link_libraries(detect_object_video YoloV8_TRT)

add_executable(serialize_engine src/serialize_engine.cpp)
target_include_directories(serialize_engine PRIVATE ${YOLO_INCLUDE_DIRS})
ament_target_dependencies(serialize_engine rclcpp)
target_link_libraries(serialize_engine YoloV8_TRT)

add_executable(yolo_node src/yolo_node.cpp)
target_include_directories(yolo_node PRIVATE ${YOLO_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(yolo_node rclcpp cv_bridge vision_msgs sensor_msgs)
target_link_libraries(yolo_node ${OpenCV_LIBS} yaml-cpp YoloV8_TRT)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  ${PROJECT_NAME}/detection_visualizer_node.py
  ${PROJECT_NAME}/image_loader_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS
    serialize_engine
    yolo_node
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY
    config
    launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()