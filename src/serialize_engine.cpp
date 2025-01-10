#include <rclcpp/rclcpp.hpp>
#include "yolo_tensorrt_cpp/cmd_line_util.h"
#include "yolo_tensorrt_cpp/yolov8.h"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  YoloV8Config config;
  std::string onnx_model_path;
  std::string trt_model_path{};
  std::string input_image{};

  if (!parseArguments(argc, argv, config, onnx_model_path, trt_model_path, input_image)) {
    return -1;
  }

  YoloV8 yolov8(onnx_model_path, trt_model_path, config);

  return 0;
}