#pragma once

#include "yolo_tensorrt_cpp/yolov8.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace yolo {
class YoloNode : public rclcpp::Node {
public:
  struct Config {
    std::string onnx_path;
    std::string trt_engine_path;
    std::string data_yaml_path;
    std::string image_topic;

    std::vector<std::string> class_names;
    std::unordered_map<int, std::string> id_class_map;
  };

  YoloNode();

private:
  void parseDataYaml();
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg);
  vision_msgs::msg::Detection2DArray yoloObjectsToDetectionMessage(const rclcpp::Time &stamp,
                                                                   const std::vector<Object> &objects) const;

private:
  Config config_;
  std::unique_ptr<YoloV8> yolo_ = nullptr;

  // Subscribers.
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_ = nullptr;

  // Publishers.
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_ = nullptr;
};

}  // namespace yolo