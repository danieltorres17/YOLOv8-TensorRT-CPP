#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "yolo_tensorrt_cpp/yolov8.h"

namespace yolo {

class YoloLcNode : public rclcpp_lifecycle::LifecycleNode {
public:
  struct Config {
    std::string onnx_path;
    std::string trt_engine_path;
    std::string data_yaml_path;
    std::string image_topic;

    std::vector<std::string> class_names;
    std::unordered_map<int, std::string> id_class_map;
  };

  explicit YoloLcNode(const rclcpp::NodeOptions& options);

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State& state);

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State& state);

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State& state);

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_shutdown(
      const rclcpp_lifecycle::State& state);

  bool parseDataYaml();

  void leftImageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg);

  vision_msgs::msg::Detection2DArray yoloObjectsToDetectionMessage(const rclcpp::Time& stamp,
                                                                   const std::vector<Object>& objects) const;

private:
  Config config_;
  std::unique_ptr<YoloV8> yolo_ = nullptr;

  // Subscribers.
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_ = nullptr;

  // Publishers.
  rclcpp_lifecycle::LifecyclePublisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_ =
      nullptr;
};

}  // namespace yolo

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(yolo::YoloLcNode)