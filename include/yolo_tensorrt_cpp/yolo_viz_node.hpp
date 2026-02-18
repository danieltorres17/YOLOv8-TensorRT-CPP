#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>

namespace yolo {

class YoloVizNode : public rclcpp::Node {
public:
  struct NodeConfig {
    std::string input_image_topic = "/zed/zed_node/left/image_rect_color";
    std::string detections_topic = "/detections_output";
    std::string output_image_topic = "/detections_output/image";
  };

  explicit YoloVizNode(const rclcpp::NodeOptions& options);

  void imageDetectionsCallback(const sensor_msgs::msg::Image::SharedPtr image_msg,
                               const vision_msgs::msg::Detection2DArray::SharedPtr detections_msg);

private:
  using ImageDetectionsSyncPolicy =
      message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, vision_msgs::msg::Detection2DArray>;

  NodeConfig config_;

  // Subscribers.
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_ = nullptr;
  std::shared_ptr<message_filters::Subscriber<vision_msgs::msg::Detection2DArray>> detections_sub_ = nullptr;
  std::shared_ptr<message_filters::Synchronizer<ImageDetectionsSyncPolicy>> sync_ = nullptr;

  // Publishers.
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr output_image_pub_ = nullptr;
};

}  // namespace yolo

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(yolo::YoloVizNode)