#include "yolo_tensorrt_cpp/yolo_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <filesystem>
namespace fs = std::filesystem;

namespace yolo {

YoloNode::YoloNode() : Node("yolo_node") {
  // Declare onnx, engine and data yaml filepath parameters.
  this->declare_parameter("onnx_path", "");
  this->declare_parameter("trt_engine_path", "");
  this->declare_parameter("data_yaml_path", "");
  this->declare_parameter("image_topic", "");

  // Get parameters.
  config_.onnx_path = this->get_parameter("onnx_path").as_string();
  config_.trt_engine_path = this->get_parameter("trt_engine_path").as_string();
  config_.data_yaml_path = this->get_parameter("data_yaml_path").as_string();
  config_.image_topic = this->get_parameter("image_topic").as_string();

  RCLCPP_INFO(this->get_logger(), "Onnx path: %s", config_.onnx_path.c_str());
  RCLCPP_INFO(this->get_logger(), "Trt engine path: %s", config_.trt_engine_path.c_str());
  RCLCPP_INFO(this->get_logger(), "Data yaml path: %s", config_.data_yaml_path.c_str());
  RCLCPP_INFO(this->get_logger(), "Image sub topic: %s", config_.image_topic.c_str());

  // Initialize YOLO inference object.
  parseDataYaml();
  YoloV8Config yolo_config;
  yolo_config.classNames = config_.class_names;
  yolo_ = std::make_unique<YoloV8>("", config_.trt_engine_path, yolo_config);

  // Subscribers.
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      config_.image_topic, 10, std::bind(&YoloNode::imageCallback, this, std::placeholders::_1));

  // Publishers.
  detections_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/detections_output", 1);
}

void YoloNode::parseDataYaml() {
  if (!fs::exists(config_.data_yaml_path)) {
    RCLCPP_FATAL(this->get_logger(), "Unable to find data configuration file: %s",
                 config_.data_yaml_path.c_str());
  }

  YAML::Node config = YAML::LoadFile(config_.data_yaml_path);
  if (config["names"]) {
    config_.class_names = config["names"].as<std::vector<std::string>>();
  } else {
    RCLCPP_FATAL(this->get_logger(), "Unable to find 'names' field in configuration file.");
  }

  for (size_t ii = 0; ii < config_.class_names.size(); ii++) {
    config_.id_class_map.insert({static_cast<int>(ii), config_.class_names.at(ii)});
  }
}

void YoloNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg) {
  cv::Mat image = cv_bridge::toCvShare(image_msg)->image;
  if (image.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Received empty image.");

    return;
  }

  auto obj_det_start = std::chrono::high_resolution_clock::now();
  const auto objects = yolo_->detectObjects(image);
  auto obj_det_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> obj_det_processing_elapsed = obj_det_end - obj_det_start;
  RCLCPP_DEBUG(this->get_logger(), "Detection processing duration: %f seconds",
               obj_det_processing_elapsed.count());

  // Publish detections.
  if (detections_pub_->get_subscription_count() > 0) {
    const vision_msgs::msg::Detection2DArray detections_msg =
        yoloObjectsToDetectionMessage(image_msg->header.stamp, objects);
    detections_pub_->publish(detections_msg);
  }
}

vision_msgs::msg::Detection2DArray YoloNode::yoloObjectsToDetectionMessage(
    const rclcpp::Time &stamp, const std::vector<Object> &objects) const {
  vision_msgs::msg::Detection2DArray detections_array_msg;
  detections_array_msg.detections.reserve(objects.size());
  detections_array_msg.header.stamp = stamp;

  for (const auto &obj : objects) {
    vision_msgs::msg::Detection2D det_msg;
    det_msg.header.stamp = stamp;
    det_msg.id = config_.id_class_map.at(obj.label);
    det_msg.bbox.center.position.x = obj.rect.x + (obj.rect.width / 2.0);
    det_msg.bbox.center.position.y = obj.rect.y + (obj.rect.height / 2.0);
    det_msg.bbox.size_x = obj.rect.width;
    det_msg.bbox.size_y = obj.rect.height;

    vision_msgs::msg::ObjectHypothesisWithPose obj_hyp_msg;
    obj_hyp_msg.hypothesis.class_id = config_.id_class_map.at(obj.label);
    obj_hyp_msg.hypothesis.score = obj.probability;
    det_msg.results.push_back(obj_hyp_msg);

    detections_array_msg.detections.push_back(det_msg);
  }

  return detections_array_msg;
}

}  // namespace yolo

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolo::YoloNode>());
  rclcpp::shutdown();

  return 0;
}
