#include "yolo_tensorrt_cpp/yolo_lc_node.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <filesystem>
namespace fs = std::filesystem;

namespace yolo {

YoloLcNode::YoloLcNode(const rclcpp::NodeOptions& options)
  : rclcpp_lifecycle::LifecycleNode("yolo_lc_node", options) {}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn YoloLcNode::on_configure(
    const rclcpp_lifecycle::State& state) {
  RCLCPP_INFO(this->get_logger(), "Configuring Yolo Lifecycle Node.");
  LifecycleNode::on_configure(state);

  // Declare parameters.
  this->declare_parameter("image_topic", "");
  this->declare_parameter("onnx_path", "");
  this->declare_parameter("trt_engine_path", "");
  this->declare_parameter("data_yaml_path", "");

  // Get parameters.
  config_.image_topic = this->get_parameter("image_topic").as_string();
  config_.onnx_path = this->get_parameter("onnx_path").as_string();
  config_.trt_engine_path = this->get_parameter("trt_engine_path").as_string();
  config_.data_yaml_path = this->get_parameter("data_yaml_path").as_string();

  RCLCPP_INFO(this->get_logger(), "Image topic: %s", config_.image_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "ONNX path: %s",
              fs::is_symlink(config_.onnx_path) ? fs::read_symlink(config_.onnx_path).c_str()
                                                : config_.onnx_path.c_str());
  RCLCPP_INFO(this->get_logger(), "TRT engine path: %s",
              fs::is_symlink(config_.trt_engine_path) ? fs::read_symlink(config_.trt_engine_path).c_str()
                                                      : config_.trt_engine_path.c_str());
  RCLCPP_INFO(this->get_logger(), "Data yaml path: %s",
              fs::is_symlink(config_.data_yaml_path) ? fs::read_symlink(config_.data_yaml_path).c_str()
                                                     : config_.data_yaml_path.c_str());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn YoloLcNode::on_activate(
    const rclcpp_lifecycle::State& state) {
  RCLCPP_INFO(this->get_logger(), "Activating Yolo Lifecycle Node.");
  LifecycleNode::on_activate(state);

  // Initialize YOLO object.
  const bool data_yaml_parsed = parseDataYaml();
  if (!data_yaml_parsed) {
    RCLCPP_ERROR(this->get_logger(), "Failed to parse data yaml file. Cannot activate node.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
  }
  YoloV8Config yolo_config;
  yolo_config.classNames = config_.class_names;
  yolo_ = std::make_unique<YoloV8>("", config_.trt_engine_path, yolo_config);

  // Subscribers.
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      config_.image_topic, 10, std::bind(&YoloLcNode::leftImageCallback, this, std::placeholders::_1));

  // Publishers.
  detections_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/detections_output", 1);
  detections_pub_->on_activate();

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn YoloLcNode::on_deactivate(
    const rclcpp_lifecycle::State& state) {
  RCLCPP_INFO(this->get_logger(), "Deactivating Yolo Lifecycle Node.");
  LifecycleNode::on_deactivate(state);

  detections_pub_->on_deactivate();
  detections_pub_.reset();
  image_sub_.reset();
  yolo_.reset();

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn YoloLcNode::on_shutdown(
    const rclcpp_lifecycle::State& state) {
  RCLCPP_INFO(this->get_logger(), "Shutting down Yolo Lifecycle Node.");
  LifecycleNode::on_shutdown(state);

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

bool YoloLcNode::parseDataYaml() {
  if (!fs::exists(config_.data_yaml_path)) {
    RCLCPP_FATAL(this->get_logger(), "Unable to find data configuration file: %s",
                 config_.data_yaml_path.c_str());
    return false;
  }

  YAML::Node config = YAML::LoadFile(config_.data_yaml_path);
  if (config["names"]) {
    config_.class_names = config["names"].as<std::vector<std::string>>();
  } else {
    RCLCPP_FATAL(this->get_logger(), "Unable to find 'names' field in configuration file.");
    return false;
  }

  for (size_t ii = 0; ii < config_.class_names.size(); ii++) {
    config_.id_class_map.insert({static_cast<int>(ii), config_.class_names.at(ii)});
  }

  return true;
}

void YoloLcNode::leftImageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg) {
  const std::string encoding_str = image_msg->encoding;
  cv::Mat image = cv_bridge::toCvShare(image_msg, "bgr8")->image;
  if (image.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Received empty image.");

    return;
  }

  // Convert to GPU mat.
  cv::cuda::GpuMat gpu_image;
  gpu_image.upload(image);

  if (encoding_str == "rgb8") {
    cv::cuda::cvtColor(gpu_image, gpu_image, cv::COLOR_RGB2BGR);
  }

  auto obj_det_start = std::chrono::high_resolution_clock::now();
  const auto objects = yolo_->detectObjects(gpu_image);
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

vision_msgs::msg::Detection2DArray YoloLcNode::yoloObjectsToDetectionMessage(
    const rclcpp::Time& stamp, const std::vector<Object>& objects) const {
  vision_msgs::msg::Detection2DArray detections_array_msg;
  detections_array_msg.detections.reserve(objects.size());
  detections_array_msg.header.stamp = stamp;

  for (const auto& obj : objects) {
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