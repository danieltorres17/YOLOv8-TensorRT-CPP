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
  this->declare_parameter("apply_preprocessing", false);
  this->declare_parameter("preprocessing_yaml_path", "");

  // Get parameters.
  config_.onnx_path = this->get_parameter("onnx_path").as_string();
  config_.trt_engine_path = this->get_parameter("trt_engine_path").as_string();
  config_.data_yaml_path = this->get_parameter("data_yaml_path").as_string();
  config_.image_topic = this->get_parameter("image_topic").as_string();
  config_.apply_preprocessing = this->get_parameter("apply_preprocessing").as_bool();
  config_.preprocessing_yaml_path = this->get_parameter("preprocessing_yaml_path").as_string();

  RCLCPP_INFO(this->get_logger(), "Onnx path: %s", config_.onnx_path.c_str());
  RCLCPP_INFO(this->get_logger(), "Trt engine path: %s", config_.trt_engine_path.c_str());
  RCLCPP_INFO(this->get_logger(), "Data yaml path: %s", config_.data_yaml_path.c_str());
  RCLCPP_INFO(this->get_logger(), "Image sub topic: %s", config_.image_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "Apply preprocessing: %s", config_.apply_preprocessing ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Preprocessing yaml path: %s", config_.preprocessing_yaml_path.c_str());

  // Preprocessing.
  if (config_.apply_preprocessing) {
    preprocessing_params_ = loadPreprocessingParams(config_.preprocessing_yaml_path);
    if (!preprocessing_params_) {
      RCLCPP_WARN(this->get_logger(),
                  "Preprocessing is enabled but failed to load parameters. Disabling preprocessing.");
      config_.apply_preprocessing = false;
    } else {
      RCLCPP_INFO(this->get_logger(), "Loaded preprocessing parameters from: %s",
                  config_.preprocessing_yaml_path.c_str());
    }
  }

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
  if (config_.apply_preprocessing) {
    preprocess_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/preprocessed_image", 1);
  }
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

std::optional<YoloNode::PreprocessingParams> YoloNode::loadPreprocessingParams(
    const std::string& yaml_path) const {
  if (!fs::exists(yaml_path)) {
    RCLCPP_WARN(this->get_logger(), "Unable to find preprocessing configuration file: %s", yaml_path.c_str());

    return std::nullopt;
  }

  try {
    YAML::Node node = YAML::LoadFile(yaml_path);
    PreprocessingParams params;
    params.brightness = node["brightness"].as<int>(params.brightness);
    params.contrast = node["contrast"].as<int>(params.contrast);
    params.gamma = node["gamma"].as<int>(params.gamma);
    params.saturation = node["saturation"].as<int>(params.saturation);

    return params;
  } catch (const YAML::Exception& e) {
    std::cerr << "Error loading adjustments from YAML: " << e.what() << std::endl;

    return std::nullopt;
  }
}

void YoloNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg) {
  const std::string encoding_str = image_msg->encoding;
  cv::Mat image = cv_bridge::toCvShare(image_msg)->image;
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

  // Apply preprocessing if enabled.
  if (config_.apply_preprocessing && preprocessing_params_) {
    gpu_image = applyPreprocessing(gpu_image, *preprocessing_params_);

    if (preprocess_img_pub_->get_subscription_count() > 0) {
      cv::Mat preprocessed_image;
      gpu_image.download(preprocessed_image);

      sensor_msgs::msg::Image preprocessed_msg =
          *cv_bridge::CvImage(image_msg->header, "bgr8", preprocessed_image).toImageMsg();
      preprocess_img_pub_->publish(std::move(preprocessed_msg));
    }
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

cv::cuda::GpuMat YoloNode::applyPreprocessing(const cv::cuda::GpuMat& input_image,
                                              const PreprocessingParams& params) const {
  cv::cuda::GpuMat output_image;

  // Brightness and contrast adjustment.
  const double alpha = params.contrast / 50.0;
  const double beta = params.brightness - 50;
  input_image.convertTo(output_image, -1, alpha, beta);

  // Gamma correction.
  const double gamma = params.gamma / 50.0;
  cv::Mat lut(1, 256, CV_8UC1);
  for (int i = 0; i < 256; i++) {
    lut.at<uchar>(i) = cv::saturate_cast<uchar>(pow(i / 255.0, 1.0 / gamma) * 255.0);
  }

  const auto lut_cuda = cv::cuda::createLookUpTable(lut);
  lut_cuda->transform(output_image, output_image);

  // Saturation adjustment.
  cv::cuda::GpuMat hsv_image;
  cv::cuda::cvtColor(output_image, hsv_image, cv::COLOR_BGR2HSV);
  std::vector<cv::cuda::GpuMat> hsv_channels;
  cv::cuda::split(hsv_image, hsv_channels);
  cv::cuda::multiply(hsv_channels[1], params.saturation / 50.0, hsv_channels[1]);
  cv::cuda::merge(hsv_channels, hsv_image);
  cv::cuda::cvtColor(hsv_image, output_image, cv::COLOR_HSV2BGR);

  return output_image;
}

vision_msgs::msg::Detection2DArray YoloNode::yoloObjectsToDetectionMessage(
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

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolo::YoloNode>());
  rclcpp::shutdown();

  return 0;
}
