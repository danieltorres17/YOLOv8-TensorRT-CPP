#include "yolo_tensorrt_cpp/yolo_viz_node.hpp"

#include <cv_bridge/cv_bridge.hpp>

namespace yolo {

YoloVizNode::YoloVizNode(const rclcpp::NodeOptions& options) : Node("yolo_viz_node", options) {
  // Declare parameters.
  this->declare_parameter("input_image_topic", config_.input_image_topic);
  this->declare_parameter("detections_topic", config_.detections_topic);
  this->declare_parameter("output_image_topic", config_.output_image_topic);

  // Get parameters.
  config_.input_image_topic = this->get_parameter("input_image_topic").as_string();
  config_.detections_topic = this->get_parameter("detections_topic").as_string();
  config_.output_image_topic = this->get_parameter("output_image_topic").as_string();

  RCLCPP_INFO(this->get_logger(), "Input image topic: %s", config_.input_image_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "Detections topic: %s", config_.detections_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "Output image topic: %s", config_.output_image_topic.c_str());

  // Subscribers.
  image_sub_ =
      std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, config_.input_image_topic);
  detections_sub_ = std::make_shared<message_filters::Subscriber<vision_msgs::msg::Detection2DArray>>(
      this, config_.detections_topic);
  sync_ = std::make_shared<message_filters::Synchronizer<ImageDetectionsSyncPolicy>>(
      ImageDetectionsSyncPolicy(10), *image_sub_, *detections_sub_);
  sync_->registerCallback(&YoloVizNode::imageDetectionsCallback, this);

  // Publishers.
  output_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(config_.output_image_topic, 1);
}

void YoloVizNode::imageDetectionsCallback(
    const sensor_msgs::msg::Image::SharedPtr image_msg,
    const vision_msgs::msg::Detection2DArray::SharedPtr detections_msg) {
  if (output_image_pub_->get_subscription_count() == 0) {
    // No subscribers, skip processing.
    return;
  }

  const std::string encoding_str = image_msg->encoding;
  cv::Mat image = cv_bridge::toCvShare(image_msg)->image;
  const cv::Scalar color(0, 255, 0);     // Green color for bounding boxes.
  const cv::Scalar text_color(0, 0, 0);  // Black color for text.
  const int thickness = 2;               // Thickness of bounding box lines.
  const float font_scale = 0.65f;        // Font scale for labels.

  for (size_t ii = 0; ii < detections_msg->detections.size(); ++ii) {
    const auto& detection_msg = detections_msg->detections.at(ii);
    const float tl_x = detection_msg.bbox.center.position.x - detection_msg.bbox.size_x / 2.0f;
    const float tl_y = detection_msg.bbox.center.position.y - detection_msg.bbox.size_y / 2.0f;
    const cv::Rect2f bbox_cv(tl_x, tl_y, detection_msg.bbox.size_x, detection_msg.bbox.size_y);

    cv::rectangle(image, bbox_cv, color, thickness);

    // Add label and confidence.
    const std::string label = cv::format("%s: %.2f", detection_msg.results.at(0).hypothesis.class_id.c_str(),
                                         detection_msg.results.at(0).hypothesis.score);
    int baseline = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, 2, &baseline);
    cv::Point2i label_tl(bbox_cv.tl().x - 1, bbox_cv.tl().y - label_size.height - baseline);

    if (label_tl.y < 0) {
      label_tl.y = bbox_cv.tl().y + label_size.height + baseline;
    }
    cv::rectangle(image, cv::Rect(label_tl, cv::Size(label_size.width, label_size.height + baseline)), color,
                  cv::FILLED);
    cv::putText(image, label, label_tl + cv::Point2i(0, label_size.height), cv::FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, 2);
  }

  const sensor_msgs::msg::Image output_image_msg =
      *cv_bridge::CvImage(image_msg->header, encoding_str, image).toImageMsg();
  output_image_pub_->publish(std::move(output_image_msg));
}

}  // namespace yolo