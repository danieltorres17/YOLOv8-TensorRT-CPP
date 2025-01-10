#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

import os
from pathlib import Path


class ImageLoaderNode(Node):
  def __init__(self) -> None:
    super().__init__('image_loader_node')
    self.image_publisher = self.create_publisher(Image, '/image', 10)
    self.timer_period: float = 0.5
    self.timer = self.create_timer(self.timer_period, self.timerCallback)
    self.bridge = CvBridge()

    data_dir: Path = Path(os.getenv('DATA_DIR'))
    images_dir: Path = data_dir / 'synthetic_dataset/train/images'
    assert images_dir.exists(), f'Unable to find images in specified directory: {images_dir.as_posix()}'

    self.count: int = 0
    self.images: list[Path] = []
    for filename in sorted(os.listdir(images_dir)):
      if filename.endswith(('.png', '.jpg')):
        self.images.append(images_dir / filename)

    assert len(self.images) > 0, f'No images in specified directory: {images_dir.as_posix()}'

  def timerCallback(self) -> None:
    if self.count >= len(self.images):
      self.count = 0

    img = cv2.imread(self.images[self.count])
    header_msg = Header()
    header_msg.stamp = self.get_clock().now().to_msg()
    msg = self.bridge.cv2_to_imgmsg(img, encoding='rgb8', header=header_msg)

    self.image_publisher.publish(msg)
    self.count += 1


def main(args=None):
  rclpy.init(args=args)
  image_loader_node = ImageLoaderNode()

  rclpy.spin(image_loader_node)
  rclpy.shutdown()


if __name__ == '__main__':
  main()
