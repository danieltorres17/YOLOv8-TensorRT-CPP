from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_path

import os


def generate_launch_description():
  yolo_pkg = get_package_share_path('yolo_tensorrt_cpp')
  preprocessing_yaml_path = os.path.join(yolo_pkg, 'config', 'preprocessing.yaml')

  ld = LaunchDescription()
  image_input_topic = LaunchConfiguration('image_input_topic')
  onnx_path = LaunchConfiguration('onnx_path')
  trt_engine_path = LaunchConfiguration('trt_engine_path')
  data_yaml_path = LaunchConfiguration('data_yaml_path')
  apply_preprocessing = LaunchConfiguration('apply_preprocessing')
  node = LaunchConfiguration('node')
  viz = LaunchConfiguration('viz')

  ld.add_action(
    DeclareLaunchArgument(
      'image_input_topic',
      default_value='/left/image_rect',
      description='Image input topic.',
    )
  )
  ld.add_action(
    DeclareLaunchArgument(
      'onnx_path',
      description='Absolute path to the model .onnx file.',
    )
  )
  ld.add_action(
    DeclareLaunchArgument(
      'trt_engine_path',
      description='Absolute path to the model TensorRT engine file.',
    )
  )
  ld.add_action(
    DeclareLaunchArgument(
      'data_yaml_path',
      default_value=os.path.join(yolo_pkg, 'config', 'example_data.yaml'),
      description='Absolute path to data.yaml file containing the class names.',
    )
  )
  ld.add_action(
    DeclareLaunchArgument(
      'apply_preprocessing',
      default_value='False',
      description='Whether to apply preprocessing to input images before inference.',
    )
  )
  ld.add_action(
    DeclareLaunchArgument(
      'node',
      default_value='True',
      description='Launch the yolo node. Set to false to only visualize existing detections.',
    )
  )
  ld.add_action(
    DeclareLaunchArgument(
      'viz',
      default_value='True',
      description='Launch the RQT image viz node to visualize detections.',
    )
  )

  ld.add_action(
    Node(
      name='yolo_node',
      namespace='',
      package='yolo_tensorrt_cpp',
      executable='yolo_node',
      parameters=[
        {
          'onnx_path': onnx_path,
          'trt_engine_path': trt_engine_path,
          'data_yaml_path': data_yaml_path,
          'image_topic': image_input_topic,
          'apply_preprocessing': apply_preprocessing,
          'preprocessing_yaml_path': preprocessing_yaml_path,
        }
      ],
      condition=IfCondition(node),
    )
  )

  ld.add_action(
    Node(
      name='detection_visualizer_node',
      package='yolo_tensorrt_cpp',
      executable='detection_visualizer_node.py',
      condition=IfCondition(viz),
      remappings=[('/image', image_input_topic)],
    )
  )

  ld.add_action(
    Node(
      package='rqt_image_view',
      executable='rqt_image_view',
      name='image_view',
      arguments=['/yolo_processed_image'],
      condition=IfCondition(viz),
    )
  )

  return ld
