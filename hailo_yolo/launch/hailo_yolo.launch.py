from launch import LaunchDescription, LaunchService
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable(name='ROS_DOMAIN_ID', value='11'),
        Node(
            package='hailo_yolo',  # パッケージ名
            executable='yolo',  # エントリポイントで定義したノード名
            name='yolo',  # ノード名
            output='screen'
        ),
        Node(
            package='hailo_yolo',
            executable='yolo_overlay_node',
            name='yolo_overlay',
            output='screen'
        )
    ])


    
