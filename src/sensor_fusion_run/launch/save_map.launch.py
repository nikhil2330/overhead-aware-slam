"""Save an OccupancyGrid topic as a PGM + YAML pair.

Run in a second terminal while the mapping session is active.

Supported maps
--------------
SLAM base map (test.launch.py or lidar3d_slam):
  ros2 launch sensor_fusion_run save_map.launch.py output:=/tmp/slam_map

Augmented sensor-fusion map — occupancy_map.py paints the camera+lidar
detected objects from object_location.py onto the SLAM base map.
This is the main research contribution map from test.launch.py:
  ros2 launch sensor_fusion_run save_map.launch.py \\
      map_topic:=/occupancy_map \\
      output:=/tmp/augmented_map \\
      transient:=false

3D lidar pointcloud occupancy (lidar3d_slam / lidar3d_rtabmap):
  ros2 launch sensor_fusion_run save_map.launch.py \\
      map_topic:=/pointcloud_occupancy_2d \\
      output:=/tmp/pointcloud_map \\
      transient:=false

RTAB-Map grid (lidar3d_rtabmap):
  ros2 launch sensor_fusion_run save_map.launch.py \\
      output:=/tmp/rtabmap_map \\
      transient:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    output = LaunchConfiguration('output')
    map_topic = LaunchConfiguration('map_topic')
    transient = LaunchConfiguration('transient')

    return LaunchDescription([
        DeclareLaunchArgument(
            'output',
            default_value='/tmp/saved_map',
            description='Output path without extension (.pgm + .yaml will be written).',
        ),
        DeclareLaunchArgument(
            'map_topic',
            default_value='/map',
            description=(
                'OccupancyGrid topic to save.  '
                'Use /pointcloud_occupancy_2d for the augmented map.'
            ),
        ),
        DeclareLaunchArgument(
            'transient',
            default_value='true',
            description=(
                'true  = topic uses Transient Local QoS (SLAM Toolbox, RTAB-Map). '
                'false = topic uses Volatile QoS (pointcloud_occupancy_2d).'
            ),
        ),

        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                '-f', output,
                '--ros-args',
                '-p', ['map_subscribe_transient_local:=', transient],
                '-r', ['map:=', map_topic],
            ],
            output='screen',
        ),
    ])
