"""
3-D LiDAR SLAM — slam_toolbox async online mapping
===================================================

Usage
-----
  ros2 launch sensor_fusion_run lidar3d_slam.launch.py [world_name:=house3.world]

  Optional args
  -------------
  world_name    Gazebo world file in sensor_fusion_desc/worlds   [house2.world]
  use_rviz      Launch RViz                                       [true]

Save the filled 2-D map (separate terminal, while session is running):
  ros2 launch sensor_fusion_run save_map.launch.py \\
      output:=~/slam_filled_map \\
      map_topic:=/pointcloud_occupancy_2d \\
      transient:=false

Save the raw SLAM map:
  ros2 launch sensor_fusion_run save_map.launch.py output:=~/slam_raw_map
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    desc_pkg = FindPackageShare('sensor_fusion_desc')
    world_name = LaunchConfiguration('world_name')
    use_rviz   = LaunchConfiguration('use_rviz')

    world   = PathJoinSubstitution([desc_pkg, 'worlds', world_name])
    models  = PathJoinSubstitution([desc_pkg, 'models'])
    worlds  = PathJoinSubstitution([desc_pkg, 'worlds'])

    gz_launch   = PathJoinSubstitution(
        [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py']
    )
    robot_sdf   = PathJoinSubstitution([desc_pkg, 'models', 'robot_3d_lidar', 'model.sdf'])
    slam_params = PathJoinSubstitution([desc_pkg, 'config', 'slam_toolbox_3d_lidar.yaml'])
    rviz_cfg    = PathJoinSubstitution([desc_pkg, 'config', 'rviz_lidar3d.rviz'])

    # ------------------------------------------------------------------ #
    # Teleop in its own gnome-terminal window                              #
    # ------------------------------------------------------------------ #
    teleop = ExecuteProcess(
        cmd=[
            'gnome-terminal', '--',
            'bash', '-c',
            'source /opt/ros/jazzy/setup.bash && '
            'source ~/sensor_fusion/miniature-waffle/install/setup.bash && '
            'echo "=== Use WASD / arrow keys to drive the robot ===" && '
            'ros2 run teleop_twist_keyboard teleop_twist_keyboard '
            '--ros-args -r cmd_vel:=/cmd_vel'
        ],
        output='screen',
    )

    # ------------------------------------------------------------------ #
    # Delayed nodes                                                        #
    # ------------------------------------------------------------------ #

    # pointcloud_occupancy: starts after Gazebo / TF settle (~8 s wall)
    # decay_seconds=0 → observations accumulate permanently as robot explores.
    # morph_close_cells=13 → 13×0.05 = 0.65 m closing radius → fills entire
    # bed / table / bookshelf interiors once the perimeter is seen.
    # projection_padding_cells=2 → each voxel expands to 5×5 cells (0.25 m)
    # which helps fill from limited single-side views.
    occ_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='sensor_fusion_nodes',
                executable='pointcloud_occupancy',
                name='pointcloud_occupancy',
                output='screen',
                parameters=[{
                    'pointcloud_topic':        '/lidar/points',
                    'map_frame':               'odom',
                    'resolution':              0.05,
                    'z_resolution':            0.08,
                    'width_m':                 16.0,
                    'height_m':                14.0,
                    'origin_x':                -8.0,
                    'origin_y':                -7.0,
                    'min_z':                   0.08,
                    'max_z':                   2.20,
                    'projection_min_z':        0.10,
                    'projection_max_z':        1.80,
                    'projection_padding_cells': 0,
                    'min_observations':        2,
                    'decay_seconds':           0.0,
                    'max_points_per_cloud':    30000,
                    'publish_period':          1.0,
                    'morph_close_cells':       7,
                    'occupancy_2d_topic':      '/pointcloud_occupancy_2d',
                    'occupancy_3d_topic':      '/pointcloud_occupancy_3d',
                    'occupied_voxels_topic':   '/occupied_voxels',
                    'use_sim_time':            True,
                }],
            ),
        ],
    )

    # map_filler: applies same closing to the SLAM /map topic, giving a
    # filled version that can also be saved as a PGM.
    filler_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='sensor_fusion_nodes',
                executable='map_filler',
                name='map_filler',
                output='screen',
                parameters=[{
                    'input_topic':   '/map',
                    'output_topic':  '/map_filled',
                    'close_cells':   13,
                    'use_sim_time':  True,
                }],
            ),
        ],
    )

    # slam_toolbox: starts after TF tree is stable (~15 s wall)
    slam_node = TimerAction(
        period=15.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('slam_toolbox'),
                        'launch',
                        'online_async_launch.py',
                    ])
                ),
                launch_arguments={
                    'use_sim_time':     'true',
                    'slam_params_file': slam_params,
                }.items(),
            ),
        ],
    )

    return LaunchDescription([
        # ---- args ---- #
        DeclareLaunchArgument(
            'world_name', default_value='house2.world',
            description='World file inside sensor_fusion_desc/worlds/',
        ),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Whether to start RViz.',
        ),

        # ---- env ---- #
        SetEnvironmentVariable('QT_QPA_PLATFORM', 'xcb'),
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            [desc_pkg, TextSubstitution(text=':'),
             models,   TextSubstitution(text=':'),
             worlds],
        ),
        SetEnvironmentVariable(
            'IGN_GAZEBO_RESOURCE_PATH',
            [desc_pkg, TextSubstitution(text=':'),
             models,   TextSubstitution(text=':'),
             worlds],
        ),

        # ---- Gazebo ---- #
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch),
            launch_arguments={
                'gz_args':          [TextSubstitution(text='-r '), world],
                'on_exit_shutdown': 'true',
            }.items(),
        ),

        # ---- spawn robot ---- #
        Node(
            package='ros_gz_sim', executable='create', output='screen',
            arguments=[
                '-name', 'robot_3d_lidar', '-file', robot_sdf,
                '-x', '1.2265', '-y', '-0.9923', '-z', '0.01', '-Y', '3.1164',
            ],
        ),

        # ---- ROS–Gazebo bridge ---- #
        Node(
            package='ros_gz_bridge', executable='parameter_bridge', output='screen',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/model/robot_3d_lidar/pose@geometry_msgs/msg/Pose[gz.msgs.Pose',
                '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
        ),

        # ---- odometry TF (Gazebo pose → odom → base_footprint) ---- #
        Node(
            package='sensor_fusion_nodes', executable='odom_tf', output='screen',
            parameters=[{
                'odom_frame':  'odom',
                'base_frame':  'base_footprint',
                'pose_topic':  '/model/robot_3d_lidar/pose',
                'use_sim_time': True,
            }],
        ),

        # ---- static TFs ---- #
        Node(
            package='tf2_ros', executable='static_transform_publisher', output='screen',
            arguments=['0', '0', '0.865', '0', '0', '0',
                       'base_footprint', 'lidar_3d_link'],
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='tf2_ros', executable='static_transform_publisher', output='screen',
            arguments=['0', '0', '1.075', '0', '0.28', '0',
                       'base_footprint', 'camera_link'],
            parameters=[{'use_sim_time': True}],
        ),

        # ---- 3-D lidar → ROS PointCloud2 + 2-D scan ---- #
        # scan_height_min/max are in the LIDAR LOCAL FRAME.
        # ±0.12 m captures near-horizontal beams that reliably hit walls at
        # 1–8 m range without the "above-wall" issue.
        Node(
            package='sensor_fusion_nodes',
            executable='gz_lidar3d_to_pointcloud',
            name='gz_lidar3d_to_pointcloud',
            output='screen',
            parameters=[{
                'gz_topic':          '/lidar_3d',
                'pointcloud_topic':  '/lidar/points',
                'scan_topic':        '/scan',
                'frame_id':          'lidar_3d_link',
                # Wider height band → more rings hit walls per scan →
                # better scan-match signal → less jitter.
                # At 5 m range, ±0.22 m stays well below the 2.6 m walls.
                'scan_height_min':   -0.22,
                'scan_height_max':    0.22,
                'point_stride':       1,
                'use_sim_time':       True,
            }],
        ),

        # ---- RViz ---- #
        Node(
            package='rviz2', executable='rviz2', name='rviz2',
            output='screen',
            arguments=['-d', rviz_cfg, '-f', 'map'],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(use_rviz),
        ),

        # ---- teleop in separate window ---- #
        teleop,

        # ---- delayed nodes ---- #
        occ_node,
        filler_node,
        slam_node,
    ])
