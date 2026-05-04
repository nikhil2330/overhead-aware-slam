"""
3-D LiDAR RTAB-Map mapping
===========================

Usage
-----
  ros2 launch sensor_fusion_run lidar3d_rtabmap.launch.py [world_name:=house3.world]

  Optional args
  -------------
  world_name         Gazebo world file in sensor_fusion_desc/worlds [house2.world]
  use_rviz           Launch RViz                                      [true]
  use_rtabmap_viz    Launch native RTAB-Map GUI                       [false]
  use_icp_odometry   Use RTAB-Map ICP odometry instead of Gazebo pose [false]

Save the filled 2-D map (separate terminal, while session is running):
  ros2 launch sensor_fusion_run save_map.launch.py \\
      output:=~/rtab_filled_map \\
      map_topic:=/pointcloud_occupancy_2d \\
      transient:=false

Save the raw RTAB-Map occupancy grid:
  ros2 launch sensor_fusion_run save_map.launch.py output:=~/rtab_raw_map transient:=true
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
    world_name        = LaunchConfiguration('world_name')
    use_rviz          = LaunchConfiguration('use_rviz')
    use_rtabmap_viz   = LaunchConfiguration('use_rtabmap_viz')
    use_icp_odometry  = LaunchConfiguration('use_icp_odometry')

    world   = PathJoinSubstitution([desc_pkg, 'worlds', world_name])
    models  = PathJoinSubstitution([desc_pkg, 'models'])
    worlds  = PathJoinSubstitution([desc_pkg, 'worlds'])

    gz_launch     = PathJoinSubstitution(
        [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py']
    )
    robot_sdf     = PathJoinSubstitution([desc_pkg, 'models', 'robot_3d_lidar', 'model.sdf'])
    rtab_params   = PathJoinSubstitution([desc_pkg, 'config', 'rtabmap_3d_lidar.yaml'])
    rviz_cfg      = PathJoinSubstitution([desc_pkg, 'config', 'rviz_lidar3d.rviz'])

    # ------------------------------------------------------------------ #
    # Teleop                                                               #
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
    # RTAB-Map ICP odometry (optional, use_icp_odometry:=true)            #
    # ------------------------------------------------------------------ #
    rtab_odom = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='rtabmap_odom',
                executable='icp_odometry',
                name='icp_odometry',
                output='screen',
                parameters=[rtab_params],
                remappings=[
                    ('scan_cloud',  '/lidar/points'),
                    ('odom',        '/icp_odom'),
                    ('odom_info',   '/icp_odom_info'),
                ],
                condition=IfCondition(use_icp_odometry),
            ),
        ],
    )

    # ------------------------------------------------------------------ #
    # RTAB-Map SLAM core                                                   #
    # ------------------------------------------------------------------ #
    # Odometry source:
    #   use_icp_odometry=false (default): Gazebo ground-truth /odom.
    #     The bridge already publishes /odom from gz.msgs.Odometry.
    #     odom_tf also provides the odom→base_footprint TF so the full
    #     TF chain exists from the first Gazebo pose message.
    #   use_icp_odometry=true: ICP odom node publishes /icp_odom.
    # Both cases use frame "odom" so the RViz fixed frame never changes.
    rtab_slam = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='rtabmap_slam',
                executable='rtabmap',
                name='rtabmap',
                output='screen',
                parameters=[rtab_params],
                remappings=[
                    ('scan_cloud',  '/lidar/points'),
                    # /pose_odom is published by odom_tf with get_clock().now()
                    # stamps — same time source as /lidar/points.  This ensures
                    # RTAB-Map's map/TF timestamps are consistent so the RViz
                    # message filter can always find TF for the map message.
                    ('odom',        '/pose_odom'),
                    ('map',         '/map'),
                ],
                arguments=['--delete_db_on_start'],
            ),
        ],
    )

    # ------------------------------------------------------------------ #
    # RTAB-Map native visualizer (optional)                               #
    # ------------------------------------------------------------------ #
    rtab_viz = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='rtabmap_viz',
                executable='rtabmap_viz',
                name='rtabmap_viz',
                output='screen',
                parameters=[rtab_params],
                remappings=[
                    ('scan_cloud', '/lidar/points'),
                    ('odom',       '/pose_odom'),
                ],
                condition=IfCondition(use_rtabmap_viz),
            ),
        ],
    )

    # ------------------------------------------------------------------ #
    # Pointcloud occupancy: permanent, morphologically-closed 2-D map.    #
    # decay_seconds=0  → voxels accumulate forever as robot explores.     #
    # morph_close_cells=13 @ 0.05 m/cell = 0.65 m closing radius.        #
    # This reliably fills full bed / table / bookshelf footprints.        #
    # ------------------------------------------------------------------ #
    occ_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='sensor_fusion_nodes',
                executable='pointcloud_occupancy',
                name='pointcloud_occupancy',
                output='screen',
                parameters=[{
                    'pointcloud_topic':         '/lidar/points',
                    'map_frame':               'odom',
                    'resolution':               0.05,
                    'z_resolution':             0.08,
                    'width_m':                  16.0,
                    'height_m':                 14.0,
                    'origin_x':                 -8.0,
                    'origin_y':                 -7.0,
                    'min_z':                    0.08,
                    'max_z':                    2.20,
                    'projection_min_z':         0.10,
                    'projection_max_z':         1.80,
                    'projection_padding_cells': 0,
                    'min_observations':         2,
                    'decay_seconds':            0.0,
                    'max_points_per_cloud':     30000,
                    'publish_period':           1.0,
                    'morph_close_cells':        7,
                    'occupancy_2d_topic':       '/pointcloud_occupancy_2d',
                    'occupancy_3d_topic':       '/pointcloud_occupancy_3d',
                    'occupied_voxels_topic':    '/occupied_voxels',
                    'use_sim_time':             True,
                }],
            ),
        ],
    )

    # ------------------------------------------------------------------ #
    # map_filler: applies morphological closing to RTAB-Map /map output.  #
    # ------------------------------------------------------------------ #
    filler_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='sensor_fusion_nodes',
                executable='map_filler',
                name='map_filler',
                output='screen',
                parameters=[{
                    'input_topic':  '/map',
                    'output_topic': '/map_filled',
                    'close_cells':  13,
                    'use_sim_time': True,
                }],
            ),
        ],
    )

    return LaunchDescription([
        # ---- args ---- #
        DeclareLaunchArgument(
            'world_name', default_value='house3.world',
            description='World file inside sensor_fusion_desc/worlds/',
        ),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Whether to launch RViz.',
        ),
        DeclareLaunchArgument(
            'use_rtabmap_viz', default_value='true',
            description='Whether to launch the native RTAB-Map GUI.',
        ),
        DeclareLaunchArgument(
            'use_icp_odometry', default_value='false',
            description=(
                'Use RTAB-Map ICP odometry. '
                'Default false uses Gazebo pose for stable mapping in sim.'
            ),
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

        # ---- Gazebo pose → odom → base_footprint TF + /pose_odom topic ---- #
        # odom_topic='/pose_odom': odom_tf also publishes nav_msgs/Odometry
        # stamped with get_clock().now() — the same time source used by
        # gz_lidar3d_to_pointcloud for the PointCloud2 stamp.  This keeps
        # RTAB-Map's approx_sync pairs and its map/TF publications at the
        # same timestamp, preventing the "timestamp earlier than TF cache"
        # drop in RViz.
        Node(
            package='sensor_fusion_nodes', executable='odom_tf', output='screen',
            parameters=[{
                'odom_frame':   'odom',
                'base_frame':   'base_footprint',
                'pose_topic':   '/model/robot_3d_lidar/pose',
                'odom_topic':   '/pose_odom',
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
        Node(
            package='sensor_fusion_nodes',
            executable='gz_lidar3d_to_pointcloud',
            name='gz_lidar3d_to_pointcloud',
            output='screen',
            parameters=[{
                'gz_topic':         '/lidar_3d',
                'pointcloud_topic': '/lidar/points',
                'scan_topic':       '/scan',
                'frame_id':         'lidar_3d_link',
                'scan_height_min':  -0.12,
                'scan_height_max':   0.12,
                'point_stride':      1,
                'use_sim_time':      True,
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

        # ---- teleop ---- #
        teleop,

        # ---- delayed nodes ---- #
        rtab_odom,
        rtab_slam,
        rtab_viz,
        occ_node,
        filler_node,
    ])
