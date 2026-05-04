from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    pkg = FindPackageShare('sensor_fusion_desc')

    world = PathJoinSubstitution([pkg, 'worlds', 'house3.world'])
    models = PathJoinSubstitution([pkg, 'models'])
    worlds = PathJoinSubstitution([pkg, 'worlds'])

    gz_launch = PathJoinSubstitution(
        [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py']
    )

    robot_sdf = PathJoinSubstitution([pkg, 'models', 'robot', 'model.sdf'])
    slam_params = PathJoinSubstitution([pkg, 'config', 'slam_toolbox.yaml'])
    object_location_params = PathJoinSubstitution([pkg, 'config', 'object_location.yaml'])

    slam_delayed = TimerAction(
        period=6.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('slam_toolbox'),
                        'launch',
                        'online_async_launch.py'
                    ])
                ),
                launch_arguments={
                    'use_sim_time': 'true',
                    'slam_params_file': slam_params,
                }.items(),
            )
        ]
    )

    return LaunchDescription([
        SetEnvironmentVariable('QT_QPA_PLATFORM', 'xcb'),

        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            [pkg, TextSubstitution(text=':'), models, TextSubstitution(text=':'), worlds]
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch),
            launch_arguments={
                'gz_args': [TextSubstitution(text='-r '), world],
                'on_exit_shutdown': 'true',
            }.items(),
        ),

        Node(
            package='ros_gz_sim',
            executable='create',
            output='screen',
            arguments=[
                '-name', 'robot',
                '-file', robot_sdf,
                '-x', '1.2265', '-y', '-0.9923', '-z', '0.01', '-Y', '3.1164'
            ],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', PathJoinSubstitution([pkg, 'config', 'rviz2.rviz'])],
            parameters=[{'use_sim_time': True}],
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            output='screen',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/model/robot/pose@geometry_msgs/msg/Pose[gz.msgs.Pose',
                '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
        ),

        Node(
            package='sensor_fusion_nodes',
            executable='odom_tf',
            output='screen',
            parameters=[{
                'odom_frame': 'odom',
                'base_frame': 'base_footprint',
                'pose_topic': '/model/robot/pose',
                'use_sim_time': True,
            }],
        ),

        # Node(
        #     package='sensor_fusion_nodes',
        #     executable='object_detection',
        #     output='screen',
        #     parameters=[{
        #         'score_thresh': 0.58,
        #         'label_filter': 'chair',
        #         'use_sim_time': True,
        #     }],
        # ),

        Node(
            package='sensor_fusion_nodes',
            executable='object_detction2',
            output='screen',
            parameters=[{
                'score_thresh': 0.58,
                'allowed_labels': 'chair,table',
                'show_window': True,
                'use_sim_time': True,
            }],
        ),

        # Node(
        #     package='sensor_fusion_nodes',
        #     executable='yolo_detection',
        #     name='yolo_detection',
        #     output='screen',
        #     parameters=[{
        #         'score_thresh': 0.58,
        #         'allowed_labels': 'chair,table',
        #         'show_window': True,
        #         'use_sim_time': True,
        #     }],
        # ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0.25', '0', '0', '0', 'base_footprint', 'lidar_link'],
            parameters=[{'use_sim_time': True}],
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0.1075', '0', '0.28', '0', 'base_footprint', 'camera_link'],
            parameters=[{'use_sim_time': True}],
        ),

        Node(
            package='sensor_fusion_nodes',
            executable='object_location',
            output='screen',
            parameters=[
                object_location_params,
                {'use_sim_time': True},
            ],
        ),

        Node(
            package='sensor_fusion_nodes',
            executable='occupancy_map',
            output='screen',
            parameters=[{
                'use_sim_time': True,
            }],
        ),

        

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'lidar_link', 'robot/lidar_link/lidar'],
            parameters=[{'use_sim_time': True}],
        ),

        slam_delayed,
    ])
