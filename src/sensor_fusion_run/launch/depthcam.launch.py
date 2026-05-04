from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    desc_pkg = FindPackageShare('sensor_fusion_desc')

    world = PathJoinSubstitution([desc_pkg, 'worlds', 'house2.world'])
    models = PathJoinSubstitution([desc_pkg, 'models'])
    worlds = PathJoinSubstitution([desc_pkg, 'worlds'])

    gz_launch = PathJoinSubstitution(
        [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py']
    )
    robot_sdf = PathJoinSubstitution([desc_pkg, 'models', 'robot_rgbd', 'model.sdf'])

    return LaunchDescription([
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            [desc_pkg, TextSubstitution(text=':'), models, TextSubstitution(text=':'), worlds]
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
                '-name', 'robot_rgbd',
                '-file', robot_sdf,
                '-x', '1.2265', '-y', '-0.9923', '-z', '0.01', '-Y', '3.1164'
            ],
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
                '/model/robot_rgbd/pose@geometry_msgs/msg/Pose[gz.msgs.Pose',
                '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
            remappings=[
                ('/camera/image', '/camera/image_raw'),
            ],
        ),

        Node(
            package='sensor_fusion_nodes',
            executable='rgbd_cam_view',
            output='screen',
            parameters=[{
                'rgb_topic': '/camera/image_raw',
                'depth_topic': '/camera/depth_image',
                'window_name': 'RGB-D Camera View',
                'max_depth_m': 10.0,
                'use_sim_time': True,
            }],
        ),

        Node(
            package='sensor_fusion_nodes',
            executable='odom_tf',
            output='screen',
            parameters=[{
                'odom_frame': 'odom',
                'base_frame': 'base_footprint',
                'pose_topic': '/model/robot_rgbd/pose',
                'use_sim_time': True,
            }],
        ),

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
            package='tf2_ros',
            executable='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'lidar_link', 'robot_rgbd/lidar_link/lidar'],
            parameters=[{'use_sim_time': True}],
        ),
    ])
