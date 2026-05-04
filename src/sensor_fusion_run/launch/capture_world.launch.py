from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('sensor_fusion_desc')

    world = PathJoinSubstitution([pkg, 'worlds', 'house2.world'])
    models = PathJoinSubstitution([pkg, 'models'])
    worlds = PathJoinSubstitution([pkg, 'worlds'])

    gz_launch = PathJoinSubstitution(
        [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py']
    )
    robot_sdf = PathJoinSubstitution([pkg, 'models', 'robot', 'model.sdf'])

    return LaunchDescription([
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
            package='ros_gz_bridge',
            executable='parameter_bridge',
            output='screen',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            ],
        ),
    ])
