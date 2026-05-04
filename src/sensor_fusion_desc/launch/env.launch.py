from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg = FindPackageShare('sensor_fusion_desc')

    # Easy environment switch:
    #   ros2 launch sensor_fusion_desc env.launch.py world_name:=house.world
    #   ros2 launch sensor_fusion_desc env.launch.py world_name:=house2.world
    #   ros2 launch sensor_fusion_desc env.launch.py world_name:=house3.world
    world_name = LaunchConfiguration('world_name')
    world = PathJoinSubstitution([pkg, 'worlds', world_name])
    models = PathJoinSubstitution([pkg, 'models'])
    worlds = PathJoinSubstitution([pkg, 'worlds'])

    gz_launch = PathJoinSubstitution(
        [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world_name',
            default_value='house3.world',
            description='Gazebo world file to load from sensor_fusion_desc/worlds.',
        ),

        SetEnvironmentVariable('QT_QPA_PLATFORM', 'xcb'),

        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            [pkg, TextSubstitution(text=':'), models, TextSubstitution(text=':'), worlds]
        ),
        SetEnvironmentVariable(
            'IGN_GAZEBO_RESOURCE_PATH',
            [pkg, TextSubstitution(text=':'), models, TextSubstitution(text=':'), worlds]
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch),
            launch_arguments={
                'gz_args': [TextSubstitution(text='-r '), world],
                'on_exit_shutdown': 'true',
            }.items(),
        ),

    ])
