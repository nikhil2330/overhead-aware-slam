import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _as_bool(value):
    return value.strip().lower() in ("1", "true", "yes", "on")


def _resolve_file(path_text):
    path = Path(os.path.expanduser(path_text))
    if not path.is_absolute():
        path = Path(os.getcwd()) / path
    return path.resolve()


def _float_launch_arg(context, name):
    value = context.perform_substitution(LaunchConfiguration(name)).strip()
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric, got '{value}'") from exc


def _origin_value(raw_origin):
    origin = raw_origin.strip()
    if origin.startswith("[") and origin.endswith("]"):
        return origin

    parts = [part.strip() for part in origin.split(",")]
    if len(parts) != 3:
        raise RuntimeError(
            "map_origin must be three comma-separated values, for example "
            "'-5.958,-4.195,0.0'"
        )
    return f"[{parts[0]}, {parts[1]}, {parts[2]}]"


def _map_yaml_for_nav2(context):
    map_arg = context.perform_substitution(LaunchConfiguration("map")).strip()
    if not map_arg:
        raise RuntimeError("map launch argument cannot be empty")

    map_path = _resolve_file(map_arg)
    if not map_path.exists():
        raise RuntimeError(f"Map file does not exist: {map_path}")

    if map_path.suffix.lower() in (".yaml", ".yml"):
        return str(map_path)

    if map_path.suffix.lower() not in (".pgm", ".png"):
        raise RuntimeError(
            "Nav2 maps must be a ROS map .yaml file, or a .pgm/.png image "
            "with map_resolution and map_origin launch arguments."
        )

    resolution = context.perform_substitution(
        LaunchConfiguration("map_resolution")
    ).strip()
    origin = _origin_value(
        context.perform_substitution(LaunchConfiguration("map_origin"))
    )
    occupied_thresh = context.perform_substitution(
        LaunchConfiguration("map_occupied_thresh")
    ).strip()
    free_thresh = context.perform_substitution(
        LaunchConfiguration("map_free_thresh")
    ).strip()

    ros_home = Path(os.environ.get("ROS_HOME", "~/.ros")).expanduser()
    output_dir = ros_home / "sensor_fusion_nav2_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = output_dir / f"{map_path.stem}_nav2.yaml"
    image_path = str(map_path).replace("'", "''")
    yaml_path.write_text(
        "\n".join(
            [
                f"image: '{image_path}'",
                "mode: trinary",
                f"resolution: {resolution}",
                f"origin: {origin}",
                "negate: 0",
                f"occupied_thresh: {occupied_thresh}",
                f"free_thresh: {free_thresh}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(yaml_path)


def _launch_nav2_stack(context, *args, **kwargs):
    map_yaml = _map_yaml_for_nav2(context)
    params_file = _resolve_file(
        context.perform_substitution(LaunchConfiguration("params_file"))
    )
    if not params_file.exists():
        raise RuntimeError(f"Nav2 params file does not exist: {params_file}")

    use_sim_time = _as_bool(
        context.perform_substitution(LaunchConfiguration("use_sim_time"))
    )
    autostart = _as_bool(
        context.perform_substitution(LaunchConfiguration("autostart"))
    )
    log_level = context.perform_substitution(LaunchConfiguration("log_level"))

    # map_server + navigation nodes all managed by one lifecycle manager.
    # AMCL is intentionally omitted: in Gazebo simulation, odom_tf publishes
    # odom→base_footprint using the Gazebo ground-truth pose, and we publish
    # a static map→odom identity transform because the robot always spawns at
    # the same position as when the evaluation map was recorded.
    # smoother_server omitted — it hangs on lifecycle change_state in Jazzy
    # and blocks all subsequent nodes from activating.  Path smoothing is
    # optional; NavFn planner output is good enough for evaluation.
    lifecycle_navigation_nodes = [
        "map_server",
        "controller_server",
        "planner_server",
        "behavior_server",
        "velocity_smoother",
        "bt_navigator",
        "waypoint_follower",
    ]

    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
    common_params = [str(params_file), {"use_sim_time": use_sim_time}]
    nav_node_kwargs = {
        "output": "screen",
        "arguments": ["--ros-args", "--log-level", log_level],
    }

    return [
        # map→odom: identity because Gazebo ground-truth odom and the map
        # share the same world origin when the robot starts at the spawn pose.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_odom",
            arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
            parameters=[{"use_sim_time": use_sim_time}],
            output="screen",
        ),
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            parameters=common_params + [{"yaml_filename": map_yaml}],
            remappings=remappings,
            **nav_node_kwargs,
        ),
        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            parameters=common_params,
            remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
            **nav_node_kwargs,
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            parameters=common_params,
            remappings=remappings,
            **nav_node_kwargs,
        ),
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            parameters=common_params,
            remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
            **nav_node_kwargs,
        ),
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            parameters=common_params,
            remappings=remappings,
            **nav_node_kwargs,
        ),
        Node(
            package="nav2_waypoint_follower",
            executable="waypoint_follower",
            name="waypoint_follower",
            parameters=common_params,
            remappings=remappings,
            **nav_node_kwargs,
        ),
        Node(
            package="nav2_velocity_smoother",
            executable="velocity_smoother",
            name="velocity_smoother",
            parameters=common_params,
            remappings=remappings
            + [("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")],
            **nav_node_kwargs,
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            parameters=[
                {"use_sim_time": use_sim_time},
                {"autostart": autostart},
                {"node_names": lifecycle_navigation_nodes},
            ],
            arguments=["--ros-args", "--log-level", log_level],
            output="screen",
        ),
    ]




def generate_launch_description():
    desc_pkg = FindPackageShare("sensor_fusion_desc")

    world_name = LaunchConfiguration("world_name")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")

    world = PathJoinSubstitution([desc_pkg, "worlds", world_name])
    models = PathJoinSubstitution([desc_pkg, "models"])
    worlds = PathJoinSubstitution([desc_pkg, "worlds"])
    robot_sdf = PathJoinSubstitution([desc_pkg, "models", "robot", "model.sdf"])
    default_map = PathJoinSubstitution([desc_pkg, "maps", "lidar_map_v1.yaml"])
    default_params = PathJoinSubstitution([desc_pkg, "config", "nav2_params.yaml"])
    rviz_config = PathJoinSubstitution([desc_pkg, "config", "nav2.rviz"])

    gz_launch = PathJoinSubstitution(
        [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
    )

    nav2_delayed = TimerAction(
        period=3.0,
        actions=[
            OpaqueFunction(function=_launch_nav2_stack),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world_name",
                default_value="house2.world",
                description="Gazebo world file from sensor_fusion_desc/worlds.",
            ),
            DeclareLaunchArgument(
                "map",
                default_value=default_map,
                description=(
                    "Full path to a ROS map YAML file. A raw .pgm/.png also works "
                    "when map_resolution and map_origin are correct."
                ),
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Full path to the Nav2 parameters YAML.",
            ),
            DeclareLaunchArgument(
                "map_resolution",
                default_value="0.05",
                description="Resolution used when map points directly to a .pgm/.png.",
            ),
            DeclareLaunchArgument(
                "map_origin",
                default_value="-5.958,-4.195,0.0",
                description="Origin used when map points directly to a .pgm/.png.",
            ),
            DeclareLaunchArgument(
                "map_occupied_thresh",
                default_value="0.65",
                description="Occupied threshold for auto-generated map YAML files.",
            ),
            DeclareLaunchArgument(
                "map_free_thresh",
                default_value="0.196",
                description="Free threshold for auto-generated map YAML files.",
            ),
            DeclareLaunchArgument(
                "spawn_x",
                default_value="1.2265",
                description="Initial Gazebo robot x pose in the map/world frame.",
            ),
            DeclareLaunchArgument(
                "spawn_y",
                default_value="-0.9923",
                description="Initial Gazebo robot y pose in the map/world frame.",
            ),
            DeclareLaunchArgument(
                "spawn_z",
                default_value="0.01",
                description="Initial Gazebo robot z pose.",
            ),
            DeclareLaunchArgument(
                "spawn_yaw",
                default_value="3.1164",
                description="Initial Gazebo robot yaw in radians.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use Gazebo simulation clock.",
            ),
            DeclareLaunchArgument(
                "autostart",
                default_value="true",
                description="Automatically configure and activate Nav2 lifecycle nodes.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Start RViz with the Nav2 display config.",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="ROS log level for Nav2 nodes.",
            ),
            SetEnvironmentVariable(
                "GZ_SIM_RESOURCE_PATH",
                [
                    desc_pkg,
                    TextSubstitution(text=":"),
                    models,
                    TextSubstitution(text=":"),
                    worlds,
                ],
            ),
            SetEnvironmentVariable(
                "IGN_GAZEBO_RESOURCE_PATH",
                [
                    desc_pkg,
                    TextSubstitution(text=":"),
                    models,
                    TextSubstitution(text=":"),
                    worlds,
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(gz_launch),
                launch_arguments={
                    "gz_args": [TextSubstitution(text="-r "), world],
                    "on_exit_shutdown": "true",
                }.items(),
            ),
            Node(
                package="ros_gz_sim",
                executable="create",
                output="screen",
                arguments=[
                    "-name",
                    "robot",
                    "-file",
                    robot_sdf,
                    "-x",
                    LaunchConfiguration("spawn_x"),
                    "-y",
                    LaunchConfiguration("spawn_y"),
                    "-z",
                    LaunchConfiguration("spawn_z"),
                    "-Y",
                    LaunchConfiguration("spawn_yaw"),
                ],
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                output="screen",
                arguments=[
                    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                    "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                    "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
                    "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                    "/model/robot/pose@geometry_msgs/msg/Pose[gz.msgs.Pose",
                ],
            ),
            Node(
                package="sensor_fusion_nodes",
                executable="odom_tf",
                output="screen",
                parameters=[
                    {
                        "odom_frame": "odom",
                        "base_frame": "base_footprint",
                        "pose_topic": "/model/robot/pose",
                        "use_sim_time": use_sim_time,
                    }
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                output="screen",
                arguments=[
                    "0",
                    "0",
                    "0.25",
                    "0",
                    "0",
                    "0",
                    "base_footprint",
                    "lidar_link",
                ],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                output="screen",
                arguments=[
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "lidar_link",
                    "robot/lidar_link/lidar",
                ],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config, "-f", "map"],
                parameters=[{"use_sim_time": use_sim_time}],
                condition=IfCondition(use_rviz),
            ),
            nav2_delayed,
        ]
    )
