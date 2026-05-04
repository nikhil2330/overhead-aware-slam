"""Record navigation waypoints by clicking in RViz2.

Run this WHILE the nav2_static_map.launch.py session is open.
In RViz2, use the  ➤ Nav2 Goal  tool (the arrow-and-dot button in the toolbar)
and click+drag on the map to set a pose. Each click is recorded here.

Usage:
  source install/setup.bash
  python3 tools/record_waypoints.py house2        # saves to eval_goals_house2.yaml
  python3 tools/record_waypoints.py house3        # saves to eval_goals_house3.yaml
  python3 tools/record_waypoints.py myname        # saves to eval_goals_myname.yaml

Press Ctrl-C when you are done clicking.
"""

import math
import pathlib
import sys

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import yaml


class WaypointRecorder(Node):

    def __init__(self, world_name: str):
        super().__init__('waypoint_recorder')
        self.world_name = world_name
        self.waypoints = []

        # Catch goals from standard RViz2 "2D Goal Pose" tool (G key)
        self.sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self._pose_cb,
            10,
        )

        # Also catch goals sent directly via the Nav2 Panel / Nav2 Goal tool
        # by acting as an action server (intercepts before Nav2 bt_navigator)
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            self._action_cb,
        )

        self.get_logger().info(
            '\n=== Waypoint Recorder ===\n'
            'TWO ways to record a waypoint:\n'
            '  1. Press G in RViz2 → click+drag on map  (2D Goal Pose tool)\n'
            '  2. Use the Nav2 Panel goal button\n'
            'Each click prints here. Ctrl-C when done.'
        )

    def _action_cb(self, goal_handle: ServerGoalHandle):
        """Intercept NavigateToPose action goals (from Nav2 Panel / plugin)."""
        pose = goal_handle.request.pose
        self._record(pose.pose.position.x, pose.pose.position.y,
                     pose.pose.orientation.z, pose.pose.orientation.w)
        # Immediately abort so Nav2's real bt_navigator can handle it
        goal_handle.abort()
        return NavigateToPose.Result()

    def _pose_cb(self, msg: PoseStamped):
        """Intercept /goal_pose topic (from RViz2 2D Goal Pose tool)."""
        self._record(msg.pose.position.x, msg.pose.position.y,
                     msg.pose.orientation.z, msg.pose.orientation.w)

    def _record(self, x: float, y: float, qz: float, qw: float):
        # Deduplicate: ignore if within 0.05 m of the last recorded waypoint
        if self.waypoints:
            last = self.waypoints[-1]
            if math.hypot(x - last['x'], y - last['y']) < 0.05:
                return
        yaw   = math.degrees(2.0 * math.atan2(qz, qw))
        idx   = len(self.waypoints) + 1
        label = f'wp_{idx:02d}'
        entry = {'x': round(x, 3), 'y': round(y, 3),
                 'yaw': round(yaw, 1), 'label': label}
        self.waypoints.append(entry)
        self.get_logger().info(
            f'  [{idx:2d}] {label}  x={x:.3f}  y={y:.3f}  yaw={yaw:.1f}°'
        )

    def _cb(self, msg: PoseStamped):
        pass  # kept for compatibility

    def save(self):
        out_dir = pathlib.Path(__file__).parent.parent / \
                  'src' / 'sensor_fusion_desc' / 'config'
        out_path = out_dir / f'eval_goals_{self.world_name}.yaml'

        data = {
            'goals':              self.waypoints,
            'trials':             3,
            'collision_range_m':  0.20,
            'goal_timeout_s':     60.0,
            'stuck_timeout_s':    12.0,
        }
        with open(out_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f'\nSaved {len(self.waypoints)} waypoints → {out_path}')
        print('\nGoals YAML:')
        for wp in self.waypoints:
            print(f"  - {{x: {wp['x']:.3f}, y: {wp['y']:.3f}, "
                  f"yaw: {wp['yaw']:.1f}, label: \"{wp['label']}\"}}")


def main():
    world = sys.argv[1] if len(sys.argv) > 1 else 'house2'
    rclpy.init()
    node = WaypointRecorder(world)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
