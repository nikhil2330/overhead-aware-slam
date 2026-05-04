"""Nav2 navigation evaluator with stuck-recovery.

Flow per goal:
  1. Send NavigateToPose goal.
  2. If robot doesn't move > STUCK_DIST_M for STUCK_TIMEOUT_S seconds → stuck.
  3. On stuck: cancel goal, backup, spin, retry once.
  4. If stuck again on retry → skip, mark collision.

Metrics
-------
success         – Nav2 returned SUCCEEDED
collision       – Nav2 failed AND robot moved >= COLLISION_DIST from start
costmap_failure – Nav2 failed AND robot moved  < COLLISION_DIST from start
time_s          – wall-clock seconds the goal was active (total, inc. retry)
path_dist_m     – net displacement start → end using true Gazebo pose
"""

import csv
import math
import pathlib
import random
import subprocess
import threading
import time

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

COLLISION_DIST  = 0.05   # moved >= this → collision
STUCK_DIST_M    = 0.05   # linear movement threshold (m) for stuck detection
STUCK_ANG_RAD   = 0.15   # angular movement threshold (rad ~8.6°) for stuck detection
STUCK_TIMEOUT_S = 9.0    # seconds with no linear OR angular movement → stuck
SAME_AREA_M     = 0.5    # if new stuck is within this of a previous stuck → same spot → skip


class NavEvaluator(Node):

    def __init__(self):
        super().__init__('nav_evaluator')

        self.declare_parameter('goals_file',     '')
        self.declare_parameter('method_name',    'unknown')
        self.declare_parameter('output_csv',     '/tmp/nav_eval_results.csv')
        self.declare_parameter('goal_timeout_s', 120.0)
        self.declare_parameter('trials',         1)
        self.declare_parameter('spawn_x',        1.2265)
        self.declare_parameter('spawn_y',       -0.9923)
        self.declare_parameter('world_name',     'default')

        self.method_name  = self.get_parameter('method_name').value
        self.output_csv   = self.get_parameter('output_csv').value
        self.goal_timeout = float(self.get_parameter('goal_timeout_s').value)
        self.spawn_x      = float(self.get_parameter('spawn_x').value)
        self.spawn_y      = float(self.get_parameter('spawn_y').value)
        self.world_name   = self.get_parameter('world_name').value

        goals_file  = self.get_parameter('goals_file').value
        self.goals  = self._load_goals(goals_file)
        self.trials = int(self.get_parameter('trials').value)

        self._lock     = threading.Lock()
        self._start_x  = None
        self._start_y  = None
        self._cur_x    = None
        self._cur_y    = None
        self._cur_yaw  = None
        self._active   = False
        self._results  = []

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Pose, '/model/robot/pose', self._pose_cb, 10)
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        threading.Thread(target=self._eval_loop, daemon=True).start()

        self.get_logger().info(
            f'NavEvaluator — method={self.method_name} '
            f'goals={len(self.goals)} trials={self.trials} '
            f'timeout={self.goal_timeout}s  stuck_timeout={STUCK_TIMEOUT_S}s'
        )

    def _pose_cb(self, msg: Pose):
        x   = msg.position.x
        y   = msg.position.y
        yaw = 2.0 * math.atan2(msg.orientation.z, msg.orientation.w)
        with self._lock:
            self._cur_x   = x
            self._cur_y   = y
            self._cur_yaw = yaw
            if self._active and self._start_x is None:
                self._start_x = x
                self._start_y = y

    def _eval_loop(self):
        self.get_logger().info('Waiting for Nav2 …')
        if not self._nav_client.wait_for_server(timeout_sec=60.0):
            self.get_logger().error('Nav2 not available.')
            return

        self.get_logger().info('Nav2 ready — starting evaluation.')

        for trial in range(1, self.trials + 1):
            self.get_logger().info(f'=== Trial {trial}/{self.trials} ===')
            for goal_cfg in self.goals:
                result = self._run_goal(goal_cfg, trial)
                self._results.append(result)
                self.get_logger().info(
                    f'  [{goal_cfg["label"]}] '
                    f'success={result["success"]} '
                    f'collision={result["collision"]} '
                    f'costmap_fail={result["costmap_failure"]} '
                    f'dist={result["path_dist_m"]:.2f}m '
                    f'time={result["time_s"]:.1f}s'
                )

        self._save_results()
        self.get_logger().info(f'Done → {self.output_csv}')
        self._reset_robot()

    # ------------------------------------------------------------------
    # Goal execution
    # ------------------------------------------------------------------

    def _run_goal(self, goal_cfg: dict, trial: int) -> dict:
        label = goal_cfg['label']
        gx    = float(goal_cfg['x'])
        gy    = float(goal_cfg['y'])
        yaw   = math.radians(float(goal_cfg.get('yaw', 0.0)))

        self.get_logger().info(f'  → {label}  ({gx:.2f}, {gy:.2f})')

        t0 = time.monotonic()

        # Set active before first attempt so pose_cb records start position
        with self._lock:
            self._active  = True
            self._start_x = None
            self._start_y = None

        success         = False
        collision_count = 0
        last_stuck      = None  # position of most recent stuck event

        # Keep retrying the same goal as long as each new stuck is in a different area.
        # Same area as the last stuck → give up on this goal.
        # Total wall-clock time still bounded by goal_timeout (120 s).
        while True:
            goal_msg    = self._make_goal(gx, gy, yaw)
            send_future = self._nav_client.send_goal_async(goal_msg)

            if not self._poll(send_future, timeout=10.0):
                break
            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                break

            result_future = goal_handle.get_result_async()
            success, reason = self._wait(result_future, goal_handle, t0)

            if success:
                break
            if reason != 'stuck':
                break  # timeout or clean plan failure

            # Stuck — compare with last stuck position
            with self._lock:
                cx_now, cy_now = self._cur_x, self._cur_y

            if last_stuck is not None:
                dist_to_last = math.hypot(
                    (cx_now or 0) - last_stuck[0],
                    (cy_now or 0) - last_stuck[1]
                )
                if dist_to_last < SAME_AREA_M:
                    print(f'  [same area] Stuck at same spot again — skipping goal', flush=True)
                    break

            # New area → collision, recover, keep trying
            collision_count += 1
            last_stuck = (cx_now, cy_now)
            print(f'  [COLLISION {collision_count}] Stuck at new area ({cx_now:.2f},{cy_now:.2f}) — recovering and retrying', flush=True)
            self._do_recovery()

        with self._lock:
            self._active = False

        return self._result(trial, label, gx, gy, success, t0, collision_count=collision_count)

    def _make_goal(self, gx: float, gy: float, yaw: float) -> NavigateToPose.Goal:
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp    = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = gx
        goal_msg.pose.pose.position.y = gy
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        goal_msg.pose.pose.orientation.z = sy
        goal_msg.pose.pose.orientation.w = cy
        return goal_msg

    def _wait(self, result_future, goal_handle, t0) -> tuple:
        """Wait for result_future. Returns (success: bool, reason: str).

        reason is one of: 'success', 'failed', 'timeout', 'stuck'
        """
        stuck_ref_x   = None
        stuck_ref_y   = None
        stuck_ref_yaw = None
        stuck_since   = None
        last_print    = t0

        while not result_future.done():
            time.sleep(0.2)
            now     = time.monotonic()
            elapsed = now - t0

            with self._lock:
                sx, sy   = self._start_x,  self._start_y
                cx, cy   = self._cur_x,    self._cur_y
                cyaw     = self._cur_yaw

            # Update stuck reference — reset timer if robot moved OR rotated
            if cx is not None and cyaw is not None:
                if stuck_ref_x is None:
                    stuck_ref_x, stuck_ref_y, stuck_ref_yaw = cx, cy, cyaw
                    stuck_since = now
                else:
                    moved_lin = math.hypot(cx - stuck_ref_x, cy - stuck_ref_y) > STUCK_DIST_M
                    ang_diff  = abs((cyaw - stuck_ref_yaw + math.pi) % (2 * math.pi) - math.pi)
                    moved_ang = ang_diff > STUCK_ANG_RAD
                    if moved_lin or moved_ang:
                        stuck_ref_x, stuck_ref_y, stuck_ref_yaw = cx, cy, cyaw
                        stuck_since = now

            # Periodic position print
            if now - last_print >= 3.0:
                dist_now = (math.hypot(cx - sx, cy - sy)
                            if (sx is not None and cx is not None) else 0.0)
                print(f'  [pos] elapsed={elapsed:.1f}s  '
                      f'cur=({cx:.2f},{cy:.2f})  '
                      f'displacement={dist_now:.3f}m', flush=True)
                last_print = now

            # Stuck check — only after 5 s so robot has time to start moving
            if (stuck_since is not None and elapsed > 5.0
                    and now - stuck_since > STUCK_TIMEOUT_S):
                print(f'  [stuck] Position unchanged for {STUCK_TIMEOUT_S:.0f}s — cancelling goal', flush=True)
                goal_handle.cancel_goal_async()
                self._poll(result_future, timeout=3.0)
                return False, 'stuck'

            # Hard timeout
            if elapsed > self.goal_timeout:
                self.get_logger().info('  Timeout — cancelling')
                goal_handle.cancel_goal_async()
                self._poll(result_future, timeout=3.0)
                return False, 'timeout'

        status  = result_future.result().status
        success = (status == GoalStatus.STATUS_SUCCEEDED)
        return success, 'success' if success else 'failed'

    def _do_recovery(self):
        """Backup 4 s then spin 5 s to escape a stuck position."""
        msg = Twist()

        # Backup
        msg.linear.x = -0.15
        t = time.monotonic()
        while time.monotonic() - t < 4.0:
            self._cmd_pub.publish(msg)
            time.sleep(0.05)

        self._cmd_pub.publish(Twist())
        time.sleep(0.3)

        # Spin — random direction each time
        msg = Twist()
        msg.angular.z = 1.0 * random.choice([-1, 1])
        t = time.monotonic()
        while time.monotonic() - t < 2.5:
            self._cmd_pub.publish(msg)
            time.sleep(0.05)

        self._cmd_pub.publish(Twist())
        time.sleep(0.3)

    # ------------------------------------------------------------------
    # Result / IO
    # ------------------------------------------------------------------

    def _result(self, trial, label, gx, gy, success, t0, collision_count=0):
        with self._lock:
            sx, sy = self._start_x, self._start_y
            cx, cy = self._cur_x,   self._cur_y

        dist = (math.hypot(cx - sx, cy - sy)
                if (sx is not None and cx is not None) else 0.0)

        collision       = collision_count > 0
        costmap_failure = (not success) and (not collision)

        outcome = ('SUCCESS' if success
                   else 'COLLISION' if collision
                   else 'COSTMAP_FAIL')
        s_str = f'({sx:.2f},{sy:.2f})' if sx is not None else 'None'
        c_str = f'({cx:.2f},{cy:.2f})' if cx is not None else 'None'
        print(f'  [result] {label}  outcome={outcome}  collisions={collision_count}  '
              f'dist={dist:.3f}m  time={time.monotonic()-t0:.1f}s  '
              f'start={s_str}  end={c_str}', flush=True)

        return {
            'method':          self.method_name,
            'trial':           trial,
            'goal_label':      label,
            'goal_x':          round(gx, 3),
            'goal_y':          round(gy, 3),
            'success':         int(success),
            'collision':       int(collision),
            'collision_count': collision_count,
            'costmap_failure': int(costmap_failure),
            'time_s':          round(time.monotonic() - t0, 2),
            'path_dist_m':     round(dist, 3),
        }

    def _poll(self, future, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    def _save_results(self):
        if not self._results:
            return
        path = pathlib.Path(self.output_csv).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fields       = list(self._results[0].keys())
        write_header = not path.exists()
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerows(self._results)
        self._print_summary()

    def _print_summary(self):
        rs = self._results
        n  = len(rs)
        if n == 0:
            return
        self.get_logger().info('--- Summary ---')
        self.get_logger().info(f'  Method:          {self.method_name}')
        self.get_logger().info(f'  Goals:           {n}')
        self.get_logger().info(f'  Success rate:    {100*sum(r["success"] for r in rs)/n:.1f}%')
        self.get_logger().info(f'  Collision rate:  {100*sum(r["collision"] for r in rs)/n:.1f}%  (total hits: {sum(r["collision_count"] for r in rs)})')
        self.get_logger().info(f'  Costmap failure: {100*sum(r["costmap_failure"] for r in rs)/n:.1f}%')
        self.get_logger().info(f'  Avg time:        {sum(r["time_s"] for r in rs)/n:.1f} s')
        self.get_logger().info(f'  Avg path dist:   {sum(r["path_dist_m"] for r in rs)/n:.2f} m')

    def _reset_robot(self):
        req = (f'name: "robot" '
               f'position: {{x: {self.spawn_x}, y: {self.spawn_y}, z: 0.05}} '
               f'orientation: {{x: 0, y: 0, z: 0, w: 1}}')
        try:
            subprocess.run(
                ['gz', 'service', '-s', f'/world/{self.world_name}/set_pose',
                 '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                 '--timeout', '2000', '--req', req],
                timeout=5, capture_output=True)
            self.get_logger().info('Robot returned to spawn.')
        except Exception as e:
            self.get_logger().warn(f'Teleport failed: {e}')

    def _load_goals(self, path: str) -> list:
        if not path:
            self.get_logger().error('goals_file not set.')
            return []
        p = pathlib.Path(path).expanduser()
        if not p.exists():
            self.get_logger().error(f'Goals file not found: {p}')
            return []
        with open(p) as f:
            data = yaml.safe_load(f)
        if 'goal_timeout_s' in data:
            self.goal_timeout = float(data['goal_timeout_s'])
        if 'trials' in data:
            self.trials = int(data['trials'])
        goals = data.get('goals', [])
        self.get_logger().info(f'Loaded {len(goals)} goals from {p}')
        return goals


def main():
    rclpy.init()
    node = NavEvaluator()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
