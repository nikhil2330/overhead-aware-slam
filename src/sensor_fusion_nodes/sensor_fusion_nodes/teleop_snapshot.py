import os
import select
import sys
import termios
import threading
import tty

import cv2

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image


DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        '..',
        'snapshots'
    )
)


class TeleopSnapshot(Node):

    def __init__(self):
        super().__init__('teleop_snapshot')

        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.motion_lock = threading.Lock()

        self.latest_frame = None
        self.latest_stamp = None
        self.latest_key = None
        self.last_saved_key = None
        self.saved_count = 0
        self.preview_ok = False

        self.linear_cmd = 0.0
        self.angular_cmd = 0.0
        self.quit_requested = False

        self.init_params()
        self.init_output_dir()
        self.init_ros()
        self.start_keyboard_thread()

    def init_params(self):
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('output_dir', DEFAULT_OUTPUT_DIR)
        self.declare_parameter('image_extension', 'jpg')
        self.declare_parameter('show_preview', True)
        self.declare_parameter('preview_window_name', 'Camera Preview')
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('angular_speed', 1.0)
        self.declare_parameter('publish_rate_hz', 10.0)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.output_dir = os.path.abspath(
            str(self.get_parameter('output_dir').value)
        )

        image_extension = str(self.get_parameter('image_extension').value)
        self.image_extension = image_extension.lower().lstrip('.')
        if self.image_extension not in ('jpg', 'jpeg', 'png'):
            self.get_logger().warn(
                f'unsupported image_extension={image_extension}, using jpg'
            )
            self.image_extension = 'jpg'

        self.show_preview = bool(self.get_parameter('show_preview').value)
        self.preview_window_name = str(
            self.get_parameter('preview_window_name').value
        )
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.angular_speed = float(self.get_parameter('angular_speed').value)
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.publish_period = 1.0 / max(1.0, publish_rate_hz)

    def init_output_dir(self):
        self.image_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)

        existing = [
            name for name in os.listdir(self.image_dir)
            if name.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.saved_count = len(existing)

    def init_ros(self):
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            10
        )
        self.publish_timer = self.create_timer(
            self.publish_period,
            self.publish_cmd
        )
        self.preview_timer = self.create_timer(
            0.03,
            self.update_preview
        )

        if self.show_preview:
            self.init_preview_window()

        self.get_logger().info(
            f'teleop_snapshot image_topic={self.image_topic} '
            f'cmd_vel_topic={self.cmd_vel_topic}'
        )
        self.get_logger().info(f'saving images to {self.image_dir}')
        if self.show_preview:
            self.get_logger().info(
                f'showing live preview in window "{self.preview_window_name}"'
            )
        self.get_logger().info(
            'controls: w forward, x back, a left, d right, s stop, Enter save, q quit'
        )

    def init_preview_window(self):
        try:
            cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
            self.preview_ok = True
        except Exception as exc:
            self.preview_ok = False
            self.get_logger().warn(
                f'preview window disabled: {exc}'
            )

    def stamp_key(self, stamp):
        return f'{stamp.sec}_{stamp.nanosec:09d}'

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        with self.frame_lock:
            self.latest_frame = frame
            self.latest_stamp = msg.header.stamp
            self.latest_key = self.stamp_key(msg.header.stamp)

    def set_motion(self, linear, angular):
        with self.motion_lock:
            self.linear_cmd = linear
            self.angular_cmd = angular

    def publish_cmd(self):
        twist = Twist()

        with self.motion_lock:
            twist.linear.x = self.linear_cmd
            twist.angular.z = self.angular_cmd

        self.cmd_pub.publish(twist)

    def make_preview_frame(self, frame):
        preview = frame.copy()
        lines = [
            'w/x move  a/d turn  s stop',
            'Enter save image  q quit',
            f'saved: {self.saved_count}',
        ]

        for i, text in enumerate(lines):
            y = 30 + i * 28
            cv2.putText(
                preview,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        return preview

    def update_preview(self):
        if not self.show_preview or not self.preview_ok:
            return

        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        try:
            preview = self.make_preview_frame(frame)
            cv2.imshow(self.preview_window_name, preview)
            cv2.waitKey(1)
        except Exception as exc:
            self.preview_ok = False
            self.get_logger().warn(
                f'preview update failed, disabling window: {exc}'
            )

    def save_snapshot(self):
        with self.frame_lock:
            if self.latest_frame is None or self.latest_stamp is None:
                self.get_logger().info('[SKIP] no image received yet')
                return

            if self.latest_key == self.last_saved_key:
                self.get_logger().info('[SKIP] latest frame already saved')
                return

            frame = self.latest_frame.copy()
            stamp = self.latest_stamp
            key = self.latest_key

        stem = (
            f'frame_{self.saved_count:06d}_'
            f'{stamp.sec}_{stamp.nanosec:09d}'
        )
        image_path = os.path.join(
            self.image_dir,
            f'{stem}.{self.image_extension}'
        )

        cv2.imwrite(image_path, frame)
        self.last_saved_key = key
        self.saved_count += 1

        self.get_logger().info(f'[SAVE] {image_path}')

    def process_key(self, key):
        if key in ('w', 'W'):
            self.set_motion(self.linear_speed, 0.0)
        elif key in ('x', 'X'):
            self.set_motion(-self.linear_speed, 0.0)
        elif key in ('a', 'A'):
            self.set_motion(0.0, self.angular_speed)
        elif key in ('d', 'D'):
            self.set_motion(0.0, -self.angular_speed)
        elif key in ('s', 'S', ' '):
            self.set_motion(0.0, 0.0)
        elif key in ('\r', '\n'):
            self.save_snapshot()
        elif key in ('q', 'Q', '\x03'):
            self.quit_requested = True
            self.set_motion(0.0, 0.0)
            self.publish_cmd()
            rclpy.shutdown()

    def start_keyboard_thread(self):
        if not sys.stdin.isatty():
            self.get_logger().warn(
                'stdin is not a tty, keyboard teleop is disabled'
            )
            return

        keyboard_thread = threading.Thread(
            target=self.keyboard_loop,
            daemon=True
        )
        keyboard_thread.start()

    def keyboard_loop(self):
        try:
            settings = termios.tcgetattr(sys.stdin)
        except Exception as exc:
            self.get_logger().warn(f'keyboard setup failed: {exc}')
            return

        try:
            tty.setraw(sys.stdin.fileno())

            while rclpy.ok() and not self.quit_requested:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue

                key = sys.stdin.read(1)
                self.process_key(key)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    def destroy_node(self):
        self.set_motion(0.0, 0.0)
        self.publish_cmd()
        if self.preview_ok:
            cv2.destroyWindow(self.preview_window_name)
        super().destroy_node()


def main():
    rclpy.init()
    node = TeleopSnapshot()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
