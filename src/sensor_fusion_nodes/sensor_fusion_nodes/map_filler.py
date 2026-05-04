import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

try:
    from scipy.ndimage import binary_closing as _scipy_binary_closing
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


class MapFiller(Node):
    """Subscribe to an OccupancyGrid, apply morphological closing to fill
    unknown interior cells, and republish on a new topic."""

    def __init__(self):
        super().__init__('map_filler')

        self.declare_parameter('input_topic', '/map')
        self.declare_parameter('output_topic', '/map_filled')
        self.declare_parameter('close_cells', 9)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.close_cells = max(0, int(self.get_parameter('close_cells').value))

        # Publisher: volatile is fine — we republish every incoming map.
        pub_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Subscriber: TRANSIENT_LOCAL so we receive the latched last-message
        # that slam_toolbox (>=2.8) and rtabmap publish on /map.
        # Also compatible with VOLATILE publishers (we just don't get the
        # retained copy, but new publications arrive normally).
        sub_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pub = self.create_publisher(OccupancyGrid, self.output_topic, pub_qos)
        self.sub = self.create_subscription(
            OccupancyGrid, self.input_topic, self.map_callback, sub_qos
        )

        self.get_logger().info(
            f'MapFiller: {self.input_topic} → {self.output_topic} '
            f'(close_cells={self.close_cells}, scipy={_HAVE_SCIPY})'
        )

    def map_callback(self, msg):
        try:
            self._process(msg)
        except Exception as exc:
            self.get_logger().error(f'map_filler error: {exc}')

    def _process(self, msg):
        w = msg.info.width
        h = msg.info.height

        if w == 0 or h == 0:
            try:
                self.pub.publish(msg)
            except RCLError:
                pass
            return

        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)

        if self.close_cells > 1:
            occupied = grid == 100
            if np.any(occupied):
                struct = np.ones((self.close_cells, self.close_cells), dtype=bool)
                if _HAVE_SCIPY:
                    closed = _scipy_binary_closing(occupied, structure=struct)
                else:
                    closed = self._numpy_close(occupied, self.close_cells)

                # Only fill UNKNOWN (-1) cells, leave FREE (0) cells as free.
                fill = closed & (grid == -1)
                grid[fill] = 100

        out = OccupancyGrid()
        out.header = msg.header
        out.info = msg.info
        out.data = grid.flatten().tolist()
        try:
            self.pub.publish(out)
        except RCLError:
            pass

    @staticmethod
    def _numpy_close(mask, n):
        half = n // 2
        dilated = np.zeros_like(mask)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                s = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
                if dy > 0:
                    s[:dy, :] = False
                elif dy < 0:
                    s[dy:, :] = False
                if dx > 0:
                    s[:, :dx] = False
                elif dx < 0:
                    s[:, dx:] = False
                dilated |= s
        eroded = np.ones_like(dilated)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                s = np.roll(np.roll(dilated, dy, axis=0), dx, axis=1)
                if dy > 0:
                    s[:dy, :] = True
                elif dy < 0:
                    s[dy:, :] = True
                if dx > 0:
                    s[:, :dx] = True
                elif dx < 0:
                    s[:, dx:] = True
                eroded &= s
        return eroded


def main():
    rclpy.init()
    node = MapFiller()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
