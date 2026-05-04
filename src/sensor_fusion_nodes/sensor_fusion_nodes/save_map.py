"""Save any OccupancyGrid topic to a standard PGM + YAML map pair.

Usage
-----
  ros2 run sensor_fusion_nodes save_map <topic> <output_prefix>

Examples
--------
  # SLAM / RTAB-Map filled map (Transient Local — session must be running):
  ros2 run sensor_fusion_nodes save_map /map_filled  ~/slam_map_v1

  # Raw SLAM / RTAB-Map map:
  ros2 run sensor_fusion_nodes save_map /map         ~/slam_raw_v1

  # Pointcloud accumulation map (Volatile — session must be running):
  ros2 run sensor_fusion_nodes save_map /pointcloud_occupancy_2d ~/occ_map_v1
"""

import pathlib
import sys

import rclpy
import yaml
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)


class MapSaver(Node):

    def __init__(self, topic: str, output: str):
        super().__init__('map_saver')
        self._output = pathlib.Path(output).expanduser()
        self._topic = topic
        self._saved = False

        # Try TRANSIENT_LOCAL so we receive the retained message immediately
        # even if the topic was last published several seconds ago.
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._sub = self.create_subscription(OccupancyGrid, topic, self._cb, qos)
        self.get_logger().info(f'Waiting for map on {topic} ...')

        # Give DDS 8 seconds to discover the publisher and deliver the
        # retained message — much safer than nav2's default 2 s timeout.
        self._timer = self.create_timer(8.0, self._on_timeout)

    def _cb(self, msg: OccupancyGrid):
        if self._saved:
            return
        self._saved = True
        try:
            self._write(msg)
        except Exception as exc:
            self.get_logger().error(f'Write failed: {exc}')
            sys.exit(1)
        sys.exit(0)

    def _on_timeout(self):
        if not self._saved:
            self.get_logger().error(
                f'No message received on {self._topic} within 8 s.\n'
                '  → Make sure the mapping session is still running.\n'
                '  → Or check: ros2 topic info ' + self._topic
            )
            sys.exit(1)

    def _write(self, msg: OccupancyGrid):
        w   = msg.info.width
        h   = msg.info.height
        res = msg.info.resolution
        ox  = msg.info.origin.position.x
        oy  = msg.info.origin.position.y

        pgm_path  = self._output.with_suffix('.pgm')
        yaml_path = self._output.with_suffix('.yaml')

        pgm_path.parent.mkdir(parents=True, exist_ok=True)

        # Write PGM (rows stored bottom-up in OccupancyGrid, top-down in PGM)
        with open(pgm_path, 'wb') as f:
            f.write(f'P5\n{w} {h}\n255\n'.encode())
            for row in range(h - 1, -1, -1):
                row_bytes = bytearray()
                for col in range(w):
                    v = msg.data[row * w + col]
                    if v < 0:
                        row_bytes.append(205)   # unknown  → grey
                    elif v >= 65:
                        row_bytes.append(0)     # occupied → black
                    else:
                        row_bytes.append(254)   # free     → white
                f.write(row_bytes)

        # Write YAML (standard nav2 / ROS map format)
        meta = {
            'image':           pgm_path.name,
            'resolution':      float(res),
            'origin':          [float(ox), float(oy), 0.0],
            'negate':          0,
            'occupied_thresh': 0.65,
            'free_thresh':     0.25,
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(meta, f, default_flow_style=False)

        self.get_logger().info(f'Saved → {pgm_path}')
        self.get_logger().info(f'      → {yaml_path}')


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    topic  = sys.argv[1]
    output = sys.argv[2]

    rclpy.init(args=None)
    node = MapSaver(topic, output)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
