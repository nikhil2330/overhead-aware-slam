import math

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

try:
    from gz.msgs10.laserscan_pb2 import LaserScan as GzLaserScan
    from gz.transport13 import Node as GzNode
except ImportError:
    GzLaserScan = None
    GzNode = None


class GzLidar3dToPointCloud(Node):
    """Convert Gazebo's full 3D LaserScan message into ROS PointCloud2 and LaserScan."""

    def __init__(self):
        super().__init__('gz_lidar3d_to_pointcloud')

        self.declare_parameter('gz_topic', '/lidar_3d')
        self.declare_parameter('pointcloud_topic', '/lidar/points')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('frame_id', 'lidar_3d_link')
        self.declare_parameter('publish_pointcloud', True)
        self.declare_parameter('publish_scan', True)
        self.declare_parameter('scan_height_min', -0.20)
        self.declare_parameter('scan_height_max', 0.20)
        self.declare_parameter('point_stride', 1)

        self.gz_topic = self.get_parameter('gz_topic').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_pointcloud = self.get_parameter('publish_pointcloud').value
        self.publish_scan = self.get_parameter('publish_scan').value
        self.scan_height_min = self.get_parameter('scan_height_min').value
        self.scan_height_max = self.get_parameter('scan_height_max').value
        self.point_stride = max(1, int(self.get_parameter('point_stride').value))
        self.shutting_down = False

        if GzNode is None or GzLaserScan is None:
            raise RuntimeError(
                'Gazebo Python transport bindings are not available. '
                'Install the gz transport Python bindings for this ROS/Gazebo distro.'
            )

        lidar_qos = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.pointcloud_pub = None
        if self.publish_pointcloud:
            self.pointcloud_pub = self.create_publisher(
                PointCloud2,
                self.pointcloud_topic,
                lidar_qos,
            )

        self.scan_pub = None
        if self.publish_scan:
            self.scan_pub = self.create_publisher(
                LaserScan,
                self.scan_topic,
                lidar_qos,
            )

        self.gz_node = GzNode()
        self.gz_node.subscribe(GzLaserScan, self.gz_topic, self.lidar_callback)

        self.get_logger().info(
            f'Subscribed to Gazebo 3D lidar {self.gz_topic}; '
            f'publishing point cloud {self.pointcloud_topic} and scan {self.scan_topic}'
        )

    def make_header(self, msg):
        header = Header()
        # Always use the ROS clock (sim time when use_sim_time=true).
        # GZ transport delivers buffered messages stamped from very early in the
        # simulation (e.g. 0.003 s), which causes every TF message-filter lookup
        # to fail with "timestamp earlier than all data in the transform cache".
        header.stamp = self.get_clock().now().to_msg()
        if self.frame_id:
            header.frame_id = self.frame_id
        else:
            header.frame_id = msg.frame.replace('::', '/')
        return header

    @staticmethod
    def angle_at(min_angle, step, index):
        return min_angle + step * index

    def lidar_callback(self, msg):
        if self.shutting_down or not rclpy.ok():
            return

        horizontal_count = int(msg.count)
        vertical_count = int(msg.vertical_count) if msg.vertical_count else 1

        if horizontal_count <= 0 or vertical_count <= 0:
            return

        ranges = msg.ranges
        expected_count = horizontal_count * vertical_count
        if len(ranges) < horizontal_count:
            return
        if len(ranges) < expected_count:
            vertical_count = 1
            expected_count = horizontal_count

        horizontal_step = float(msg.angle_step)
        if horizontal_step == 0.0 and horizontal_count > 1:
            horizontal_step = (
                float(msg.angle_max) - float(msg.angle_min)
            ) / float(horizontal_count - 1)

        vertical_step = float(msg.vertical_angle_step)
        if vertical_step == 0.0 and vertical_count > 1:
            vertical_step = (
                float(msg.vertical_angle_max) - float(msg.vertical_angle_min)
            ) / float(vertical_count - 1)

        range_min = float(msg.range_min)
        range_max = float(msg.range_max)
        header = self.make_header(msg)

        points = []
        scan_ranges = [math.inf] * horizontal_count

        cos_h = [
            math.cos(self.angle_at(float(msg.angle_min), horizontal_step, i))
            for i in range(horizontal_count)
        ]
        sin_h = [
            math.sin(self.angle_at(float(msg.angle_min), horizontal_step, i))
            for i in range(horizontal_count)
        ]

        for v_index in range(vertical_count):
            vertical_angle = self.angle_at(
                float(msg.vertical_angle_min),
                vertical_step,
                v_index,
            )
            cos_v = math.cos(vertical_angle)
            sin_v = math.sin(vertical_angle)
            row_offset = v_index * horizontal_count

            for h_index in range(horizontal_count):
                range_index = row_offset + h_index
                if range_index >= expected_count:
                    break

                ray_range = float(ranges[range_index])
                if not math.isfinite(ray_range):
                    continue
                if ray_range < range_min or ray_range > range_max:
                    continue

                xy_range = ray_range * cos_v
                z = ray_range * sin_v
                x = xy_range * cos_h[h_index]
                y = xy_range * sin_h[h_index]

                if (
                    self.publish_pointcloud
                    and h_index % self.point_stride == 0
                    and v_index % self.point_stride == 0
                ):
                    points.append((x, y, z))

                if self.publish_scan and self.scan_height_min <= z <= self.scan_height_max:
                    if xy_range < scan_ranges[h_index]:
                        scan_ranges[h_index] = xy_range

        if self.publish_pointcloud and self.pointcloud_pub is not None:
            cloud = point_cloud2.create_cloud_xyz32(header, points)
            try:
                self.pointcloud_pub.publish(cloud)
            except RCLError:
                return

        if self.publish_scan and self.scan_pub is not None:
            scan = LaserScan()
            scan.header = header
            scan.angle_min = float(msg.angle_min)
            scan.angle_max = float(msg.angle_max)
            scan.angle_increment = horizontal_step
            scan.time_increment = 0.0
            scan.scan_time = 0.0
            scan.range_min = range_min
            scan.range_max = range_max
            scan.ranges = scan_ranges
            try:
                self.scan_pub.publish(scan)
            except RCLError:
                return


def main():
    rclpy.init()
    node = GzLidar3dToPointCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutting_down = True
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
