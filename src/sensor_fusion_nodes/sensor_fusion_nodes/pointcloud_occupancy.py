import math
import time

import numpy as np
import rclpy

try:
    from scipy.ndimage import binary_closing as _scipy_binary_closing
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy._rclpy_pybind11 import RCLError
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class PointCloudOccupancy(Node):
    """Build a persistent 2D occupancy projection and 3D voxel marker grid."""

    def __init__(self):
        super().__init__('pointcloud_occupancy')

        self.declare_parameter('pointcloud_topic', '/lidar/points')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('resolution', 0.08)
        self.declare_parameter('z_resolution', 0.12)
        self.declare_parameter('width_m', 14.0)
        self.declare_parameter('height_m', 10.0)
        self.declare_parameter('origin_x', -7.0)
        self.declare_parameter('origin_y', -5.0)
        self.declare_parameter('min_z', 0.08)
        self.declare_parameter('max_z', 2.40)
        self.declare_parameter('projection_min_z', 0.10)
        self.declare_parameter('projection_max_z', 1.35)
        self.declare_parameter('projection_padding_cells', 1)
        self.declare_parameter('decay_seconds', 0.0)
        self.declare_parameter('min_observations', 1)
        self.declare_parameter('max_points_per_cloud', 18000)
        self.declare_parameter('max_marker_voxels', 12000)
        self.declare_parameter('publish_period', 0.50)
        self.declare_parameter('occupancy_2d_topic', '/pointcloud_occupancy_2d')
        self.declare_parameter('occupancy_3d_topic', '/pointcloud_occupancy_3d')
        self.declare_parameter('occupied_voxels_topic', '/occupied_voxels')
        self.declare_parameter('unknown_value', -1)
        self.declare_parameter('morph_close_cells', 0)

        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.resolution = float(self.get_parameter('resolution').value)
        self.z_resolution = float(self.get_parameter('z_resolution').value)
        self.width_m = float(self.get_parameter('width_m').value)
        self.height_m = float(self.get_parameter('height_m').value)
        self.origin_x = float(self.get_parameter('origin_x').value)
        self.origin_y = float(self.get_parameter('origin_y').value)
        self.min_z = float(self.get_parameter('min_z').value)
        self.max_z = float(self.get_parameter('max_z').value)
        self.projection_min_z = float(self.get_parameter('projection_min_z').value)
        self.projection_max_z = float(self.get_parameter('projection_max_z').value)
        self.projection_padding_cells = max(
            0,
            int(self.get_parameter('projection_padding_cells').value),
        )
        self.decay_seconds = float(self.get_parameter('decay_seconds').value)
        self.min_observations = int(self.get_parameter('min_observations').value)
        self.max_points_per_cloud = int(self.get_parameter('max_points_per_cloud').value)
        self.max_marker_voxels = int(self.get_parameter('max_marker_voxels').value)
        self.publish_period = float(self.get_parameter('publish_period').value)
        self.occupancy_2d_topic = self.get_parameter('occupancy_2d_topic').value
        self.occupancy_3d_topic = self.get_parameter('occupancy_3d_topic').value
        self.occupied_voxels_topic = self.get_parameter('occupied_voxels_topic').value
        self.unknown_value = int(self.get_parameter('unknown_value').value)
        self.morph_close_cells = max(0, int(self.get_parameter('morph_close_cells').value))

        self.width = int(math.ceil(self.width_m / self.resolution))
        self.height = int(math.ceil(self.height_m / self.resolution))

        self.voxels = {}
        self.last_tf_warning = 0.0
        self.shutting_down = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        reliable_qos = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data,
        )
        self.grid_pub = self.create_publisher(OccupancyGrid, self.occupancy_2d_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.occupancy_3d_topic, 10)
        self.voxel_cloud_pub = self.create_publisher(
            PointCloud2,
            self.occupied_voxels_topic,
            reliable_qos,
        )

        self.timer = self.create_timer(self.publish_period, self.publish_maps)

        self.get_logger().info(
            f'Building 3D voxel occupancy from {self.pointcloud_topic}; '
            f'publishing {self.occupancy_2d_topic}, {self.occupancy_3d_topic}, '
            f'and {self.occupied_voxels_topic}'
        )

    @staticmethod
    def transform_to_matrix(transform):
        q = transform.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            x, y, z, w = 0.0, 0.0, 0.0, 1.0
        else:
            x, y, z, w = x / norm, y / norm, z / norm, w / norm

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rotation = np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ])

        t = transform.transform.translation
        translation = np.array([t.x, t.y, t.z])
        return rotation, translation

    def warn_tf(self, message):
        now = time.monotonic()
        if now - self.last_tf_warning > 5.0:
            self.get_logger().warn(message)
            self.last_tf_warning = now

    def cloud_callback(self, msg):
        if self.shutting_down or not rclpy.ok():
            return

        if not msg.header.frame_id:
            self.warn_tf('Skipping point cloud with empty frame_id.')
            return

        points = point_cloud2.read_points(
            msg,
            field_names=['x', 'y', 'z'],
            skip_nans=True,
        )
        if points.size == 0:
            return

        xyz = np.column_stack((points['x'], points['y'], points['z'])).astype(
            np.float64,
            copy=False,
        )

        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        if xyz.size == 0:
            return

        if xyz.shape[0] > self.max_points_per_cloud > 0:
            stride = int(math.ceil(xyz.shape[0] / self.max_points_per_cloud))
            xyz = xyz[::stride]

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                msg.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as exc:
            self.warn_tf(
                f'Waiting for TF {self.map_frame} <- {msg.header.frame_id}: {exc}'
            )
            return

        rotation, translation = self.transform_to_matrix(transform)
        map_points = xyz @ rotation.T + translation

        z_mask = (map_points[:, 2] >= self.min_z) & (map_points[:, 2] <= self.max_z)
        map_points = map_points[z_mask]
        if map_points.size == 0:
            return

        x_indices = np.floor((map_points[:, 0] - self.origin_x) / self.resolution)
        y_indices = np.floor((map_points[:, 1] - self.origin_y) / self.resolution)
        z_indices = np.floor((map_points[:, 2] - self.min_z) / self.z_resolution)

        x_indices = x_indices.astype(np.int32)
        y_indices = y_indices.astype(np.int32)
        z_indices = z_indices.astype(np.int32)

        in_bounds = (
            (x_indices >= 0)
            & (x_indices < self.width)
            & (y_indices >= 0)
            & (y_indices < self.height)
            & (z_indices >= 0)
        )
        if not np.any(in_bounds):
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        keys = zip(
            x_indices[in_bounds].tolist(),
            y_indices[in_bounds].tolist(),
            z_indices[in_bounds].tolist(),
        )

        for key in keys:
            hits, _ = self.voxels.get(key, (0, now))
            self.voxels[key] = (hits + 1, now)

    def active_voxels(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.decay_seconds > 0.0:
            cutoff = now - self.decay_seconds
            stale = [key for key, (_, last_seen) in self.voxels.items() if last_seen < cutoff]
            for key in stale:
                del self.voxels[key]

        return [
            key
            for key, (hits, _) in self.voxels.items()
            if hits >= self.min_observations
        ]

    def voxel_center(self, key):
        x_idx, y_idx, z_idx = key
        return (
            self.origin_x + (x_idx + 0.5) * self.resolution,
            self.origin_y + (y_idx + 0.5) * self.resolution,
            self.min_z + (z_idx + 0.5) * self.z_resolution,
        )

    def publish_maps(self):
        if self.shutting_down or not rclpy.ok():
            return

        voxels = self.active_voxels()
        now_msg = self.get_clock().now().to_msg()

        self.publish_2d_grid(voxels, now_msg)
        self.publish_3d_markers(voxels, now_msg)
        self.publish_voxel_cloud(voxels, now_msg)

    def publish_2d_grid(self, voxels, stamp):
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = self.map_frame
        grid.info.map_load_time = stamp
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0

        data = [self.unknown_value] * (self.width * self.height)
        for key in voxels:
            x_idx, y_idx, _ = key
            _, _, z = self.voxel_center(key)
            if z < self.projection_min_z or z > self.projection_max_z:
                continue

            for dx in range(-self.projection_padding_cells, self.projection_padding_cells + 1):
                for dy in range(-self.projection_padding_cells, self.projection_padding_cells + 1):
                    padded_x = x_idx + dx
                    padded_y = y_idx + dy
                    if 0 <= padded_x < self.width and 0 <= padded_y < self.height:
                        data[padded_y * self.width + padded_x] = 100

        if self.morph_close_cells > 1:
            try:
                data = self._morph_close_2d(data)
            except Exception as exc:
                self.get_logger().warn(f'morph_close_2d failed: {exc}', throttle_duration_sec=10.0)

        grid.data = data
        try:
            self.grid_pub.publish(grid)
        except RCLError:
            return

    def _morph_close_2d(self, data):
        """Apply binary morphological closing to fill holes in occupied cells."""
        n = self.morph_close_cells
        grid_2d = np.array(data, dtype=np.int8).reshape(self.height, self.width)
        occupied = grid_2d == 100

        if not np.any(occupied):
            return data

        struct = np.ones((n, n), dtype=bool)
        if _HAVE_SCIPY:
            closed = _scipy_binary_closing(occupied, structure=struct)
        else:
            half = n // 2
            dilated = np.zeros_like(occupied)
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    s = np.roll(np.roll(occupied, dy, axis=0), dx, axis=1)
                    if dy > 0:
                        s[:dy, :] = False
                    elif dy < 0:
                        s[dy:, :] = False
                    if dx > 0:
                        s[:, :dx] = False
                    elif dx < 0:
                        s[:, dx:] = False
                    dilated |= s
            closed = np.ones_like(dilated)
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
                    closed &= s

        fill_mask = closed & (grid_2d == self.unknown_value)
        grid_2d[fill_mask] = 100
        return grid_2d.flatten().tolist()

    def publish_3d_markers(self, voxels, stamp):
        marker_array = MarkerArray()

        clear = Marker()
        clear.header.frame_id = self.map_frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = stamp
        marker.ns = 'pointcloud_occupancy_voxels'
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.resolution
        marker.scale.y = self.resolution
        marker.scale.z = self.z_resolution
        marker.color.r = 0.05
        marker.color.g = 0.62
        marker.color.b = 0.95
        marker.color.a = 0.46

        marker_voxels = voxels
        if len(marker_voxels) > self.max_marker_voxels > 0:
            stride = int(math.ceil(len(marker_voxels) / self.max_marker_voxels))
            marker_voxels = marker_voxels[::stride]

        for key in marker_voxels:
            x, y, z = self.voxel_center(key)
            marker.points.append(Point(x=x, y=y, z=z))

        marker_array.markers.append(marker)
        try:
            self.marker_pub.publish(marker_array)
        except RCLError:
            return

    def publish_voxel_cloud(self, voxels, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.map_frame
        centers = [self.voxel_center(key) for key in voxels]
        cloud = point_cloud2.create_cloud_xyz32(header, centers)
        try:
            self.voxel_cloud_pub.publish(cloud)
        except RCLError:
            return


def main():
    rclpy.init()
    node = PointCloudOccupancy()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutting_down = True
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
