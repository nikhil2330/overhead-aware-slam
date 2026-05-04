import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import Pose, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class Odom_tf(Node):
    def __init__(self):
        super().__init__("odom_tf")

        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("pose_topic", "/model/robot/pose")
        # If non-empty, also publish nav_msgs/Odometry on this topic.
        # Use this instead of Gazebo's /odom to guarantee that the
        # Odometry stamp always matches get_clock().now() — the same
        # time source used by gz_lidar3d_to_pointcloud.  This keeps
        # RTAB-Map's approx_sync pairs temporally consistent with the
        # TF it publishes, avoiding the "timestamp earlier than TF cache"
        # drop in RViz.
        self.declare_parameter("odom_topic", "")

        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.pose_topic = self.get_parameter("pose_topic").value
        odom_topic     = self.get_parameter("odom_topic").value

        self.broadcaster = TransformBroadcaster(self)

        self.odom_pub = None
        if odom_topic:
            self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)

        self.sub = self.create_subscription(
            Pose, self.pose_topic, self.pose_cb, 50
        )

        extra = f" + Odometry on {odom_topic}" if odom_topic else ""
        self.get_logger().info(
            f"Publishing TF: {self.odom_frame} -> {self.base_frame} "
            f"from {self.pose_topic}{extra}"
        )

    def pose_cb(self, msg: Pose):
        now = self.get_clock().now().to_msg()

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = msg.position.x
        t.transform.translation.y = msg.position.y
        t.transform.translation.z = msg.position.z
        t.transform.rotation = msg.orientation
        self.broadcaster.sendTransform(t)

        if self.odom_pub is not None:
            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = self.odom_frame
            odom.child_frame_id = self.base_frame
            odom.pose.pose.position.x = msg.position.x
            odom.pose.pose.position.y = msg.position.y
            odom.pose.pose.position.z = msg.position.z
            odom.pose.pose.orientation = msg.orientation
            self.odom_pub.publish(odom)


def main():
    rclpy.init()
    node = Odom_tf()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
