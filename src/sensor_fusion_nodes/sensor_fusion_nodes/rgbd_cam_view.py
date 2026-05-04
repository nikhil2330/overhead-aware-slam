import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RGBDViewer(Node):
    def __init__(self):
        super().__init__('rgbd_viewer')

        self.bridge = CvBridge()

        self.rgb_frame = None
        self.depth_frame = None

        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.rgb_callback,
            10
        )

        self.create_subscription(
            Image,
            '/camera/depth_image',
            self.depth_callback,
            10
        )

        cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)

    def rgb_callback(self, msg: Image):
        self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('RGB', self.rgb_frame)
        cv2.waitKey(1)

    def depth_callback(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth = np.array(depth, dtype=np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        if depth.size > 0:
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            self.depth_frame = depth_vis.astype(np.uint8)
        
        cv2.imshow('Depth', self.depth_frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = RGBDViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()