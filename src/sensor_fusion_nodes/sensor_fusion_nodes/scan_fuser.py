#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray, Bool


class ScanFuser(Node):
    def __init__(self):
        super().__init__('scan_fuser')

        self.xmin = 0
        self.xmax = 0
        self.imw = 0
        self.ok = False

        self.hfov = 1.5
        self.min_pts = 2
        self.margin = 0.25
        self.far_margin = 0.20

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Int32MultiArray, '/bbox_px', self.bbox_cb, 10)
        self.create_subscription(Bool, '/bbox_valid', self.valid_cb, 10)

        self.pub = self.create_publisher(LaserScan, '/scan_fused', 10)

        self.get_logger().info('simple scan_fuser started')

    def bbox_cb(self, msg):
        if len(msg.data) >= 3:
            self.xmin = int(msg.data[0])
            self.xmax = int(msg.data[1])
            self.imw = int(msg.data[2])

    def valid_cb(self, msg):
        self.ok = bool(msg.data)

    def scan_cb(self, msg):
        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.intensities = list(msg.intensities)

        r = np.array(msg.ranges, dtype=np.float32)

        if (not self.ok) or self.imw <= 0 or self.xmax <= self.xmin:
            out.ranges = r.tolist()
            self.pub.publish(out)
            return

        c = (self.xmin + self.xmax) * 0.5
        w = self.xmax - self.xmin

        theta = ((c / self.imw) - 0.5) * self.hfov
        half = 0.5 * (w / self.imw) * self.hfov

        i0 = int((theta - half - msg.angle_min) / msg.angle_increment)
        i1 = int((theta + half - msg.angle_min) / msg.angle_increment)

        i0 = max(0, min(i0, len(r) - 1))
        i1 = max(0, min(i1, len(r) - 1))

        if i1 < i0:
            i0, i1 = i1, i0

        vals = []
        for i in range(i0, i1 + 1):
            if math.isfinite(float(r[i])) and msg.range_min <= r[i] <= msg.range_max:
                vals.append(float(r[i]))

        if len(vals) < self.min_pts:
            out.ranges = r.tolist()
            self.pub.publish(out)
            return

        d0 = min(vals)

        near = []
        for v in vals:
            if v <= d0 + self.margin:
                near.append(v)

        if len(near) < self.min_pts:
            out.ranges = r.tolist()
            self.pub.publish(out)
            return

        d = float(np.median(near))
        self.get_logger().info(
            f'i0={i0} i1={i1} wedge={i1-i0+1} support={len(vals)} near={len(near)} d={d:.2f}'
        )
        rep = 0
        for i in range(i0, i1 + 1):
            if (not math.isfinite(float(r[i]))) or r[i] > d + self.far_margin:
                r[i] = d
                rep += 1
        self.get_logger().info(f'replaced={rep}')

        self.get_logger().info(
            f'bbox_valid=True xmin={self.xmin} xmax={self.xmax} imw={self.imw} '
            f'i0={i0} i1={i1} d={d:.2f}'
        )

        out.ranges = r.tolist()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanFuser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()