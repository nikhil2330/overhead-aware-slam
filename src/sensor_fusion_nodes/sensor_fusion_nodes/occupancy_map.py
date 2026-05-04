import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray


class OccupancyMap(Node):

    def __init__(self):
        super().__init__('occupancy_map')

        # latest slam map
        self.map_msg = None

        # subscribes to base slam map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # subscribes to confirmed object geometry from object_location
        self.obj_sub = self.create_subscription(
            MarkerArray,
            '/confirmed_object_geometry',
            self.object_callback,
            10
        )

        # publishes semantic occupancy map painted on top of slam map
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/occupancy_map',
            10
        )

    def map_callback(self, msg):
        # stores latest base map
        self.map_msg = msg

    def world_to_grid(self, x, y, origin_x, origin_y, resolution):
        # converts world coords in map frame to occupancy grid cell coords
        gx = int((x - origin_x) / resolution)
        gy = int((y - origin_y) / resolution)
        return gx, gy

    def in_bounds(self, gx, gy, width, height):
        # checks if a grid cell lies inside map bounds
        return 0 <= gx < width and 0 <= gy < height

    def set_occ(self, data, gx, gy, width, height):
        # marks a cell occupied if it is inside bounds
        if not self.in_bounds(gx, gy, width, height):
            return
        idx = gy * width + gx
        data[idx] = 100

    def inflate_cell(self, data, gx, gy, width, height, radius=1):
        # inflates one occupied cell by a small square radius
        # later this could become class based
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx = gx + dx
                ny = gy + dy
                if self.in_bounds(nx, ny, width, height):
                    idx = ny * width + nx
                    data[idx] = 100

    def draw_line(self, data, x0, y0, x1, y1, width, height):
        # draws a line between 2 grid cells
        # used for center to support rays and support boundary edges
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy
        x, y = x0, y0

        while True:
            self.set_occ(data, x, y, width, height)

            if x == x1 and y == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x += sx

            if e2 < dx:
                err += dx
                y += sy

    def point_in_polygon(self, x, y, poly):
        # checks if a grid point lies inside a polygon
        # used for filling support polygon area
        inside = False
        n = len(poly)
        j = n - 1

        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]

            intersect = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi
            )
            if intersect:
                inside = not inside

            j = i

        return inside

    def fill_polygon(self, data, poly, width, height):
        # fills interior of support boundary polygon    
        # current logic works best when support points roughly explain footprint which is type 1
        # later type 1 vs type 2 logic will come
        if len(poly) < 3:
            return

        min_x = max(0, min(p[0] for p in poly))
        max_x = min(width - 1, max(p[0] for p in poly))
        min_y = max(0, min(p[1] for p in poly))
        max_y = min(height - 1, max(p[1] for p in poly))

        for gx in range(min_x, max_x + 1):
            for gy in range(min_y, max_y + 1):
                if self.point_in_polygon(gx + 0.5, gy + 0.5, poly):
                    self.set_occ(data, gx, gy, width, height)

    def object_callback(self, msg: MarkerArray):
        if self.map_msg is None:
            return

        # create output map using latest slam map as base
        new_map = OccupancyGrid()
        new_map.header.stamp = self.get_clock().now().to_msg()
        new_map.header.frame_id = 'map'
        new_map.info = self.map_msg.info

        # copy base map data then paint semantic obstacles onto it
        data = list(self.map_msg.data)

        resolution = new_map.info.resolution
        width = new_map.info.width
        height = new_map.info.height
        origin_x = new_map.info.origin.position.x
        origin_y = new_map.info.origin.position.y

        # each marker is one confirmed object
        # marker pose stores geom centroid in map frame
        # marker points store local support relative to that centroid
        for marker in msg.markers:

            cx = marker.pose.position.x
            cy = marker.pose.position.y

            # convert object center to grid cell
            cgx, cgy = self.world_to_grid(
                cx, cy,
                origin_x, origin_y,
                resolution
            )

            if not self.in_bounds(cgx, cgy, width, height):
                continue

            # always paint center cell first
            self.set_occ(data, cgx, cgy, width, height)

            # reconstruct support points back into world map frame
            support_world = []
            for p in marker.points:
                wx = cx + p.x
                wy = cy + p.y
                support_world.append((wx, wy))

            # convert support world points into grid cells
            support_grid = []
            for wx, wy in support_world:
                gx, gy = self.world_to_grid(wx, wy, origin_x, origin_y, resolution)
                if self.in_bounds(gx, gy, width, height):
                    support_grid.append((gx, gy))

            # if no support cells survived then just inflate center
            # this keeps at least a small occupied blob for the object
            if not support_grid:
                self.inflate_cell(data, cgx, cgy, width, height, radius=2)
                continue

            # unique cells
            support_grid = list(dict.fromkeys(support_grid))

            # sort support boundary around centroid
            # current simple ordering works decently for chair like objects
            # later large objects may need better ordering or class based logic
            support_grid.sort(
                key=lambda p: math.atan2(p[1] - cgy, p[0] - cgx)
            )

            # draw center-to-support rays
            # helps connect sparse support back to object center
            for sgx, sgy in support_grid:
                self.draw_line(data, cgx, cgy, sgx, sgy, width, height)

            # draw outer boundary loop
            # makes a closed support outline
            if len(support_grid) >= 2:
                for i in range(len(support_grid)):
                    x0, y0 = support_grid[i]
                    x1, y1 = support_grid[(i + 1) % len(support_grid)]
                    self.draw_line(data, x0, y0, x1, y1, width, height)

            # fill the support polygon
            # current fill is for type 1 style objects
            if len(support_grid) >= 3:
                self.fill_polygon(data, support_grid, width, height)

            # small final inflation
            for sgx, sgy in support_grid:
                self.inflate_cell(data, sgx, sgy, width, height, radius=0)

            # final center inflation for a more stable occupied core
            self.inflate_cell(data, cgx, cgy, width, height, radius=2)

        new_map.data = data
        self.map_pub.publish(new_map)


def main():
    rclpy.init()
    node = OccupancyMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()