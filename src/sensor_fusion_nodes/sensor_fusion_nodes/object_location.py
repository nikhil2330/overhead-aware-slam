import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, CameraInfo
from sensor_fusion_msgs.msg import Detection
from nav_msgs.msg import OccupancyGrid

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import ColorRGBA
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs


class ObjectLocation(Node):

    def __init__(self):
        super().__init__('object_location')
        # last detection
        self.last_detection = None

        self.cam_info_ok = False
        self.base_map = None
        # focal lengths
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0

        #s scan buffer for storing scans based onnn timestamp
        self.scan_buffer = []
        # max amount of scans
        self.max_scan_buffer = 5

        self.init_params()
        self.init_memory()
        self.init_ros()

    def init_params(self):

        # sync params

        # max allowed detection/scan time mismatch
        self.sync_max_det_scan_dt = 0.08

        # class footprint priors in the map plane
        # user-facing params are:
        # s_value: expected short side in meters
        # l_value: expected long side in meters
        # size_range: symmetric fractional tolerance around those values
        self.builtin_default_profile = {
            's_value': 0.60,
            'l_value': 0.75,
            'size_range': 0.30,
        }
        self.builtin_class_profiles = {
            'chair': {
                's_value': 0.56,
                'l_value': 0.60,
                'size_range': 0.25,
            },
            'table': {
                's_value': 0.90,
                'l_value': 1.35,
                'size_range': 0.35,
            },
        }
        self.profile_fallback_warned = set()

        self.declare_parameter('profile_classes', 'chair,table')
        configured_classes = self.parse_profile_class_names(
            self.get_parameter('profile_classes').value
        )

        self.default_class_profile = self.load_profile_param_group(
            'default_profile',
            self.builtin_default_profile
        )
        self.class_profiles = {}

        for class_name in configured_classes:
            defaults = dict(self.builtin_default_profile)
            defaults.update(self.builtin_class_profiles.get(class_name, {}))
            self.class_profiles[class_name] = self.load_profile_param_group(
                f'class_profiles.{class_name}',
                defaults
            )

        # wedge params

        # amount to trim from bbox edges before converting to lidar wedge
        # class based 
        self.wedge_bbox_trim_frac = 0.12

        # max half-angle allowed around bbox center
        # class based
        self.wedge_max_half_angle = 0.20

        # front layer selection params

        # depth size used to split wedge support into shallow layers
        # class based 
        self.layer_depth_band = 0.10

        # max number of front layers to consider
        self.layer_max_layers = 3

        # extra depth allowed beyond front layer
        # class based
        self.layer_max_extra_depth_close = 0.10
        self.layer_max_extra_depth_med = 0.18
        self.layer_max_extra_depth_far = 0.35

        # base clustering params

        # cluster points inside a kept layer using scan continuity
        # depth continuity and xy continuity
        # class based
        self.cluster_scan_gap = 2
        self.cluster_depth_jump = 0.12
        self.cluster_point_gap = 0.15
        self.cluster_min_pts = 2

        # front subcluster params

        # tighter clustering for front-most centroid points
        # class based
        self.front_cluster_scan_gap = 1
        self.front_cluster_depth_jump = 0.035
        self.front_cluster_point_gap = 0.08

        # front band kept near min depth for centroiding
        # class based
        self.front_band_close = 0.03
        self.front_band_med = 0.05
        self.front_band_far = 0.08

        # cluster validity params

        # hard gates for rejecting implausible support clusters
        # class based
        self.cluster_gate_min_pts = 2
        self.cluster_gate_max_center_angle_error = 0.22
        self.cluster_gate_max_diag = 0.75
        self.cluster_gate_max_range_span = 0.30

        # front cluster preference params

        # if a valid cluster is both near bbox center and near the wedge front,
        # prefer it over a deeper cluster
        # class based
        self.front_select_angle_gate = 0.12
        self.front_select_close_r = 1.8
        self.front_select_med_r = 3.0
        self.front_select_depth_margin_close = 0.08
        self.front_select_depth_margin_med = 0.14
        self.front_select_depth_margin_far = 0.25

        # memory params

        # temp object must be seen this many times before promotion
        # class based
        self.memory_confirm_hits = 3

        # temp association radius
        # class based
        self.memory_temp_match_radius = 0.35

        # confirmed association radius
        # class based
        self.memory_confirmed_match_radius = 0.65

        # remove stale temp objects after this many seconds
        self.memory_temp_timeout = 0.6

        # suppress creating a new object if already extremely close
        # class based
        self.memory_too_close_confirmed_radius = 0.45

        # merge nearby confirmed duplicates
        # class based
        self.memory_merge_radius = 0.35

        # small weak supports near existing objects are suppressed
        # class based
        self.memory_new_object_suppress_radius = 0.70
        self.memory_new_object_suppress_small_n = 3

        # support params

        # max support points stored per object
        self.support_max_points_per_object = 200
        self.support_raw_max_points_per_object = 1500

        # support points closer than this are treated as duplicates
        # class based
        self.support_merge_radius = 0.06

        # keep only new support near current geometry center
        # class based
        self.support_keep_radius = 0.55

        # trim full support cloud around geometry center during refresh
        # class based
        self.support_trim_radius = 0.60

        # geometry publishing params

        # margin added to percentile-based support radius when publishing
        self.publish_support_radius_margin = 0.10

        # if too few local support points remain, publish full support instead
        self.publish_support_min_keep = 3

        # temporal stability params
        self.assoc_centroid_alpha = 0.08
        self.geom_centroid_alpha = 0.18
        self.max_geom_centroid_step = 0.10
        self.observation_center_pull_gain = 0.55

        # cautious type-2 promotion params
        self.type_obs_alpha = 0.20
        self.type_score_gate = 3.8
        self.type2_fill_low = 0.40
        self.type2_gap_ratio = 1.80
        self.type2_gap_abs = 0.28
        self.type2_extreme_score_gate = 3.0
        self.type2_extreme_fill_low = 0.44
        self.type2_extreme_gap_ratio = 2.35
        self.type2_extreme_gap_abs = 0.45
        self.type2_extreme_min_updates = 8
        self.type2_extreme_min_view_bins = 2
        self.type2_extreme_min_hits = 5
        self.type2_mismatch_fill_low = 0.42
        self.type2_mismatch_gap_ratio = 1.75
        self.type2_mismatch_gap_abs = 0.26
        self.type2_mismatch_min_hits = 5
        self.type2_confirmed_min_updates = 4
        self.type2_support_sufficient_major_frac = 0.68
        self.type2_support_sufficient_fill = 0.48
        self.type2_compact_gate_scale = 0.68
        self.type2_compact_ratio_scale = 0.90
        self.type2_compact_min_abs_gap = 0.16
        self.type2_compact_fill_boost = 0.12
        self.type2_round_ratio = 1.95
        self.type2_round_support_ratio = 1.20
        self.type_view_bin_count = 8
        self.type_min_view_bins = 3
        self.type_min_updates = 11
        # multi-view consistency: both the fraction AND a hard fill cap are
        # required — prevents objects that only occasionally look sparse from
        # getting the bonus
        self.type2_multi_view_min_frac = 0.60
        self.type2_multi_view_fill_cap = 0.28
        self.type2_multi_view_bonus = 0.35
        # persistence: score floor below the demote gate so demotion is still
        # possible but takes many more updates to accumulate clear-type1 hits
        self.type2_score_floor = 1.5
        self.type2_score_decay = 0.14
        # once type2, use a much slower alpha for geometry so the displayed
        # obstacle size stays stable across views
        self.type2_geom_alpha = 0.06
        self.type1_demote_score_gate = 1.8
        self.type1_demote_fill_high = 0.62
        self.type1_demote_gap_ratio = 1.45
        self.type1_demote_gap_abs = 0.20
        self.type1_demote_clear_hits = 10
        self.type1_demote_min_updates_after_type2 = 12

        # structural background rejection params
        self.map_occupied_thresh = 65
        self.structural_run_scale = 1.35
        self.structural_min_run_m = 0.90
        self.structural_max_run_m = 1.60
        self.structural_diag_run_relax = 0.88
        self.structural_local_occ_cells = 5
        self.structural_center_keep_frac = 0.22
        self.structural_view_across_scale = 0.26
        self.structural_view_back_pad = 0.04
        self.structural_view_back_cap = 0.22

        # cross-class separation params
        self.cross_class_context_pad = 0.06
        self.cross_class_support_radius_scale = 0.65
        self.foreign_large_claim_ratio = 1.15
        self.foreign_claim_margin = 0.04
        self.same_class_fragment_margin = 0.16
        self.same_class_fragment_min_fraction = 0.50
        self.same_class_fragment_radius_scale = 0.95

        loaded_classes = ', '.join(sorted(self.class_profiles))
        self.get_logger().info(
            f'class profiles loaded default='
            f'(s={self.default_class_profile["s_value"]:.2f}, '
            f'l={self.default_class_profile["l_value"]:.2f}, '
            f'size_range={self.default_class_profile["size_range"]:.2f}) '
            f'classes=[{loaded_classes}]'
        )

    def parse_profile_class_names(self, value):

        if isinstance(value, str):
            items = value.split(',')
        else:
            items = value

        names = []
        for item in items:
            name = str(item).strip()
            if name:
                names.append(name)

        return names

    def load_profile_param_group(self, prefix, defaults):

        self.declare_parameter(f'{prefix}.s_value', float(defaults['s_value']))
        self.declare_parameter(f'{prefix}.l_value', float(defaults['l_value']))
        self.declare_parameter(f'{prefix}.size_range', float(defaults['size_range']))

        raw_profile = {
            's_value': float(self.get_parameter(f'{prefix}.s_value').value),
            'l_value': float(self.get_parameter(f'{prefix}.l_value').value),
            'size_range': float(self.get_parameter(f'{prefix}.size_range').value),
        }

        return self.build_profile(prefix, raw_profile)

    def build_profile(self, profile_name, raw_profile):

        s_value = max(0.0, float(raw_profile['s_value']))
        l_value = max(0.0, float(raw_profile['l_value']))
        size_range = max(0.0, float(raw_profile['size_range']))

        if l_value < s_value:
            s_value, l_value = l_value, s_value
            self.get_logger().warn(
                f'{profile_name}: swapped s_value/l_value to keep l_value >= s_value'
            )

        s_range = (
            max(0.0, s_value * (1.0 - size_range)),
            s_value * (1.0 + size_range)
        )
        l_range = (
            max(0.0, l_value * (1.0 - size_range)),
            l_value * (1.0 + size_range)
        )

        return {
            's_value': s_value,
            'l_value': l_value,
            'size_range': size_range,
            's_range': s_range,
            'l_range': l_range,
        }

    def init_memory(self):
        # temp objects waiting for enough hits
        self.temp_objects = []
        # confirmed object list storing stable objects
        self.confirmed_objects = []
        # id for confirmed objects
        self.next_object_id = 0

    # subscribe to camera , lidar scan, and publishes markers for rviz and geometry for map
    def init_ros(self):

        self.create_subscription(
            Detection,
            '/detections',
            self.detection_cb,
            10
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_cb,
            10
        )

        self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_cb,
            10
        )

        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_frame = 'map'

        self.marker_pub = self.create_publisher(Marker, '/object_marker', 10)
        self.geometry_pub = self.create_publisher(MarkerArray, '/confirmed_object_geometry', 10)

        self.create_timer(1.0, self.log_object_count)
        self.create_timer(0.5, self.publish_confirmed_object_geometry)

    def camera_info_cb(self, msg: CameraInfo):

        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

        if not self.cam_info_ok:
            self.cam_info_ok = True
            hfov = 2.0 * math.atan((msg.width * 0.5) / self.fx)

            self.get_logger().info(
                f'camera_info fx={self.fx:.2f} fy={self.fy:.2f} '
                f'cx={self.cx:.2f} cy={self.cy:.2f} hfov={hfov:.3f}'
            )

    def detection_cb(self, msg: Detection):

        self.last_detection = msg
        self.try_process()

    def map_callback(self, msg: OccupancyGrid):

        self.base_map = msg

    def stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def scan_cb(self, msg: LaserScan):
        self.scan_buffer.append(msg)

        if len(self.scan_buffer) > self.max_scan_buffer:
            self.scan_buffer.pop(0)

    def log_object_count(self):

        self.get_logger().info(
            f'OBJECTS confirmed={len(self.confirmed_objects)} '
            f'candidates={len(self.temp_objects)}'
        )

    # gets current time in seconds as float
    def now_sec(self):

        return self.get_clock().now().nanoseconds * 1e-9

    # converts pixel u coords to angle in lidar frame assuming camera and lidar are aligned on x and y
    def pixel_to_bearing(self, u):

        return -math.atan((u - self.cx) / self.fx)

    # limits number of stored support points
    def cap_points(self, pts, limit=None):

        if limit is None:
            limit = self.support_max_points_per_object

        if len(pts) <= limit:
            return pts

        return pts[-limit:]

    # just gets the centroid of pts in cluster
    def cluster_centroid(self, pts):

        x = sum(p[0] for p in pts) / len(pts)
        y = sum(p[1] for p in pts) / len(pts)
        return (x, y)

    # computes center of bounding box around support points
    def support_box_center(self, pts):

        if not pts:
            return None

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        return (
            0.5 * (min(xs) + max(xs)),
            0.5 * (min(ys) + max(ys))
        )

    def geometry_center_from_assoc_and_support(self, class_name, assoc_centroid, support_pts):

        support_center = self.support_box_center(support_pts)
        if support_center is None:
            return assoc_centroid

        return support_center

    def blend_point(self, old_pt, new_pt, alpha):

        if old_pt is None:
            return new_pt

        if new_pt is None:
            return old_pt

        alpha = max(0.0, min(1.0, alpha))
        return (
            (1.0 - alpha) * old_pt[0] + alpha * new_pt[0],
            (1.0 - alpha) * old_pt[1] + alpha * new_pt[1],
        )

    def blend_scalar(self, old_value, new_value, alpha):

        if new_value is None:
            return old_value

        if old_value is None:
            return new_value

        alpha = max(0.0, min(1.0, alpha))
        return (1.0 - alpha) * old_value + alpha * new_value

    def limit_point_step(self, old_pt, new_pt, max_step):

        if old_pt is None or new_pt is None or max_step <= 0.0:
            return new_pt

        dx = new_pt[0] - old_pt[0]
        dy = new_pt[1] - old_pt[1]
        dist = math.hypot(dx, dy)

        if dist <= max_step or dist <= 1e-9:
            return new_pt

        scale = max_step / dist
        return (
            old_pt[0] + dx * scale,
            old_pt[1] + dy * scale,
        )

    def smooth_point_update(self, old_pt, new_pt, alpha, max_step):

        blended = self.blend_point(old_pt, new_pt, alpha)
        return self.limit_point_step(old_pt, blended, max_step)

    # appends new support points while avoiding duplicates
    def append_unique_points(self, old_pts, new_pts, limit=None):

        if not old_pts:
            return self.cap_points(list(new_pts), limit=limit)

        out = list(old_pts)

        for p in new_pts:
            keep = True

            for q in out:
                if math.hypot(p[0] - q[0], p[1] - q[1]) < self.support_merge_radius:
                    keep = False
                    break

            if keep:
                out.append(p)

        return self.cap_points(out, limit=limit)

    def cap_points_preserve(self, pts, preserve_pts, limit):

        if len(pts) <= limit:
            return list(pts)

        kept = []

        for p in preserve_pts:
            if len(kept) >= limit:
                break

            duplicate = False
            for q in kept:
                if math.hypot(p[0] - q[0], p[1] - q[1]) < self.support_merge_radius:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(p)

        for p in reversed(pts):
            if len(kept) >= limit:
                break

            duplicate = False
            for q in kept:
                if math.hypot(p[0] - q[0], p[1] - q[1]) < self.support_merge_radius:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(p)

        return kept

    # keeps only support points within a radius of a center point
    def filter_points_near_center(self, pts, center, radius):

        if center is None:
            return list(pts)

        out = []

        for p in pts:
            if math.hypot(p[0] - center[0], p[1] - center[1]) <= radius:
                out.append(p)

        return out

    def filter_points_near_any_center(self, pts, centers, radius):

        valid_centers = [c for c in centers if c is not None]
        if not valid_centers:
            return list(pts)

        out = []

        for p in pts:
            for center in valid_centers:
                if math.hypot(p[0] - center[0], p[1] - center[1]) <= radius:
                    out.append(p)
                    break

        return out

    def support_principal_angle(self, pts, center=None):

        if not pts or len(pts) < 2:
            return 0.0

        if center is None:
            center = self.support_box_center(pts)

        if center is None:
            return 0.0

        xx = 0.0
        yy = 0.0
        xy = 0.0

        for p in pts:
            dx = p[0] - center[0]
            dy = p[1] - center[1]
            xx += dx * dx
            yy += dy * dy
            xy += dx * dy

        if abs(xx - yy) < 1e-9 and abs(xy) < 1e-9:
            return 0.0

        return 0.5 * math.atan2(2.0 * xy, xx - yy)

    def support_axis_spans(self, pts, center=None, angle=None):

        if not pts:
            return 0.0, 0.0, 0.0

        if center is None:
            center = self.support_box_center(pts)

        if center is None:
            return 0.0, 0.0, 0.0

        if angle is None:
            angle = self.support_principal_angle(pts, center)

        ca = math.cos(angle)
        sa = math.sin(angle)
        us = []
        vs = []

        for p in pts:
            dx = p[0] - center[0]
            dy = p[1] - center[1]
            us.append(dx * ca + dy * sa)
            vs.append(-dx * sa + dy * ca)

        return max(us) - min(us), max(vs) - min(vs), angle

    def support_major_minor(self, pts, center=None, angle=None):

        if not pts:
            return 0.0, 0.0, 0.0

        if center is None:
            center = self.support_box_center(pts)

        if center is None:
            return 0.0, 0.0, 0.0

        span_u, span_v, span_angle = self.support_axis_spans(pts, center, angle)

        if span_v > span_u:
            span_u, span_v = span_v, span_u
            span_angle = self.normalize_half_turn(span_angle + 0.5 * math.pi)

        return span_u, span_v, span_angle

    def local_support_hypothesis(self, class_name, center, support_pts, pad=0.0):

        if center is None:
            center = self.support_box_center(support_pts)

        if center is None or not support_pts:
            return None

        major_span, minor_span, angle = self.support_major_minor(
            support_pts,
            center
        )
        profile = self.get_class_profile(class_name)
        min_half = max(0.06, 0.18 * profile['s_value'])
        max_half = 0.5 * profile['l_range'][1] + pad

        half_l = min(max_half, max(min_half, 0.5 * major_span + pad))
        half_s = min(max_half, max(min_half, 0.5 * minor_span + pad))

        if half_s > half_l:
            half_l, half_s = half_s, half_l
            angle = self.normalize_half_turn(angle + 0.5 * math.pi)

        return {
            'center': center,
            'angle': angle,
            'half_l': half_l,
            'half_s': half_s,
        }

    def filter_points_in_class_footprint(self, pts, center, class_name, angle=None):

        if center is None or not pts:
            return list(pts)

        if angle is None:
            angle = self.support_principal_angle(pts, center)

        profile = self.get_class_profile(class_name)
        half_s = 0.5 * profile['s_range'][1] + 0.10
        half_l = 0.5 * profile['l_range'][1] + 0.10

        ca = math.cos(angle)
        sa = math.sin(angle)
        out = []

        for p in pts:
            dx = p[0] - center[0]
            dy = p[1] - center[1]
            u = dx * ca + dy * sa
            v = -dx * sa + dy * ca

            fits_primary = abs(u) <= half_l and abs(v) <= half_s
            fits_swapped = abs(u) <= half_s and abs(v) <= half_l

            if fits_primary or fits_swapped:
                out.append(p)

        if out:
            return out

        return list(pts)

    def observation_centroid_from_support(self, det, centroid_pts, geometry_pts, theta_center):

        if centroid_pts:
            front_centroid = self.cluster_centroid(centroid_pts)
        elif geometry_pts:
            front_centroid = self.cluster_centroid(geometry_pts)
        else:
            return None

        if len(geometry_pts) < 2:
            return front_centroid

        geom_center = self.support_box_center(geometry_pts)
        if geom_center is None:
            return front_centroid

        major_span, minor_span, angle = self.support_axis_spans(
            geometry_pts,
            geom_center
        )
        profile = self.get_class_profile(det.class_name)
        aspect_ratio = self.class_aspect_ratio(det.class_name)
        elongation = self.class_elongation(det.class_name)

        if aspect_ratio <= 1.10:
            return front_centroid

        short_err = abs(major_span - profile['s_value']) / max(0.05, profile['s_value'])
        long_err = abs(major_span - profile['l_value']) / max(0.05, profile['l_value'])

        if long_err <= short_err:
            anchor_span = profile['l_value']
            target_transverse = profile['s_value']
        else:
            anchor_span = profile['s_value']
            target_transverse = profile['l_value']

        if major_span < 0.35 * anchor_span:
            return geom_center

        transverse_cov = 0.0
        if target_transverse > 0.0:
            transverse_cov = min(1.0, minor_span / target_transverse)
        anchor_cov = min(1.0, major_span / max(0.05, anchor_span))
        missing_transverse = max(0.0, target_transverse - minor_span)
        center_pull = elongation * anchor_cov * max(0.0, 1.0 - transverse_cov)

        if missing_transverse <= 0.05 or center_pull <= 0.08:
            return geom_center

        inward = (-math.sin(angle), math.cos(angle))
        center_ray = (math.cos(theta_center), math.sin(theta_center))

        if inward[0] * center_ray[0] + inward[1] * center_ray[1] < 0.0:
            inward = (-inward[0], -inward[1])

        offset = min(
            0.50 * target_transverse,
            self.observation_center_pull_gain * center_pull * (0.50 * missing_transverse + 0.03)
        )

        return (
            geom_center[0] + offset * inward[0],
            geom_center[1] + offset * inward[1]
        )

    def normalize_half_turn(self, angle):

        while angle >= 0.5 * math.pi:
            angle -= math.pi

        while angle < -0.5 * math.pi:
            angle += math.pi

        return angle

    def long_axis_angle_from_support(self, class_name, pts, center=None):

        if not pts or len(pts) < 2:
            return 0.0

        if center is None:
            center = self.support_box_center(pts)

        if center is None:
            return 0.0

        major_span, _, angle = self.support_axis_spans(pts, center)
        profile = self.get_class_profile(class_name)

        long_err = abs(major_span - profile['l_value']) / max(0.05, profile['l_value'])
        short_err = abs(major_span - profile['s_value']) / max(0.05, profile['s_value'])

        if short_err < long_err:
            angle += 0.5 * math.pi

        return self.normalize_half_turn(angle)

    def footprint_hypothesis(self, class_name, center, support_pts):

        if center is None:
            center = self.support_box_center(support_pts)

        if center is None:
            return None

        profile = self.get_class_profile(class_name)
        support_center = self.support_box_center(support_pts)
        angle = self.long_axis_angle_from_support(
            class_name,
            support_pts,
            support_center
        )

        expand = 1.0 + 0.50 * profile['size_range']

        return {
            'center': center,
            'angle': angle,
            'half_l': 0.5 * profile['l_value'] * expand,
            'half_s': 0.5 * profile['s_value'] * expand,
        }

    def support_claim_hypothesis(self, obj):

        support_pts = list(obj.get('support_points', []))
        center = obj.get('geom_centroid')
        hypothesis = self.local_support_hypothesis(
            obj['class_name'],
            center,
            support_pts,
            pad=self.cross_class_context_pad
        )
        if hypothesis is not None:
            return hypothesis

        hypothesis = self.footprint_hypothesis(
            obj['class_name'],
            center,
            support_pts
        )
        if hypothesis is None:
            return None

        return {
            'center': hypothesis['center'],
            'angle': hypothesis['angle'],
            'half_l': max(0.08, 0.72 * hypothesis['half_l']),
            'half_s': max(0.08, 0.72 * hypothesis['half_s']),
        }

    def oriented_box_corners(self, hypothesis):

        center = hypothesis['center']
        angle = hypothesis['angle']
        half_l = hypothesis['half_l']
        half_s = hypothesis['half_s']

        long_axis = (math.cos(angle), math.sin(angle))
        short_axis = (-math.sin(angle), math.cos(angle))
        corners = []

        for du in (-half_l, half_l):
            for dv in (-half_s, half_s):
                corners.append((
                    center[0] + du * long_axis[0] + dv * short_axis[0],
                    center[1] + du * long_axis[1] + dv * short_axis[1],
                ))

        return corners

    def project_points_on_axis(self, pts, axis):

        values = [p[0] * axis[0] + p[1] * axis[1] for p in pts]
        return min(values), max(values)

    def intervals_overlap(self, a_min, a_max, b_min, b_max):

        return max(a_min, b_min) <= min(a_max, b_max)

    def footprint_hypotheses_overlap(self, hyp_a, hyp_b):

        if hyp_a is None or hyp_b is None:
            return False

        pts_a = self.oriented_box_corners(hyp_a)
        pts_b = self.oriented_box_corners(hyp_b)
        axes = [
            (math.cos(hyp_a['angle']), math.sin(hyp_a['angle'])),
            (-math.sin(hyp_a['angle']), math.cos(hyp_a['angle'])),
            (math.cos(hyp_b['angle']), math.sin(hyp_b['angle'])),
            (-math.sin(hyp_b['angle']), math.cos(hyp_b['angle'])),
        ]

        for axis in axes:
            a_min, a_max = self.project_points_on_axis(pts_a, axis)
            b_min, b_max = self.project_points_on_axis(pts_b, axis)
            if not self.intervals_overlap(a_min, a_max, b_min, b_max):
                return False

        return True

    def supports_same_object(self, class_name, center_a, pts_a, center_b, pts_b):

        if center_a is None or center_b is None:
            return False

        center_gap = math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
        if center_gap > 1.10 * self.class_max_diag(class_name):
            return False

        hyp_a = self.footprint_hypothesis(class_name, center_a, pts_a)
        hyp_b = self.footprint_hypothesis(class_name, center_b, pts_b)

        return self.footprint_hypotheses_overlap(hyp_a, hyp_b)

    def support_fraction_in_hypothesis(self, pts, hypothesis, margin=0.0):

        if not pts or hypothesis is None:
            return 0.0

        inside = 0

        for pt in pts:
            if self.point_in_footprint_hypothesis(pt, hypothesis, margin=margin):
                inside += 1

        return inside / max(1, len(pts))

    def support_fragment_belongs_to_object(self, class_name, centroid, support_pts, obj):

        if not support_pts or centroid is None:
            return False

        if obj.get('class_name') != class_name:
            return False

        max_dist = self.same_class_fragment_radius_scale * self.class_max_diag(class_name)
        geom_center = obj.get('geom_centroid')
        assoc_center = obj.get('assoc_centroid')
        nearest_center_dist = 1e9

        for center in (geom_center, assoc_center):
            if center is None:
                continue
            nearest_center_dist = min(
                nearest_center_dist,
                math.hypot(centroid[0] - center[0], centroid[1] - center[1])
            )

        if nearest_center_dist > max_dist:
            return False

        hypotheses = []
        for center in (geom_center, assoc_center):
            hypothesis = self.footprint_hypothesis(
                class_name,
                center,
                obj.get('support_points', [])
            )
            if hypothesis is not None:
                hypotheses.append(hypothesis)

        if not hypotheses:
            return False

        margin = self.same_class_fragment_margin

        for hypothesis in hypotheses:
            support_fraction = self.support_fraction_in_hypothesis(
                support_pts,
                hypothesis,
                margin=margin
            )
            centroid_inside = self.point_in_footprint_hypothesis(
                centroid,
                hypothesis,
                margin=margin
            )

            if support_fraction >= self.same_class_fragment_min_fraction:
                return True

            if centroid_inside and support_fraction > 0.0:
                return True

        return False

    def point_in_footprint_hypothesis(self, pt, hypothesis, margin=0.0):

        if hypothesis is None:
            return False

        dx = pt[0] - hypothesis['center'][0]
        dy = pt[1] - hypothesis['center'][1]
        ca = math.cos(hypothesis['angle'])
        sa = math.sin(hypothesis['angle'])
        u = dx * ca + dy * sa
        v = -dx * sa + dy * ca

        return (
            abs(u) <= hypothesis['half_l'] + margin
            and abs(v) <= hypothesis['half_s'] + margin
        )

    def foreign_support_radius_for_class(self, class_name):

        s_max, _ = self.class_max_sides(class_name)
        return max(0.08, 0.20 * s_max)

    def foreign_support_radius_between_classes(self, class_name, other_class_name):

        radius = self.foreign_support_radius_for_class(other_class_name)
        if class_name != other_class_name:
            radius *= self.cross_class_support_radius_scale

        return max(0.06, radius)

    def object_claim_diag(self, obj):

        class_name = obj.get('class_name')
        claim_diag = self.class_max_diag(class_name)
        support_pts = obj.get('support_points', [])
        if not support_pts:
            return claim_diag

        center = obj.get('geom_centroid')
        if center is None:
            center = self.support_box_center(support_pts)

        if center is None:
            return claim_diag

        major, minor, _ = self.support_major_minor(support_pts, center)
        return max(claim_diag, math.hypot(major, minor))

    def other_object_is_larger_context(self, class_name, obj):

        current_diag = self.class_max_diag(class_name)
        other_diag = self.object_claim_diag(obj)

        return other_diag > self.foreign_large_claim_ratio * current_diag

    def support_point_claimed_by_other_object(self, class_name, pt, owner_object_id=None):

        for obj in self.confirmed_objects:
            if owner_object_id is not None and obj.get('object_id') == owner_object_id:
                continue

            if obj.get('class_name') == class_name:
                continue

            larger_context = self.other_object_is_larger_context(class_name, obj)
            if larger_context:
                hypothesis = self.footprint_hypothesis(
                    obj['class_name'],
                    obj.get('geom_centroid'),
                    obj.get('support_points', [])
                )
                if self.point_in_footprint_hypothesis(
                    pt,
                    hypothesis,
                    margin=0.01
                ):
                    return True
                continue

            hypothesis = self.support_claim_hypothesis(obj)
            if self.point_in_footprint_hypothesis(
                pt,
                hypothesis,
                margin=self.foreign_claim_margin
            ):
                return True

            keep_radius = self.foreign_support_radius_between_classes(
                class_name,
                obj['class_name']
            )
            for other_pt in obj.get('support_points', []):
                if math.hypot(pt[0] - other_pt[0], pt[1] - other_pt[1]) <= keep_radius:
                    return True

        return False

    def count_foreign_support_points(self, class_name, pts, owner_object_id=None):

        if not pts:
            return 0

        count = 0
        for pt in pts:
            if self.support_point_claimed_by_other_object(
                class_name,
                pt,
                owner_object_id=owner_object_id
            ):
                count += 1

        return count

    def view_frame_axes(self, center, sensor_origin):

        if center is None or sensor_origin is None:
            return None, None, None

        dx = center[0] - sensor_origin[0]
        dy = center[1] - sensor_origin[1]
        range_m = math.hypot(dx, dy)

        if range_m <= 1e-6:
            return None, None, None

        view_axis = (dx / range_m, dy / range_m)
        across_axis = (-view_axis[1], view_axis[0])
        return view_axis, across_axis, range_m

    def point_view_coords(self, pt, center, sensor_origin):

        view_axis, across_axis, _ = self.view_frame_axes(center, sensor_origin)
        if view_axis is None:
            return None, None

        rel_sensor_x = pt[0] - sensor_origin[0]
        rel_sensor_y = pt[1] - sensor_origin[1]
        rel_center_x = pt[0] - center[0]
        rel_center_y = pt[1] - center[1]

        along = rel_sensor_x * view_axis[0] + rel_sensor_y * view_axis[1]
        across = rel_center_x * across_axis[0] + rel_center_y * across_axis[1]
        return along, across

    def support_view_span(self, center, sensor_origin, support_pts):

        if not support_pts:
            return 0.0

        _, across_axis, _ = self.view_frame_axes(center, sensor_origin)
        if across_axis is None:
            return 0.0

        values = []

        for pt in support_pts:
            dx = pt[0] - center[0]
            dy = pt[1] - center[1]
            values.append(dx * across_axis[0] + dy * across_axis[1])

        if not values:
            return 0.0

        return max(values) - min(values)

    def clone_observation_state(self, state=None):

        copied = {
            'updates': 0,
            'view_bins': set(),
            'confirmed_updates': 0,
            'type_score': 0.0,
            'type2_mismatch_hits': 0,
            'type2_extreme_hits': 0,
            'type1_clear_hits': 0,
            'type2_updates': 0,
            'bbox_span': None,
            'support_major': None,
            'support_minor': None,
            'support_view_span': None,
            'support_fill': 1.0,
        }

        if state is None:
            return copied

        copied['updates'] = int(state.get('updates', 0))
        copied['view_bins'] = set(state.get('view_bins', set()))
        copied['confirmed_updates'] = int(state.get('confirmed_updates', 0))
        copied['type_score'] = float(state.get('type_score', 0.0))
        copied['type2_mismatch_hits'] = int(state.get('type2_mismatch_hits', 0))
        copied['type2_extreme_hits'] = int(state.get('type2_extreme_hits', 0))
        copied['type1_clear_hits'] = int(state.get('type1_clear_hits', 0))
        copied['type2_updates'] = int(state.get('type2_updates', 0))
        copied['bbox_span'] = state.get('bbox_span')
        copied['support_major'] = state.get('support_major')
        copied['support_minor'] = state.get('support_minor')
        copied['support_view_span'] = state.get('support_view_span')
        copied['support_fill'] = float(state.get('support_fill', 1.0))
        return copied

    def build_observation_meta(self, class_name, center, support_pts, bbox_span, sensor_origin=None):

        if center is None:
            center = self.support_box_center(support_pts)

        support_major = 0.0
        support_minor = 0.0
        support_angle = 0.0
        if support_pts and center is not None:
            support_major, support_minor, support_angle = self.support_major_minor(
                support_pts,
                center
            )

        support_view_span = support_major
        if support_pts and center is not None and sensor_origin is not None:
            view_span = self.support_view_span(center, sensor_origin, support_pts)
            if view_span > 0.0:
                support_view_span = view_span

        # A zero or near-zero support_view_span means the lidar found one point
        # (or none) on a narrow pedestal — that is maximally sparse, not
        # "unknown".  Use 0.0 so the fill signal correctly drives type2
        # evidence instead of the no-data default of 1.0.
        support_fill = 1.0
        if bbox_span is not None and bbox_span > 0.05:
            if support_view_span > 0.0:
                support_fill = support_view_span / max(0.05, bbox_span)
            else:
                support_fill = 0.0

        view_angle = None
        if sensor_origin is not None and center is not None:
            view_angle = math.atan2(
                center[1] - sensor_origin[1],
                center[0] - sensor_origin[0]
            )

        return {
            'class_name': class_name,
            'bbox_span': bbox_span,
            'support_major': support_major,
            'support_minor': support_minor,
            'support_view_span': support_view_span,
            'support_fill': support_fill,
            'support_angle': support_angle,
            'view_angle': view_angle,
        }

    def observation_meta_from_context(self, class_name, center, support_pts, obs_context):

        if obs_context is None:
            return None

        return self.build_observation_meta(
            class_name,
            center,
            support_pts,
            obs_context.get('bbox_span'),
            sensor_origin=obs_context.get('sensor_origin')
        )

    def record_object_view_bin(self, obs_state, view_angle):

        if view_angle is None:
            return

        bin_count = max(1, self.type_view_bin_count)
        angle = (view_angle + math.pi) % (2.0 * math.pi)
        bin_idx = int(bin_count * angle / (2.0 * math.pi)) % bin_count
        obs_state['view_bins'].add(bin_idx)

    def initialize_object_observation_state(self, obs_meta=None):

        obs_state = self.clone_observation_state()

        if obs_meta is None:
            return obs_state

        obs_state['updates'] = 1
        self.record_object_view_bin(obs_state, obs_meta.get('view_angle'))
        obs_state['bbox_span'] = obs_meta.get('bbox_span')
        obs_state['support_major'] = obs_meta.get('support_major')
        obs_state['support_minor'] = obs_meta.get('support_minor')
        obs_state['support_view_span'] = obs_meta.get('support_view_span')
        obs_state['support_fill'] = obs_meta.get('support_fill', 1.0)
        return obs_state

    def type2_gate_scale_for_class(self, class_name):

        if class_name is None:
            return 1.0

        default_diag = math.hypot(
            self.default_class_profile['s_range'][1],
            self.default_class_profile['l_range'][1]
        )
        class_diag = self.class_max_diag(class_name)

        if default_diag <= 0.05:
            return 1.0

        if class_diag <= 1.05 * default_diag:
            return self.type2_compact_gate_scale

        if class_diag >= 1.55 * default_diag:
            return 1.0

        t = (class_diag - 1.05 * default_diag) / (0.50 * default_diag)
        return self.type2_compact_gate_scale + t * (1.0 - self.type2_compact_gate_scale)

    def type2_gap_abs_gate_for_class(self, class_name, base_gate):

        scale = self.type2_gate_scale_for_class(class_name)
        return max(self.type2_compact_min_abs_gap, base_gate * scale)

    def type2_gap_ratio_gate_for_class(self, class_name, base_gate):

        scale = self.type2_gate_scale_for_class(class_name)
        ratio_scale = self.type2_compact_ratio_scale + 0.10 * scale
        return max(1.45, base_gate * ratio_scale)

    def type2_score_gate_for_class(self, class_name, base_gate):

        scale = self.type2_gate_scale_for_class(class_name)
        return max(3.0, base_gate * (0.82 + 0.18 * scale))

    def type2_fill_gate_for_class(self, class_name, base_fill):
        # Compact objects (small chairs) have center-pedestal support that
        # produces fill ~0.22–0.26 — larger than a round table (~0.10) but
        # still clearly sparse. Allow a modest boost to the fill threshold so
        # the EMA can converge within the minimum update window.
        scale = self.type2_gate_scale_for_class(class_name)
        if scale >= 0.85:
            return base_fill
        boost = self.type2_compact_fill_boost * (1.0 - scale / 0.85)
        return min(base_fill + boost, 0.52)

    def type2_min_updates_for_class(self, class_name):

        if self.type2_gate_scale_for_class(class_name) < 0.85:
            return max(self.type2_confirmed_min_updates + 3, self.type_min_updates - 2)

        return self.type_min_updates

    def type2_min_view_bins_for_class(self, class_name):

        if self.type2_gate_scale_for_class(class_name) < 0.85:
            return max(2, self.type_min_view_bins - 1)

        return self.type_min_view_bins

    def type2_mismatch_hits_for_class(self, class_name):

        if self.type2_gate_scale_for_class(class_name) < 0.85:
            return max(3, self.type2_mismatch_min_hits - 1)

        return self.type2_mismatch_min_hits

    def type1_support_is_sufficient(self, class_name, obs_state):

        if class_name is None or obs_state is None:
            return False

        profile = self.get_class_profile(class_name)
        support_major = obs_state.get('support_major')
        support_minor = obs_state.get('support_minor')
        bbox_span = obs_state.get('bbox_span')
        support_fill = obs_state.get('support_fill', 1.0)

        if support_major is None or support_major <= 0.05:
            return False

        if bbox_span is not None and bbox_span > 0.05:
            gap_ratio = bbox_span / max(0.05, support_major)
            gap_abs = bbox_span - support_major
            if (
                gap_ratio >= self.type2_gap_ratio_gate_for_class(
                    class_name,
                    self.type2_mismatch_gap_ratio
                )
                and gap_abs >= self.type2_gap_abs_gate_for_class(
                    class_name,
                    self.type2_mismatch_gap_abs
                )
                and support_fill <= self.type2_mismatch_fill_low
            ):
                return False

        support_minor = max(0.0, support_minor if support_minor is not None else 0.0)
        major_floor = self.type2_support_sufficient_major_frac * profile['s_value']
        minor_floor = 0.25 * profile['s_value']

        if support_major >= major_floor and support_fill >= self.type2_support_sufficient_fill:
            return True

        if support_major >= major_floor and support_minor >= minor_floor:
            return True

        return (
            support_major >= 0.52 * profile['l_value']
            and support_fill >= self.type2_support_sufficient_fill
        )

    def type2_ready_for_promotion(self, obs_state, class_name=None):

        if obs_state is None:
            return False

        if obs_state.get('confirmed_updates', 0) < self.type2_confirmed_min_updates:
            return False

        bbox_span = obs_state.get('bbox_span')
        support_major = obs_state.get('support_major')
        support_view_span = obs_state.get('support_view_span')
        support_fill = obs_state.get('support_fill', 1.0)
        gap_support_span = support_view_span
        if gap_support_span is None or gap_support_span <= 0.05:
            gap_support_span = support_major

        # require a meaningful bbox; floor support span so a single lidar
        # point on a narrow pedestal still produces valid gap evidence
        if bbox_span is None or bbox_span <= 0.15:
            return False

        if gap_support_span is None or gap_support_span < 0.0:
            gap_support_span = 0.0
        gap_support_span_eff = max(0.05, gap_support_span)

        gap_ratio = bbox_span / gap_support_span_eff
        gap_abs = bbox_span - gap_support_span_eff
        gap_ratio_gate = self.type2_gap_ratio_gate_for_class(
            class_name,
            self.type2_gap_ratio
        )
        gap_abs_gate = self.type2_gap_abs_gate_for_class(
            class_name,
            self.type2_gap_abs
        )
        extreme_ratio_gate = self.type2_gap_ratio_gate_for_class(
            class_name,
            self.type2_extreme_gap_ratio
        )
        extreme_abs_gate = self.type2_gap_abs_gate_for_class(
            class_name,
            self.type2_extreme_gap_abs
        )
        score_gate = self.type2_score_gate_for_class(
            class_name,
            self.type_score_gate
        )
        extreme_score_gate = self.type2_score_gate_for_class(
            class_name,
            self.type2_extreme_score_gate
        )

        if self.type1_support_is_sufficient(class_name, obs_state):
            return False

        fill_gate = self.type2_fill_gate_for_class(class_name, self.type2_fill_low)
        fill_gate_extreme = self.type2_fill_gate_for_class(
            class_name, self.type2_extreme_fill_low
        )

        extreme_ready = (
            obs_state.get('updates', 0) >= self.type2_extreme_min_updates
            and len(obs_state.get('view_bins', set())) >= self.type2_extreme_min_view_bins
            and obs_state.get('type2_extreme_hits', 0) >= self.type2_extreme_min_hits
            and gap_ratio >= extreme_ratio_gate
            and gap_abs >= extreme_abs_gate
            and support_fill <= fill_gate_extreme
            and obs_state.get('type_score', 0.0) >= extreme_score_gate
        )
        if extreme_ready:
            return True

        if obs_state.get('updates', 0) < self.type2_min_updates_for_class(class_name):
            return False

        if len(obs_state.get('view_bins', set())) < self.type2_min_view_bins_for_class(class_name):
            return False

        # standard promotion path
        std_ready = (
            obs_state.get('type2_mismatch_hits', 0) >= self.type2_mismatch_hits_for_class(class_name)
            and gap_ratio >= gap_ratio_gate
            and gap_abs >= gap_abs_gate
            and support_fill <= fill_gate
            and obs_state.get('type_score', 0.0) >= score_gate
        )
        if std_ready:
            return True

        # multi-view fast path: consistent sparse support seen from ≥2 view
        # bins, even if score hasn't fully peaked yet. Requires a tighter fill.
        n_updates = obs_state.get('updates', 0)
        n_views = len(obs_state.get('view_bins', set()))
        mismatch_frac = (
            obs_state.get('type2_mismatch_hits', 0) / max(1, n_updates)
        )
        mv_fill_gate = self.type2_fill_gate_for_class(
            class_name, self.type2_fill_low * 0.78
        )
        # mv_fill_gate is already tight (0.78 * fill_gate), so this fast path
        # only fires for objects with consistently very sparse support across
        # multiple view angles — not objects that look sparse from one side
        multi_view_ready = (
            n_views >= 2
            and support_fill <= mv_fill_gate
            and mismatch_frac >= self.type2_multi_view_min_frac
            and gap_ratio >= gap_ratio_gate
            and gap_abs >= gap_abs_gate
            and obs_state.get('type_score', 0.0) >= score_gate * 0.85
            and obs_state.get('type2_mismatch_hits', 0) >= max(
                3, self.type2_mismatch_hits_for_class(class_name)
            )
        )
        return multi_view_ready

    def type2_ready_for_demotion(self, obs_state):

        if obs_state is None:
            return False

        if obs_state.get('type2_updates', 0) < self.type1_demote_min_updates_after_type2:
            return False

        if obs_state.get('type1_clear_hits', 0) < self.type1_demote_clear_hits:
            return False

        if obs_state.get('type_score', 0.0) > self.type1_demote_score_gate:
            return False

        bbox_span = obs_state.get('bbox_span')
        support_major = obs_state.get('support_major')
        support_view_span = obs_state.get('support_view_span')
        support_fill = obs_state.get('support_fill', 1.0)
        gap_support_span = support_view_span
        if gap_support_span is None or gap_support_span <= 0.05:
            gap_support_span = support_major

        if bbox_span is None or gap_support_span is None or gap_support_span <= 0.05:
            return False

        gap_ratio = bbox_span / max(0.05, gap_support_span)
        gap_abs = bbox_span - gap_support_span

        return (
            support_fill >= self.type1_demote_fill_high
            or (
                gap_ratio <= self.type1_demote_gap_ratio
                and gap_abs <= self.type1_demote_gap_abs
            )
        )

    def update_object_observation_state(self, obj, obs_meta):

        obs_state = self.clone_observation_state(obj.get('obs_state'))
        obj['obs_state'] = obs_state

        if obs_meta is None:
            return

        obs_state['updates'] += 1
        self.record_object_view_bin(obs_state, obs_meta.get('view_angle'))
        if obj.get('object_id') is not None:
            obs_state['confirmed_updates'] = min(
                1000,
                obs_state.get('confirmed_updates', 0) + 1
            )

        # geometry measurements update slowly once type2 so the displayed
        # obstacle size stays stable; fill still tracks at normal speed so
        # the type-classification evidence remains responsive
        geom_alpha = (
            self.type2_geom_alpha
            if obj.get('obstacle_type', 1) == 2
            else self.type_obs_alpha
        )
        obs_state['bbox_span'] = self.blend_scalar(
            obs_state.get('bbox_span'),
            obs_meta.get('bbox_span'),
            geom_alpha
        )
        obs_state['support_major'] = self.blend_scalar(
            obs_state.get('support_major'),
            obs_meta.get('support_major'),
            geom_alpha
        )
        obs_state['support_minor'] = self.blend_scalar(
            obs_state.get('support_minor'),
            obs_meta.get('support_minor'),
            geom_alpha
        )
        obs_state['support_view_span'] = self.blend_scalar(
            obs_state.get('support_view_span'),
            obs_meta.get('support_view_span'),
            geom_alpha
        )
        obs_state['support_fill'] = self.blend_scalar(
            obs_state.get('support_fill'),
            obs_meta.get('support_fill', 1.0),
            self.type_obs_alpha
        )

        bbox_span = obs_meta.get('bbox_span')
        support_major = max(0.0, obs_meta.get('support_major', 0.0))
        support_minor = max(0.0, obs_meta.get('support_minor', 0.0))
        support_view_span = obs_meta.get('support_view_span')
        support_fill = obs_meta.get('support_fill', 1.0)
        evidence = 0.0
        current_clear_type1 = False
        class_name = obj.get('class_name')

        gap_support_span = support_view_span
        if gap_support_span is None or gap_support_span <= 0.05:
            gap_support_span = support_major

        # A near-zero or sub-threshold support span (single lidar point on a
        # narrow pedestal) is itself strong type2 evidence — floor at 0.05
        # rather than skipping evidence entirely.  Require bbox_span > 0.15
        # to avoid computing ratios on noise.
        if gap_support_span is None or gap_support_span < 0.0:
            gap_support_span = 0.0
        gap_support_span_eff = max(0.05, gap_support_span)

        if bbox_span is not None and bbox_span > 0.15:
            gap_ratio = bbox_span / gap_support_span_eff
            gap_abs = bbox_span - gap_support_span
            round_ratio = bbox_span / max(0.05, support_minor)
            gap_ratio_gate = self.type2_gap_ratio_gate_for_class(
                class_name,
                self.type2_gap_ratio
            )
            gap_abs_gate = self.type2_gap_abs_gate_for_class(
                class_name,
                self.type2_gap_abs
            )
            mismatch_ratio_gate = self.type2_gap_ratio_gate_for_class(
                class_name,
                self.type2_mismatch_gap_ratio
            )
            mismatch_abs_gate = self.type2_gap_abs_gate_for_class(
                class_name,
                self.type2_mismatch_gap_abs
            )
            extreme_ratio_gate = self.type2_gap_ratio_gate_for_class(
                class_name,
                self.type2_extreme_gap_ratio
            )
            extreme_abs_gate = self.type2_gap_abs_gate_for_class(
                class_name,
                self.type2_extreme_gap_abs
            )
            current_clear_type1 = (
                support_fill >= self.type1_demote_fill_high
                or (
                    gap_ratio <= self.type1_demote_gap_ratio
                    and gap_abs <= self.type1_demote_gap_abs
                )
            )
            fill_gate_mismatch = self.type2_fill_gate_for_class(
                class_name, self.type2_mismatch_fill_low
            )
            fill_gate_extreme = self.type2_fill_gate_for_class(
                class_name, self.type2_extreme_fill_low
            )
            persistent_mismatch = (
                gap_ratio >= mismatch_ratio_gate
                and gap_abs >= mismatch_abs_gate
                and support_fill <= fill_gate_mismatch
                and not current_clear_type1
            )
            extreme_mismatch = (
                gap_ratio >= extreme_ratio_gate
                and gap_abs >= extreme_abs_gate
                and support_fill <= fill_gate_extreme
            )

            if persistent_mismatch:
                obs_state['type2_mismatch_hits'] = min(
                    30,
                    obs_state.get('type2_mismatch_hits', 0) + 1
                )
                evidence += 0.55
            else:
                obs_state['type2_mismatch_hits'] = max(
                    0,
                    obs_state.get('type2_mismatch_hits', 0) - 1
                )

            if extreme_mismatch:
                obs_state['type2_extreme_hits'] = min(
                    20,
                    obs_state.get('type2_extreme_hits', 0) + 1
                )
                evidence += 1.2
            else:
                obs_state['type2_extreme_hits'] = max(
                    0,
                    obs_state.get('type2_extreme_hits', 0) - 1
                )

            if gap_ratio >= gap_ratio_gate and gap_abs >= gap_abs_gate:
                evidence += 1.4 * min(1.8, gap_ratio / gap_ratio_gate)

            fill_gate = self.type2_fill_gate_for_class(class_name, self.type2_fill_low)
            if support_fill < fill_gate:
                evidence += 0.9 + 0.9 * (
                    (fill_gate - support_fill)
                    / max(0.05, fill_gate)
                )

            if round_ratio >= self.type2_round_ratio:
                evidence += 0.45 * min(1.6, round_ratio / self.type2_round_ratio)

            # multi-view consistency bonus: the current raw fill must be
            # clearly sparse (below fill_cap) AND most historical updates must
            # have been mismatch hits. The hard fill cap stops objects that
            # only occasionally look sparse from collecting the bonus.
            n_updates = obs_state.get('updates', 0)
            n_views = len(obs_state.get('view_bins', set()))
            mismatch_frac = (
                obs_state.get('type2_mismatch_hits', 0) / max(1, n_updates)
            )
            if (
                n_views >= 2
                and support_fill <= self.type2_multi_view_fill_cap
                and mismatch_frac >= self.type2_multi_view_min_frac
            ):
                evidence += self.type2_multi_view_bonus
        else:
            obs_state['type2_mismatch_hits'] = max(
                0,
                obs_state.get('type2_mismatch_hits', 0) - 1
            )
            obs_state['type2_extreme_hits'] = max(
                0,
                obs_state.get('type2_extreme_hits', 0) - 1
            )

        if evidence > 0.0:
            obs_state['type_score'] = min(8.0, obs_state['type_score'] + 0.36 * evidence)
        else:
            # type2 objects decay more slowly so accumulated confidence
            # is not lost after a handful of occluded or partial views
            is_type2 = obj.get('obstacle_type', 1) == 2
            decay = self.type2_score_decay if is_type2 else 0.24
            obs_state['type_score'] = max(0.0, obs_state['type_score'] - decay)

        # once promoted, keep the score above a floor so a few bad views
        # cannot immediately undo many consistent type2 observations
        if obj.get('obstacle_type', 1) == 2:
            obs_state['type_score'] = max(
                self.type2_score_floor, obs_state['type_score']
            )
            obs_state['type2_updates'] = min(
                1000,
                obs_state.get('type2_updates', 0) + 1
            )
            if current_clear_type1:
                obs_state['type1_clear_hits'] = min(
                    20,
                    obs_state.get('type1_clear_hits', 0) + 1
                )
                obs_state['type_score'] = max(
                    self.type2_score_floor,
                    obs_state['type_score'] - 0.12
                )
            else:
                obs_state['type1_clear_hits'] = max(
                    0,
                    obs_state.get('type1_clear_hits', 0) - 1
                )

            if self.type2_ready_for_demotion(obs_state):
                obj['obstacle_type'] = 1
                obs_state['type2_updates'] = 0
                obs_state['type1_clear_hits'] = 0
                obs_state['type2_mismatch_hits'] = 0
                obs_state['type2_extreme_hits'] = 0
                self.get_logger().info(
                    f'[TYPE1 DEMOTE] class={obj.get("class_name", "?")} '
                    f'updates={obs_state.get("updates", 0)} '
                    f'confirmed_updates={obs_state.get("confirmed_updates", 0)} '
                    f'views={len(obs_state.get("view_bins", set()))} '
                    f'score={obs_state.get("type_score", 0.0):.2f} '
                    f'bbox={obs_state.get("bbox_span") if obs_state.get("bbox_span") is not None else float("nan"):.2f} '
                    f'view_span={obs_state.get("support_view_span") if obs_state.get("support_view_span") is not None else float("nan"):.2f} '
                    f'fill={obs_state.get("support_fill", 1.0):.2f}'
                )
            return

        obs_state['type2_updates'] = 0
        obs_state['type1_clear_hits'] = 0

        if self.type2_ready_for_promotion(obs_state, obj.get('class_name')):
            obj['obstacle_type'] = 2
            obs_state['type2_updates'] = 0
            obs_state['type1_clear_hits'] = 0
            self.get_logger().info(
                f'[TYPE2 PROMOTE] class={obj.get("class_name", "?")} '
                f'updates={obs_state.get("updates", 0)} '
                f'confirmed_updates={obs_state.get("confirmed_updates", 0)} '
                f'views={len(obs_state.get("view_bins", set()))} '
                f'score={obs_state.get("type_score", 0.0):.2f} '
                f'mismatch_hits={obs_state.get("type2_mismatch_hits", 0)} '
                f'extreme_hits={obs_state.get("type2_extreme_hits", 0)} '
                f'bbox={obs_state.get("bbox_span") if obs_state.get("bbox_span") is not None else float("nan"):.2f} '
                f'view_span={obs_state.get("support_view_span") if obs_state.get("support_view_span") is not None else float("nan"):.2f} '
                f'fill={obs_state.get("support_fill", 1.0):.2f}'
            )

    def effective_object_hypothesis(self, obj):

        center = obj.get('geom_centroid')
        support_pts = list(obj.get('support_points', []))
        support_hypothesis = self.local_support_hypothesis(
            obj['class_name'],
            center,
            support_pts,
            pad=0.02
        )

        if obj.get('obstacle_type', 1) != 2:
            return support_hypothesis

        obs_state = obj.get('obs_state', {})
        bbox_span = obs_state.get('bbox_span')
        support_major = obs_state.get('support_major') or 0.0
        support_minor = obs_state.get('support_minor') or 0.0
        # A narrow-pedestal type2 object (e.g. office chair) will have
        # support_major ≈ 0 — that is correct and expected.  Do NOT fall back
        # to the tiny local_support_hypothesis; use bbox_span to infer extent.
        if center is None or bbox_span is None or bbox_span <= 0.10:
            return support_hypothesis

        profile = self.get_class_profile(obj['class_name'])
        support_angle = 0.0
        if support_hypothesis is not None:
            support_angle = support_hypothesis['angle']
        elif support_pts:
            _, _, support_angle = self.support_major_minor(support_pts, center)

        major_span = max(
            support_major,
            min(1.15 * profile['l_range'][1], bbox_span)
        )
        gap_ratio = bbox_span / max(0.05, support_major)

        if gap_ratio >= self.type2_round_ratio:
            minor_span = max(
                support_minor if support_minor is not None else 0.0,
                min(1.05 * profile['l_range'][1], 0.80 * bbox_span)
            )
        else:
            minor_span = max(
                support_minor if support_minor is not None else 0.0,
                min(profile['l_range'][1], max(profile['s_value'], 0.58 * bbox_span))
            )

        if minor_span > major_span:
            major_span, minor_span = minor_span, major_span
            support_angle = self.normalize_half_turn(support_angle + 0.5 * math.pi)

        return {
            'center': center,
            'angle': support_angle,
            'half_l': 0.5 * major_span,
            'half_s': 0.5 * minor_span,
        }

    def outline_points_from_hypothesis(self, hypothesis, obstacle_type=1):

        if hypothesis is None:
            return []

        if obstacle_type != 2:
            return self.oriented_box_corners(hypothesis)

        center = hypothesis['center']
        angle = hypothesis['angle']
        half_l = hypothesis['half_l']
        half_s = hypothesis['half_s']
        ca = math.cos(angle)
        sa = math.sin(angle)
        points = []

        for i in range(12):
            theta = 2.0 * math.pi * i / 12.0
            u = half_l * math.cos(theta)
            v = half_s * math.sin(theta)
            points.append((
                center[0] + u * ca - v * sa,
                center[1] + u * sa + v * ca,
            ))

        return points

    def map_world_to_grid(self, x, y):

        if self.base_map is None:
            return None

        info = self.base_map.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        return gx, gy

    def map_in_bounds(self, gx, gy):

        if self.base_map is None:
            return False

        info = self.base_map.info
        return 0 <= gx < info.width and 0 <= gy < info.height

    def map_is_occupied_cell(self, gx, gy):

        if self.base_map is None or not self.map_in_bounds(gx, gy):
            return False

        idx = gy * self.base_map.info.width + gx
        return self.base_map.data[idx] >= self.map_occupied_thresh

    def map_occ_run_cells(self, gx, gy, dx, dy, max_steps):

        if not self.map_is_occupied_cell(gx, gy):
            return 0

        length = 1

        for sign in (-1, 1):
            x = gx
            y = gy

            for _ in range(max_steps):
                x += sign * dx
                y += sign * dy
                if not self.map_is_occupied_cell(x, y):
                    break
                length += 1

        return length

    def map_local_occ_count(self, gx, gy, radius_cells):

        count = 0

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if self.map_is_occupied_cell(gx + dx, gy + dy):
                    count += 1

        return count

    def point_on_structural_map(self, x, y, class_name):

        cell = self.map_world_to_grid(x, y)
        if cell is None:
            return False

        gx, gy = cell
        if not self.map_is_occupied_cell(gx, gy):
            return False

        resolution = max(1e-6, self.base_map.info.resolution)
        run_thresh_m = max(
            self.structural_min_run_m,
            self.structural_run_scale * self.class_max_diag(class_name)
        )
        run_thresh_m = min(self.structural_max_run_m, run_thresh_m)
        max_steps = max(4, int(run_thresh_m / resolution) + 2)

        horiz_run_m = self.map_occ_run_cells(gx, gy, 1, 0, max_steps) * resolution
        vert_run_m = self.map_occ_run_cells(gx, gy, 0, 1, max_steps) * resolution
        diag1_run_m = self.map_occ_run_cells(gx, gy, 1, 1, max_steps) * resolution
        diag2_run_m = self.map_occ_run_cells(gx, gy, 1, -1, max_steps) * resolution

        axis_run_m = max(horiz_run_m, vert_run_m)
        diag_run_m = max(diag1_run_m, diag2_run_m)

        if axis_run_m < run_thresh_m and diag_run_m < self.structural_diag_run_relax * run_thresh_m:
            return False

        local_radius = max(1, int(0.10 / resolution))
        return self.map_local_occ_count(gx, gy, local_radius) >= self.structural_local_occ_cells

    def structural_back_margin_for_view(self, class_name, center, sensor_origin):

        _, _, range_m = self.view_frame_axes(center, sensor_origin)
        if range_m is None:
            return None

        profile = self.get_class_profile(class_name)
        base_margin = self.max_extra_depth_for_range(range_m) + self.structural_view_back_pad
        class_cap = 0.30 * profile['s_value']

        return min(
            self.structural_view_back_cap,
            max(0.08, min(class_cap, base_margin))
        )

    def structural_point_behind_visible_front(
        self,
        class_name,
        center,
        sensor_origin,
        structural_pt,
        anchor_pts
    ):

        if center is None or sensor_origin is None or not anchor_pts:
            return False

        profile = self.get_class_profile(class_name)
        across_tol = max(0.08, self.structural_view_across_scale * profile['s_value'])
        back_margin = self.structural_back_margin_for_view(
            class_name,
            center,
            sensor_origin
        )
        if back_margin is None:
            return False

        along_pt, across_pt = self.point_view_coords(
            structural_pt,
            center,
            sensor_origin
        )
        if along_pt is None:
            return False

        front_along = None
        global_front = None

        for anchor_pt in anchor_pts:
            along_anchor, across_anchor = self.point_view_coords(
                anchor_pt,
                center,
                sensor_origin
            )
            if along_anchor is None:
                continue

            if global_front is None or along_anchor < global_front:
                global_front = along_anchor

            if abs(across_anchor - across_pt) <= across_tol:
                if front_along is None or along_anchor < front_along:
                    front_along = along_anchor

        if front_along is None:
            front_along = global_front

        if front_along is None:
            return False

        return along_pt > front_along + back_margin

    def exclude_support_on_structural_map(self, class_name, centroid, pts, sensor_origin=None):

        if self.base_map is None or not pts:
            return list(pts)

        profile = self.get_class_profile(class_name)
        center_keep_radius = max(
            0.10,
            self.structural_center_keep_frac * profile['s_value']
        )
        link_radius = max(0.08, 0.12 * profile['s_value'])

        non_struct_pts = []
        structural_pts = []

        for pt in pts:
            if self.point_on_structural_map(pt[0], pt[1], class_name):
                structural_pts.append(pt)
            else:
                non_struct_pts.append(pt)

        if not structural_pts:
            return list(pts)

        if not non_struct_pts:
            if centroid is None:
                return []
            return [
                pt for pt in structural_pts
                if math.hypot(pt[0] - centroid[0], pt[1] - centroid[1]) <= center_keep_radius
            ]

        center_ref = centroid
        if center_ref is None:
            center_ref = self.support_box_center(non_struct_pts)

        max_object_radius = 0.0
        if center_ref is not None:
            max_object_radius = max(
                math.hypot(pt[0] - center_ref[0], pt[1] - center_ref[1])
                for pt in non_struct_pts
            )

        keep = list(non_struct_pts)

        for pt in structural_pts:
            if center_ref is not None and math.hypot(pt[0] - center_ref[0], pt[1] - center_ref[1]) <= center_keep_radius:
                keep.append(pt)
                continue

            if self.structural_point_behind_visible_front(
                class_name,
                center_ref,
                sensor_origin,
                pt,
                non_struct_pts
            ):
                continue

            if (
                center_ref is not None
                and math.hypot(pt[0] - center_ref[0], pt[1] - center_ref[1]) > max_object_radius + link_radius
            ):
                continue

            near_object = False
            for obj_pt in non_struct_pts:
                if math.hypot(pt[0] - obj_pt[0], pt[1] - obj_pt[1]) <= link_radius:
                    near_object = True
                    break

            if near_object:
                keep.append(pt)

        return keep

    def count_structural_support_points(self, class_name, pts):

        if self.base_map is None or not pts:
            return 0

        count = 0

        for pt in pts:
            if self.point_on_structural_map(pt[0], pt[1], class_name):
                count += 1

        return count

    def exclude_support_from_other_objects(self, class_name, pts, owner_object_id=None):

        if not pts:
            return []

        out = []

        for pt in pts:
            if not self.support_point_claimed_by_other_object(
                class_name,
                pt,
                owner_object_id=owner_object_id
            ):
                out.append(pt)

        return out

    def derive_support_geometry(self, class_name, assoc_centroid, raw_pts, owner_object_id=None):

        pts = list(raw_pts)
        if not pts:
            return [], assoc_centroid

        pts = self.exclude_support_from_other_objects(
            class_name,
            pts,
            owner_object_id=owner_object_id
        )

        support_center = self.support_box_center(pts)
        center = self.geometry_center_from_assoc_and_support(
            class_name,
            assoc_centroid,
            pts
        )
        if center is None:
            center = support_center

        if center is None:
            return [], assoc_centroid

        pts = self.exclude_support_on_structural_map(
            class_name,
            center,
            pts
        )
        if not pts:
            return [], center

        angle = self.support_principal_angle(pts, support_center)
        pts = self.filter_points_in_class_footprint(
            pts,
            center,
            class_name,
            angle
        )
        if not pts:
            return [], center

        center = self.geometry_center_from_assoc_and_support(
            class_name,
            assoc_centroid,
            pts
        )
        if center is None:
            return [], assoc_centroid

        pts = self.filter_points_near_center(
            pts,
            center,
            self.support_trim_radius_for_class(class_name)
        )
        if not pts:
            return [], center

        angle = self.support_principal_angle(pts, center)
        pts = self.filter_points_in_class_footprint(
            pts,
            center,
            class_name,
            angle
        )
        pts = self.exclude_support_on_structural_map(
            class_name,
            center,
            pts
        )
        if not pts:
            return [], center

        center = self.geometry_center_from_assoc_and_support(
            class_name,
            assoc_centroid,
            pts
        )
        return pts, center

    def prune_structural_support_memory(self, obj, filtered_pts):

        raw_pts = list(obj.get('raw_support_points', []))
        if self.base_map is None or not raw_pts or not filtered_pts:
            return

        profile = self.get_class_profile(obj['class_name'])
        center = obj.get('geom_centroid')
        link_radius = max(0.08, 0.15 * profile['s_value'])
        center_keep_radius = max(0.10, 0.18 * profile['s_value'])
        kept = []

        for pt in raw_pts:
            if not self.point_on_structural_map(pt[0], pt[1], obj['class_name']):
                kept.append(pt)
                continue

            near_filtered = False
            for good_pt in filtered_pts:
                if math.hypot(pt[0] - good_pt[0], pt[1] - good_pt[1]) <= link_radius:
                    near_filtered = True
                    break

            if near_filtered:
                kept.append(pt)
                continue

            if (
                center is not None
                and len(filtered_pts) < 4
                and math.hypot(pt[0] - center[0], pt[1] - center[1]) <= center_keep_radius
            ):
                kept.append(pt)

        if not kept:
            kept = list(filtered_pts)

        obj['raw_support_points'] = self.cap_points_preserve(
            kept,
            filtered_pts,
            self.support_raw_max_points_per_object
        )

    def prune_foreign_support_memory(self, obj, filtered_pts):

        raw_pts = list(obj.get('raw_support_points', []))
        if not raw_pts or not filtered_pts:
            return

        kept = [
            pt for pt in raw_pts
            if not self.support_point_claimed_by_other_object(
                obj['class_name'],
                pt,
                owner_object_id=obj.get('object_id')
            )
        ]

        if len(kept) == len(raw_pts) or not kept:
            return

        obj['raw_support_points'] = self.cap_points_preserve(
            kept,
            filtered_pts,
            self.support_raw_max_points_per_object
        )

    def should_keep_current_geometry(self, obj, current_pts, filtered_pts):

        if not current_pts:
            return False

        class_name = obj['class_name']
        current_foreign = self.count_foreign_support_points(
            class_name,
            current_pts,
            owner_object_id=obj.get('object_id')
        )
        filtered_foreign = self.count_foreign_support_points(
            class_name,
            filtered_pts,
            owner_object_id=obj.get('object_id')
        )

        if current_foreign > filtered_foreign:
            return False

        current_score = self.support_coverage_score(current_pts)
        candidate_score = self.support_coverage_score(filtered_pts)

        if candidate_score >= 0.72 * current_score:
            return False

        current_struct = self.count_structural_support_points(class_name, current_pts)
        candidate_struct = self.count_structural_support_points(class_name, filtered_pts)

        if (
            self.base_map is not None
            and candidate_score >= 0.52 * current_score
            and current_struct >= candidate_struct + 2
        ):
            return False

        return True

    def convex_hull_points(self, pts):

        if len(pts) <= 2:
            return list(pts)

        pts = sorted(set((round(p[0], 5), round(p[1], 5)) for p in pts))
        if len(pts) <= 2:
            return list(pts)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    def support_coverage_score(self, pts):

        if not pts:
            return 0.0

        center = self.support_box_center(pts)
        major_span, minor_span, _ = self.support_axis_spans(pts, center)
        return (
            major_span * minor_span
            + 0.35 * max(major_span, minor_span)
            + 0.01 * len(pts)
        )

    def merge_raw_support_points(self, obj, new_pts):

        raw_pts = self.append_unique_points(
            obj.get('raw_support_points', obj.get('support_points', [])),
            new_pts,
            limit=None
        )

        preserve_pts = self.convex_hull_points(obj.get('support_points', []))
        return self.cap_points_preserve(
            raw_pts,
            preserve_pts,
            self.support_raw_max_points_per_object
        )

    # gets the best scan and time for detection
    def get_best_scan_for_detection(self, det):
        if not self.scan_buffer:
            return None, None

        # converts header timestamp to seconds in float
        det_t = self.stamp_to_sec(det.header.stamp)

        best_scan = None
        best_dt = 1e9

        # iterates through scan finds scan with closest timestamp to detection
        for scan in self.scan_buffer:
            scan_t = self.stamp_to_sec(scan.header.stamp)
            dt = abs(det_t - scan_t)

            if dt < best_dt:
                best_dt = dt
                best_scan = scan

        return best_scan, best_dt

    # returns range bucket used by front logic
    def range_bucket(self, r):
        if r < self.front_select_close_r:
            return 'close'
        elif r < self.front_select_med_r:
            return 'med'
        return 'far'

    # this is for clusters
    def depth_margin_for_range(self, r):
        bucket = self.range_bucket(r)

        # if object is close allow only small depth margin
        if bucket == 'close':
            return self.front_select_depth_margin_close
        # if object is med
        elif bucket == 'med':
            return self.front_select_depth_margin_med
        # if object is far
        return self.front_select_depth_margin_far

    # this is for selecting points before clustering
    def max_extra_depth_for_range(self, r):
        bucket = self.range_bucket(r)

        # if object is close allow only small extra depth
        if bucket == 'close':
            return self.layer_max_extra_depth_close
        # if object is medium
        elif bucket == 'med':
            return self.layer_max_extra_depth_med
        # if object is far
        return self.layer_max_extra_depth_far

    # gets front band for centroid points based on depth
    def front_band_for_range(self, r):
        bucket = self.range_bucket(r)

        if bucket == 'close':
            return self.front_band_close
        elif bucket == 'med':
            return self.front_band_med
        return self.front_band_far

    def get_class_profile(self, class_name):

        if class_name in self.class_profiles:
            return self.class_profiles[class_name]

        if class_name not in self.profile_fallback_warned:
            self.get_logger().warn(
                f'no class profile for "{class_name}", using default_profile'
            )
            self.profile_fallback_warned.add(class_name)

        return self.default_class_profile

    def class_nominal_sides(self, class_name):

        profile = self.get_class_profile(class_name)
        return profile['s_value'], profile['l_value']

    def class_max_sides(self, class_name):

        profile = self.get_class_profile(class_name)
        return profile['s_range'][1], profile['l_range'][1]

    def class_max_diag(self, class_name):

        s_max, l_max = self.class_max_sides(class_name)
        return math.hypot(s_max, l_max)

    def class_aspect_ratio(self, class_name):

        s_value, l_value = self.class_nominal_sides(class_name)
        return l_value / max(0.05, s_value)

    def class_elongation(self, class_name):

        aspect_ratio = self.class_aspect_ratio(class_name)
        return max(0.0, min(1.0, aspect_ratio - 1.0))

    def class_prefers_grouping(self, class_name):

        return (
            self.class_aspect_ratio(class_name) >= 1.20
            and self.class_max_diag(class_name) >= 1.20
        )

    def class_size_scale(self, class_name):

        default_max_side = max(
            self.default_class_profile['s_range'][1],
            self.default_class_profile['l_range'][1]
        )
        class_max_side = max(self.class_max_sides(class_name))

        if default_max_side <= 0.0:
            return 1.0

        return max(0.75, min(2.0, class_max_side / default_max_side))

    def wedge_trim_frac_for_class(self, class_name):

        size_scale = max(1.0, self.class_size_scale(class_name))
        return max(0.04, self.wedge_bbox_trim_frac / size_scale)

    def wedge_max_half_angle_for_class(self, class_name):

        size_scale = max(1.0, self.class_size_scale(class_name))
        return min(0.38, self.wedge_max_half_angle * math.sqrt(size_scale))

    def wedge_expand_angle_for_class(self, class_name):

        size_scale = self.class_size_scale(class_name)
        if size_scale <= 1.0:
            return 0.0

        return min(0.06, 0.01 + 0.02 * (size_scale - 1.0))

    def cluster_max_diag_for_class(self, class_name):

        return max(self.cluster_gate_max_diag, self.class_max_diag(class_name))

    def temp_match_radius_for_class(self, class_name):

        s_max, l_max = self.class_max_sides(class_name)
        elongation = self.class_elongation(class_name)
        radius = max(
            self.memory_temp_match_radius,
            0.50 * max(s_max, l_max),
            (0.25 + 0.25 * elongation) * self.class_max_diag(class_name)
        )
        return radius

    def confirmed_match_radius_for_class(self, class_name):

        s_max, l_max = self.class_max_sides(class_name)
        elongation = self.class_elongation(class_name)
        radius = max(
            self.memory_confirmed_match_radius,
            0.65 * max(s_max, l_max),
            (0.55 + 0.15 * elongation) * self.class_max_diag(class_name)
        )
        return radius

    def too_close_radius_for_class(self, class_name):

        s_max, l_max = self.class_max_sides(class_name)
        elongation = self.class_elongation(class_name)
        radius = max(
            self.memory_too_close_confirmed_radius,
            0.40 * max(s_max, l_max),
            (0.20 + 0.20 * elongation) * self.class_max_diag(class_name)
        )
        return radius

    def merge_radius_for_class(self, class_name):

        s_max, l_max = self.class_max_sides(class_name)
        elongation = self.class_elongation(class_name)
        radius = max(
            self.memory_merge_radius,
            0.45 * max(s_max, l_max),
            (0.50 + 0.12 * elongation) * self.class_max_diag(class_name)
        )
        return radius

    def new_object_suppress_radius_for_class(self, class_name):

        return max(
            self.memory_new_object_suppress_radius,
            0.60 * self.class_max_diag(class_name)
        )

    def support_keep_radius_for_class(self, class_name):

        return max(
            self.support_keep_radius,
            0.45 * self.class_max_diag(class_name)
        )

    def support_trim_radius_for_class(self, class_name):

        scale = 0.60
        if self.class_prefers_grouping(class_name):
            scale = 0.78

        return max(
            self.support_trim_radius,
            scale * self.class_max_diag(class_name)
        )

    def support_reanchor_radius_for_class(self, class_name):

        scale = 0.70
        if self.class_prefers_grouping(class_name):
            scale = 0.85

        return max(
            self.support_keep_radius_for_class(class_name),
            scale * self.class_max_diag(class_name)
        )

    def group_stats(self, pts):

        # get x y depth and angle values from points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        rs = [p[2] for p in pts]
        angs = [p[3] for p in pts]

        # width and height of cluster spread
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        ang_span = max(angs) - min(angs)

        # diagonal of cluster
        diag = math.hypot(dx, dy)
        # depth spread
        r_span = max(rs) - min(rs)

        # average angle of cluster
        mean_ang = sum(angs) / len(angs)

        # average depth
        mean_r = sum(rs) / len(rs)
        # minimum depth
        min_r = min(rs)

        return {
            'dx': dx,
            'dy': dy,
            'ang_span': ang_span,
            'diag': diag,
            'r_span': r_span,
            'mean_ang': mean_ang,
            'mean_r': mean_r,
            'min_r': min_r,
            'n': len(pts),
        }

    def valid_group(self, stats, theta_center, class_name):

        # compact objects (narrow pedestal) may give a single lidar return;
        # allow 1-point clusters so that return is not silently discarded
        gate_min_pts = (
            1 if self.type2_gate_scale_for_class(class_name) < 0.85
            else self.cluster_gate_min_pts
        )
        if stats['n'] < gate_min_pts:
            return False

        # reject if cluster angle too far from center bbox
        if abs(stats['mean_ang'] - theta_center) > self.cluster_gate_max_center_angle_error:
            return False

        # reject if cluster is too large
        if stats['diag'] > self.cluster_max_diag_for_class(class_name):
            return False

        # reject if cluster depth spread is too large
        if stats['r_span'] > self.cluster_gate_max_range_span:
            return False

        return True

    def score_cluster(self, stats, theta_center, class_name):

        # angle error from bbox center
        angle_err = abs(stats['mean_ang'] - theta_center)
        max_diag = max(0.10, self.class_max_diag(class_name))
        norm_diag = stats['diag'] / max_diag
        norm_span = stats['r_span'] / max(0.05, self.cluster_gate_max_range_span)

        return (
            # more points
            2.5 * stats['n']
            # - angle mismatch
            - 13.0 * angle_err
            # - far front point
            - 2.8 * stats['min_r']
            # - far average depth
            - 1.0 * stats['mean_r']
            # scale geometry penalties by class footprint so larger classes
            # are not punished as harshly as chair-sized objects
            - 3.0 * norm_diag
            - 2.0 * norm_span
        )

    def bbox_span_m_for_detection(self, det, range_m):

        if range_m is None or not math.isfinite(range_m) or range_m <= 0.0:
            return None

        xmin = float(det.xmin)
        xmax = float(det.xmax)
        if xmax < xmin:
            xmin, xmax = xmax, xmin

        theta_left = self.pixel_to_bearing(xmin)
        theta_right = self.pixel_to_bearing(xmax)
        theta_span = abs(theta_right - theta_left)

        if theta_span <= 0.0:
            return None

        return 2.0 * range_m * math.tan(0.5 * theta_span)

    def span_m_for_angle_bounds(self, theta_min, theta_max, range_m):

        if range_m is None or not math.isfinite(range_m) or range_m <= 0.0:
            return None

        theta_span = abs(theta_max - theta_min)
        if theta_span <= 0.0:
            return None

        return 2.0 * range_m * math.tan(0.5 * theta_span)

    def bbox_consistency_for_stats(self, det, stats):

        bbox_span = self.bbox_span_m_for_detection(det, stats['mean_r'])
        if bbox_span is None:
            return {
                'bbox_span': None,
                'support_span': 0.0,
                'support_fill': 1.0,
                'penalty': 0.0,
                'hard_reject': False,
            }

        profile = self.get_class_profile(det.class_name)
        s_min, _ = profile['s_range']
        _, l_max = profile['l_range']

        min_span = max(0.80 * s_min, 0.70 * profile['s_value'])
        max_span = 1.35 * l_max
        support_span = max(stats['dx'], stats['dy'], 0.75 * stats['diag'])
        support_fill = support_span / max(0.05, bbox_span)

        penalty = 0.0

        if bbox_span < min_span:
            penalty += 12.0 * (min_span - bbox_span) / max(0.10, min_span)

        if bbox_span > max_span:
            penalty += 3.0 * (bbox_span - max_span) / max(0.10, max_span)

        elongation = self.class_elongation(det.class_name)
        min_fill = 0.20 + 0.10 * elongation
        if bbox_span >= min_span and support_fill < min_fill:
            penalty += 6.0 * (min_fill - support_fill) / max(0.05, min_fill)

        hard_reject = (
            elongation >= 0.25
            and bbox_span < min_span
        )

        return {
            'bbox_span': bbox_span,
            'support_span': support_span,
            'support_fill': support_fill,
            'penalty': penalty,
            'hard_reject': hard_reject,
        }

    # builds cluster candidate once so we dont recompute stats repeatedly
    def build_cluster_candidate(self, pts, theta_center, class_name):
        stats = self.group_stats(pts)
        angle_err = abs(stats['mean_ang'] - theta_center)
        ok = self.valid_group(stats, theta_center, class_name)
        score = self.score_cluster(stats, theta_center, class_name)

        return {
            'pts': pts,
            'stats': stats,
            'angle_err': angle_err,
            'score': score,
            'valid': ok,
        }

    # ros marker with basic parameters
    def make_marker(self, ns, marker_id, frame_id, marker_type, color, scale_x):

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = int(marker_id)
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.scale.x = scale_x
        marker.color = color

        return marker

    # publishes estimated object centroid marker in rviz
    def publish_marker(self, object_id, x, y, frame_id):

        marker = self.make_marker(
            ns='object_estimate',
            marker_id=object_id,
            frame_id=frame_id,
            marker_type=Marker.SPHERE,
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            scale_x=0.2
        )

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        self.marker_pub.publish(marker)

    # publishes wedge lines in rviz showing lidar search region
    def publish_wedge(self, theta_min, theta_max, frame_id):

        for i, theta in enumerate([theta_min, theta_max]):

            marker = self.make_marker(
                ns='bbox_wedge',
                marker_id=i,
                frame_id=frame_id,
                marker_type=Marker.LINE_STRIP,
                color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
                scale_x=0.02
            )

            p0 = Point()
            p0.x = 0.0
            p0.y = 0.0
            p0.z = 0.0

            p1 = Point()
            p1.x = 5.0 * math.cos(theta)
            p1.y = 5.0 * math.sin(theta)
            p1.z = 0.0

            marker.points = [p0, p1]
            self.marker_pub.publish(marker)

    # gets all support points in that wedge
    def extract_support_points(self, scan, theta_min, theta_max):

        pts = []

        # i is index and r is distance
        for i, r in enumerate(scan.ranges):

            if not math.isfinite(r):
                continue

            if r < scan.range_min or r > scan.range_max:
                continue

            # calculates each angle for each point
            angle = scan.angle_min + i * scan.angle_increment

            # if angle within wedge, get x and y coord and append
            if theta_min <= angle <= theta_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                pts.append((x, y, r, angle, i))

        return pts

    # generic cluster splitting helper so normal cluster and tight front cluster use same logic
    def split_by_thresholds(self, pts, scan_gap, depth_jump, point_gap, min_pts):

        if not pts:
            return []

        # sort points by scan index
        pts = sorted(pts, key=lambda p: p[4])

        clusters = []
        current = [pts[0]]
        for pt in pts[1:]:
            prev = current[-1]
            # groupd nearby points into clusters based on scan index, depth jump and xy distance
            same_cluster = (
                # lidar beams should be close in index
                (pt[4] - prev[4]) <= scan_gap and
                # depth should be minimal
                abs(pt[2] - prev[2]) <= depth_jump and
                # xy distance should be minimal
                math.hypot(pt[0] - prev[0], pt[1] - prev[1]) <= point_gap
            )

            # if sat then add th epoint or else new cluster
            if same_cluster:
                current.append(pt)
            else:
                clusters.append(current)
                current = [pt]

        clusters.append(current)
        # remove small clusters
        return [c for c in clusters if len(c) >= min_pts]

    def cluster_points(self, pts, min_pts=None):

        if min_pts is None:
            min_pts = self.cluster_min_pts
        return self.split_by_thresholds(
            pts,
            self.cluster_scan_gap,
            self.cluster_depth_jump,
            self.cluster_point_gap,
            min_pts
        )

    def build_depth_layer_clusters(self, raw_pts, max_layers=3, min_pts=None):
        if not raw_pts:
            return []

        # sort by depth
        pts = sorted(raw_pts, key=lambda p: p[2])

        layers = []
        current = [pts[0]]

        # for every point, we build depth layers based on the depth band
        for pt in pts[1:]:
            prev = current[-1]
            # if point within band, we add it if not add a new layer
            if abs(pt[2] - prev[2]) <= self.layer_depth_band:
                current.append(pt)
            else:
                layers.append(current)
                current = [pt]

        layers.append(current)

        if not layers:
            return []

        # get the front layer
        front_r = min(p[2] for p in layers[0])

        # based on front layer, calc max extra depth we can use
        max_extra_depth = self.max_extra_depth_for_range(front_r)

        kept_layers = []

        for layer in layers[:max_layers]:
            # closest point in layer
            layer_min_r = min(p[2] for p in layer)
            # keep only the front layers
            if layer_min_r <= front_r + max_extra_depth:
                kept_layers.append(layer)

        candidate_clusters = []

        for layer in kept_layers:
            # we cluster those points in that layer and then return the clusters
            clusters = self.cluster_points(layer, min_pts=min_pts)
            candidate_clusters.extend(clusters)

        return candidate_clusters

    def split_cluster_tight(self, pts):
        return self.split_by_thresholds(
            pts,
            self.front_cluster_scan_gap,
            self.front_cluster_depth_jump,
            self.front_cluster_point_gap,
            1
        )

    # gets the front points of the best cluster
    def front_subcluster_points(self, pts):
        if not pts:
            return []
        # closest depth
        min_r = min(p[2] for p in pts)

        # front band size based on distance (class based as well)
        front_band = self.front_band_for_range(min_r)

        # keep points very close to the front
        front_pts = [p for p in pts if p[2] <= min_r + front_band]
        if not front_pts:
            front_pts = pts

        # split front points into smaller clusters
        subclusters = self.split_cluster_tight(front_pts)

        best = None
        best_min_r = 1e9
        best_mean_r = 1e9
        # choose the front most cluster
        for sc in subclusters:
            # closest point
            sc_min_r = min(p[2] for p in sc)
            # average depth
            sc_mean_r = sum(p[2] for p in sc) / len(sc)

            # prefer smallest min depth
            # tie break using mean depth
            if sc_min_r < best_min_r or (abs(sc_min_r - best_min_r) < 1e-6 and sc_mean_r < best_mean_r):
                best = sc
                best_min_r = sc_min_r
                best_mean_r = sc_mean_r

        return best if best else front_pts

    def geometry_support_points(self, pts, centroid_pts):
        # centroiding wants the front-most support, but geometry should keep
        # the fuller selected cluster so larger obstacles do not collapse to
        # a tiny front patch in memory and occupancy painting.
        if pts:
            return list(pts)
        return list(centroid_pts)

    def merge_group_points(self, candidates):
        pts = []
        used = set()

        for cand in candidates:
            for pt in cand['pts']:
                key = pt[4]
                if key in used:
                    continue
                used.add(key)
                pts.append(pt)

        pts.sort(key=lambda p: p[4])
        return pts

    def build_candidate_groups(self, candidates, depth_band):
        groups = []

        for cand in sorted(candidates, key=lambda c: c['stats']['min_r']):
            placed = False

            for group in groups:
                if abs(cand['stats']['min_r'] - group['front_r']) <= depth_band:
                    group['candidates'].append(cand)
                    group['front_r'] = min(group['front_r'], cand['stats']['min_r'])
                    placed = True
                    break

            if not placed:
                groups.append({
                    'front_r': cand['stats']['min_r'],
                    'candidates': [cand],
                })

        return groups

    def score_candidate_group(self, pts, theta_center, wedge_width, class_name):
        stats = self.group_stats(pts)
        angle_err = abs(stats['mean_ang'] - theta_center)
        ang_fill = stats['ang_span'] / max(0.05, wedge_width)
        diag_fill = stats['diag'] / max(0.10, self.class_max_diag(class_name))

        score = (
            2.5 * stats['n']
            + 8.0 * ang_fill
            + 4.0 * diag_fill
            - 10.0 * angle_err
            - 1.2 * stats['min_r']
            - 0.4 * stats['mean_r']
        )

        return {
            'pts': pts,
            'stats': stats,
            'angle_err': angle_err,
            'ang_fill': ang_fill,
            'diag_fill': diag_fill,
            'score': score,
        }

    def select_grouped_cluster(self, candidates, theta_center, det, raw_pts):
        if not candidates or not raw_pts:
            return None, -1e9

        wedge_width = max(p[3] for p in raw_pts) - min(p[3] for p in raw_pts)
        class_name = det.class_name
        groups = self.build_candidate_groups(
            candidates,
            depth_band=max(self.layer_depth_band, 0.12)
        )

        best_group = None

        for i, group in enumerate(groups):
            pts = self.merge_group_points(group['candidates'])
            scored = self.score_candidate_group(
                pts,
                theta_center,
                wedge_width,
                class_name
            )
            consistency = self.bbox_consistency_for_stats(det, scored['stats'])
            scored['consistency'] = consistency
            scored['score'] -= consistency['penalty']

            self.get_logger().info(
                f'[GROUP {i}] class={class_name} clusters={len(group["candidates"])} '
                f'n={scored["stats"]["n"]} min_r={scored["stats"]["min_r"]:.2f} '
                f'diag={scored["stats"]["diag"]:.2f} ang_fill={scored["ang_fill"]:.2f} '
                f'bbox_span={consistency["bbox_span"]:.2f} '
                f'support_fill={consistency["support_fill"]:.2f} '
                f'angle_err={scored["angle_err"]:.3f} score={scored["score"]:.2f}'
            )

            if consistency['hard_reject']:
                self.get_logger().info(
                    f'[GROUP {i}] class={class_name} reject bbox_span={consistency["bbox_span"]:.2f} '
                    f'below class minimum'
                )
                continue

            if best_group is None or scored['score'] > best_group['score']:
                best_group = scored

        if best_group is None:
            return None, -1e9

        return best_group['pts'], best_group['score']

    def select_best_cluster(self, clusters, theta_center, det, raw_pts=None):

        det_msg = det if hasattr(det, 'class_name') else None
        class_name = det_msg.class_name if det_msg is not None else str(det)

        candidates = []

        for i, pts in enumerate(clusters):

            cand = self.build_cluster_candidate(pts, theta_center, class_name)
            stats = cand['stats']
            if det_msg is not None:
                consistency = self.bbox_consistency_for_stats(det_msg, stats)
            else:
                consistency = {
                    'bbox_span': None,
                    'support_span': 0.0,
                    'support_fill': 1.0,
                    'penalty': 0.0,
                    'hard_reject': False,
                }
            cand['consistency'] = consistency
            cand['score'] -= consistency['penalty']
            bbox_span_str = 'none'
            if consistency['bbox_span'] is not None:
                bbox_span_str = f'{consistency["bbox_span"]:.2f}'

            if consistency['hard_reject']:
                cand['valid'] = False

            self.get_logger().info(
                f'[CLUSTER {i}] n={stats["n"]} '
                f'ang_err={cand["angle_err"]:.3f} min_r={stats["min_r"]:.2f} mean_r={stats["mean_r"]:.2f} '
                f'diag={stats["diag"]:.2f} span={stats["r_span"]:.2f} '
                f'bbox_span={bbox_span_str} '
                f'fill={consistency["support_fill"]:.2f} '
                f'valid={cand["valid"]} score={cand["score"]:.2f}'
            )
            # only keeps valid clusters
            if cand['valid']:
                candidates.append(cand)

        if not candidates:
            return None, -1e9

        if det_msg is not None and self.class_prefers_grouping(class_name):
            best_pts, best_score = self.select_grouped_cluster(
                candidates,
                theta_center,
                det_msg,
                raw_pts
            )
            if best_pts is not None:
                return best_pts, best_score

        if raw_pts:
            # closest point in wedge
            wedge_front_r = min(p[2] for p in raw_pts)

            # the depth margin from the front
            front_margin = self.depth_margin_for_range(wedge_front_r)

            # prefers clusters near bbox center and the front
            front_candidates = [
                c for c in candidates
                if c['angle_err'] <= self.front_select_angle_gate
                and c['stats']['min_r'] <= wedge_front_r + front_margin
            ]

            if front_candidates:
                # sorts based on how front it is and
                front_candidates.sort(
                    key=lambda c: (
                        c['stats']['min_r'],
                        c['angle_err'],
                        -c['stats']['n'],
                        c['stats']['diag']
                    )
                )
                # returns highest score cluster
                best = front_candidates[0]
                return best['pts'], best['score']
        # return highest score cluster
        best = max(candidates, key=lambda c: c['score'])
        return best['pts'], best['score']

    # transforms  xy point from lidar pov to map frame pov
    def transform_xy_point_to_map(self, x, y, frame, stamp):
        pt = PointStamped()
        pt.header.frame_id = frame
        pt.header.stamp = stamp
        pt.point.x = x
        pt.point.y = y
        pt.point.z = 0.0

        map_pt = self.tf_buffer.transform(pt, self.map_frame)
        return (map_pt.point.x, map_pt.point.y)

    # transforms all the supoort points using the previous function
    def transform_support_points_to_map(self, pts, frame, stamp):

        out = []

        for p in pts:
            try:
                mx, my = self.transform_xy_point_to_map(p[0], p[1], frame, stamp)
                out.append((mx, my))
            except Exception:
                continue

        return out

    # create new temp object from current observation
    def make_temp_object(self, class_name, centroid, support_pts, now, obs_meta=None):

        raw_support = self.append_unique_points(
            [],
            support_pts,
            limit=self.support_raw_max_points_per_object
        )
        filtered_support, filtered_center = self.derive_support_geometry(
            class_name,
            centroid,
            raw_support
        )
        if not filtered_support:
            filtered_support = list(raw_support)
            filtered_center = centroid

        stored_obs_meta = self.observation_meta_from_context(
            class_name,
            filtered_center,
            filtered_support,
            obs_meta
        )

        return {
            'class_name': class_name,
            'assoc_centroid': centroid,
            'hits': 1,
            'last_seen': now,
            'obstacle_type': 1,
            'obs_state': self.initialize_object_observation_state(stored_obs_meta),
            'raw_support_points': raw_support,
            'support_points': filtered_support
        }

    def make_confirmed_object(
        self,
        class_name,
        centroid,
        support_pts,
        now,
        obs_meta=None,
        obs_state=None,
        obstacle_type=1
    ):
        raw_support = self.append_unique_points(
            [],
            support_pts,
            limit=self.support_raw_max_points_per_object
        )
        obj = {
            'object_id': self.next_object_id,
            'class_name': class_name,
            'assoc_centroid': centroid,
            'geom_centroid': centroid,
            'obstacle_type': obstacle_type,
            'obs_state': self.clone_observation_state(obs_state),
            'raw_support_points': raw_support,
            'support_points': list(raw_support),
            'last_seen': now
        }

        self.next_object_id += 1
        self.refresh_geometry(obj)

        if obs_state is None:
            stored_obs_meta = self.observation_meta_from_context(
                class_name,
                obj.get('geom_centroid'),
                obj.get('support_points', []),
                obs_meta
            )
            obj['obs_state'] = self.initialize_object_observation_state(stored_obs_meta)

        return obj

    def refresh_geometry(self, obj):
        raw_support = list(obj.get('raw_support_points', obj['support_points']))
        if not raw_support:
            return

        filtered_pts, center = self.derive_support_geometry(
            obj['class_name'],
            obj['assoc_centroid'],
            raw_support,
            owner_object_id=obj.get('object_id')
        )

        if not filtered_pts or center is None:
            return

        current_pts = list(obj.get('support_points', []))
        current_center = obj.get('geom_centroid')
        if (
            current_pts
            and current_center is not None
            and self.should_keep_current_geometry(
                obj,
                current_pts,
                filtered_pts
            )
        ):
            obj['support_points'] = current_pts
            obj['geom_centroid'] = self.smooth_point_update(
                current_center,
                current_center,
                self.geom_centroid_alpha,
                self.max_geom_centroid_step
            )
            self.prune_structural_support_memory(obj, current_pts)
            self.prune_foreign_support_memory(obj, current_pts)
            return

        obj['support_points'] = filtered_pts
        obj['geom_centroid'] = self.smooth_point_update(
            obj.get('geom_centroid'),
            center,
            self.geom_centroid_alpha,
            self.max_geom_centroid_step
        )
        self.prune_structural_support_memory(obj, filtered_pts)
        self.prune_foreign_support_memory(obj, filtered_pts)

    def cluster_is_edge_weak(self, pts, theta_min, theta_max, theta_center):
        if len(pts) > 2:
            return False
        # get only angle
        mean_ang = self.group_stats(pts)['mean_ang']
        # width of wedge
        wedge_width = theta_max - theta_min

        edge_margin = 0.2 * wedge_width
        # cluster near left edge
        near_left = abs(mean_ang - theta_min) < edge_margin
        # cluster near right edge
        near_right = abs(mean_ang - theta_max) < edge_margin
        # cluster if far from center (this is class based)
        far_from_center = abs(mean_ang - theta_center) > 0.12
        # weak if near wedge edge and far from bbox center
        return (near_left or near_right) and far_from_center

    def prune_temp_objects(self, now):

        # remove temp objects not seen recently
        self.temp_objects = [
            obj for obj in self.temp_objects
            if (now - obj['last_seen']) <= self.memory_temp_timeout
        ]

    # gets smallest distance to a confirmed object using assoc and geometry center
    def confirmed_match_distance(self, centroid, obj):
        d_assoc = math.hypot(
            centroid[0] - obj['assoc_centroid'][0],
            centroid[1] - obj['assoc_centroid'][1]
        )
        d_geom = math.hypot(
            centroid[0] - obj['geom_centroid'][0],
            centroid[1] - obj['geom_centroid'][1]
        )
        return min(d_assoc, d_geom)

    # class based check if new cnetroid is too close
    def too_close_to_confirmed(self, class_name, centroid, support_pts=None, radius=None):

        if radius is None:
            radius = self.too_close_radius_for_class(class_name)

        for obj in self.confirmed_objects:
            if obj['class_name'] != class_name:
                continue

            if support_pts and self.supports_same_object(
                class_name,
                centroid,
                support_pts,
                obj['assoc_centroid'],
                obj['support_points']
            ):
                return True

            if math.hypot(
                centroid[0] - obj['geom_centroid'][0],
                centroid[1] - obj['geom_centroid'][1]
            ) < radius:
                return True

        return False

    def find_best_confirmed_match(self, class_name, centroid, support_pts, radius):

        best = None
        best_dist = 1e9
        # in all confirmed objects
        for obj in self.confirmed_objects:
            # only of that specific class
            if obj['class_name'] != class_name:
                continue

            d = self.confirmed_match_distance(centroid, obj)
            overlap_match = self.supports_same_object(
                class_name,
                centroid,
                support_pts,
                obj['assoc_centroid'],
                obj['support_points']
            )
            fragment_match = self.support_fragment_belongs_to_object(
                class_name,
                centroid,
                support_pts,
                obj
            )

            # keep closest and within radius
            if not overlap_match and not fragment_match and d >= radius:
                continue

            eff_dist = d
            if overlap_match or fragment_match:
                eff_dist = min(eff_dist, 0.35 * d)

            if eff_dist < best_dist:
                best = obj
                best_dist = eff_dist
        # returning the match and distance
        return best, best_dist

    def should_suppress_new_object(self, class_name, centroid, support_pts):
        # small suport means most likely weak new object
        small_support = len(support_pts) <= self.memory_new_object_suppress_small_n

        if not small_support:
            return False

        suppress_radius = self.new_object_suppress_radius_for_class(class_name)

        # surpress if object too clsoe to already existing object
        for obj in self.confirmed_objects:
            if obj['class_name'] != class_name:
                continue

            if self.supports_same_object(
                class_name,
                centroid,
                support_pts,
                obj['assoc_centroid'],
                obj['support_points']
            ):
                return True

            d = self.confirmed_match_distance(centroid, obj)
            if d < suppress_radius:
                return True

        # suppress if too close to temp object
        for obj in self.temp_objects:
            if obj['class_name'] != class_name:
                continue

            if self.supports_same_object(
                class_name,
                centroid,
                support_pts,
                obj['assoc_centroid'],
                obj['support_points']
            ):
                return True

            d = math.hypot(
                centroid[0] - obj['assoc_centroid'][0],
                centroid[1] - obj['assoc_centroid'][1]
            )
            if d < suppress_radius:
                return True

        return False

    def merge_confirmed_objects(self):

        merged = []
        used = set()
        # in all confirmed objects
        for i, base in enumerate(self.confirmed_objects):

            if i in used:
                continue

            for j in range(i + 1, len(self.confirmed_objects)):

                if j in used:
                    continue

                other = self.confirmed_objects[j]

                if base['class_name'] != other['class_name']:
                    continue
                # get distances of the centroids
                d_geom = math.hypot(
                    base['geom_centroid'][0] - other['geom_centroid'][0],
                    base['geom_centroid'][1] - other['geom_centroid'][1]
                )
                d_assoc = math.hypot(
                    base['assoc_centroid'][0] - other['assoc_centroid'][0],
                    base['assoc_centroid'][1] - other['assoc_centroid'][1]
                )

                merge_radius = self.merge_radius_for_class(base['class_name'])

                overlap_match = self.supports_same_object(
                    base['class_name'],
                    base['assoc_centroid'],
                    base['support_points'],
                    other['assoc_centroid'],
                    other['support_points']
                )
                fragment_match = (
                    self.support_fragment_belongs_to_object(
                        base['class_name'],
                        other['assoc_centroid'],
                        other['support_points'],
                        base
                    )
                    or self.support_fragment_belongs_to_object(
                        base['class_name'],
                        base['assoc_centroid'],
                        base['support_points'],
                        other
                    )
                )

                # skip if too much
                if (
                    not overlap_match
                    and not fragment_match
                    and d_geom > merge_radius
                    and d_assoc > merge_radius
                ):
                    continue

                self.get_logger().info(
                    f'[MERGE] keep={base["object_id"]} remove={other["object_id"]}'
                )
                base_n = max(1, len(base.get('raw_support_points', base['support_points'])))
                other_n = max(1, len(other.get('raw_support_points', other['support_points'])))
                total_n = base_n + other_n

                base['assoc_centroid'] = (
                    (base_n * base['assoc_centroid'][0] + other_n * other['assoc_centroid'][0]) / total_n,
                    (base_n * base['assoc_centroid'][1] + other_n * other['assoc_centroid'][1]) / total_n,
                )
                base['last_seen'] = max(base['last_seen'], other['last_seen'])
                base['obstacle_type'] = max(
                    base.get('obstacle_type', 1),
                    other.get('obstacle_type', 1)
                )
                base_obs = self.clone_observation_state(base.get('obs_state'))
                other_obs = self.clone_observation_state(other.get('obs_state'))
                base_obs['updates'] = max(
                    base_obs.get('updates', 0),
                    other_obs.get('updates', 0)
                )
                base_obs['confirmed_updates'] = max(
                    base_obs.get('confirmed_updates', 0),
                    other_obs.get('confirmed_updates', 0)
                )
                base_obs['view_bins'].update(other_obs.get('view_bins', set()))
                base_obs['type_score'] = max(
                    base_obs.get('type_score', 0.0),
                    other_obs.get('type_score', 0.0)
                )
                base_obs['type2_mismatch_hits'] = max(
                    base_obs.get('type2_mismatch_hits', 0),
                    other_obs.get('type2_mismatch_hits', 0)
                )
                base_obs['type2_extreme_hits'] = max(
                    base_obs.get('type2_extreme_hits', 0),
                    other_obs.get('type2_extreme_hits', 0)
                )
                base_obs['type1_clear_hits'] = min(
                    base_obs.get('type1_clear_hits', 0),
                    other_obs.get('type1_clear_hits', 0)
                )
                base_obs['type2_updates'] = max(
                    base_obs.get('type2_updates', 0),
                    other_obs.get('type2_updates', 0)
                )
                for key in ('bbox_span', 'support_major', 'support_minor'):
                    if base_obs.get(key) is None:
                        base_obs[key] = other_obs.get(key)
                if base_obs.get('support_view_span') is None:
                    base_obs['support_view_span'] = other_obs.get('support_view_span')
                base_obs['support_fill'] = min(
                    base_obs.get('support_fill', 1.0),
                    other_obs.get('support_fill', 1.0)
                )
                base['obs_state'] = base_obs
                # merge support points into base object
                base['raw_support_points'] = self.merge_raw_support_points(
                    base,
                    other.get('raw_support_points', other['support_points'])
                )

                # refresh merged geometry
                self.refresh_geometry(base)
                used.add(j)

            merged.append(base)

        self.confirmed_objects = merged

    def reconcile_confirmed_support_ownership(self):

        if len(self.confirmed_objects) < 2:
            return

        for obj in list(self.confirmed_objects):
            raw_pts = list(obj.get('raw_support_points', obj.get('support_points', [])))
            if not raw_pts:
                continue

            kept = [
                pt for pt in raw_pts
                if not self.support_point_claimed_by_other_object(
                    obj['class_name'],
                    pt,
                    owner_object_id=obj.get('object_id')
                )
            ]

            removed = len(raw_pts) - len(kept)
            if removed <= 0 or not kept:
                continue

            obj['raw_support_points'] = self.cap_points_preserve(
                kept,
                obj.get('support_points', []),
                self.support_raw_max_points_per_object
            )
            self.refresh_geometry(obj)
            self.get_logger().info(
                f'[OWNERSHIP PRUNE] id={obj["object_id"]} '
                f'class={obj["class_name"]} removed={removed}'
            )

    #updates the values of a confirmed object
    def update_confirmed_object(self, obj, centroid, support_pts, now, obs_meta=None):

        support_pts = self.exclude_support_from_other_objects(
            obj['class_name'],
            support_pts,
            owner_object_id=obj.get('object_id')
        )
        support_pts = self.exclude_support_on_structural_map(
            obj['class_name'],
            centroid,
            support_pts
        )
        if not support_pts:
            self.get_logger().info(
                f'[CONFIRMED MATCH] id={obj["object_id"]} no support after foreign exclusion'
            )
            self.publish_marker(
                obj['object_id'],
                obj['geom_centroid'][0],
                obj['geom_centroid'][1],
                self.map_frame
            )
            return

        obj['last_seen'] = now

        obj['assoc_centroid'] = self.blend_point(
            obj['assoc_centroid'],
            centroid,
            self.assoc_centroid_alpha
        )
        observed_geom = self.geometry_center_from_assoc_and_support(
            obj['class_name'],
            centroid,
            support_pts
        )

        # allow a confirmed object to grow from a new view instead of locking
        # onto only the very first side it saw.
        support_pts = self.filter_points_near_any_center(
            support_pts,
            [
                obj['geom_centroid'],
                obj['assoc_centroid'],
                centroid,
                observed_geom,
            ],
            self.support_reanchor_radius_for_class(obj['class_name'])
        )

        if observed_geom is not None:
            support_pts = self.filter_points_in_class_footprint(
                support_pts,
                observed_geom,
                obj['class_name']
            )

        if not support_pts:
            self.get_logger().info(
                f'[CONFIRMED MATCH] id={obj["object_id"]} no new local support kept'
            )
            # publish the marker in rviz
            self.publish_marker(
                obj['object_id'],
                obj['geom_centroid'][0],
                obj['geom_centroid'][1],
                self.map_frame
            )
            return
        # add new support points
        obj['raw_support_points'] = self.merge_raw_support_points(
            obj,
            support_pts
        )

        # refreshes the geometry
        self.refresh_geometry(obj)
        mature_obs_meta = self.observation_meta_from_context(
            obj['class_name'],
            obj.get('geom_centroid'),
            obj.get('support_points', []),
            obs_meta
        )
        self.update_object_observation_state(obj, mature_obs_meta)

        self.get_logger().info(
            f'[CONFIRMED MATCH] id={obj["object_id"]} '
            f'assoc=({obj["assoc_centroid"][0]:.2f},{obj["assoc_centroid"][1]:.2f}) '
            f'geom=({obj["geom_centroid"][0]:.2f},{obj["geom_centroid"][1]:.2f}) '
            f'support={len(obj["support_points"])} '
            f'type={obj.get("obstacle_type", 1)}'
        )

        self.publish_marker(
            obj['object_id'],
            obj['geom_centroid'][0],
            obj['geom_centroid'][1],
            self.map_frame
        )

    def update_temp_or_promote(self, obj, centroid, support_pts, now, obs_meta=None):

        support_pts = self.exclude_support_from_other_objects(
            obj['class_name'],
            support_pts
        )
        support_pts = self.exclude_support_on_structural_map(
            obj['class_name'],
            centroid,
            support_pts
        )
        if not support_pts:
            return

        # increase hit count
        obj['hits'] += 1

        # update last seen time
        obj['last_seen'] = now

        obj['assoc_centroid'] = self.blend_point(
            obj['assoc_centroid'],
            centroid,
            0.25
        )

        # add new support points
        obj['raw_support_points'] = self.merge_raw_support_points(
            obj,
            support_pts
        )
        obj['support_points'], _ = self.derive_support_geometry(
            obj['class_name'],
            obj['assoc_centroid'],
            obj['raw_support_points']
        )
        mature_obs_meta = self.observation_meta_from_context(
            obj['class_name'],
            obj['assoc_centroid'],
            obj.get('support_points', []),
            obs_meta
        )
        self.update_object_observation_state(obj, mature_obs_meta)

        # if enough hits becomes confirmed
        if obj['hits'] >= self.memory_confirm_hits:
            confirmed = self.make_confirmed_object(
                obj['class_name'],
                obj['assoc_centroid'],
                obj['raw_support_points'],
                now,
                obs_meta=obs_meta,
                obs_state=obj.get('obs_state'),
                obstacle_type=obj.get('obstacle_type', 1)
            )

            self.get_logger().info(
                f'[PROMOTE] id={confirmed["object_id"]} '
                f'class={confirmed["class_name"]} '
                f'centroid=({centroid[0]:.2f},{centroid[1]:.2f}) '
                f'type={confirmed.get("obstacle_type", 1)}'
            )

            self.confirmed_objects.append(confirmed)
            self.temp_objects.remove(obj)
            self.merge_confirmed_objects()
            self.reconcile_confirmed_support_ownership()

    def update_objects(self, class_name, centroid, support_pts, now, obs_meta=None):
        support_pts = self.exclude_support_from_other_objects(
            class_name,
            support_pts
        )
        support_pts = self.exclude_support_on_structural_map(
            class_name,
            centroid,
            support_pts
        )
        if not support_pts:
            return

        # remove old temp
        self.prune_temp_objects(now)
        # see if we can match with an already existing object
        match, _ = self.find_best_confirmed_match(
            class_name,
            centroid,
            support_pts,
            self.confirmed_match_radius_for_class(class_name)
        )
        # if match we update the object
        if match:
            self.update_confirmed_object(match, centroid, support_pts, now, obs_meta=obs_meta)
            self.reconcile_confirmed_support_ownership()
            return

        # if its too close to a confirmed object we leave it
        if self.too_close_to_confirmed(class_name, centroid, support_pts):
            return

        # then we see if we can match it to a temp object
        for obj in self.temp_objects:
            if obj['class_name'] != class_name:
                continue

            if (
                self.supports_same_object(
                    class_name,
                    centroid,
                    support_pts,
                    obj['assoc_centroid'],
                    obj['support_points']
                )
                or self.support_fragment_belongs_to_object(
                    class_name,
                    centroid,
                    support_pts,
                    obj
                )
                or math.hypot(
                    centroid[0] - obj['assoc_centroid'][0],
                    centroid[1] - obj['assoc_centroid'][1]
                ) < self.temp_match_radius_for_class(class_name)
            ):
                self.update_temp_or_promote(
                    obj,
                    centroid,
                    support_pts,
                    now,
                    obs_meta=obs_meta
                )
                return

        # if its a weak object too close to existing we surpress it
        if self.should_suppress_new_object(class_name, centroid, support_pts):
            self.get_logger().info(
                f'[SUPPRESS NEW] class={class_name} '
                f'centroid=({centroid[0]:.2f},{centroid[1]:.2f})'
            )
            return

        self.get_logger().info(
            f'[NEW TEMP] class={class_name} '
            f'centroid=({centroid[0]:.2f},{centroid[1]:.2f})'
        )
        # if none above then create the temp object
        self.temp_objects.append(
            self.make_temp_object(
                class_name,
                centroid,
                support_pts,
                now,
                obs_meta=obs_meta
            )
        )

    #creates a wedge based on bounding box
    def compute_wedge(self, det):

        # gets pixel coords
        xmin = float(det.xmin)
        xmax = float(det.xmax)
        center_x = float(det.center_x)

        if xmax < xmin:
            xmin, xmax = xmax, xmin

        class_name = det.class_name
        trim_frac = self.wedge_trim_frac_for_class(class_name)
        max_half_angle = self.wedge_max_half_angle_for_class(class_name)
        expand_angle = self.wedge_expand_angle_for_class(class_name)

        # gets the width and then trims the edge of the bbox width(might change for other objects class absed)
        bbox_w = max(1.0, xmax - xmin)
        trim = trim_frac * bbox_w
        u_left = xmin + trim
        u_right = xmax - trim

        if u_right <= u_left:
            u_left = xmin
            u_right = xmax

        # convers pixel coords to angles
        theta_left = self.pixel_to_bearing(u_left)
        theta_right = self.pixel_to_bearing(u_right)
        theta_center = self.pixel_to_bearing(center_x)

        theta_min = min(theta_left, theta_right)
        theta_max = max(theta_left, theta_right)
        theta_min -= expand_angle
        theta_max += expand_angle

        # caps the angle to be a certain range from the center, this will be class based(currently only for chairs)
        theta_min = max(theta_min, theta_center - max_half_angle)
        theta_max = min(theta_max, theta_center + max_half_angle)

        if theta_max <= theta_min:
            fallback_half_angle = min(0.08, max(0.05, 0.4 * max_half_angle))
            theta_min = theta_center - fallback_half_angle
            theta_max = theta_center + fallback_half_angle

        return theta_min, theta_max, theta_center

    # returns  support points around geometry centroid
    def get_filtered_local_support_points(self, obj):

        center = obj['geom_centroid']
        pts = list(obj['support_points'])

        if not pts or center is None:
            return []

        if obj.get('obstacle_type', 1) == 2:
            hypothesis = self.effective_object_hypothesis(obj)
            outline = self.outline_points_from_hypothesis(
                hypothesis,
                obstacle_type=2
            )
            return [(p[0] - center[0], p[1] - center[1]) for p in outline]

        angle = self.support_principal_angle(pts, center)
        pts = self.filter_points_in_class_footprint(
            pts,
            center,
            obj['class_name'],
            angle
        )

        hull = self.convex_hull_points(pts)
        if len(hull) >= self.publish_support_min_keep:
            pts = hull

        kept = [(p[0] - center[0], p[1] - center[1]) for p in pts]

        return kept

    # publishes calues for every object and creates a marker and its support points as well
    def publish_confirmed_object_geometry(self):

        arr = MarkerArray()

        for obj in self.confirmed_objects:
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'confirmed_object_geometry'
            marker.id = int(obj['object_id'])
            marker.type = Marker.POINTS
            marker.action = Marker.ADD

            marker.pose.position.x = obj['geom_centroid'][0]
            marker.pose.position.y = obj['geom_centroid'][1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.05
            marker.scale.y = 0.05

            marker.color.r = 0.0
            marker.color.g = 0.0 if obj.get('obstacle_type', 1) == 1 else 0.45
            marker.color.b = 1.0
            marker.color.a = 1.0

            local_supports = self.get_filtered_local_support_points(obj)

            for lx, ly in local_supports:
                p = Point()
                p.x = lx
                p.y = ly
                p.z = 0.0
                marker.points.append(p)

            arr.markers.append(marker)

        self.geometry_pub.publish(arr)

    # main function to use camera detection to estimate object geometry
    def try_process(self):

        # checks if camera is working
        if not self.cam_info_ok:
            return

        # checks if camera is working
        if self.last_detection is None:
            return

        det = self.last_detection
        now = self.now_sec()

        # gets the best scan and time of detection
        scan, scan_dt = self.get_best_scan_for_detection(det)

        if scan is None:
            self.get_logger().info('[REJECT] no scan available')
            self.last_detection = None
            return

        # checks if there is a time mismatch
        if scan_dt > self.sync_max_det_scan_dt:
            self.get_logger().info(
                f'[REJECT] det/scan mismatch dt={scan_dt:.3f}'
            )
            self.last_detection = None
            return

        # computes the fov for the lidar using the bbox
        theta_min, theta_max, theta_center = self.compute_wedge(det)

        # computes all lidar points in that wedge
        raw_pts = self.extract_support_points(scan, theta_min, theta_max)

        # compact objects (e.g. office chairs) may give a single lidar return
        # on their narrow center pedestal — use min_pts=1 so that point is not
        # discarded before it can drive type2 evidence accumulation
        cluster_min = (
            1 if self.type2_gate_scale_for_class(det.class_name) < 0.85
            else self.cluster_min_pts
        )
        clusters = self.build_depth_layer_clusters(
            raw_pts,
            self.layer_max_layers,
            min_pts=cluster_min
        )

        self.get_logger().info(
            f'[DET] class={det.class_name} dt={scan_dt:.3f} '
            f'raw={len(raw_pts)} clusters={len(clusters)} '
            f'wedge=[{theta_min:.3f},{theta_max:.3f}] center={theta_center:.3f}'
        )

        if not clusters:
            self.last_detection = None
            return

        # gets the best cluster[s] and its score
        best_pts, best_score = self.select_best_cluster(
            clusters,
            theta_center,
            det,
            raw_pts=raw_pts
        )

        if best_pts is None:
            self.get_logger().info('[REJECT] no valid cluster')
            self.last_detection = None
            return

        # if the best cluster chosen is weak, it then rejects it
        weak_edge_cluster = self.cluster_is_edge_weak(
            best_pts, theta_min, theta_max, theta_center
        )

        # gets more of the front facing points from the cluster
        centroid_pts = self.front_subcluster_points(best_pts)
        geometry_pts = self.geometry_support_points(best_pts, centroid_pts)
        # use a class-aware observation centroid so large tables can associate
        # to their center instead of locking each visible side as its own object.
        centroid_lidar = self.observation_centroid_from_support(
            det,
            centroid_pts,
            geometry_pts,
            theta_center
        )

        if centroid_lidar is None:
            self.get_logger().info('[REJECT] no centroid from support')
            self.last_detection = None
            return

        try:
            # converts centroid from lidar frame to map frame
            map_point = self.transform_xy_point_to_map(
                centroid_lidar[0],
                centroid_lidar[1],
                scan.header.frame_id,
                scan.header.stamp
            )

            # converts support points from lidar frame to map frame
            support_map = self.transform_support_points_to_map(
                geometry_pts,
                scan.header.frame_id,
                scan.header.stamp
            )

        except Exception:
            self.get_logger().info('[REJECT] tf transform failed')
            self.last_detection = None
            return

        if not support_map:
            self.get_logger().info('[REJECT] no support points in map')
            self.last_detection = None
            return

        # if cluster was weak then it checks if the cluster was near or part of a object that already exists
        # if it does then it allows the update and if not its rejected
        if weak_edge_cluster:
            match, _ = self.find_best_confirmed_match(
                det.class_name,
                map_point,
                support_map,
                self.confirmed_match_radius_for_class(det.class_name)
            )

            if match is None:
                self.get_logger().info('[REJECT] weak edge cluster for new object')
                self.last_detection = None
                return

        self.get_logger().info(
            f'[BEST] n={len(best_pts)} front_n={len(centroid_pts)} geom_n={len(geometry_pts)} score={best_score:.2f} '
            f'lidar=({centroid_lidar[0]:.2f},{centroid_lidar[1]:.2f}) '
            f'map=({map_point[0]:.2f},{map_point[1]:.2f}) '
            f'weak_edge={weak_edge_cluster}'
        )

        sensor_origin = None
        try:
            sensor_origin = self.transform_xy_point_to_map(
                0.0,
                0.0,
                scan.header.frame_id,
                scan.header.stamp
            )
        except Exception:
            pass

        support_map = self.exclude_support_on_structural_map(
            det.class_name,
            map_point,
            support_map,
            sensor_origin=sensor_origin
        )

        if not support_map:
            self.get_logger().info('[REJECT] no support after structural wall filtering')
            self.last_detection = None
            return

        best_stats = self.group_stats(best_pts)
        type_bbox_span = self.span_m_for_angle_bounds(
            theta_min,
            theta_max,
            best_stats['mean_r']
        )
        obs_meta = {
            'bbox_span': type_bbox_span,
            'sensor_origin': sensor_origin,
        }

        mature_obs_meta = self.observation_meta_from_context(
            det.class_name,
            map_point,
            support_map,
            obs_meta
        )
        if mature_obs_meta is not None:
            bbox_span = mature_obs_meta.get('bbox_span')
            view_span = mature_obs_meta.get('support_view_span')
            self.get_logger().info(
                f'[TYPE OBS] bbox={bbox_span if bbox_span is not None else float("nan"):.2f} '
                f'view_span={view_span if view_span is not None else 0.0:.2f} '
                f'fill={mature_obs_meta["support_fill"]:.2f}'
            )

        # this updates the object geometry into memomry
        self.update_objects(
            det.class_name,
            map_point,
            support_map,
            now,
            obs_meta=obs_meta
        )

        # publishes the object geometry for visualization later
        self.publish_confirmed_object_geometry()

        # publishes the wedge for temporary visualization
        self.publish_wedge(theta_min, theta_max, scan.header.frame_id)
        self.last_detection = None


def main():

    rclpy.init()

    node = ObjectLocation()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
