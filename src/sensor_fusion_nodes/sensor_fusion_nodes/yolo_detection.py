import os

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from ultralytics import YOLO
from sensor_fusion_msgs.msg import Detection


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('yolo_detection')

        self.bridge = CvBridge()

        share_dir = get_package_share_directory('sensor_fusion_nodes')
        model_dir = os.path.join(share_dir, 'models')
        model_path = os.path.join(model_dir, 'best.pt')

        self.declare_parameter("score_thresh", 0.5)
        self.declare_parameter("allowed_labels", "chair,table")
        self.declare_parameter("show_window", False)

        self.score_thresh = float(self.get_parameter("score_thresh").value)
        allowed_labels = str(self.get_parameter("allowed_labels").value).strip()
        self.allowed_labels = {
            label.strip() for label in allowed_labels.split(',') if label.strip()
        }
        self.show_window = bool(self.get_parameter("show_window").value)

        self.model = YOLO(model_path)

        self.det_pub = self.create_publisher(Detection, "/detections", 10)
        self.valid_pub = self.create_publisher(Bool, "/bbox_valid", 10)

        image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            image_qos
        )

        if self.show_window:
            cv2.namedWindow("YOLO Detector", cv2.WINDOW_NORMAL)

    def class_name(self, class_id):
        names = self.model.names

        if isinstance(names, dict):
            return str(names.get(class_id, class_id))

        if class_id < 0 or class_id >= len(names):
            return ""

        return str(names[class_id])

    def listener_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        imH, imW = frame.shape[:2]

        results = self.model.predict(
            source=frame,
            conf=self.score_thresh,
            verbose=False,
        )

        found = False
        best_det = None
        best_score = -1.0

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                for i in range(len(scores)):
                    score = float(scores[i])
                    if score < self.score_thresh or score <= best_score:
                        continue

                    class_id = int(classes[i])
                    label = self.class_name(class_id)
                    if not label:
                        continue

                    if self.allowed_labels and label not in self.allowed_labels:
                        continue

                    x1, y1, x2, y2 = boxes_xyxy[i]

                    xmin = int(max(0, x1))
                    ymin = int(max(0, y1))
                    xmax = int(min(imW, x2))
                    ymax = int(min(imH, y2))

                    cx = 0.5 * (xmin + xmax)
                    cy = 0.5 * (ymin + ymax)

                    best_det = {
                        "label": label,
                        "score": score,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "cx": cx,
                        "cy": cy,
                    }
                    best_score = score
                    found = True

        self.valid_pub.publish(Bool(data=found))

        if found and best_det is not None:
            det_msg = Detection()
            det_msg.header = msg.header
            det_msg.class_name = best_det["label"]
            det_msg.score = best_det["score"]

            det_msg.xmin = best_det["xmin"]
            det_msg.ymin = best_det["ymin"]
            det_msg.xmax = best_det["xmax"]
            det_msg.ymax = best_det["ymax"]

            det_msg.image_width = imW
            det_msg.image_height = imH

            det_msg.center_x = float(best_det["cx"])
            det_msg.center_y = float(best_det["cy"])

            self.det_pub.publish(det_msg)

            cv2.rectangle(
                frame,
                (best_det["xmin"], best_det["ymin"]),
                (best_det["xmax"], best_det["ymax"]),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f'{best_det["label"]} {best_det["score"]:.2f}',
                (best_det["xmin"], max(0, best_det["ymin"] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            cv2.circle(
                frame,
                (int(best_det["cx"]), int(best_det["cy"])),
                4,
                (0, 0, 255),
                -1
            )

        if self.show_window:
            cv2.imshow("YOLO Detector", frame)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = ObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
