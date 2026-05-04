import os
import cv2
import numpy as np
import tensorflow as tf

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from sensor_fusion_msgs.msg import Detection


class ObjectDetector2(Node):

    def __init__(self):
        super().__init__('object_detction2')

        self.bridge = CvBridge()

        self.declare_parameter("score_thresh", 0.5)
        self.declare_parameter("allowed_labels", "chair,table")
        self.declare_parameter("show_window", False)

        share_dir = get_package_share_directory('sensor_fusion_nodes')
        model_dir = os.path.join(share_dir, 'models')

        model_path = os.path.join(model_dir, 'detect1.tflite')
        labels_path = os.path.join(model_dir, 'labelmap1.txt')

        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        if self.labels and self.labels[0] == '???':
            self.labels.pop(0)

        self.score_thresh = float(self.get_parameter("score_thresh").value)
        allowed_labels = str(self.get_parameter("allowed_labels").value).strip()
        self.allowed_labels = {
            label.strip() for label in allowed_labels.split(',') if label.strip()
        }
        self.show_window = bool(self.get_parameter("show_window").value)

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.height = int(self.input_details[0]['shape'][1])
        self.width = int(self.input_details[0]['shape'][2])
        self.boxes_idx, self.classes_idx, self.scores_idx, self.count_idx = self.resolve_output_indices()

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
            cv2.namedWindow("Object Detector 2", cv2.WINDOW_NORMAL)

    def resolve_output_indices(self):
        first_name = self.output_details[0]['name']

        if 'StatefulPartitionedCall' in first_name:
            return 1, 3, 0, 2

        if 'TFLite_Detection_PostProcess' in first_name:
            return 0, 1, 2, 3

        boxes_idx = None
        count_idx = None

        for idx, detail in enumerate(self.output_details):
            shape = tuple(int(dim) for dim in detail['shape'])
            if len(shape) == 3 and shape[-1] == 4:
                boxes_idx = idx
            elif int(np.prod(shape)) == 1:
                count_idx = idx

        vector_indices = [
            idx for idx, detail in enumerate(self.output_details)
            if idx not in {boxes_idx, count_idx} and len(detail['shape']) >= 2
        ]

        classes_idx = vector_indices[0] if len(vector_indices) > 0 else None
        scores_idx = vector_indices[1] if len(vector_indices) > 1 else None
        return boxes_idx, classes_idx, scores_idx, count_idx

    def flatten_output(self, tensor):
        array = np.asarray(tensor)
        if array.ndim > 1 and array.shape[0] == 1:
            return array[0]
        return array

    def listener_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        imH, imW = frame.shape[:2]

        frame_resized = cv2.resize(frame, (self.width, self.height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(frame_rgb, axis=0)

        if self.input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        outputs = [
            self.interpreter.get_tensor(detail['index'])
            for detail in self.output_details
        ]

        boxes = self.flatten_output(outputs[self.boxes_idx])
        classes = self.flatten_output(outputs[self.classes_idx])
        scores = self.flatten_output(outputs[self.scores_idx])

        detection_count = min(len(boxes), len(classes), len(scores))
        if self.count_idx is not None:
            detection_count = min(
                detection_count,
                max(0, int(round(float(np.asarray(outputs[self.count_idx]).reshape(-1)[0]))))
            )

        found = False
        best_det = None
        best_score = -1.0

        for i in range(detection_count):
            score = float(scores[i])
            if score < self.score_thresh or score <= best_score:
                continue

            class_id = int(round(float(classes[i])))
            if class_id < 0 or class_id >= len(self.labels):
                continue

            label = self.labels[class_id]
            if self.allowed_labels and label not in self.allowed_labels:
                continue

            ymin = int(max(0, boxes[i][0] * imH))
            xmin = int(max(0, boxes[i][1] * imW))
            ymax = int(min(imH, boxes[i][2] * imH))
            xmax = int(min(imW, boxes[i][3] * imW))

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
            cv2.imshow("Object Detector 2", frame)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = ObjectDetector2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
