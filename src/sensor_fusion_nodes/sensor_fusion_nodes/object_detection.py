import os
import cv2
import numpy as np
import tensorflow as tf

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from sensor_fusion_msgs.msg import Detection


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        self.bridge = CvBridge()

        share_dir = get_package_share_directory('sensor_fusion_nodes')
        model_dir = os.path.join(share_dir, 'models')

        model_path = os.path.join(model_dir, 'detect.tflite')
        labels_path = os.path.join(model_dir, 'labelmap.txt')

        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        if self.labels and self.labels[0] == '???':
            self.labels.pop(0)

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.height = int(self.input_details[0]['shape'][1])
        self.width = int(self.input_details[0]['shape'][2])

        self.declare_parameter("score_thresh", 0.5)
        self.declare_parameter("label_filter", "")

        self.score_thresh = float(self.get_parameter("score_thresh").value)
        self.label_filter = str(self.get_parameter("label_filter").value)

        self.det_pub = self.create_publisher(Detection, "/detections", 10)
        self.valid_pub = self.create_publisher(Bool, "/bbox_valid", 10)

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10
        )

        cv2.namedWindow("Object Detector", cv2.WINDOW_NORMAL)

    def listener_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        imH, imW = frame.shape[:2]

        frame_resized = cv2.resize(frame, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if self.input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        found = False
        best_det = None

        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.score_thresh:
                continue

            class_id = int(classes[i])
            if class_id < 0 or class_id >= len(self.labels):
                continue

            label = self.labels[class_id]
            if self.label_filter and label != self.label_filter:
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
            found = True
            break

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

        cv2.imshow("Object Detector", frame)
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
