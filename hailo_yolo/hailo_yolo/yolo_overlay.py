import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from yolo_msgs.msg import BoolArray, BoundingBox2P, Hue, YoloDetection, YoloDetectionArray

# COCO labels (80 classes)
CLASS_LABELS = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ] 

class YoloOverlayNode(Node):
    def __init__(self):
        super().__init__('yolo_overlay_node')
        self.bridge = CvBridge()

        # Subscriberの設定（message_filtersを使う）
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.image_sub = Subscriber(self, Image, '/realsense/raw_image')
        self.det_sub = Subscriber(self, YoloDetectionArray, '/yolo/detections')

        # ApproximateTimeSynchronizerで同期
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.det_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synced_callback)

        # オーバーレイ画像のPublisher
        self.pub = self.create_publisher(Image, '/overlay_image', 10)

        self.get_logger().info("YOLO Overlay Node Started")

    def synced_callback(self, image_msg, detection_msg):
        # ROSイメージ → OpenCV画像へ変換
        if not image_msg.encoding:
            self.get_logger().warn("Received image with empty encoding. Skipping frame.")
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except:
            self.get_logger().warn(f"cv_bridge failed to convert image")
            return

        # YOLOの検出結果を重ねる
        height_img, width_img, _ = cv_image.shape
        for detection in detection_msg.detections:
            # center_x = float((x_min + x_max) / 2)
            # center_y = float((y_min + y_max) / 2)
            # width = float(x_max - x_min)
            # height = float(y_max - y_min)

            x_min, y_min, x_max, y_max = detection.bboxparam

            # 左上の点に変換（画像座標系における始点）
            w = x_max - x_min
            h = y_max - y_min
            label_id = CLASS_LABELS[detection.label]
            score = detection.score

            # 緑色の矩形とラベル
            cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # (image, pt1:四角形の頂点, pt2:pt1の対角の四角形の頂点, color, thickness:optional, )
            cv2.putText(cv_image, f"ID:{label_id} ({score:.2f})", (x_min, max(y_min - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # self.get_logger().info(f"Detection: {label_id}")
            self.get_logger().info(f"[Overlay] x={x_min+w/2}, y={y_min+h/2}, w={w}, h={h}, label={label_id}, score={score:.2f}")


        # オーバーレイ画像をpublish
        overlay_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        overlay_msg.header = image_msg.header  # 時刻とフレーム情報を一致させる
        self.pub.publish(overlay_msg)
        # self.get_logger().info('overlay_msg was sent')

def main(args=None):
    rclpy.init(args=args)
    node = YoloOverlayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
