import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from sensor_msgs.msg import Image
import cv2
import numpy as np
import threading
import time
from cv_bridge import CvBridge

import pyrealsense2 as rs
import degirum as dg

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

PUB_TIME = 1/30 # Publish frequency [s]

# -----------------------------------------------------------------------------------------------
# Yolo Inference Node
# -----------------------------------------------------------------------------------------------
class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo')
        
        self.rs_height = 480
        self.rs_width = 640
        rs_FPS = 30

        # Realsense pipeline setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.rs_width, self.rs_height, rs.format.bgr8, rs_FPS)
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # Uncomment if depth is needed
        self.bridge = CvBridge()
        
        # Hailo inference via Degirum PySDK
        self.hef_path = '/root/myPython_test/model_zoo'
        self.model_name = 'yolov8n_seg'

        self.get_logger().info(f"YoloInferenceNode initialized. Using HEF: {self.model_name} into {self.hef_path}")
        self.get_logger().info(f"RealSense configured for: {self.rs_width}x{self.rs_height} @ {rs_FPS} FPS")

        # Load model
        self.model = dg.load_model(
            model_name = self.model_name,
            inference_host_address = '@local',
            zoo_url = self.hef_path
        )

        try:
            self.pipeline.start(self.config)
            self.get_logger().info("Realsense pipeline started.")
        except Exception as e:
            self.get_logger().info(f"Failed to start Realsense pipeline: {e}")
            rclpy.shutdown()
            return
        
        # Subscriber
        # self.create_subscription(Image, image_topic, self.image_callback, 10)

        # Publisher
        self.detection_pub = self.create_publisher(YoloDetectionArray, '/yolo/detections', qos_profile=5)
        self.rs_pub = self.create_publisher(Image, '/realsense/raw_image', qos_profile=5)
        self.det_msg = YoloDetectionArray()
        self.image_msg = Image()
        self.timer = self.create_timer(PUB_TIME, self.timer_callback)

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        self.shot_time = self.get_clock().now().to_msg()
        if not color_frame:
            return
        color_image = np.asanyarray(color_frame.get_data())
        results = self.model(color_image)
        detection_msg = self.postprocess_yolo_results(results,'segment')
        self.publish_results(color_image, detection_msg)

    def postprocess_yolo_results(self, raw_results, mode):
        if mode == 'object_detection': # Obsolete
            det_msg = self.object_detection(raw_results)
        elif mode == 'segment':
            det_msg = self.instance_segmentation(raw_results)
        else:
            raise NameError(f'No mode such a {mode}.')
        
        return det_msg
    
    def object_detection(self, results):
        """
        現状はPublishの整合性が取れていないため使用不可
        Convert to ros message from hailo-formated inference result.
        raw_results: type:nd.ndarray, row:CLASS_LABEL, col:[BBOX:param(ymin,xmin,ymax,xmax),score], 
        height:Detection num, shape:80*5*100 (based on hailo output_tensor)
        """
        detection_array_msg = Detection2DArray()
        if len(results.results) == 0:
            return detection_array_msg
        

        for idx in results.results:
            class_id = idx['category_id']
            bbox = idx['bbox']
            mask = idx['mask']
            score = idx['score']
            # label = idx['label']
            x_min, y_min, x_max, y_max = bbox

            center_x = float((x_min + x_max) / 2)
            center_y = float((y_min + y_max) / 2)
            width = float(x_max - x_min)
            height = float(y_max - y_min)
            

            detection_msg = Detection2D()

            # Bounding box
            detection_msg.bbox.center.position.x = float(center_x)
            detection_msg.bbox.center.position.y = float(center_y)
            detection_msg.bbox.center.theta = 0.0
            detection_msg.bbox.size_x = float(width)
            detection_msg.bbox.size_y = float(height)

            # CLASS ID, Score
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(CLASS_LABELS[class_id])
            hypothesis.hypothesis.score = float(score)
            # hypothesis.pose.pose = 0
            # hypothesis.pose.covariance = [0.0] * 36
            detection_msg.results.append(hypothesis)
            detection_array_msg.detections.append(detection_msg)

        return detection_array_msg
    
    def instance_segmentation(self, results):
        """
            results' component:
            bbox: [x_min, y_min, x_max, y_max]
            category_id: (int) corresponed COCO Labels
            label: (str) Detected object type
            mask: (bool)
            score: (float) 0-1
            """
        detection_array_msg = YoloDetectionArray()
        if len(results.results) == 0:
            return detection_array_msg
        # Mask 累積用
        mask_set = np.zeros((self.rs_height,self.rs_width),dtype=np.uint8)
        for idx, detection in enumerate(results.results):
            detection_msg = YoloDetection()
            class_id = detection['category_id']
            bbox = detection['bbox']
            mask = detection['mask'].astype(bool)
            score = detection['score']
            # label = idx['label']
            
            # Object label COCO ID
            detection_msg.label = class_id
            # Bounding box
            # 内部のパディング処理の影響で小数点が発生するためマスク領域を損なわないようにする
            x1 = max(0, min(self.rs_width, int(np.floor(bbox[0]))))
            y1 = max(0, min(self.rs_height, int(np.floor(bbox[1]))))
            x2 = max(0, min(self.rs_width, int(np.ceil(bbox[2]))))
            y2 = max(0, min(self.rs_height, int(np.ceil(bbox[3]))))
            detection_msg.bboxparam = [x1,y1,x2,y2]
            # Mask
            """
            Mask ID
            Background = 0
            Mask1 = 1
            Mask2 = 2
            :
            MaskN = N
            """
            mask_set[mask] = idx+1 # Combine mask
            # Hue
            detection_msg.hue = [10]*180 # ここに色相ヒストグラム
            # Score
            detection_msg.score = score
            
            # (参考) BBの計算
            # center_x = float((x1 + x2) / 2)
            # center_y = float((y1 + y2) / 2)
            # width = float(x2 - x1)
            # height = float(y2 - y1)

            detection_array_msg.detections.append(detection_msg)

        # PNG pack
        _, mask_png = cv2.imencode('.png', mask_set)
        masks = mask_png.flatten()
        
        # Bit pack
        # masks = np.packbits(mask_set.flatten()) 

        # Raw data
        # masks = mask_set.flatten()         
        detection_array_msg.masks.data = masks.tolist()
        detection_array_msg.rows = mask_set.shape[0]
        detection_array_msg.cols = mask_set.shape[1]

        return detection_array_msg
    
    def publish_results(self, image_raw, detection_msg):
        # Detection Header config
        detection_msg.header.stamp = self.shot_time
        detection_msg.header.frame_id = "camera_link"

        # Image config
        ros_image = self.bridge.cv2_to_imgmsg(image_raw, encoding='bgr8')
        ros_image.header.stamp = self.shot_time
        ros_image.header.frame_id = "camera_link"
        
        self.det_msg = detection_msg
        self.image_msg = ros_image

        # Add from old timer_callback color[0][1][2]
        self.detection_pub.publish(self.det_msg)
        self.rs_pub.publish(self.image_msg)

    # def timer_callback(self):
    #     # Publish message
    #     self.detection_pub.publish(self.det_msg)
    #     self.rs_pub.publish(self.image_msg)
    #     self.mask_pub.publish(self.mask_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    try:
        rclpy.spin(node)
        # node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

# For debug
if __name__ == '__main__':
    main()
