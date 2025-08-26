import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from sensor_msgs.msg import Image
import cv2
import numpy as np
import threading
import time
from cv_bridge import CvBridge

from hailo_platform.pyhailort import pyhailort as ph
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams, OutputVStreams, 
                            VDevice)

import pyrealsense2 as rs
import degirum as dg

""" Output tensors' shape: 80 (classes) * 5 (BB info.) * 100 (upper limit of detections)"""
CLASS_LABELS = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ] # Example COCO labels (80 classes)

PUB_TIME = 1/30 # Publish frequency [s]
TH_SCORE = 0.5 # Detection rejection threshold 0-1 [-]



# -----------------------------------------------------------------------------------------------
# Yolo Inference Node
# -----------------------------------------------------------------------------------------------
class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo')
        # Parameters
        # self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('hef_path', '/root/ros2_ws/src/hailo_yolo/resource/yolov8n.hef')
        # image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.hef_path = self.get_parameter('hef_path').get_parameter_value().string_value
        self.rs_height = 480
        self.rs_width = 640
        rs_FPS = 30

        # Realsense pipeline setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.rs_width, self.rs_height, rs.format.bgr8, rs_FPS)
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # Uncomment if depth is needed
        self.bridge = CvBridge()
        

        # Subscriber
        # self.create_subscription(Image, image_topic, self.image_callback, 10)

        # Publisher
        self.detection_pub = self.create_publisher(Detection2DArray, '/yolo/detections', qos_profile=5)
        self.rs_pub = self.create_publisher(Image, '/realsense/raw_image', qos_profile=5)
        self.det_msg = Detection2DArray()
        self.image_msg = Image()
        self.timer = self.create_timer(PUB_TIME, self.timer_callback)
        

        # Hailo object initialize
        self.input_vstream_info = None
        self.output_vstream_info = None
        self.input_height = 0
        self.input_width = 0
        
        # Padding image
        self.scale = 0
        self.pad_top = 0
        self.pad_left = 0

        self.get_logger().info(f"YoloInferenceNode initialized. Using HEF: {self.hef_path}")
        self.get_logger().info(f"RealSense configured for: {self.rs_width}x{self.rs_height} @ {rs_FPS} FPS")


    def run(self):
        try:
            self.pipeline.start(self.config)
            self.get_logger().info("Realsense pipeline started.")
        except Exception as e:
            self.get_logger().info(f"Failed to start Realsense pipeline: {e}")
            rclpy.shutdown()
            return
        
        # Hailo setup
        self.get_logger().info("Setting up Hailo...")
        params = VDevice.create_params() # Default parameters
        params.group_id = "SHARED"
        params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
        target = VDevice(params=params)
        infer_model = target.create_infer_model(self.hef_path)
        myInferModel = ph.InferModel(infer_model, self.hef_path)
        myInferModel.set_batch_size(batch_size=1) # 1picture-processing/execution
        # myInferModel.set_power_mode()
        # my_configured_inferModel = myInferModel.configure()
        myhef = myInferModel.hef

        # configure_params = ConfigureParams.create_from_hef(hef=myhef, interface=HailoStreamInterface.PCIe)
        # network_groups = target.configure(myhef, configure_params)
        network_groups = target.configure(myhef)
        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_params = InputVStreamParams.make(self.network_group, quantized=False,
                                                    format_type=FormatType.FLOAT32,queue_size=5)
        self.output_vstream_params = OutputVStreamParams.make(self.network_group, quantized=False,
                                                        format_type=FormatType.FLOAT32,queue_size=5)
        self.input_vstream_info = myhef.get_input_vstream_infos()[0]
        self.output_vstream_info = myhef.get_output_vstream_infos()[0]

        self.input_height, self.input_width = self.input_vstream_info.shape[:2]

        try:
            # Create inference pipeline
            with ph.InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params, tf_nms_format=False) as self.infer_pipeline:
                if self.hef_path.rfind('seg') == -1:
                    self.infer_pipeline.set_nms_score_threshold(TH_SCORE)
                with self.network_group.activate(self.network_group_params):
                    self.get_logger().info('Start streaming.')
                    while rclpy.ok():
                        rclpy.spin_once(self, timeout_sec=0.01)
                        # Get frames from Realsense
                        frames = self.pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        self.shot_time = self.get_clock().now().to_msg()
                        if not color_frame:
                            continue
                        color_image = np.asarray(color_frame.get_data())

                        # Padding image
                        resized_image, self.scale, self.pad_top, self.pad_left = self.resize_with_letterbox(color_image,target_shape=(1,self.input_height,self.input_width,3))

                        # Execute inference
                        results = self.run_yolo(resized_image)
                        detection_msg = self.postprocess_yolo_results(results,'object_detection')

                        # Publish results
                        self.publish_results(np.asanyarray(color_frame.get_data()), detection_msg)


        finally:
            self.pipeline.stop()

    def resize_with_letterbox(self, image, target_shape=(1,640,640,3), padding_value=(0, 0, 0)):
        """
        Resizes an image with letterboxing to fit the target size, preserving aspect ratio.
        
        Parameters:
            image (numpy.asanyarray): Input image.
            target_shape (tuple): Target shape in NHWC format (batch_size, target_height, target_width, channels).
            padding_value (tuple): RGB values for padding (default is black padding).
            
        Returns:
            letterboxed_image (ndarray): The resized image with letterboxing.
            scale (float): Scaling ratio applied to the original image.
            pad_top (int): Padding applied to the top.
            pad_left (int): Padding applied to the left.
        """        
        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get the original image dimensions (height, width, channels)
        h, w, c = image.shape
        
        # Extract target height and width from target_shape (NHWC format)
        target_height, target_width = target_shape[1], target_shape[2]
        
        # Calculate the scaling factors for width and height
        scale_x = target_width / w
        scale_y = target_height / h
        
        # Choose the smaller scale factor to preserve the aspect ratio
        scale = min(scale_x, scale_y)
        
        # Calculate the new dimensions based on the scaling factor
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image to the new dimensions
        resized_image = cv2.resize(image, (new_w, new_h),interpolation=cv2.INTER_LINEAR)
        
        # Create a new image with the target size, filled with the padding value
        letterboxed_image = np.full((target_height, target_width, c), padding_value, dtype=np.uint8)
        
        # Compute the position where the resized image should be placed (padding)
        pad_top = (target_height - new_h) // 2
        pad_left = (target_width - new_w) // 2
        
        # Place the resized image onto the letterbox background
        letterboxed_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_image

        expanded_dims_image = np.expand_dims(letterboxed_image, axis=0)
        final_image = expanded_dims_image.astype(np.float32)
        
        # Return the letterboxed image, scaling ratio, and padding (top, left)
        return final_image, scale, pad_top, pad_left
        


    def run_yolo(self, image):
        """Performs inference using the managed infer_pipeline."""
        input_data = {self.input_vstream_info.name: image}
        infer_results = self.infer_pipeline.infer(input_data)
        
        
        return infer_results

    def postprocess_yolo_results(self, raw_results, mode):
        if mode == 'object_detection':
            msg = self.od(raw_results)
        elif mode == 'segment':
            msg = self.seg(raw_results)
        else:
            raise NameError(f'No mode such a {mode}.')
        return msg
    
    def od(self, raw_results):
        """
        Convert to ros message from hailo-formated inference result.
        raw_results: type:nd.ndarray, row:CLASS_LABEL, col:[BBOX:param(ymin,xmin,ymax,xmax),score], 
        height:Detection num, shape:80*5*100 (based on hailo output_tensor)
        """
        detection_array_msg = Detection2DArray()
        results = raw_results[self.output_vstream_info.name][0]
        for class_id, arr in enumerate(results):
            if arr.shape[0] == 0:
                continue # No detection

            for det in arr:
                ymin, xmin, ymax, xmax, score = det
                xy = np.array([[xmin,xmax],[ymin,ymax]])
                xy = xy*[self.input_width, self.input_height]
                # Scaling for original frame resolution. e.x.)640*640->1280*720
                center_x, center_y, width, height = self.reverse_rescale_bboxes((xy[0,0],xy[1,0],xy[0,1],xy[1,1]),(self.rs_height,self.rs_width))

                detection_msg = Detection2D()

                # Bounding box
                detection_msg.bbox.center.position.x = center_x
                detection_msg.bbox.center.position.y = center_y
                detection_msg.bbox.center.theta = 0.0
                detection_msg.bbox.size_x = width
                detection_msg.bbox.size_y = height

                # CLASS ID, Score
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(CLASS_LABELS[class_id])
                hypothesis.hypothesis.score = float(score)
                # hypothesis.pose.pose = 0
                # hypothesis.pose.covariance = [0.0] * 36
                detection_msg.results.append(hypothesis)
                detection_array_msg.detections.append(detection_msg)

        return detection_array_msg
    
    def seg(self, raw_results):
        detection_array_msg = Detection2DArray()
        print(self.output_vstream_info.name)
        results = raw_results[self.output_vstream_info.name][0]
        for class_id, arr in enumerate(results):
            if arr.shape[0] == 0:
                continue # No detection

            for det in arr:
                ymin, xmin, ymax, xmax, score = det
                xy = np.array([[xmin,xmax],[ymin,ymax]])
                xy = xy*[self.input_width, self.input_height]
                # Scaling for original frame resolution. e.x.)640*640->1280*720
                center_x, center_y, width, height = self.reverse_rescale_bboxes((xy[0,0],xy[1,0],xy[0,1],xy[1,1]),(self.rs_height,self.rs_width))

                detection_msg = Detection2D()

                # Bounding box
                detection_msg.bbox.center.position.x = center_x
                detection_msg.bbox.center.position.y = center_y
                detection_msg.bbox.center.theta = 0.0
                detection_msg.bbox.size_x = width
                detection_msg.bbox.size_y = height

                # CLASS ID, Score
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(CLASS_LABELS[class_id])
                hypothesis.hypothesis.score = float(score)
                # hypothesis.pose.pose = 0
                # hypothesis.pose.covariance = [0.0] * 36
                detection_msg.results.append(hypothesis)
                detection_array_msg.detections.append(detection_msg)

        return detection_array_msg
    
    def reverse_rescale_bboxes(self, bbox, original_shape):
        """
        Reverse rescales bounding boxes from the letterbox image to the original image.

        Parameters:
            bbox (tuple): (x1, y1, x2, y2)
            original_shape (tuple): The shape (height, width) of the original image before resizing.

        Returns:
            center_x, center_y, width, height
        """
        orig_h, orig_w = original_shape  # original image height and width

        
        # Reverse padding
        x1, y1, x2, y2 = bbox
        x1 -= self.pad_left
        y1 -= self.pad_top
        x2 -= self.pad_left
        y2 -= self.pad_top
        
        # Reverse scaling
        x1 = (x1 / self.scale)
        y1 = (y1 / self.scale)
        x2 = (x2 / self.scale)
        y2 = (y2 / self.scale)
        
        # Clip the bounding box to make sure it fits within the original image dimensions
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        # BBOX parameter for Detection2DArray msg
        center_x = float((x1 + x2) / 2)
        center_y = float((y1 + y2) / 2)
        width = float(x2 - x1)
        height = float(y2 - y1)
        
        return center_x, center_y, width, height
    
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

    def timer_callback(self):
        # Publish message
        self.detection_pub.publish(self.det_msg)
        self.rs_pub.publish(self.image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    try:
        # rclpy.spin(node)
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

# For debug
if __name__ == '__main__':
    main()
