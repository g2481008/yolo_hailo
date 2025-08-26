import cv2
import matplotlib.pyplot as plt
import numpy as np
import degirum as dg
from pprint import pprint
import json

def main_3():
    image_path='/root/myPython_test/resource/Cat.jpg'
    print_image_size(image_path)
    original_image_array = read_image_as_rgb(image_path)
    # Load the model
    model = dg.load_model(
        model_name='yolov8n',
        inference_host_address='@local',
        zoo_url='/root/myPython_test/model_zoo/'
    )

    # Prepare the input image
    image_array, scale, pad_top, pad_left = resize_with_letterbox('/root/myPython_test/resource/Cat.jpg', model.input_shape[0])

    # Run inference
    inference_result = model(image_array)

    # Pretty print the results
    with open('/root/myPython_test/model_zoo/yolov8n/coco_labels.json', "r") as json_file:
        label_dictionary = json.load(json_file)
        detection_results = postprocess_detection_results(inference_result.results[0]['data'], model.input_shape[0], 80, label_dictionary)
        pprint(detection_results)

    # Padding sized image
    # overlay_image = overlay_bboxes_and_labels(image_array[0], detection_results)
    # display_images([overlay_image], title="Image with Bounding Boxes and Labels")

    # Original-sized image
    scaled_detections = reverse_rescale_bboxes(detection_results, scale, pad_top, pad_left,original_image_array.shape[:2])
    overlay_original_image = overlay_bboxes_and_labels(original_image_array, scaled_detections)
    display_images([overlay_original_image], title="Original Image with Bounding Boxes and Labels")
    # pprint(inference_result.results)

def main_1():
    image_path='/root/myPython_test/resource/Cat.jpg'
    print_image_size(image_path)
    original_image_array = read_image_as_rgb(image_path)
    display_images([original_image_array], title="Original Image")

def main_2():
    image_array, scale, pad_top, pad_left = resize_with_letterbox('/root/myPython_test/resource/Cat.jpg', (1, 640,640,3))
    display_images([image_array[0]])



def read_image_as_rgb(image_path):
    # Load the image in BGR format (default in OpenCV)
    image_bgr = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image_bgr is None:
        raise ValueError(f"Error: Unable to load image from path: {image_path}")
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb

def print_image_size(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image from path: {image_path}")
    else:
        # Get the image size (height, width, channels)
        height, width, channels = image.shape
        print(f"Image size: {height}x{width} (Height x Width)")

def display_images(images, title="Images", figsize=(15, 5)):
    """
    Display a list of images in a single row using Matplotlib.

    Parameters:
    - images (list): List of images (NumPy arrays) to display.
    - title (str): Title for the plot.
    - figsize (tuple): Size of the figure.
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    if num_images == 1:
        axes = [axes]  # Make it iterable for a single image
    for ax, image in zip(axes, images):
        ax.imshow(image)
        ax.axis('off')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def resize_with_letterbox(image_path, target_shape, padding_value=(0, 0, 0)):
    """
    Resizes an image with letterboxing to fit the target size, preserving aspect ratio.
    
    Parameters:
        image_path (str): Path to the input image.
        target_shape (tuple): Target shape in NHWC format (batch_size, target_height, target_width, channels).
        padding_value (tuple): RGB values for padding (default is black padding).
        
    Returns:
        letterboxed_image (ndarray): The resized image with letterboxing.
        scale (float): Scaling ratio applied to the original image.
        pad_top (int): Padding applied to the top.
        pad_left (int): Padding applied to the left.
    """
    # Load the image from the given path
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Error: Unable to load image from path: {image_path}")
    
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

    final_image = np.expand_dims(letterboxed_image, axis=0)
    
    # Return the letterboxed image, scaling ratio, and padding (top, left)
    return final_image, scale, pad_top, pad_left

def postprocess_detection_results(detection_output, input_shape, num_classes, label_dictionary, confidence_threshold=0.3):
    """
    Process the raw output tensor to produce formatted detection results.
    
    Parameters:
        detection_output (numpy.ndarray): The flattened output tensor from the model containing detection results.
        input_shape (tuple): The shape of the input image in the format (batch, input_height, input_width, channels).
        num_classes (int): The number of object classes that the model predicts.
        label_dictionary (dict): Mapping of class IDs to class labels.
        confidence_threshold (float, optional): Minimum confidence score required to keep a detection. Defaults to 0.3.

    Returns:
        list: List of dictionaries containing detection results in JSON-friendly format.
    """
    # Unpack input dimensions (batch is unused, but included for flexibility)
    batch, input_height, input_width, _ = input_shape
    
    # Initialize an empty list to store detection results
    new_inference_results = []

    # Reshape and flatten the raw output tensor for parsing
    output_array = detection_output.reshape(-1)

    # Initialize an index pointer to traverse the output array
    index = 0

    # Loop through each class ID to process its detections
    for class_id in range(num_classes):
        # Read the number of detections for this class from the output array
        num_detections = int(output_array[index])
        index += 1  # Move to the next entry in the array

        # Skip processing if there are no detections for this class
        if num_detections == 0:
            continue

        # Iterate through each detection for this class
        for _ in range(num_detections):
            # Ensure there is enough data to process the next detection
            if index + 5 > len(output_array):
                # Break to prevent accessing out-of-bounds indices
                break

            # Extract confidence score and bounding box values
            score = float(output_array[index + 4])
            y_min, x_min, y_max, x_max = map(float, output_array[index : index + 4])
            index += 5  # Move index to the next detection entry

            # Skip detections if the confidence score is below the threshold
            if score < confidence_threshold:
                continue

            # Convert bounding box coordinates to absolute pixel values
            x_min = x_min * input_width
            y_min = y_min * input_height
            x_max = x_max * input_width
            y_max = y_max * input_height

            # Create a detection result with bbox, score, and class label
            result = {
                "bbox": [x_min, y_min, x_max, y_max],  # Bounding box in pixel coordinates
                "score": score,  # Confidence score of the detection
                "category_id": class_id,  # Class ID of the detected object
                "label": label_dictionary.get(str(class_id), f"class_{class_id}"),  # Class label or fallback
            }
            new_inference_results.append(result)  # Store the formatted detection

        # Stop parsing if remaining output is padded with zeros (no more detections)
        if index >= len(output_array) or all(v == 0 for v in output_array[index:]):
            break

    # Return the final list of detection results
    return new_inference_results

import cv2
import numpy as np

def overlay_bboxes_and_labels(image, annotations, color=(0, 255, 0), font_scale=1, thickness=2):
    """
    Overlays bounding boxes and labels on the image for a list of annotations.
    
    Parameters:
        image (ndarray): The input image (in RGB format).
        annotations (list of dicts): List of dictionaries with 'bbox' (x1, y1, x2, y2) and 'label' keys.
        color (tuple): The color of the bounding box and text (default is green).
        font_scale (int): The font scale for the label (default is 1).
        thickness (int): The thickness of the bounding box and text (default is 2).
    
    Returns:
        image_with_bboxes (ndarray): The image with the bounding boxes and labels overlayed.
    """
    # Convert the image from RGB to BGR (OpenCV uses BGR by default)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Loop over each annotation (bbox and label)
    for annotation in annotations:
        bbox = annotation['bbox']  # Bounding box as (x1, y1, x2, y2)
        label = annotation['label']  # Label text
        
        # Unpack bounding box coordinates
        x1, y1, x2, y2 = bbox
        
        # Convert float coordinates to integers
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        
        # Draw the rectangle (bounding box)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)
        
        # Put the label text on the image
        cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    # Convert the image back to RGB for display or further processing
    image_with_bboxes = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_with_bboxes

def reverse_rescale_bboxes(annotations, scale, pad_top, pad_left, original_shape):
    """
    Reverse rescales bounding boxes from the letterbox image to the original image, returning new annotations.

    Parameters:
        annotations (list of dicts): List of dictionaries, each containing a 'bbox' (x1, y1, x2, y2) and other fields.
        scale (float): The scale factor used for resizing the image.
        pad_top (int): The padding added to the top of the image.
        pad_left (int): The padding added to the left of the image.
        original_shape (tuple): The shape (height, width) of the original image before resizing.

    Returns:
        new_annotations (list of dicts): New annotations with rescaled bounding boxes adjusted back to the original image.
    """
    orig_h, orig_w = original_shape  # original image height and width
    
    new_annotations = []
    
    for annotation in annotations:
        bbox = annotation['bbox']  # Bounding box as (x1, y1, x2, y2)
        
        # Reverse padding
        x1, y1, x2, y2 = bbox
        x1 -= pad_left
        y1 -= pad_top
        x2 -= pad_left
        y2 -= pad_top
        
        # Reverse scaling
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        
        # Clip the bounding box to make sure it fits within the original image dimensions
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        # Create a new annotation with the rescaled bounding box and the original label
        new_annotation = annotation.copy()
        new_annotation['bbox'] = (x1, y1, x2, y2)
        
        # Append the new annotation to the list
        new_annotations.append(new_annotation)
    
    return new_annotations

if __name__ == '__main__':
    main_3()




