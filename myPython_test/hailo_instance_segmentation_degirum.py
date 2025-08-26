import degirum as dg
import cv2

# Load the segmentation model from the model zoo.
# Replace '<path_to_model_zoo>' with the directory path to your model assets.
model = dg.load_model(
    model_name='yolov8m_seg',
    inference_host_address='@local',
    zoo_url='/root/myPython_test/model_zoo/'
)

# Run inference on an input image.
# Replace '<path_to_input_image>' with the actual path to your image.
inference_result = model('/root/myPython_test/resource/IMG_6827.jpg')

# The segmentation overlay (with masks and labels) is available via the image_overlay attribute.
cv2.imshow("Segmentation Output", inference_result.image_overlay)

# Wait until the user presses 'x' or 'q' to close the window.
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('x') or key == ord('q'):
        break

cv2.destroyAllWindows()