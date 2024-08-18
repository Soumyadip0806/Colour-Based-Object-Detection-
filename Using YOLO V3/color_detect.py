import cv2
import numpy as np
import os
import time

# Load YOLO
net = cv2.dnn.readNet(r"C:\Users\soumy\PycharmProjects\GUI pYQT\yolov3.weights", r"C:\Users\soumy\PycharmProjects\GUI pYQT\yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open(r"C:\Users\soumy\PycharmProjects\GUI pYQT\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define range of yellow color in HSV
lower_yellow = np.array([160, 50, 50])
upper_yellow = np.array([180, 255, 255])

# Output directory
output_directory = 'output/'
os.makedirs(output_directory, exist_ok=True)

# Initialize video capture device
video_path = r'C:\Users\soumy\PycharmProjects\GUI pYQT\testing.mkv'  # Replace 'your_video_file.mp4' with the path to your video file
cap = cv2.VideoCapture(video_path)

# Process each frame
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames_processed = 0
detected_cars = []
yellow_car_detected = False

start_time = time.time()
lower_val = np.array([160, 50, 50])
upper_val = np.array([180, 255, 255])
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames_processed += 1
    frame_resize = cv2.resize(frame, None, fx=0.5, fy=0.5)
    hsv = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image - any green color will show up as white
    mask = cv2.inRange(hsv, lower_val, upper_val)

    # if there are any white pixels on mask, sum will be > 0
    ColorExist = np.count_nonzero(mask)
    if (ColorExist == 0):
        continue
    blob = cv2.dnn.blobFromImage(frame_resize, 0.00392, (192, 192), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    car_detections = [detection for output in outs for detection in output if
                      np.argmax(detection[5:]) == 2 and detection[4] > 0.4]

    for detection in car_detections:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            height, width, _ = frame_resize.shape
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            car_image = frame_resize[y:y+h, x:x+w]

            if car_image.shape[0] > 0 and car_image.shape[1] > 0:
                    hsv_car = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv_car, lower_yellow, upper_yellow)
                    yellow_pixel_count = np.count_nonzero(mask)

                    if yellow_pixel_count > (w * h) * 0.5:
                        cv2.rectangle(frame_resize, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        detected_cars.append(frame_resize)
                        yellow_car_detected = True
                        print(f"Yellow car detected at frame {frames_processed}/{frame_count}")
                        break


    print(f"Processed frame {frames_processed}/{frame_count}")

# Release video capture
cap.release()

# Save detected cars
for i, car_image in enumerate(detected_cars):
    cv2.imwrite(output_directory + f"car_{i}.jpg", car_image)
    print("Saved car:", output_directory + f"car_{i}.jpg")

end_time = time.time()
print("Total processing time:", end_time - start_time, "seconds")
