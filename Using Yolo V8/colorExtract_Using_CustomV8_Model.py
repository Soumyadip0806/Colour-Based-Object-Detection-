import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# Load YOLOv8 model
def load_yolo():
    model = YOLO(r"C:\Users\soumy\PycharmProjects\GUI pYQT\best.pt")  # Update this path to the correct YOLOv8 model path if necessary
    return model

# Define color ranges for different car colors
color_ranges = {
    'red': ((160, 50, 50), (180, 255, 255)),
    'red2': ((160, 50, 50), (195, 255, 255)),
    'maroon_red': ((0, 100, 100), (10, 255, 255)),
    'blue': ((100,50,50), (130, 255, 255)),
    'brown':((20, 100, 100), (30, 255,255)),
    'maroon_green': ((40, 50, 50), (80, 255, 255)),
    'white': ((0, 0, 200), (180, 40, 255)),
    'yellow': ((20, 100, 100), (30, 255, 255)),
    'deep_black': ((0, 0, 0), (180, 255, 30))
}

# Output directory
output_directory = 'output_car_segments/'
os.makedirs(output_directory, exist_ok=True)

# Initialize video capture device
video_path = r'C:\Users\soumy\PycharmProjects\GUI pYQT\testing.mkv'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Function to save segment
def save_segment(start_time, end_time, frames, car_id):
    filename = f"car_{car_id}segment{start_time:.2f}_{end_time:.2f}.mp4"
    output_path = os.path.join(output_directory, filename)
    height, width, _ = frames[0].shape if frames else (0, 0, 0)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Saved segment {filename} from {start_time:.2f} sec to {end_time:.2f} sec")

# Initialize variables for object tracking
car_detected = False
frames_to_save = []
car_start_time = None
last_detection_frame = None
buffer_frames = 10  # Number of frames to add after last detection
next_car_id = 0  # Initialize next_car_id here

# Get user input for the desired car color
desired_color = input("Enter the color of the car you want to detect (maroon_red, maroon_green,blue,brown, white, yellow, deep_black, red, red2): ").lower()

# Check if the desired color is valid
if desired_color not in color_ranges:
    print("Error: Invalid color specified.")
    exit()

# Retrieve the color range for the desired color
color_range = color_ranges[desired_color]

# Load the YOLO model
model = load_yolo()
class_names = model.names

frame_number = 0

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip every other frame
    if frame_number % 2 != 0:
        frame_number += 1
        continue

    print(f"Processing frame {frame_number}/{frame_count}")

    # Resize frame to a fixed size for consistency
    frame_resized = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Check if the desired color is present in the frame
    lower_bound, upper_bound = color_range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    if np.count_nonzero(mask) == 0:
        if car_detected:
            last_detection_frame += 1
            frames_to_save.append(frame)
            if last_detection_frame >= buffer_frames:
                car_end_time = (frame_number - buffer_frames) / fps
                save_segment(car_start_time, car_end_time, frames_to_save, next_car_id)
                car_detected = False
                frames_to_save = []
        frame_number += 1
        continue  # Skip frame if desired color is not present

    # Detect objects in the frame using YOLOv8
    results = model(frame_resized)

    color_detected_in_frame = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = box.conf[0]

            if class_names[class_id] == 'car':
                car_image = frame_resized[y1:y2, x1:x2]
                if car_image.shape[0] > 0 and car_image.shape[1] > 0:
                    hsv_car = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)
                    car_mask = cv2.inRange(hsv_car, lower_bound, upper_bound)
                    color_pixel_count = np.count_nonzero(car_mask)
                    if color_pixel_count > (car_image.shape[0] * car_image.shape[1]) * 0.5:
                        color_detected_in_frame = True
                        last_detection_frame = 0
                        if not car_detected:
                            car_detected = True
                            car_start_time = frame_number / fps
                            next_car_id += 1
                        break
            if color_detected_in_frame:
                break

    if color_detected_in_frame or car_detected:
        frames_to_save.append(frame)

    frame_number += 1

# Check if the car was still being detected at the end of the video
if car_detected:
    car_end_time = frame_number / fps
    save_segment(car_start_time, car_end_time, frames_to_save, next_car_id)

end_time = time.time()
print("Total processing time:", end_time - start_time, "seconds")

# Release video capture device
cap.release()