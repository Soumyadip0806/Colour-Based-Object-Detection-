import sys
import os
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QComboBox, QTextEdit, QSizePolicy, QHBoxLayout, QListWidget, QListWidgetItem, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import vlc
import shutil

class VideoProcessingWorker(QThread):
    processing_finished = pyqtSignal(list)
    update_status = pyqtSignal(str)

    def __init__(self, video_path, vehicle_type, color, output_directory):
        super().__init__()
        self.video_path = video_path
        self.vehicle_type = vehicle_type
        self.color = color
        self.output_directory = output_directory

    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        model = maskrcnn_resnet50_fpn(weights=weights).to(device)
        model.eval()

        COCO_INSTANCE_CATEGORY_NAMES = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
            'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        color_ranges = {
            'red': ((160, 50, 50), (180, 255, 255)),
            'maroon_red': ((0, 100, 100), (10, 255, 255)),
            'maroon_green': ((40, 50, 50), (80, 255, 255)),
            'white': ((0, 0, 220), (180, 40, 255)),
            'yellow': ((20, 100, 100), (30, 255, 255)),
            'deep_black': ((0, 0, 0), (180, 255, 30))
        }

        os.makedirs(self.output_directory, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            self.update_status.emit("Error: Failed to open the video file.")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_skip = 2

        start_time = time.time()

        object_trackers = {}
        next_object_id = 0

        def save_segment(start_time, end_time, frames):
            filename = f"car_segment_{start_time:.2f}_{end_time:.2f}.mp4"
            output_path = os.path.join(self.output_directory, filename)
            height, width, _ = frames[0].shape if frames else (0, 0, 0)
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()
            self.update_status.emit(f"Saved segment {filename} from {start_time} sec to {end_time} sec")

        def init_object_tracker(frame, bbox):
            nonlocal next_object_id
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, bbox)
            object_trackers[next_object_id] = tracker
            next_object_id += 1

        def update_object_tracker(frame, object_id, bbox):
            tracker = object_trackers[object_id]
            success, bbox = tracker.update(frame)
            if success:
                object_trackers[object_id] = tracker
            return success, bbox

        desired_color = self.color.lower()

        if desired_color not in color_ranges:
            self.update_status.emit("Error: Invalid color specified.")
            return

        color_range = color_ranges[desired_color]

        frames_to_save = []
        car_detected = False
        car_start_time = 0
        car_end_time = 0
        frames_without_detection = 0

        for frame_number in range(0, frame_count, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break

            self.update_status.emit(f"Processing frame {frame_number}/{frame_count}")

            frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5)
            img = T.ToTensor()(frame_resized).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(img)

            color_detected_in_frame = False
            lower_bound, upper_bound = color_range
            for score, label, mask in zip(prediction[0]['scores'], prediction[0]['labels'], prediction[0]['masks']):
                if score > 0.5 and COCO_INSTANCE_CATEGORY_NAMES[label.item()] == 'car':
                    mask = mask[0, :, :].cpu().numpy()
                    mask[mask >= 0.5] = 255
                    mask[mask < 0.5] = 0
                    mask = mask.astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        car_image = frame_resized[y:y+h, x:x+w]
                        if car_image.shape[0] > 0 and car_image.shape[1] > 0:
                            hsv_car = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)
                            mask = cv2.inRange(hsv_car, lower_bound, upper_bound)
                            color_pixel_count = np.count_nonzero(mask)
                            if color_pixel_count > (w * h) * 0.5:
                                color_detected_in_frame = True
                                break
                    if color_detected_in_frame:
                        break

            if car_detected:
                success, bbox = update_object_tracker(frame_resized, 0, bbox)
                if success:
                    centroid = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                    cv2.circle(frame_resized, centroid, 4, (0, 255, 0), -1)
                    frames_without_detection = 0
                else:
                    frames_without_detection += 1

            if color_detected_in_frame and not car_detected:
                car_detected = True
                car_start_time = frame_number / fps
                x, y, w, h = cv2.boundingRect(contours[0])
                bbox = (x, y, w, h)
                init_object_tracker(frame_resized, bbox)

            if car_detected:
                frames_to_save.append(frame_resized)

            if car_detected and not color_detected_in_frame:
                frames_without_detection += 1

            if car_detected and frames_without_detection > 100:
                car_detected = False
                car_end_time = frame_number / fps
                save_segment(car_start_time, car_end_time, frames_to_save)
                frames_to_save.clear()

        if car_detected:
            car_end_time = frame_count / fps
            save_segment(car_start_time, car_end_time, frames_to_save)

        end_time = time.time()
        self.update_status.emit(f"Processing completed in {end_time - start_time:.2f} seconds")

        segments = [file for file in os.listdir(self.output_directory) if file.endswith('.mp4')]
        self.processing_finished.emit(segments)
        cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vehicle Color Detection")
        self.setGeometry(100, 100, 1200, 800)

        self.video_path = ""
        self.output_directory = "output_videos"
        self.vlc_instance = vlc.Instance()
        self.media_player = self.vlc_instance.media_player_new()

        main_layout = QVBoxLayout()
        header_layout = QVBoxLayout()

        self.header_label = QLabel("Govt. Of West Bengal")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #FFFFFF;")
        header_layout.addWidget(self.header_label)

        main_layout.addLayout(header_layout)

        content_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        self.sub_header_label_left = QLabel("Vehicle colour detection")
        self.sub_header_label_left.setAlignment(Qt.AlignCenter)
        self.sub_header_label_left.setStyleSheet("font-size: 24px; color: #FFFFFF;")
        left_layout.addWidget(self.sub_header_label_left)

        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)
        left_layout.addWidget(self.select_button)

        self.vehicle_type_label = QLabel("Select Vehicle Type:")
        left_layout.addWidget(self.vehicle_type_label)

        self.vehicle_type_combo = QComboBox()
        self.vehicle_type_combo.addItems(["Car", "Bus", "Truck"])
        left_layout.addWidget(self.vehicle_type_combo)

        self.color_label = QLabel("Select Color:")
        left_layout.addWidget(self.color_label)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["Red", "Maroon_red", "Maroon_green", "White", "Yellow", "Deep_black"])
        left_layout.addWidget(self.color_combo)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        left_layout.addWidget(self.status_text)

        self.process_button = QPushButton("Process Video")
        self.process_button.clicked.connect(self.process_video)
        left_layout.addWidget(self.process_button)

        self.video_list = QListWidget()
        left_layout.addWidget(self.video_list)

        self.download_button = QPushButton("Download Selected Video")
        self.download_button.clicked.connect(self.download_video)
        left_layout.addWidget(self.download_button)

        self.show_video_button = QPushButton("Show Selected Video")
        self.show_video_button.clicked.connect(self.show_video)
        left_layout.addWidget(self.show_video_button)

        content_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()

        self.sub_header_label_right = QLabel("Show output video here")
        self.sub_header_label_right.setAlignment(Qt.AlignCenter)
        self.sub_header_label_right.setStyleSheet("font-size: 24px; color: #FFFFFF;")
        right_layout.addWidget(self.sub_header_label_right)

        self.video_frame = QLabel()
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setMinimumSize(640, 700)
        right_layout.addWidget(self.video_frame)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self.set_position)
        right_layout.addWidget(self.slider)
        self.slider.setVisible(False)  # Initially hide the slider

        video_control_layout = QVBoxLayout()

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_video)
        video_control_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_video)
        video_control_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        video_control_layout.addWidget(self.stop_button)

        right_layout.addLayout(video_control_layout)

        content_layout.addLayout(right_layout)

        main_layout.addLayout(content_layout)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.setStyleSheet("""
            QWidget {
                background-color: rgb(1,4,46);
                color: #FFFFFF;
            }
            QPushButton {
                font-size: 18px;
                padding: 12px;
                margin: 10px;
                background-color: #7289DA;
                color: white;
                border: 2px solid #7289DA;
                border-radius: 10px;
            }
            QTextEdit {
                font-size: 18px;
                padding: 12px;
                border: 2px solid #cccccc;
                border-radius: 10px;
                margin: 10px;
                background-color: rgb(4, 26, 21);
                color: white;
            }
            QListWidget {
                font-size: 18px;
                padding: 12px;
                border: 2px solid #cccccc;
                border-radius: 10px;
                margin: 10px;
                background-color: rgb(4, 26, 21);
                color: white;
            }
            QListWidget::item:selected {
                background-color: #5865F2;
                color: white;
            }
            QLabel#video_frame {
                border: 2px solid #cccccc;
                border-radius: 10px;
                margin: 10px;
            }
        """)

        self.media_player.set_hwnd(int(self.video_frame.winId()))

        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.update_slider)

    def select_video(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Videos (*.mp4 *.avi *.mov)")
        if file_dialog.exec_():
            self.video_path = file_dialog.selectedFiles()[0]
            self.status_text.append(f"Selected video: {self.video_path}")

    def process_video(self):
        if not self.video_path:
            self.status_text.append("Error: No video selected.")
            return

        vehicle_type = self.vehicle_type_combo.currentText()
        color = self.color_combo.currentText()

        self.worker = VideoProcessingWorker(self.video_path, vehicle_type, color, self.output_directory)
        self.worker.processing_finished.connect(self.on_processing_finished)
        self.worker.update_status.connect(self.status_text.append)
        self.worker.start()

    def on_processing_finished(self, segments):
        self.video_list.clear()
        for segment in segments:
            item = QListWidgetItem(segment)
            self.video_list.addItem(item)
        self.status_text.append("Processing finished.")

    def download_video(self):
        selected_item = self.video_list.currentItem()
        if not selected_item:
            self.status_text.append("Error: No video selected for download.")
            return

        filename = selected_item.text()
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Video", filename, "Videos (*.mp4);;All Files (*)", options=options)
        if save_path:
            shutil.copy(os.path.join(self.output_directory, filename), save_path)
            self.status_text.append(f"Downloaded video saved to: {save_path}")

    def show_video(self):
        selected_item = self.video_list.currentItem()
        if not selected_item:
            self.status_text.append("Error: No video selected to show.")
            return

        video_path = os.path.join(self.output_directory, selected_item.text())
        media = self.vlc_instance.media_new(video_path)
        self.media_player.set_media(media)
        self.media_player.play()
        self.status_text.append(f"Loaded video for playing: {video_path}")

        self.slider.setVisible(True)  # Show the slider when the video starts playing
        self.timer.start()

    def play_video(self):
        self.media_player.play()
        self.slider.setVisible(True)  # Show the slider when the video is playing
        self.timer.start()

    def pause_video(self):
        self.media_player.pause()
        self.timer.stop()

    def stop_video(self):
        self.media_player.stop()
        self.timer.stop()
        self.slider.setValue(0)
        self.slider.setVisible(False)  # Hide the slider when the video is stopped

    def set_position(self, position):
        self.media_player.set_position(position / 1000.0)

    def update_slider(self):
        self.slider.setValue(int(self.media_player.get_position() * 1000))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
