from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QListWidget, QPushButton, QComboBox, QGridLayout, QVBoxLayout, QHBoxLayout, QStackedLayout, QMessageBox, QDialog, QListWidgetItem
from PyQt6.QtCore import pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont
from collections import deque
import json
import os
import sys
import cv2
import threading
from ultralytics import YOLO
import torch
import time
from datetime import datetime
import queue
import csv
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

#Modular Detector class, ensuring only one instance of model is loaded into gpu memory
class Weapon_Detector:
    def __init__(self):
        self.model = YOLO(r"model\weapon_detection.pt").to(DEVICE)
        self.detection_results = []

    def predict(self, frames):
        results = self.model.predict(frames, conf=0.50, iou=0.45, device=DEVICE, verbose=False, imgsz=640)
        detections = []
        for result in results:
            boxes = []
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                cls_idx = int(box.cls[0].item())
                conf = box.conf[0].item()
                label = self.model.names[cls_idx]
                boxes.append({
                    "coords": coords,
                    "label": label,
                    "conf": conf
                })
            detections.append(boxes)
        self.detection_results = detections
        return self.detection_results #returning a list boxes


#Generate Alarms
class Alarm:
    def __init__(self, cam_id, source_name, message, clip_path, status="Pending", alarm_id=None, timestamp=None):
        self.cam_id = cam_id
        self.source_name = source_name
        self.message = message
        self.clip_path = clip_path
        self.status = status
        self.id = alarm_id if alarm_id else f"alarm_{int(time.time())}_{cam_id}"
        self.timestamp = timestamp if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.is_recording = False
    
    def set_status(self, status):
        self.status = status

    def to_dict(self):
        return {
            "id": self.id,
            "cam_id": self.cam_id,
            "source_name": self.source_name,
            "message": self.message,
            "clip_path": self.clip_path,
            "status": self.status,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            cam_id=data.get("cam_id"),
            source_name=data.get("source_name"),
            message=data.get("message"),
            clip_path=data.get("clip_path"),
            status=data.get("status", "Pending"),
            alarm_id=data.get("id"),
            timestamp=data.get("timestamp")
        )

#Save a clip around the detection for the human in the middle
class ClipRecorder(threading.Thread):
    TARGET_FPS = 10.0
    RESOLUTION = (320, 240)

    def __init__(self, alarm, inference_worker, cam_id, output_path, pre_frames, duration_sec=5, fps=25.0):
        super().__init__()
        self.alarm = alarm
        self.inference_worker = inference_worker
        self.cam_id = cam_id
        self.output_path = output_path
        self.pre_frames = list(pre_frames) # Copy the buffer frames
        self.duration_sec = duration_sec
        self.fps = fps
        
    def run(self):
        self.alarm.is_recording = True
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, self.TARGET_FPS, self.RESOLUTION)
            if not out.isOpened():
                print(f"Error opening VideoWriter for path: {self.output_path}")
                return
                
            # Write downsampled and resized pre-trigger frames
            orig_pre_count = len(self.pre_frames)
            step = self.fps / self.TARGET_FPS
            target_pre_count = int(orig_pre_count / step)
            
            for i in range(target_pre_count):
                idx = int(i * step)
                if idx < orig_pre_count:
                    resized = cv2.resize(self.pre_frames[idx], self.RESOLUTION)
                    out.write(resized)
                
            # Record post-trigger annotated frames directly from inference_worker at target_fps
            post_count = 0
            target_post_frames = int(self.duration_sec * self.TARGET_FPS)
            
            last_recorded_frame_id = self.inference_worker.annotated_frame_ids[self.cam_id]
            frames_since_last_write = 0
            
            while post_count < target_post_frames and self.inference_worker.running:
                current_frame_id = self.inference_worker.annotated_frame_ids[self.cam_id]
                if current_frame_id != last_recorded_frame_id:
                    frames_since_last_write += (current_frame_id - last_recorded_frame_id)
                    last_recorded_frame_id = current_frame_id
                    
                    if frames_since_last_write >= step:
                        frame = self.inference_worker.latest_annotated_frames[self.cam_id]
                        if frame is not None:
                            resized = cv2.resize(frame, self.RESOLUTION)
                            out.write(resized)
                            post_count += 1
                            frames_since_last_write -= step
                else:
                    time.sleep(0.005)
                    
            out.release()
            print(f"Alarm annotated clip successfully written to {self.output_path}")
        except Exception as e:
            print(f"Error during video recording: {e}")
        finally:
            self.alarm.is_recording = False


#Verification By the Human in the middle
class AlarmVerificationDialog(QDialog):
    def __init__(self, alarm, dashboard_parent):
        super().__init__(dashboard_parent)
        self.alarm = alarm
        self.dashboard_parent = dashboard_parent
        self.setWindowTitle(f"Verify Security Breach Incident: {alarm.id}")
        self.resize(800, 560)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
                font-family: 'Segoe UI', Roboto, sans-serif;
            }
            QLabel {
                color: #f8fafc;
            }
            QPushButton {
                font-weight: bold;
                border-radius: 6px;
                padding: 10px 18px;
                font-size: 13px;
                border: none;
            }
        """)

        # Main Layout (Split Pane)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Left Panel (Video Player + Pause/Play Button)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        video_title = QLabel("Recorded Threat Anomaly:")
        video_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #94a3b8;")
        left_panel.addWidget(video_title)

        self.video_label = QLabel()
        self.video_label.setFixedSize(480, 360)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #020617; border: 1px solid #334155; border-radius: 8px;")
        self.video_label.setText("Loading Threat Video Clip...")
        left_panel.addWidget(self.video_label)
        
        self.btn_pause_play = QPushButton("⏸️ Pause Video")
        self.btn_pause_play.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
            }
        """)
        self.btn_pause_play.setEnabled(False)
        self.btn_pause_play.clicked.connect(self.toggle_pause_play)
        left_panel.addWidget(self.btn_pause_play)
        
        main_layout.addLayout(left_panel, 60)

        # Right Panel (Metadata & Action Buttons)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        right_panel.setContentsMargins(10, 0, 0, 0)

        meta_title = QLabel("Incident Information:")
        meta_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 5px;")
        right_panel.addWidget(meta_title)

        # Metadata Labels
        fields = [
            ("Incident ID", alarm.id),
            ("Timestamp", alarm.timestamp),
            ("Source Channel", alarm.source_name),
            ("Anomaly Type", alarm.message.split(" — ")[0]),
        ]
        
        for name, value in fields:
            row = QHBoxLayout()
            lbl_name = QLabel(f"{name}:")
            lbl_name.setStyleSheet("font-weight: bold; color: #94a3b8; font-size: 12px;")
            lbl_val = QLabel(str(value))
            lbl_val.setStyleSheet("color: #f8fafc; font-size: 12px;")
            lbl_val.setWordWrap(True)
            row.addWidget(lbl_name, 35)
            row.addWidget(lbl_val, 65)
            right_panel.addLayout(row)

        # Status Display
        status_row = QHBoxLayout()
        lbl_status_title = QLabel("Current Status:")
        lbl_status_title.setStyleSheet("font-weight: bold; color: #94a3b8; font-size: 12px;")
        self.lbl_status_val = QLabel(alarm.status.upper())
        status_row.addWidget(lbl_status_title, 35)
        status_row.addWidget(self.lbl_status_val, 65)
        right_panel.addLayout(status_row)
        
        self.update_status_style()

        right_panel.addStretch()

        # Action Buttons Container
        actions_title = QLabel("Security Action Controls:")
        actions_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 5px;")
        right_panel.addWidget(actions_title)

        # Buttons
        self.btn_confirm = QPushButton("✔️ Confirm (True Positive)")
        self.btn_confirm.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        self.btn_confirm.clicked.connect(self.confirm_alarm)
        right_panel.addWidget(self.btn_confirm)

        self.btn_dismiss = QPushButton("❌ Dismiss (False Positive)")
        self.btn_dismiss.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:pressed {
                background-color: #b91c1c;
            }
        """)
        self.btn_dismiss.clicked.connect(self.dismiss_alarm)
        right_panel.addWidget(self.btn_dismiss)

        self.btn_close = QPushButton("Close")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: #475569;
                color: white;
            }
            QPushButton:hover {
                background-color: #334155;
            }
        """)
        self.btn_close.clicked.connect(self.close)
        right_panel.addWidget(self.btn_close)

        main_layout.addLayout(right_panel, 40)

        # Set up Video Capture Loop
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_frame)
        
        if alarm.is_recording:
            self.video_label.setText("Recording threat video clip...\nPlease wait a few seconds.")
            self.check_recording_timer = QTimer(self)
            self.check_recording_timer.timeout.connect(self.check_if_recorded)
            self.check_recording_timer.start(500)
        else:
            self.start_video_playback()

    def check_if_recorded(self):
        if not self.alarm.is_recording:
            self.check_recording_timer.stop()
            self.start_video_playback()

    def start_video_playback(self):
        if os.path.exists(self.alarm.clip_path):
            self.cap = cv2.VideoCapture(self.alarm.clip_path)
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            timer_ms = int(1000.0 / self.video_fps)
            self.timer.start(timer_ms)
            self.btn_pause_play.setEnabled(True)
        else:
            self.video_label.setText("Threat Video Clip Not Found or Still Generating...")

    def update_status_style(self):
        status = self.alarm.status
        self.lbl_status_val.setText(status.upper())
        if status == "Pending":
            self.lbl_status_val.setStyleSheet("color: #f97316; font-weight: bold; font-size: 13px;")
        elif status == "True Positive":
            self.lbl_status_val.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 13px;")
        elif status == "False Positive":
            self.lbl_status_val.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 13px; text-decoration: line-through;")

    def play_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # Video ended - loop back to beginning (plays in a loop!)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            
            if ret and frame is not None:
                frame = cv2.resize(frame, (480, 360))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qImg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qImg))

    def toggle_pause_play(self):
        if self.cap is not None and self.cap.isOpened():
            if self.timer.isActive():
                self.timer.stop()
                self.btn_pause_play.setText("▶️ Play Video")
                self.btn_pause_play.setStyleSheet("""
                    QPushButton {
                        background-color: #10b981;
                        color: white;
                    }
                    QPushButton:hover {
                        background-color: #059669;
                    }
                """)
            else:
                self.timer.start()
                self.btn_pause_play.setText("⏸️ Pause Video")
                self.btn_pause_play.setStyleSheet("""
                    QPushButton {
                        background-color: #3b82f6;
                        color: white;
                    }
                    QPushButton:hover {
                        background-color: #2563eb;
                    }
                """)

    def confirm_alarm(self):
        self.dashboard_parent.update_alarm_status(self.alarm.id, "True Positive")
        self.update_status_style()

    def dismiss_alarm(self):
        self.dashboard_parent.update_alarm_status(self.alarm.id, "False Positive")
        self.update_status_style()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()
    
#A worker thread will capture frames from the source and update buffer
class Camera_Worker:
    def __init__(self, camid, source):
        self.camid = camid
        self.source = source
        self.running = True
        
        # 1. Request Hardware Accelerated Decoding (Uses NVDEC on RTX GPUs if supported)
        self.cap = cv2.VideoCapture(source, cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        
        # Check if source is a live camera (int) or video file (string/path)
        self.is_live = isinstance(source, int) or str(source).isdigit()
        
        print(f"Initializing Source: {source} | Live: {self.is_live}")
        
        self.latest_frame = None
        self.frame_buffer = deque(maxlen=150)
        self.frame_id = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0 # Fallback if fps fails
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        if self.cap.isOpened():
            print(f"Camera {self.camid} initialized")
            
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                if not self.is_live:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video files
                continue
            
            # 2. Downsample immediately to save RAM bandwidth across threads
            # 640x480 is standard for YOLO and SlowFast (which downsizes to 224 anyway)
            frame = cv2.resize(frame, (640, 480))
            self.latest_frame = frame
            self.frame_buffer.append(frame.copy())
            self.frame_id += 1
            
            # 3. Only throttle video files. Live cameras MUST run unthrottled.
            if not self.is_live:
                time.sleep(1.0 / self.fps)
    
    def get_latest_frame(self):
        return self.latest_frame
    
    def stop(self):
        self.running = False
        self.cap.release()

class Inference_Worker:
    def __init__(self, source, camera_workers):
        self.camera_workers = camera_workers
        self.running = True
        self.No_of_camera = len(source)
        self.display_frames = [None for _ in camera_workers]

        # Initialize the Abstraction Layer — weapon detection only
        self.weapon_detector = Weapon_Detector()

        # Thread-safe queue for the weapon inference background thread
        self.weapon_queue = queue.Queue()

        # Shared atomic reference for weapon results
        self.latest_weapon_boxes = [[] for _ in range(self.No_of_camera)]

        self.last_frame_ids = [-1 for _ in camera_workers]

        # Rolling buffers for fully annotated BGR frames
        self.annotated_buffers = [deque(maxlen=150) for _ in range(self.No_of_camera)] #Buffer to store annotated frames(last 150 frames) which will be saved when alarm goes off
        self.latest_annotated_frames = [None for _ in range(self.No_of_camera)]
        self.annotated_frame_ids = [0 for _ in range(self.No_of_camera)]

        # FPS tracking
        self.fps_timer = time.time()
        self.fps_counter = 0
        self.current_fps = 25.0

        # Spawn background threads
        self.weapon_thread = threading.Thread(target=self.run_weapon_inference)
        self.weapon_thread.start()

        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run_weapon_inference(self):
        print("Weapon Inference Thread Up")
        while self.running:
            try:
                frames = self.weapon_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            try:
                # Run weapon detector (GPU inference, fast)
                weapon_boxes = self.weapon_detector.predict(frames)
                self.latest_weapon_boxes = weapon_boxes
            except Exception as e:
                print(f"Error in weapon inference: {e}")
            finally:
                self.weapon_queue.task_done()

    def run(self):
        print("Inference Worker Up")
        while self.running:
            # Avoid busy waiting and duplicate processing
            current_frame_ids = [worker.frame_id for worker in self.camera_workers]
            if current_frame_ids == self.last_frame_ids:
                time.sleep(0.002)
                continue
            self.last_frame_ids = current_frame_ids

            frames = []
            for worker in self.camera_workers:
                frame = worker.get_latest_frame()
                if frame is None:
                    # Create a standard 640x480 black placeholder frame
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "CONNECTING...", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    frames.append(placeholder)
                else:
                    frames.append(frame)
                
            start = time.time()

            # Behavior inference runs in its own dedicated thread with direct camera polling.
            # Weapon detection still uses a bounded queue to stay real-time.

            # 2. Push to weapon queue selectively (size limit of 2) to ensure bounding boxes stay real-time and responsive
            if self.weapon_queue.qsize() < 2:
                self.weapon_queue.put(frames)

            # Fetch latest weapon results
            weapon_boxes = self.latest_weapon_boxes
            
            # Calculate real-time FPS
            self.fps_counter += 1
            now = time.time()
            if now - self.fps_timer >= 1.0:
                self.current_fps = self.fps_counter / (now - self.fps_timer)
                self.fps_counter = 0
                self.fps_timer = now

            # Render Overlays
            for i in range(self.No_of_camera):
                canvas = frames[i].copy() # Copy raw frame to draw overlays in main loop safely
                orig_w = canvas.shape[1]
                
                # Draw latest weapon boxes
                for box in weapon_boxes[i]:
                    coords = box["coords"]
                    cv2.rectangle(canvas, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
                    cv2.putText(canvas, f"{box['label'].upper()} {box['conf']:.2f}", (coords[0], coords[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                # Overlay: weapon alert or all-clear
                if len(weapon_boxes[i]) > 0:
                    weapon_names = ", ".join([box["label"].upper() for box in weapon_boxes[i]])
                    box_color = (0, 0, 255)
                    overlay_text = f"{weapon_names}"
                else:
                    box_color = (0, 255, 0)
                    overlay_text = "STATUS: NORMAL"

                cv2.rectangle(canvas, (0, 0), (orig_w, 45), (15, 15, 15), -1)
                cv2.putText(canvas, overlay_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)
                
                # Draw dynamic FPS overlay inside the frame (top-right corner)
                fps_text = f"{self.current_fps:.1f} FPS"
                
                # Draw green FPS text (BGR: 0, 255, 0)
                cv2.putText(canvas, fps_text, (orig_w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

                # PRE-PROCESS FOR GUI HERE (Takes the load off the main UI thread)
                rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                self.display_frames[i] = rgb_canvas
                
                # Capture annotated canvas to thread-safe rolling buffers for video clips
                self.latest_annotated_frames[i] = canvas.copy()
                self.annotated_frame_ids[i] += 1
                self.annotated_buffers[i].append(canvas.copy())
                
            elapsed = time.time() - start
            print("Inference Batch Latency:", elapsed)
            
            # Cap frame rate at ~25 FPS to ensure smooth video playback
            time.sleep(max(0.001, 0.040 - elapsed))

    def get_processed_frames(self):
        return self.display_frames
    
    def stop(self):
        self.running = False
        if hasattr(self, 'weapon_thread'):
            self.weapon_thread.join()

class CameraLabel(QLabel):
    #custom clickable label

    clicked = pyqtSignal(int) #self-defined signal and pass cam_id as an argument

    def __init__(self,cam_id):
        super().__init__()
        self.cam_id = cam_id
        
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.cam_id)
    
class SecurityDashboard(QWidget):
    def __init__(self, config_file="cctv.txt"):
        super().__init__()
        self.resize(1300, 850)
        self.setWindowTitle("SiteSecure VISION - Advanced Surveillance Command Center")

        self.config_file = config_file
        
        self.cctvList = [] # First four are CCTV(Video for now) and rest are the attached cameras
        self.load_cctv()
        self.get_available_cameras()
        self.setup_camera_workers()
        self.setup_inference_worker()
        self.selected_camera = -1
        self.previous_selected = 0
        
        # State machine tracking for active alarms to prevent spamming
        self.active_alert_states = {}
        
        self.setup_ui()
        self.apply_styles()

        self.alarm_records = {}
        self.load_alarms_from_json()

        self.gui_timer = QTimer()
        self.gui_timer.timeout.connect(self.update_gui)
        self.gui_timer.start(33)
        print(self.cctvList)
        print(self.camera_workers)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f8fafc;
                font-family: 'Segoe UI', Roboto, sans-serif;
                color: #1e293b;
            }
            
            /* Sidebar Container Layout Card */
            QWidget#sidebar_container {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
            
            QLabel#sidebar_title {
                font-size: 16px;
                font-weight: bold;
                color: #0f172a;
                padding-bottom: 5px;
            }
            
            /* Sky Blue Refresh Button */
            QPushButton#refresh_btn {
                background-color: #0284c7;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                border: none;
            }
            QPushButton#refresh_btn:hover {
                background-color: #0369a1;
            }
            QPushButton#refresh_btn:pressed {
                background-color: #075985;
            }
            
            /* CCTV list items style */
            QListWidget#cctv_list {
                background-color: transparent;
                border: none;
                outline: none;
            }
            QListWidget#cctv_list::item {
                background-color: #f1f5f9;
                color: #334155;
                font-weight: bold;
                margin-bottom: 8px;
                border-radius: 8px;
            }
            QListWidget#cctv_list::item:hover {
                background-color: #e2e8f0;
            }
            QListWidget#cctv_list::item:selected {
                background-color: #0284c7;
                color: white;
            }
            
            /* Back Button */
            QPushButton#back_btn {
                background-color: #64748b;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px 15px;
                font-size: 13px;
                border: none;
            }
            QPushButton#back_btn:hover {
                background-color: #475569;
            }
            
            /* Legend Container */
            QWidget#legend_container {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
            
            /* Alarms List Widget (Amber/Orange Alarm Cards) */
            QListWidget#alarms_list {
                background-color: transparent;
                border: none;
                outline: none;
            }
            QListWidget#alarms_list::item {
                background-color: #f97316;
                color: white;
                font-weight: bold;
                font-size: 13px;
                border-radius: 6px;
                margin-bottom: 6px;
            }
            QListWidget#alarms_list::item:hover {
                background-color: #ea580c;
            }
        """)

    def setup_camera_workers(self):
        self.camera_workers = []
        i = 0
        for source in self.cctvList:
            self.camera_workers.append(Camera_Worker(i,source))
            i += 1

    def setup_inference_worker(self):
        self.inference_worker = Inference_Worker(self.cctvList, self.camera_workers)

    def refresh_cameras(self):
        self.cctv_list_widget.clear()
        self.setup_cctv_list_widget()

    def log_new_alarm(self, cam_id, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        alarm_id = f"alarm_{int(time.time())}_{cam_id}"
        clip_path = f"alerts/{alarm_id}.mp4"
        
        source_name = self.cctvList[cam_id]
        if isinstance(source_name, int):
            source_name = f"Camera {source_name}"
        else:
            source_name = source_name.split('\\')[-1].split('/')[-1]
            
        alarm = Alarm(cam_id, source_name, msg, clip_path, "Pending", alarm_id)
        self.alarm_records[alarm_id] = alarm
        
        cam_worker = self.camera_workers[cam_id]
        pre_frames = list(self.inference_worker.annotated_buffers[cam_id])
        
        recorder = ClipRecorder(alarm, self.inference_worker, cam_id, clip_path, pre_frames, duration_sec=5, fps=cam_worker.fps)
        recorder.start()
        
        item = QListWidgetItem(f"⏰ [PENDING] [{timestamp}] {msg}")
        item.setData(Qt.ItemDataRole.UserRole, alarm_id)
        item.setForeground(QColor("#f97316"))
        self.alarms.insertItem(0, item)
        
        self.save_alarms_to_json()

    def load_alarms_from_json(self):
        log_path = "alerts/alarm_log.json"
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    data = json.load(f)
                    for alarm_id, alarm_data in data.items():
                        alarm = Alarm.from_dict(alarm_data)
                        self.alarm_records[alarm_id] = alarm
                        
                        status_str = f"[{alarm.status.upper()}]"
                        timestamp_only = alarm.timestamp.split(" ")[-1] if " " in alarm.timestamp else alarm.timestamp
                        item = QListWidgetItem(f"⏰ {status_str} [{timestamp_only}] {alarm.message}")
                        item.setData(Qt.ItemDataRole.UserRole, alarm.id)
                        
                        if alarm.status == "Pending":
                            item.setForeground(QColor("#f97316"))
                        elif alarm.status == "True Positive":
                            item.setForeground(QColor("#ef4444"))
                            font = item.font()
                            font.setBold(True)
                            item.setFont(font)
                        elif alarm.status == "False Positive":
                            item.setForeground(QColor("#94a3b8"))
                            font = item.font()
                            font.setStrikeOut(True)
                            item.setFont(font)
                            
                        self.alarms.addItem(item)
            except Exception as e:
                print(f"Error loading alarms from JSON: {e}")

    def save_alarms_to_json(self):
        os.makedirs("alerts", exist_ok=True)
        log_path = "alerts/alarm_log.json"
        try:
            data = {alarm_id: alarm.to_dict() for alarm_id, alarm in self.alarm_records.items()}
            with open(log_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving alarms to JSON: {e}")

    def update_alarm_status(self, alarm_id, status):
        if alarm_id in self.alarm_records:
            alarm = self.alarm_records[alarm_id]
            alarm.status = status
            
            for i in range(self.alarms.count()):
                item = self.alarms.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == alarm_id:
                    timestamp = alarm.timestamp.split(" ")[-1] if " " in alarm.timestamp else alarm.timestamp
                    status_str = f"[{status.upper()}]"
                    item.setText(f"⏰ {status_str} [{timestamp}] {alarm.message}")
                    
                    if status == "True Positive":
                        item.setForeground(QColor("#ef4444"))
                        font = item.font()
                        font.setBold(True)
                        font.setStrikeOut(False)
                        item.setFont(font)
                    elif status == "False Positive":
                        item.setForeground(QColor("#94a3b8"))
                        font = item.font()
                        font.setBold(False)
                        font.setStrikeOut(True)
                        item.setFont(font)
                    break
                    
            self.save_alarms_to_json()

    def verify_alarm_dialog(self, item):
        alarm_id = item.data(Qt.ItemDataRole.UserRole)
        if alarm_id and alarm_id in self.alarm_records:
            alarm = self.alarm_records[alarm_id]
            dialog = AlarmVerificationDialog(alarm, self)
            dialog.exec()

    def update_gui(self):
        display_frames = self.inference_worker.get_processed_frames()
        row = 0
        col = 0
        for i in range(0, 4): # hardcoded for grid
            if i < len(display_frames) and display_frames[i] is not None:
                rgb = display_frames[i]
                h, w, ch = rgb.shape
                qImg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                
                # Make the grid labels perfectly responsive by scaling them
                pixmap = QPixmap.fromImage(qImg).scaled(
                    self.camera_labels[i].size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
                self.camera_labels[i].setPixmap(pixmap)
            else:
                self.camera_labels[i].setText("Connecting...")
            
            col += 1
            if col > 1:  # Changed to match 2x2 grid properly (col: 0, 1)
                row += 1
                col = 0
        
        # Updating the fullscreen feed
        if self.selected_camera != -1:
            if self.selected_camera < len(display_frames) and display_frames[self.selected_camera] is not None:
                rgb = display_frames[self.selected_camera]
                h, w, ch = rgb.shape
                qImg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                
                # Fullscreen feed occupies maximum layout container size
                pixmap = QPixmap.fromImage(qImg).scaled(
                    self.fullscreen_feed.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
                self.fullscreen_feed.setPixmap(pixmap)
            else:
                self.fullscreen_feed.setText("Connecting...")
            
        # DYNAMIC ALARM LOGGING & BANNER TRANSITION SYSTEM
      
        weapon_boxes = self.inference_worker.latest_weapon_boxes

        active_alarms = []
        current_alerts = { (i, "weapon"): False for i in range(self.inference_worker.No_of_camera) }

        for i in range(self.inference_worker.No_of_camera):
            source_name = self.cctvList[i]
            if isinstance(source_name, int):
                source_name = f"Camera {source_name}"
            else:
                source_name = source_name.split('\\')[-1].split('/')[-1]

            # Weapon Threat
            if len(weapon_boxes[i]) > 0:
                current_alerts[(i, "weapon")] = True
                weapon_names = ", ".join([box["label"].upper() for box in weapon_boxes[i]])
                msg = f"WEAPON DETECTED ({weapon_names}) — CAMERA: {source_name} — CRITICAL THREAT"
                active_alarms.append(msg)
                if not self.active_alert_states.get((i, "weapon"), False):
                    self.log_new_alarm(i, msg)

        # Update alarm state-machine values
        for key, val in current_alerts.items():
            self.active_alert_states[key] = val

        # Update the Alert Banner dynamically
        if not active_alarms:
            self.alert_banner.setText("SYSTEM STATUS: ALL CLEAR — MONITORING SURVEILLANCE CHANNELS")
            self.alert_banner.setStyleSheet("background-color: #22c55e; color: white; font-weight: bold; font-size: 13px; padding: 10px; border-radius: 6px;")
        else:
            self.alert_banner.setText(active_alarms[0])
            self.alert_banner.setStyleSheet("background-color: #ea580c; color: white; font-weight: bold; font-size: 13px; padding: 10px; border-radius: 6px;")

    def setup_ui(self):
        # 1. Master vertical layout
        master_vertical = QVBoxLayout()
        master_vertical.setContentsMargins(15, 15, 15, 15)
        master_vertical.setSpacing(15)
        
        # 2. Header Logo (SiteSecure VISION layout)
        header = QHBoxLayout()
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label = QLabel()
        logo_label.setText("<span style='font-size: 28px; font-weight: bold; color: #1e293b;'>👷 Site</span>"
                           "<span style='font-size: 28px; font-weight: bold; color: #0284c7;'>Secure</span> "
                           "<span style='font-size: 28px; font-weight: bold; color: #22c55e;'>VISION</span> 🎥")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(logo_label)
        master_vertical.addLayout(header)
        
        # 3. Horizontal Split layout (Sidebar vs main view)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # --- LEFT SIDEBAR PANEL ---
        sidebar_container = QWidget()
        sidebar_container.setObjectName("sidebar_container")
        sidebar = QVBoxLayout(sidebar_container)
        sidebar.setContentsMargins(15, 15, 15, 15)
        sidebar.setSpacing(10)
        
        cam_list_title = QLabel("Camera List")
        cam_list_title.setObjectName("sidebar_title")
        sidebar.addWidget(cam_list_title)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setObjectName("refresh_btn")
        self.refresh_btn.clicked.connect(self.refresh_cameras)
        sidebar.addWidget(self.refresh_btn)
        
        self.cctv_list_widget = QListWidget()
        self.cctv_list_widget.setObjectName("cctv_list")
        self.setup_cctv_list_widget()
        self.cctv_list_widget.currentRowChanged.connect(self.list_changed)
        sidebar.addWidget(self.cctv_list_widget)
        
        content_layout.addWidget(sidebar_container, 20) # 20% width
        
        # --- RIGHT SURVEILLANCE MONITOR AND ALARMS PANEL ---
        right_container = QVBoxLayout()
        right_container.setSpacing(12)
        
        self.right_pane = QStackedLayout()
        
        self.camera_grid = QWidget()
        self.setup_camera_grid_ui()
        self.camera_grid.show()
        
        self.fullscreen = QWidget()
        self.setup_fullscreen_ui()
        self.fullscreen.show()
        
        self.right_pane.addWidget(self.camera_grid)
        self.right_pane.addWidget(self.fullscreen)
        
        right_container.addLayout(self.right_pane, 75) # 75% vertical space for live feeds
        
        # Legend (Active Detections Panel)
        legend_layout = QHBoxLayout()
        legend_layout.setContentsMargins(10, 10, 10, 10)
        legend_layout.setSpacing(20)
        legend_label = QLabel("Active Detections:")
        legend_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #475569;")
        legend_layout.addWidget(legend_label)
        
        def create_legend_box(color_hex, text, border_style="none"):
            box = QLabel()
            box.setFixedSize(14, 14)
            box.setStyleSheet(f"background-color: {color_hex}; border: {border_style}; border-radius: 3px;")
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #475569;")
            
            lay = QHBoxLayout()
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)
            lay.addWidget(box)
            lay.addWidget(lbl)
            
            w = QWidget()
            w.setLayout(lay)
            return w
            
        legend_layout.addWidget(create_legend_box("#f97316", "PPE"))
        legend_layout.addWidget(create_legend_box("#ef4444", "Zone"))
        legend_layout.addWidget(create_legend_box("#991b1b", "Collision"))
        legend_layout.addWidget(create_legend_box("#ffffff", "Night Mode", "1px solid #94a3b8"))
        legend_layout.addWidget(create_legend_box("#ffffff", "Compliant", "2px solid #22c55e"))
        legend_layout.addStretch()
        
        legend_container = QWidget()
        legend_container.setObjectName("legend_container")
        legend_container.setLayout(legend_layout)
        right_container.addWidget(legend_container)
        
        # Dynamic Alert Banner
        self.alert_banner = QLabel("🟢 SYSTEM STATUS: ALL CLEAR — MONITORING SURVEILLANCE CHANNELS")
        self.alert_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_banner.setStyleSheet("background-color: #22c55e; color: white; font-weight: bold; font-size: 13px; padding: 10px; border-radius: 6px;")
        right_container.addWidget(self.alert_banner)
        
        # Alarms Incidents Card list
        self.alarms = QListWidget()
        self.alarms.setObjectName("alarms_list")
        self.alarms.itemDoubleClicked.connect(self.verify_alarm_dialog)
        
        alarms_layout = QVBoxLayout()
        alarms_layout.setSpacing(5)
        alarms_title = QLabel("🚨 Live Incident Activity Logs")
        alarms_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #475569;")
        alarms_layout.addWidget(alarms_title)
        alarms_layout.addWidget(self.alarms)
        
        right_container.addLayout(alarms_layout, 25) # 25% height for incident logging
        
        content_layout.addLayout(right_container, 80) # 80% width
        master_vertical.addLayout(content_layout)
        self.setLayout(master_vertical)

    def list_changed(self):
        if self.cctv_list_widget.currentRow() >= 0:
            row = self.cctv_list_widget.currentRow()
            self.selected_camera = row
            self.show_fullscreen(row)

    def load_cctv(self):
        # Graceful fallback logic
        if not os.path.exists(self.config_file):
            if self.config_file == "cctv.txt" and os.path.exists("cctv.csv"):
                print("cctv.txt not found, automatically falling back to cctv.csv")
                self.config_file = "cctv.csv"
            elif self.config_file == "cctv.csv" and os.path.exists("cctv.txt"):
                print("cctv.csv not found, automatically falling back to cctv.txt")
                self.config_file = "cctv.txt"
                
        if os.path.exists(self.config_file):
            try:
                # If it's a CSV file, parse it using the csv module to be highly robust
                if self.config_file.lower().endswith('.csv'):
                    with open(self.config_file, "r", encoding="utf-8-sig") as file:
                        reader = csv.reader(file)
                        for row in reader:
                            if row:
                                val = row[0].strip()
                                if val:
                                    if val.isdigit():
                                        val = int(val)
                                    elif not os.path.isabs(val):
                                        # Resolve relative path relative to the config file's directory
                                        val = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(self.config_file)), val))
                                    self.cctvList.append(val)
                else:
                    with open(self.config_file, "r", encoding="utf-8-sig") as file:
                        for line in file:
                            val = line.rstrip('\n').strip()
                            if val:
                                if val.isdigit():
                                    val = int(val)
                                elif not os.path.isabs(val):
                                    # Resolve relative path relative to the config file's directory
                                    val = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(self.config_file)), val))
                                self.cctvList.append(val)
            except Exception as e:
                print(f"Error reading configuration file {self.config_file}: {e}")
        else:
            print(f"Warning: Configuration file {self.config_file} not found. Operating with blank feed list.")

    
    def setup_cctv_list_widget(self):
        for cctv in self.cctvList:
            if isinstance(cctv,int):
                cctv = f"Camera {cctv}"
            else:
                cctv = cctv.split('\\')[-1].split('/')[-1]
            self.cctv_list_widget.addItem(cctv)

    def setup_camera_grid_ui(self):
        cam_grid = QVBoxLayout()
        cam_grid.setContentsMargins(0, 0, 0, 0)
        self.camera_grid_title = QLabel("Live Surveillance Feed Grid")
        self.camera_grid_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #334155; padding-bottom: 5px;")
        
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.setup_grid()
        
        cam_grid.addWidget(self.camera_grid_title)
        cam_grid.addLayout(self.grid)
        self.camera_grid.setLayout(cam_grid)

    def setup_fullscreen_ui(self):
        fullscreen = QVBoxLayout()
        fullscreen.setContentsMargins(0, 0, 0, 0)
        
        fullscreen_row_1 = QHBoxLayout()
        self.back_button = QPushButton("⬅ Back to Grid")
        self.back_button.setObjectName("back_btn")
        self.back_button.clicked.connect(self.show_grid) # back to grid view
        
        self.fullscreen_title = QLabel("Active Camera Feed")
        self.fullscreen_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #334155;")
        
        fullscreen_row_1.addWidget(self.back_button)
        fullscreen_row_1.addWidget(self.fullscreen_title)
        fullscreen_row_1.addStretch()

        self.fullscreen_feed = CameraLabel(-1)
        self.fullscreen_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fullscreen_feed.setMinimumSize(640, 480)
        self.fullscreen_feed.setStyleSheet("background-color: #0f172a; border: 1px solid #e2e8f0; border-radius: 8px;")

        fullscreen.addLayout(fullscreen_row_1)
        fullscreen.addWidget(self.fullscreen_feed, 1)

        self.fullscreen.setLayout(fullscreen)

    def handle_camera(self):
        # Kept for compatibility, though list_changed overrides it
        pass

    def get_available_cameras(self):
        for i in range(2):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.cctvList.append(i)
            cap.release()

    def setup_grid(self):
        self.camera_labels = []
        row = 0
        col = 0
        for i in range(0, 4):
            label = CameraLabel(i)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(320, 240)
            label.setStyleSheet("background-color: #0f172a; border: 1px solid #e2e8f0; border-radius: 8px;")
            label.clicked.connect(self.show_fullscreen)
            self.camera_labels.append(label)
            self.grid.addWidget(label, row, col)
            
            col += 1
            if col > 1:
                row += 1
                col = 0

    def show_fullscreen(self, camid):
        if camid == -1:
            print(f"Invalid Camid: {camid}")
            return
        self.selected_camera = camid
        source_name = self.cctvList[camid]
        if isinstance(source_name, int):
            source_name = f"Camera {source_name}"
        else:
            source_name = source_name.split('\\')[-1].split('/')[-1]
            
        self.fullscreen_title.setText(f"Active Camera Feed: {source_name}")
        self.right_pane.setCurrentWidget(self.fullscreen)

    def show_grid(self):
        self.right_pane.setCurrentWidget(self.camera_grid)
        self.selected_camera = -1

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', 
            "Are you sure you want to quit?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.inference_worker.stop()
            self.inference_worker.thread.join()

            for workers in self.camera_workers:
                workers.stop()
            for worker in self.camera_workers:
                worker.thread.join()
            
            event.accept()  # Let the window close
        else:
            event.ignore()  # Keep the window open


if __name__ == "__main__":
    app = QApplication([])
    config_file = "cctv.csv"
    if len(sys.argv) > 1:  
        #Config file can be passed as command line argument
        config_file = sys.argv[1]
    main = SecurityDashboard(config_file)
    main.show()
    app.exec()