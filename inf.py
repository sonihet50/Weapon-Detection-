import cv2
import threading
from ultralytics import YOLO
import numpy as np
import torch
import time

class ThreadedCamera:
    def __init__(self, source):
        # Initialize your cv2.VideoCapture here
        self.cap = cv2.VideoCapture(source)
        
        # We need a variable to hold the absolute newest frame
        self.ret, self.frame = self.cap.read()
        
        # A flag to stop the thread gracefully when we quit
        self.stopped = False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if not self.fps or self.fps == 0:
            self.fps = 30

        self.sleep_time = 1.0 / self.fps
        # Start the background thread immediately upon creation
        threading.Thread(target=self.update, args=(), daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            # --- NEW: Force the thread to wait before reading the next frame ---
            # This stops the "fast-forward" effect for local files
            time.sleep(self.sleep_time)

    def read(self):
        # Return the latest frame when the main loop asks for it
        return self.ret, self.frame

    def stop(self):
        # Set the stop flag to True and release the cap
        self.stopped = True
        self.cap.release()

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Targeting Device :",{device})

    model_path = r"D:\Weapon Detection\runs\detect\weapon_detection\yolo26_high_perf\weights\best.pt"

    model = YOLO(model_path).to(device)

    cameras = ["nigga.mp4", "thieves.mp4", "weapon_in_school.mp4","clear_pistol.mp4"]

    cam = []

    for i in range(0,len(cameras)):
        cam.append(ThreadedCamera(cameras[i])) 

    pane_width = 640
    pane_height = 360

    print("Dashboard")
    while True:
        frames = []
        for i in range(0,len(cam)):
            ret, f = cam[i].read()
            if ret and f is not None:
                frames.append(f)
            else:
                frames.append(np.zeros((pane_height, pane_width, 3), dtype=np.uint8))
        results = model.predict(frames,conf=0.60,iou=0.45,device=device,verbose=False,imgsz=640)
        annotated_frames = []
        for i in range(len(results)):
            img = results[i].plot()
            img = cv2.resize(img, (pane_width, pane_height))
            annotated_frames.append(img)

        top_row = np.hstack((annotated_frames[0], annotated_frames[1]))
        bottom_row = np.hstack((annotated_frames[2], annotated_frames[3]))

        dashboard = np.vstack((top_row, bottom_row))

        cv2.imshow("Security Dashboard", dashboard)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    for i in range(len(cam)):
        cam[i].stop()
    cv2.destroyAllWindows()
