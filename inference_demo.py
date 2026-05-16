import cv2
from ultralytics import YOLO
import torch
import time

def run_perfect_display_demo():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Targeting Device: {device}")
    
    # Load Model
    model_path = r"D:\Weapon Detection\runs\detect\weapon_detection\yolo26_high_perf\weights\best.pt"
    model = YOLO(model_path).to(device)

    # Video Source
    #source = r"D:\Weapon Detection\thieves.mp4"
    #source = r"D:\Weapon Detection\nigga.mp4"
    #source = r"D:\Weapon Detection\weapon_in_school.mp4"
    source = r"D:\Weapon Detection\clear_pistol.mp4" 
    cap = cv2.VideoCapture(0)
    
    print("Demo Live! Press 'q' to stop.")
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue

        # Inference
        results = model.predict(frame, conf=0.60, iou=0.45, device=device, verbose=False, imgsz=640)
        annotated_frame = results[0].plot()

        # --- THE SCREEN SIZE FIX ---
        # Resize the frame so it actually fits on your laptop screen
        # We set the width to 1280 and scale the height proportionally
        target_width = 1280
        aspect_ratio = target_width / annotated_frame.shape[1]
        target_height = int(annotated_frame.shape[0] * aspect_ratio)
        
        display_frame = cv2.resize(annotated_frame, (target_width, target_height))
        # ---------------------------

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Add text to the display_frame (not the massive original frame)
        cv2.putText(display_frame, f"FPS: {int(fps)} | Conf: 0.25", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Weapon Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_perfect_display_demo()