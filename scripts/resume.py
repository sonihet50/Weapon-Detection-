from ultralytics import YOLO

def resume_training():
    # Path to the 'last.pt' from your crashed run
    # It should be in weapon_detection/yolo26_high_perf/weights/last.pt
    # Use a relative path instead of the full D:/... drive path
    model = YOLO("runs/detect/weapon_detection/yolo26_high_perf/weights/last.pt")

    # Resume with a safer batch size for the final 10 epochs
    results = model.train(
        resume=True,      # This tells YOLO to pick up exactly where it left off
        batch=24,         # Lowered from 32 to avoid that VRAM spike
        workers=4         # Lowered slightly to save overhead
    )

if __name__ == "__main__":
    resume_training()