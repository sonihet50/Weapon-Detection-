from ultralytics import YOLO

model = YOLO(r"D:\Weapon Detection\runs\detect\weapon_detection\Final_yolo26n_cctv_aug\weights\best.pt")

print("Exporting 4-Stream ONNX...")
model.export(
    format="onnx",
    simplify=True,
    dynamic=False,
    batch=4,       # CRITICAL: Locks the architecture to process 4 videos simultaneously
    opset=12,
    imgsz=640
)