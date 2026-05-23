from ultralytics import YOLO

def train_model():
    # Load the pretrained YOLO26 Nano model
    model = YOLO("yolo26n.pt") 

    # Train the model
    # Note: We use the MuSGD optimizer which YOLO26 introduced for better stability
    results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,            # Increased from 16 to 32 to fill your 6GB VRAM
    workers=8,           # Uses more CPU threads to keep that 95W GPU fed
    device=0,
    optimizer="MuSGD",
    amp=True,            # Mixed precision is a must for the 30-series
    project="weapon_detection",
    name="yolo26_high_perf",
    exist_ok=True
)
    print("Training complete. Weights saved to:", results.save_dir)

if __name__ == "__main__":
    train_model()