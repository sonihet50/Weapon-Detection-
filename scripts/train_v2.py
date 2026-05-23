from ultralytics import YOLO

def train_production_model():
    model = YOLO("yolo26n.pt") 
    
    yaml_path = r"D:\Weapon Detection\Final_dataset\data.yaml"

    print("Initiating Enterprise Training Run with CCTV Augmentations...")

    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=100,           
        patience=15,          
        imgsz=640,            
        batch=16,              
        device=0,             
        workers=4,            
        project="weapon_detection",
        name="Final_yolo26n_cctv_aug", 
        amp=True,
        # --- CCTV Augmentations ---
        hsv_h=0.015,      # Hue shifts
        hsv_s=0.7,        # Saturation shifts (simulates washed-out lenses)
        hsv_v=0.4,        # Value shifts (forces learning in dark shadows and bright glare)
        translate=0.1,    # Translates image (forces learning of partially off-screen weapons)
        scale=0.5,        # Scales image (zooms out to make weapons smaller)
        mosaic=1.0,       # 100% chance to stitch 4 images together
        mixup=0.15        # 15% chance to overlay images (simulates occlusion)
    )
    
    print("\n--- Training Complete! ---")

if __name__ == "__main__":
    train_production_model()