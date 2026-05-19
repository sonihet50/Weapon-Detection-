from ultralytics import YOLO

def train_production_model():
    model = YOLO(r"D:\Weapon Detection\runs\detect\weapon_detection\v2_cctv_robust\weights\last.pt") 

    
    yaml_path = r"D:\Weapon Detection\Final_dataset\data.yaml"

    print("Initiating Enterprise Training Run...")

    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=100,           
        patience=15,          
        imgsz=640,            
        batch=32,              
        device=0,             
        workers=4,            
        project="weapon_detection",
        name="v2_cctv_robust",
        amp=True,
        resume=True            
    )
    
    print("\n--- Training Complete! ---")

if __name__ == "__main__":
    train_production_model()