import albumentations as A
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
from ultralytics.utils import colorstr

# 1. Define our custom CCTV pipeline override (UPDATED SIGNATURE)
def custom_cctv_init(self, p=1.0, *args, **kwargs):
    """Overrides the default Ultralytics Albumentations pipeline."""
    self.p = p
    self.transform = None
    prefix = colorstr("albumentations: ")
    try:
        T = [
            A.MotionBlur(blur_limit=9, p=0.4),          
            A.ImageCompression(quality_lower=20, quality_upper=60, p=0.5), 
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), 
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4)
        ]
        self.transform = A.Compose(
            T, 
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
        )
        print(f"{prefix}Successfully injected Custom CCTV Pipeline!")
    except ImportError:
        print(f"{prefix}WARNING ⚠️ albumentations not found. Run 'pip install albumentations'")

# 2. Apply the override
Albumentations.__init__ = custom_cctv_init

def train_production_model():
    model = YOLO("yolo26n.pt") 
    
    yaml_path = r"D:\Weapon Detection\Final_dataset\data.yaml"                                           

    print("Initiating Enterprise Training Run with Custom CCTV Augmentations...")

    # Train the model (YOLO will now automatically use our injected pipeline)
    results = model.train(
        data=yaml_path,
        epochs=100,           
        patience=15,          
        imgsz=640,            
        batch=16,              
        device=0,             
        workers=4,            
        project="weapon_detection",
        name="yolo26n_albumentations", 
        amp=True,
        
        # --- Standard Spatial Augmentations ---
        hsv_h=0.015,      
        hsv_s=0.7,        
        hsv_v=0.4,        
        translate=0.1,    
        scale=0.5,        
        mosaic=1.0,       
        mixup=0.15        
    )
    
    print("\n--- Training Complete! ---")

if __name__ == "__main__":
    train_production_model()