import cv2
import matplotlib.pyplot as plt
import albumentations as A

# 1. Define the exact same CCTV pipeline
cctv_transforms = A.Compose([
    A.MotionBlur(blur_limit=9, p=1.0),          # Forced to 100% for testing
    A.ImageCompression(quality_lower=20, quality_upper=60, p=1.0), 
    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0), 
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)
])

# 2. Load a test image from your dataset
image_path = r"D:\Weapon Detection\Final_dataset\train\images\den-of-thieves2018-final-gun-fight-with-police-movie-cube05971_jpg.rf.5e44a14a24bc128272aecae2ee089b70.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Apply the CCTV degradations
augmented = cctv_transforms(image=image)
degraded_image = augmented['image']

# 4. Visualize the results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Original High-Res Image")
ax[0].axis('off')

ax[1].imshow(degraded_image)
ax[1].set_title("Degraded CCTV View")
ax[1].axis('off')

plt.tight_layout()
plt.show()