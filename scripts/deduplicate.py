import os
import hashlib
import shutil
from pathlib import Path

def get_file_hash(filepath):
    """Calculates the MD5 hash of a file's contents."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def deduplicate_dataset(roboflow_dir, unannotated_dir, quarantine_dir):
    print("Step 1: Indexing Roboflow (Annotated) images...")
    roboflow_hashes = set()
    roboflow_path = Path(roboflow_dir)
    
    # Recursively find all images in the Roboflow dataset (train, val, test)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.tiff'}
    for filepath in roboflow_path.rglob('*'):
        if filepath.is_file() and filepath.suffix.lower() in image_extensions:
            file_hash = get_file_hash(filepath)
            if file_hash:
                roboflow_hashes.add(file_hash)
                
    print(f"Found {len(roboflow_hashes)} unique annotated images.")

    print("\nStep 2: Scanning Unannotated folder for duplicates...")
    os.makedirs(quarantine_dir, exist_ok=True)
    unannotated_path = Path(unannotated_dir)
    
    duplicates_found = 0
    clean_images = 0

    for filepath in unannotated_path.rglob('*'):
        if filepath.is_file() and filepath.suffix.lower() in image_extensions:
            file_hash = get_file_hash(filepath)
            
            if file_hash in roboflow_hashes:
                # It's a duplicate! Move it to quarantine.
                duplicates_found += 1
                destination = os.path.join(quarantine_dir, filepath.name)
                
                # Handle edge case where duplicate names exist in quarantine
                counter = 1
                while os.path.exists(destination):
                    name = filepath.stem
                    ext = filepath.suffix
                    destination = os.path.join(quarantine_dir, f"{name}_{counter}{ext}")
                    counter += 1
                    
                shutil.move(str(filepath), destination)
                print(f"Moved duplicate: {filepath.name}")
            else:
                # Add to hashes so we also deduplicate within the unannotated folder itself
                roboflow_hashes.add(file_hash)
                clean_images += 1

    print("\n--- Deduplication Complete ---")
    print(f"Clean unannotated images ready for pseudo-labeling: {clean_images}")
    print(f"Duplicates moved to quarantine: {duplicates_found}")

if __name__ == "__main__":
    # Update these paths to match your local directories
    ROBOFLOW_DATASET_DIR = "./roboflow_dataset"   # Folder A
    UNANNOTATED_DIR = "./unannotated_raw"         # Folder B
    QUARANTINE_DIR = "./quarantined_duplicates"   # Safe storage for duplicates

    deduplicate_dataset(ROBOFLOW_DATASET_DIR, UNANNOTATED_DIR, QUARANTINE_DIR)