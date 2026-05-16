import os
from pathlib import Path
from ultralytics import YOLO
import torch
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

def run_pseudo_labeling():

    model_path=r"D:\Weapon Detection\runs\detect\weapon_detection\yolo26_high_perf\weights\best.pt"

    model=YOLO(model_path)

    image_dir=Path(r"D:\Weapon Detection\unannotated_raw")

    output_labels=Path(r"D:\Weapon Detection\pseudo_labels")

    output_labels.mkdir(exist_ok=True)

    image_extensions={'.jpg','.jpeg','.png','.bmp','.webp',
                      '.JPG','.JPEG','.PNG'}

    image_paths=[]

    for f in image_dir.rglob("*"):
        if f.suffix in image_extensions:
            image_paths.append(f)

    print(f"Found {len(image_paths)} images")

    chunk_size=20
    count=0

    for i in range(0,len(image_paths),chunk_size):

        chunk=image_paths[i:i+chunk_size]

        results=model.predict(
            source=[str(x) for x in chunk],
            conf=0.65,
            imgsz=416,
            batch=1,
            stream=True,
            device=0,
            verbose=False
        )

        for img_path,result in zip(chunk,results):

            label_file=output_labels/(img_path.stem+".txt")

            with open(label_file,"w") as f:

                boxes=result.boxes

                if boxes is not None:

                    for box in boxes:

                        cls=int(box.cls[0])

                        x,y,w,h=box.xywhn[0].tolist()

                        f.write(
                            f"{cls} {x} {y} {w} {h}\n"
                        )

            count+=1

        print(f"Processed {count}/{len(image_paths)}")

        torch.cuda.empty_cache()
        gc.collect()

    print("Done")


if __name__=="__main__":
    run_pseudo_labeling()