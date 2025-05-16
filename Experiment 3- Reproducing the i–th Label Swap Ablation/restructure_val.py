import os
import shutil

val_dir = "./tiny-imagenet-200/val"
img_dir = os.path.join(val_dir, "images")
ann_path = os.path.join(val_dir, "val_annotations.txt")


with open(ann_path, 'r') as f:
    lines = f.readlines()
    mapping = {line.split('\t')[0]: line.split('\t')[1] for line in lines}

for img_name, class_id in mapping.items():
    class_dir = os.path.join(img_dir, class_id)
    os.makedirs(class_dir, exist_ok=True)
    src = os.path.join(img_dir, img_name)
    dst = os.path.join(class_dir, img_name)

    if os.path.exists(src):
        shutil.move(src, dst)

for f in os.listdir(img_dir):
    p = os.path.join(img_dir, f)
    if os.path.isfile(p):
        os.remove(p)

print("Reformatted val/ folder for PyTorch ImageFolder compatibility.")
