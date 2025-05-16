import os
import argparse
from shutil import copyfile

def main(val_dir, annotations_file):
    # try the usual subfolder first
    images_dir = os.path.join(val_dir, "images")
    # if it doesn't exist, fall back to val_dir itself
    if not os.path.isdir(images_dir):
        print(f"  ↳ no '{images_dir}', falling back to '{val_dir}' as image root")
        images_dir = val_dir

    # read the annotation file
    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            filename, class_name = parts[0], parts[1]
            # make class subdir if needed
            class_subdir = os.path.join(val_dir, class_name)
            os.makedirs(class_subdir, exist_ok=True)
            # move (copy) image into its class folder
            src = os.path.join(images_dir, filename)
            dst = os.path.join(class_subdir, filename)
            if not os.path.isfile(src):
                raise FileNotFoundError(f"  ↳ couldn't find {src}")
            copyfile(src, dst)

    print("Done! Your validation directory now has per-class subfolders.")
    print("  (You can safely delete the original images folder once you’ve verified everything.)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reorganize Tiny-ImageNet val into class folders"
    )
    parser.add_argument(
        "--val-dir", required=True,
        help="path to tiny-imagenet-200/val"
    )
    parser.add_argument(
        "--annotations", required=True,
        help="path to val_annotations.txt"
    )
    args = parser.parse_args()
    main(args.val_dir, args.annotations)
