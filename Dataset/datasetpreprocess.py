from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d takihasan/div2k-dataset-for-super-resolution
!unzip div2k-dataset-for-super-resolution.zip -d /content/DIV2K

!mkdir -p "/content/drive/My Drive/INTEL PROJECT/archive/DATASET/DIV2K"
!cp -r /content/DIV2K/* "/content/drive/My Drive/INTEL PROJECT/archive/DATASET/DIV2K/"

import os
import cv2
method = "bicubic"
scale_factor = 4
BASE_DIR = "/content/drive/MyDrive/INTEL PROJECT/archive/DATASET"
DIV2K_PATH = os.path.join(BASE_DIR, "DIV2K", "Dataset")
DATASET_ROOT = os.path.join(BASE_DIR, "dataset_root")
HR_DIR = os.path.join(DATASET_ROOT, "HR")
LR_DIR = os.path.join(DATASET_ROOT, "LR")
os.makedirs(HR_DIR, exist_ok=True)
os.makedirs(LR_DIR, exist_ok=True)
hr_image_paths = []
hr_folders = ["DIV2K_train_HR", "DIV2K_valid_HR"]

for folder in hr_folders:
    full_path = os.path.join(DIV2K_PATH, folder)
    if os.path.exists(full_path):
        files = sorted([f for f in os.listdir(full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for f in files:
            hr_image_paths.append(os.path.join(full_path, f))
    else:
        print(f" Folder not found: {full_path}")

print(f" Total HR images found: {len(hr_image_paths)}")
for idx, hr_path in enumerate(hr_image_paths, 1):
    img = cv2.imread(hr_path)
    if img is None:
        print(f"⚠️ Could not read: {hr_path}")
        continue
    filename = f"{idx:04d}.png"
    height, width = img.shape[:2]
    cv2.imwrite(os.path.join(HR_DIR, filename), img)
    if method == "bicubic":
        lr_img = cv2.resize(img, (width // scale_factor, height // scale_factor), interpolation=cv2.INTER_CUBIC)
    elif method == "bilateral":
        filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        lr_img = cv2.resize(filtered, (width // scale_factor, height // scale_factor), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError("❌ Invalid method")
    cv2.imwrite(os.path.join(LR_DIR, filename), lr_img)
    if idx % 100 == 0 or idx == len(hr_image_paths):
        print(f"Processed {idx}/{len(hr_image_paths)}")
print("\nHR and LR images saved with name alignment in dataset_root/")


