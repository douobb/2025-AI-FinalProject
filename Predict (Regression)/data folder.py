import os
import shutil
import random
from pathlib import Path

input_dir = Path('data/processed_images')
output_dir = Path('data_split/processed_images')
train_dir = output_dir / 'train'
test_dir = output_dir / 'test'

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

image_paths = list(input_dir.rglob('*.png'))

random.shuffle(image_paths)

split_ratio = 0.9
split_index = int(len(image_paths) * split_ratio)
train_images = image_paths[:split_index]
test_images = image_paths[split_index:]


for img_path in train_images:
    dst = train_dir / img_path.name
    shutil.copy(img_path, dst)

for img_path in test_images:
    dst = test_dir / img_path.name
    shutil.copy(img_path, dst)

print("With epicenter:")
print(f"total images: {len(image_paths)}")
print(f"train images: {len(train_images)}")
print(f"test images: {len(test_images)}")

# no_epicenter

input_dir = Path('data/processed_images_no_epicenter')
output_dir = Path('data_split/processed_images_no_epicenter')
train_dir = output_dir / 'train'
test_dir = output_dir / 'test'

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

image_paths = list(input_dir.rglob('*.png'))

random.shuffle(image_paths)

split_ratio = 0.9
split_index = int(len(image_paths) * split_ratio)
train_images = image_paths[:split_index]
test_images = image_paths[split_index:]


for img_path in train_images:
    dst = train_dir / img_path.name
    shutil.copy(img_path, dst)

for img_path in test_images:
    dst = test_dir / img_path.name
    shutil.copy(img_path, dst)

print("Without epicenter:")
print(f"total images: {len(image_paths)}")
print(f"train images: {len(train_images)}")
print(f"test images: {len(test_images)}")


