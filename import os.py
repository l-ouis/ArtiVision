import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
source_dir = 'split'
target_dir = 'data'
train_dir = os.path.join(target_dir, 'train')
test_dir = os.path.join(target_dir, 'test')

# Create target directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split ratio
train_ratio = 0.7

# Process each style folder
for style in os.listdir(source_dir):
    style_path = os.path.join(source_dir, style)
    if os.path.isdir(style_path):
        images = os.listdir(style_path)
        print(f'Processing style: {style}')
        print(f'Found images: {images}')
        if not images:
            print(f'No images found in {style_path}')
            continue

        train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)

        # Create style directories in train and test folders
        train_style_dir = os.path.join(train_dir, style)
        test_style_dir = os.path.join(test_dir, style)
        os.makedirs(train_style_dir, exist_ok=True)
        os.makedirs(test_style_dir, exist_ok=True)

        # Move images to train and test directories
        for img in train_images:
            shutil.copy(os.path.join(style_path, img), train_style_dir)
        for img in test_images:
            shutil.copy(os.path.join(style_path, img), test_style_dir)

print('Data has been successfully split into train and test sets.')