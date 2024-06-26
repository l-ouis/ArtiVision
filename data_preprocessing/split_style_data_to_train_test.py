import os
import shutil
import random


source_folder = 'split'
destination_folder = 'data'

split_ratio = 0.7


if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)


train_folder = os.path.join(destination_folder, 'train')
test_folder = os.path.join(destination_folder, 'test')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

for style_folder in os.listdir(source_folder):
    style_folder_path = os.path.join(source_folder, style_folder)
    
    if os.path.isdir(style_folder_path):

        train_style_folder = os.path.join(train_folder, style_folder)
        test_style_folder = os.path.join(test_folder, style_folder)
        
        os.makedirs(train_style_folder, exist_ok=True)
        os.makedirs(test_style_folder, exist_ok=True)
        
      
        images = [f for f in os.listdir(style_folder_path) if os.path.isfile(os.path.join(style_folder_path, f))]
        
       
        random.shuffle(images)
        
 
        split_index = int(len(images) * split_ratio)
        

        train_images = images[:split_index]
        test_images = images[split_index:]
        
  
        for image in train_images:
            shutil.move(os.path.join(style_folder_path, image), os.path.join(train_style_folder, image))
        
    
        for image in test_images:
            shutil.move(os.path.join(style_folder_path, image), os.path.join(test_style_folder, image))

print("Data split into train and test sets successfully.")