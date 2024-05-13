''' download data at https://www.kaggle.com/datasets/simolopes/wikiart-all-artpieces?resource=download
'''

import os
import shutil
import pandas as pd


csv_file = 'og data/wikiart_art_pieces.csv'
images_dir = 'og data/wikiart/wikiart'
output_dir = 'split'


df = pd.read_csv(csv_file)


os.makedirs(output_dir, exist_ok=True)

# Group images by their style
for index, row in df.iterrows():
    image_name = row['file_name']
    style = row['movement']
    
    style_dir = os.path.join(output_dir, style)
    os.makedirs(style_dir, exist_ok=True)
    
    src_path = os.path.join(images_dir, image_name)
    dest_path = os.path.join(style_dir, image_name)
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
    else:
        print(f"Image {src_path} not found")

print('Images have been successfully grouped by style.')