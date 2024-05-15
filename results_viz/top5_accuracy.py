import tensorflow as tf
import os
from PIL import Image
import numpy as np
import pandas as pd
import pickle

import sys
sys.path.append('code')
import hyperparameters as hp
from models import Basic, Advanced

# Measures accuracy for the top n predictions for a subset of images.

# just minimize this definition
with open('code/styles.pkl', 'rb') as f:
    styles = pickle.load(f)


# Load the Keras model - need to load in the loss fn as well
model = tf.keras.models.load_model('saved_models/my_model', custom_objects={'loss_fn': Basic.loss_fn})

# make predictions
def predict(model, input_image):
    if input_image.shape != (hp.img_size, hp.img_size, 3):
        raise ValueError(f'Image shape {input_image.shape} not matching model input shape {(hp.img_size, hp.img_size, 3)}')
    predictions = model.predict(input_image[None, ...])  # Add batch dimension

    top_k_predictions = tf.nn.top_k(predictions, k=10)

    top_k_values, top_k_indices = top_k_predictions.values.numpy()[0], top_k_predictions.indices.numpy()[0]

    return top_k_values, top_k_indices


csv_file = 'og data/wikiart_art_pieces.csv'
images_dir = 'og data/wikiart/wikiart'
output_dir = 'split'


df = pd.read_csv(csv_file)
rand_count = 0
filenames = []
stylelist = []

# Get a small sample
for index, row in df.iterrows():
    rand_count += 1 
    if rand_count == 200:
        rand_count = 0
        filenames.append(row['file_name'])
        stylelist.append(row['style'])

print("Gathered sample")



os.makedirs(output_dir, exist_ok=True)

top5_style_ids_total = [] # Let's store the top 5 styles of each painting so we can do more viz with PCA
top5_style_ids_result = []
total_guesses = 0
total_correct = 0

# The image saving NEEDS to be batched or you run out of memory. change batch size depending on ur RAM
batch_size = 1000
# get an image-style pair of lists
images = []
img_names = []
style_strings = [] # renamned beacuse we have style defined above
count = 0
batch_num = 0
for image_name, style in zip(filenames, stylelist):
    try:
        input_image = Image.open(os.path.join(images_dir, image_name))

        # resize and np the image
        input_image = input_image.resize((hp.img_size, hp.img_size))
        input_image = np.array(input_image)
        
        # normalize
        input_image = input_image / 255.0

        images.append(input_image)
        img_names.append(image_name)
        style_strings.append(style)
        count += 1
    except:
        print(f'Image {image_name} not found')
    if count == batch_size:
        img_batch = 'gen_data/image_batch' + str(batch_num)
        style_batch = 'gen_data/style_batch' + str(batch_num)
        with open(img_batch, 'wb') as f:
            pickle.dump(img_names, f)
        with open(style_batch, 'wb') as f:
            pickle.dump(style_strings, f)

        # Get stats on the images we have
        for image, style_name in zip(images, style_strings):
            try:
                predictions = predict(model, image)
                style_ids = predictions[1]
                top5_style_ids_total.append(style_ids)
                correct = False
                for style_id in style_ids:
                    if style_name == styles[style_id]:
                        correct = True
                        print(f'Correct: {total_correct} / {total_guesses}')
                        break
                total_guesses += 1
                if correct:
                    top5_style_ids_result.append(1)
                    total_correct += 1
                else:
                    top5_style_ids_result.append(0)
            except:
                print(f'Image {image_name} unable to predict')

        # Reset batch
        images = []
        style_strings = []
        count = 0
        batch_num += 1

# Get final batch stats
for image, style_name in zip(images, style_strings):
    try:
        predictions = predict(model, image)
        style_ids = predictions[1]
        top5_style_ids_total.append(style_ids)
        correct = False
        for style_id in style_ids:
            if style_name == styles[style_id]:
                correct = True
                print(f'Correct: {total_correct} / {total_guesses}')
                break
        total_guesses += 1
        if correct:
            top5_style_ids_result.append(1)
            total_correct += 1
        else:
            top5_style_ids_result.append(0)
    except:
        print(f'Image {image_name} unable to predict')
    

print(f'Correct: {total_correct} / {total_guesses}')

with open('gen_data/top5_style_ids_total.pkl', 'wb') as f:
    pickle.dump(top5_style_ids_total, f)
with open('gen_data/top5_style_ids_result.pkl', 'wb') as f:
    pickle.dump(top5_style_ids_result, f)
