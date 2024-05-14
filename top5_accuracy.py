import tensorflow as tf
import os
from PIL import Image
import numpy as np
import shutil
import pandas as pd
import pickle

import sys
sys.path.append('code')
import hyperparameters as hp
from models import Basic, Advanced

# just minimize this definition
styles = [
    'Abstract Art', 'Abstract Expressionism', 'Academicism', 'Action painting', 
    'American Realism', 'Analytical Cubism', 'Analytical Realism', 'Art Brut', 
    'Art Deco', 'Art Informel', 'Art Nouveau (Modern)', 'Art Singulier', 'Automatic Painting', 
    'Baroque', 'Biedermeier', 'Byzantine', 'Cartographic Art', 
    'Chernihiv school of icon painting', 'Classical Realism', 'Classicism', 'Cloisonnism', 
    'Color Field Painting', 'Conceptual Art', 'Concretism', 'Confessional Art', 
    'Constructivism', 'Contemporary', 'Contemporary Realism', 'Coptic art', 
    'Costumbrismo', 'Cretan school of icon painting', 'Crusader workshop', 'Cubism', 
    'Cubo-Expressionism', 'Cubo-Futurism', 'Cyber Art', 'Dada', 'Digital Art', 
    'Divisionism', 'Documentary photography', 'Early Byzantine (c. 330–750)', 
    'Early Christian', 'Early Renaissance', 'Environmental (Land) Art', 'Ero guro', 
    'Excessivism', 'Existential Art', 'Expressionism', 'Fantastic Realism', 'Fantasy Art', 
    'Fauvism', 'Feminist Art', 'Fiber art', 'Figurative Expressionism', 'Futurism', 
    'Galicia-Volyn school', 'Geometric', 'Gongbi', 'Gothic', 'Graffiti Art', 
    'Hard Edge Painting', 'High Renaissance', 'Hyper-Mannerism (Anachronism)', 
    'Hyper-Realism', 'Impressionism', 'Indian Space painting', 'Ink and wash painting', 
    'International Gothic', 'Intimism', 'Japonism', 'Joseon Dynasty', 'Junk Art', 
    'Kinetic Art', 'Kitsch', 'Komnenian style (1081-1185)', 'Kyiv school of icon painting', 
    'Late Byzantine/Palaeologan Renaissance (c. 1261–1453)', 
    'Latin Empire of Constantinople (1204-1261)', 'Lettrism', 'Light and Space', 
    'Lowbrow Art', 'Luminism', 'Lyrical Abstraction', 'Macedonian Renaissance (867–1056)', 
    'Macedonian school of icon painting', 'Magic Realism', 'Mail Art', 
    'Mannerism (Late Renaissance)', 'Maximalism', 'Mechanistic Cubism', 'Medieval Art', 
    'Metaphysical art', 'Middle Byzantine (c. 850–1204)', 'Minimalism', 'Miserablism', 
    'Modernism', 'Modernismo', 'Mosan art', 'Moscow school of icon painting', 'Mozarabic', 
    'Muralism', 'Native Art', 'Naturalism', 'Naïve Art (Primitivism)', 'Neo-Byzantine', 
    'Neo-Concretism', 'Neo-Dada', 'Neo-Expressionism', 'Neo-Figurative Art', 'Neo-Geo', 
    'Neo-Impressionism', 'Neo-Minimalism', 'Neo-Orthodoxism', 'Neo-Pop Art', 'Neo-Rococo', 
    'Neo-Romanticism', 'Neo-Suprematism', 'Neo-baroque', 'Neoclassicism', 'Neoplasticism', 
    'New Casualism', 'New European Painting', 'New Ink Painting', 'New Medievialism', 
    'New Realism', 'New media art', 'Northern Renaissance', 'Nouveau Réalisme', 
    'Novgorod school of icon painting', 'Op Art', 'Orientalism', 'Orphism', 'Outsider art', 
    'P&D (Pattern and Decoration)', 'Performance Art', 'Photorealism', 'Pictorialism', 
    'Pointillism', 'Pop Art', 'Post-Impressionism', 'Post-Minimalism', 
    'Post-Painterly Abstraction', 'Postcolonial art', 'Poster Art Realism', 'Precisionism', 
    'Proto Renaissance', 'Pskov school of icon painting', 'Purism', 'Queer art', 
    'Rayonism', 'Realism', 'Regionalism', 'Renaissance', 'Rococo', 'Romanesque', 
    'Romanticism', 'Safavid Period', 'Severe Style', 'Shin-hanga', 'Site-specific art', 
    'Sky Art', 'Social Realism', 'Socialist Realism', 'Sots Art', 'Spatialism', 
    'Spectralism', 'Street Photography', 'Street art', 'Stroganov school of icon painting', 
    'Stuckism', 'Sumi-e (Suiboku-ga)', 'Superflat', 'Suprematism', 'Surrealism', 
    'Symbolism', 'Synchromism', 'Synthetic Cubism', 'Synthetism', 'Tachisme', 'Tenebrism', 
    'Tonalism', 'Transautomatism', 'Transavantgarde', 'Tubism', 'Ukiyo-e', 'Unknown', 
    'Verism', 'Viking art', 'Vladimir school of icon painting', 'Vologda school of icon painting', 
    'Yaroslavl school of icon painting', 'Yoruba', 'Zen'
]


# Load the Keras model - need to load in the loss fn as well
model = tf.keras.models.load_model('saved_models/my_model', custom_objects={'loss_fn': Basic.loss_fn})

# make predictions
def predict(model, input_image):
    predictions = model.predict(input_image[None, ...])  # Add batch dimension

    top_5_predictions = tf.nn.top_k(predictions, k=10)

    top_5_values, top_5_indices = top_5_predictions.values.numpy()[0], top_5_predictions.indices.numpy()[0]

    return top_5_values, top_5_indices


csv_file = 'og data/wikiart_art_pieces.csv'
images_dir = 'og data/wikiart/wikiart'
output_dir = 'split'


df = pd.read_csv(csv_file)
rand_count = 0
filenames = []
stylelist = []
for index, row in df.iterrows():
    rand_count += 1 
    if rand_count == 150:
        rand_count = 0
        filenames.append(row['file_name'])
        stylelist.append(row['style'])
print("here!")
print(filenames, stylelist)



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
