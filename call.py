import tensorflow as tf
import os
from PIL import Image
import numpy as np

import sys
sys.path.append('code')
import hyperparameters as hp

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


# Load the Keras model
model = tf.keras.models.load_model('saved_models/model')


# Path to the test image
test_image_path = 'test_images/image'

# Check if the image exists
if os.path.exists(test_image_path):
    # Open the image
    input_image = Image.open(test_image_path)
    
    # resize and np the image
    input_image = input_image.resize((hp.img_size, hp.img_size))
    input_image = np.array(input_image)
    
    # normalize
    input_image = input_image / 255.0
else:
    print("img not found")


# Assuming the input image is already preprocessed and ready to be inputted to the model
# For example, let's assume 'input_image' is the preprocessed image tensor
# input_image = preprocess_your_image_function(your_image)

# Make predictions
predictions = model.predict(input_image[None, ...])  # Add batch dimension

# Get the top 5 class predictions
top_5_predictions = tf.nn.top_k(predictions, k=5)

for i in range(5):
    print(styles[top_5_predictions[1][0][i]])
