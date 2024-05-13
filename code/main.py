import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
from tqdm.keras import TqdmCallback

import hyperparameters as hp
from models import Basic, Advanced
from preprocess import Datasets


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
)
    parser.add_argument(
        'model_choice',
        choices=['1', '2'],
        help='1=basic model, 2=advanced model'
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Path to a directory containing images or a single image file'
    )
    
    parser.add_argument(
        '--style',
        choices=styles,
        help='Choose from the list of styles to classify your image to'
    )
    
    parser.add_argument(
        '--style2',
        choices=styles,
        help='Choose a second style to classify your image to'
    )
    
    return parser.parse_args()
    


def main():
    args = parse_args()
    
    # Load and preprocess data
    datasets = Datasets(args.input, args.model_choice)
    
    # Select model
    if args.model_choice == '1':
        model = Basic()
    else:
        model = Advanced()
    
    # Compile model
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=['accuracy']
    )

    print(model.summary())
    
    # Train model
    model.fit(
        datasets.train_data,
        epochs=hp.num_epochs,
        validation_data=datasets.test_data,
        callbacks=[TqdmCallback(verbose=1)]
    )
    
    # Evaluate model
    results = model.evaluate(datasets.test_data)
    print(f'Test Loss: {results[0]}, Test Accuracy: {results[1]}')

    # Save the trained model
    model.save('saved_model/my_model', save_format="tf")
    


if __name__ == '__main__':
    main()

