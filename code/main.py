import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
from tqdm.keras import TqdmCallback
import pickle

import hyperparameters as hp
from models import Basic, Advanced
from preprocess import Datasets


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


with open('code/styles.pkl', 'rb') as f:
    styles = pickle.load(f)


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

    model.build((None, 224, 224, 3))

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
    model.save('saved_model/basic.h5')
    


if __name__ == '__main__':
    main()

