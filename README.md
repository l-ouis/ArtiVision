# ArtiVision

## Overview
ArtiVision is a convolutional model trained on the WikiArt dataset. The primary goal is to classify images by their artistic style and visualize model results and behavior.

## Data
The dataset used in this project can be downloaded from Kaggle: [WikiArt Dataset](https://www.kaggle.com/datasets/simolopes/wikiart-all-artpieces?resource=download). It contains 176436 images and a csv mapping each to their respective info.

## Data Instructions
To preprocess the data, run the files in data_preprocessing in the order `group_by_style.py` then `split_style_data_to_train_test.py`.

## Model Instructions
Run `main.py` to train the model with choosen arguments. Edit `hyperparameters.py` based on machine capabilities. Model architecture is defined in `models.py`.

## Visualization Instructions
Visualization and various testing scripts can be found in the results_viz folder. To run top-n prediction accuracy, run `top5_accuracy.py` then either `top5_correct_analysis.py` or `top5_dr_analysis.py`.
