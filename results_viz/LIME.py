import tensorflow as tf
import os
from PIL import Image
import numpy as np
import pickle

import lime
from lime import lime_image

from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries

import sys
sys.path.append('code')
import hyperparameters as hp
from models import Basic, Advanced

with open('code/styles.pkl', 'rb') as f:
    styles = pickle.load(f)

    
# Load the Keras model - need to load in the loss fn as well


model = tf.keras.models.load_model('saved_models/my_model', custom_objects={'loss_fn': Advanced.loss_fn})


test_image_path = 'test_images/182552-magical-space-forms-1948.jpg'

if os.path.exists(test_image_path):
    input_image = Image.open(test_image_path)
    
    # resize and np the image
    input_image = input_image.resize((hp.img_size, hp.img_size))
    input_image = np.array(input_image)
    
    # normalize
    input_image = input_image / 255.0
else:
    print("img not found")


explainer = lime_image.LimeImageExplainer()

def model_predict(input_image):
    input_image = input_image.astype(np.float32) / 255.0 
    predictions = model.predict(input_image)
    return predictions

explanation = explainer.explain_instance(input_image, 
                                         model_predict,
                                         num_features=10000,
                                         num_samples=100)

img, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=True)
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Display the original input image
axs[0].imshow(input_image)
axs[0].set_title('Original Image')
axs[0].axis('off')

# Display the image returned by LIME
axs[1].imshow(img)
axs[1].set_title('LIME Image')
axs[1].axis('off')

# Display the mask returned by LIME
axs[2].imshow(mask, cmap='gray')
axs[2].set_title('LIME Mask')
axs[2].axis('off')

# Display the original image with the mask applied
axs[3].imshow(mark_boundaries(img / 255.0, mask))
axs[3].set_title('Image with LIME Mask')
axs[3].axis('off')

plt.show()

# make predictions
def predict(model, input_image):
    predictions = model.predict(input_image[None, ...])  # Add batch dimension

    top_5_predictions = tf.nn.top_k(predictions, k=5)

    top_5_values, top_5_indices = top_5_predictions.values.numpy()[0], top_5_predictions.indices.numpy()[0]

    return top_5_values, top_5_indices

predictions = predict(model, input_image)
print(predictions)

