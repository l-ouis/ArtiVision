import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import io

from tqdm import tqdm
import IPython.display

import skimage
import skimage.io
import numpy as np

mpl.rcParams['figure.figsize'] = [8, 8]


model = tf.keras.models.load_model('saved_model/model')
print(model.summary())

def visualize_cnn_layer(img, layer_name, nrows, ncols, figsize, view_img=True):
    img = img.resize((224,224))
    img = np.array(img)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    curr_layer = model.get_layer(layer_name).output
    slice_model  = tf.keras.Model(inputs=model.input, outputs=curr_layer)
    slice_output = slice_model.predict(img[None,:,:,:])

    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            curr_ax = axes[row, col]
            out = slice_output[0,:,:,idx].astype(np.uint8)
            out = Image.fromarray(out)
            out = out.resize(img.shape[:-1], resample=Image.BOX)
            curr_ax.imshow(out)
            if view_img:
                curr_ax.imshow(img, alpha=0.3)

    return fig, axes


img_file = 'image.png'
img2 = Image.open(img_file)
img2 = img2.crop((200, 200, img2.width - 200, img2.height - 200))

## Fill in layer name here, same one as model instantiated
visualize_cnn_layer(img2, 'block1_conv1', 6, 6, (15,15));

