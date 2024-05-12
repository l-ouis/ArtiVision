import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

import hyperparameters as hp

class Datasets():

    def __init__(self, data_path, model):

        self.data_path = data_path
        self.model_choice= model

        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.num_classes

        # Mean and std for standardization
        self.mean = np.zeros((hp.img_size,hp.img_size,3))
        self.std = np.ones((hp.img_size,hp.img_size,3))
        self.calc_mean_and_std()

        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), model == '2', True, True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), model == '2', False, False)

    def calc_mean_and_std(self):

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))
                    

        random.shuffle(file_list)

        file_list = file_list[:hp.preprocess_sample_size]


        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
    
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            if img.shape[-1] == 4:
                 img = img[..., :3]
            img /= 255.


            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img

       

        self.mean = np.mean(data_sample, axis = 0)
        self.std = np.std(data_sample, axis = 0)

       
    def standardize(self, img):
      
        img = (img-self.mean) /self.std     # replace this code

        return img



    def preprocess_fn(self, img):
        """ Custom preprocess function for ImageDataGenerator. """
        if self.model_choice == '2':
            img = tf.keras.applications.vgg16.preprocess_input(img)
        else:
            img = img / 255.
            img = self.standardize(img)

        if random.random() < 0.5:
            img = img + tf.random.uniform(
                (hp.img_size, hp.img_size, 3),
                minval=-0.1,
                maxval=0.1)

        if random.random() < 0.5:
            img = self.random_cutout(img)

        return img

    def random_cutout(self, img):
        """ Apply random cutout augmentation. """
        cutout_size = hp.img_size // 4
        y = random.randint(0, hp.img_size - cutout_size)
        x = random.randint(0, hp.img_size - cutout_size)
        mask = np.ones((hp.img_size, hp.img_size, 3), dtype=np.float32)
        mask[y:y+cutout_size, x:x+cutout_size, :] = 0
        img = img * mask
        return img

    def get_data(self, path, is_vgg, shuffle, augment):
     
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)
        # VGG must take images of size 224x224
        img_size = hp.img_size

        classes_for_flow = None


        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=shuffle,
            classes=classes_for_flow)

        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen
    
'''
    
data_path = "data"
model_choice = '2'  # Or any other choice
datasets = Datasets(data_path, model_choice)

# Test mean and std calculation
print("Mean:", datasets.mean)
print("Std:", datasets.std)

# Test preprocessing on sample images
sample_images, _ = next(datasets.train_data)
preprocessed_images = datasets.preprocess_fn(sample_images)

# Visualize original and preprocessed images
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, len(sample_images), figsize=(12, 4))
for i in range(len(sample_images)):
    axes[0, i].imshow(sample_images[i])
    axes[0, i].set_title("Original")
    axes[1, i].imshow(preprocessed_images[i])
    axes[1, i].set_title("Preprocessed")
plt.show()


# Test ImageDataGenerator output
data_batch, labels_batch = next(datasets.train_data)
print("Data batch shape:", data_batch.shape)
print("Labels batch shape:", labels_batch.shape)
'''