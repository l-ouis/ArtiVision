"""
Taken from Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

"""
Number of epochs. We as you to use only 50 epochs as your budget.
For extra credit, if you experiment with more complex networks, you
can change this.
"""
num_epochs = 1

"""
A critical parameter that can dramatically affect whether training
succeeds or fails. The value for this depends significantly on which
optimizer is used. Refer to the default learning rate parameter
"""
learning_rate = 1e-4

"""
Momentum on the gradient (if you use a momentum-based optimizer)
"""
momentum = 0.01

"""
Resize image size for task 1. Task 3 must have an image size of 224,
so that is hard-coded elsewhere.
"""
img_size = 224

"""
Sample size for calculating the mean and standard deviation of the
training data. This many images will be randomly seleted to be read
into memory temporarily.
"""
preprocess_sample_size = 1000

"""
Defines the number of training examples per batch.
You don't need to modify this.
"""
batch_size = 32

"""
The number of image scene classes. Don't change this.
"""
num_classes = 193
