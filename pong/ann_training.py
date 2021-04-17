"""
This script takes care of the training. 
As long as you have your data in the data/0 data/2 and data/3 folders (see data_creation.py),
you can just run this file and it will train the agent on this data.
"""

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


import pathlib
data_dir = pathlib.Path("./data")


batch_size = 8
img_height = 40
img_width = 68

#load the training data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=101,
  image_size=(img_height, img_width),
  color_mode="grayscale",
  batch_size=batch_size)

#load the validation/testing data, on which the agent will NOT learn
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=101,
  image_size=(img_height, img_width),
  color_mode="grayscale",
  batch_size=batch_size)

# This snippet of code is only there to make the learning faster by caching the datasets
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(10000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 3

# Define the model:
# Three conv2D layers with max pooling, and a final densely connected layer.

# We also add a normalization at the beginning (mapping the 0-255 pixel values to 0-1).

# Note: I tried experimenting with the depth and width of the network, 
# but it didn't change much to the performances of the network
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
  layers.Conv2D(8, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# prepare the model for training, using classical parameters,
# very often used for such classification tasks
# (our task is basically a classification between the actions)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Train for n Epochs, 10 seems like a good number, stopping before overfitting
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# save the trained model for later use
model.save("trained_agent")


#################################### Plotting Training Results ########################################
plot_results = False #turn this to true to get the plots after running the training
if plot_results:
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(10, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()