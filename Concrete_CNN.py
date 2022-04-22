# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:26:23 2022

@author: owner
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os 
import pathlib

file_path = r"C:\Users\owner\Desktop\SHRDC MIDA AIML\Deep Learning\Spyder files\Concrete Crack"
data_dir = pathlib.Path(file_path)
SEED = 12345
IMG_SIZE = (180,180)
train_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, shuffle=True, validation_split=0.2, subset="training",
                                                            seed=SEED, image_size = IMG_SIZE , batch_size=10)
val_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, shuffle=True, validation_split=0.2, subset="validation",
                                                            seed=SEED, image_size = IMG_SIZE , batch_size=10)
#%%
class_names = train_dataset.class_names
#%%
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
val_dataset = val_dataset.skip(val_batches//5)
#%%
AUTOTUNE = tf.data.AUTOTUNE

train_dataset_pf = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset_pf = val_dataset.prefetch(buffer_size = AUTOTUNE)
test_dataset_pf = test_dataset.prefetch(buffer_size=AUTOTUNE)
#%%

data_augmentation = tf.keras.Sequential()
data_augmentation.add(tf.keras.layers.RandomFlip())
data_augmentation.add(tf.keras.layers.RandomRotation(0.3))
#%%
preprocess_input = tf.keras.applications.resnet50.preprocess_input

IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.ResNet50(include_top = False, weights = "imagenet", input_shape = IMG_SHAPE)
#%%
base_model.trainable = False
base_model.summary()
#%%

global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(class_names), activation="softmax")
#%%

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_avg_layer(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
#%%
es = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 2, min_delta = 0.2)
model.compile(optimizer= "adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
#%%

loss0, accuracy0 = model.evaluate(val_dataset_pf)

print("---------------Before Training----------------")
print(f"loss = {loss0}")
print(f"Accuracy = {accuracy0}")
#%%
EPOCH = 20
BATCH_SIZE = 32

history = model.fit(train_dataset_pf, validation_data=val_dataset_pf, epochs = EPOCH, batch_size = BATCH_SIZE, callbacks = [es])
#%%
import matplotlib.pyplot as plt

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs = history.epoch

plt.plot(epochs, train_loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title("Training vs Validation loss")
plt.legend()
plt.figure()

plt.plot(epochs, train_acc, label="Training accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.title("Training vs Validation accuracy")
plt.legend()
plt.figure()

plt.show()
#%%
image_batch, label_batch = test_dataset_pf.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
class_pred = np.argmax(predictions, axis=1)

print(f"Prediction: {class_pred}")
print(f"Labels: {label_batch}")
#%%
plt.figure(figsize=(10,10))

for i in range(9):
    axs = plt.subplot(3,3,i+1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[class_pred[i]])
    plt.axis("off")
#%%

loss, accuracy = model.evaluate(test_dataset_pf)

print("---------------After Training----------------")
print(f"loss = {loss}")
print(f"Accuracy = {accuracy}")

#%%