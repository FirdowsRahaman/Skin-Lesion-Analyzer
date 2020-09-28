from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


METRICS = [
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]


def build_model(params, num_classes):
    image_shape = params['image_shape']
    image_input = keras.Input(shape=image_shape)
    
    x = layers.Conv2D(filters=64, kernel_size=(7, 7))(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    
    x = layers.Conv2D(filters=128, kernel_size=(7, 7))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    
    x = layers.Conv2D(filters=256, kernel_size=(7, 7))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    
    x = layers.Conv2D(filters=512, kernel_size=(7, 7))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=image_input, outputs=x)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=METRICS)
    
    return model