from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib
import numpy as np
import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE

class DatasetBuilder():
    def __init__(self, params):
        self.params = params
        
    
    @property
    def image_size(self):
        return list(self.params['image_shape'][:2])
    
    
    @property
    def class_names(self):
        train_data_path = pathlib.Path(self.params['train_data_path'])
        return np.array([item.name for item in train_data_path.glob('*')])
    
    
    @property
    def batch_size(self):
        return int(self.params['batch_size'])
    
    
    @property
    def train_data_dir(self):
        return pathlib.Path(self.params['train_data_path'])
    
    
    @property
    def val_data_dir(self):
        return pathlib.Path(self.params['val_data_path'])
    
    
    @property
    def num_classes(self):
        return int(self.params['num_classes'])
    
    
    def decode_image(self, image_path):
        image_bytes = tf.io.read_file(image_path)
        image_array = tf.image.decode_jpeg(image_bytes, channels=3)
        image_array = tf.image.convert_image_dtype(image_array, tf.float32)
        image_array = tf.image.resize(image_array, self.image_size)
        return image_array
    
    
    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == self.class_names
    
    
    def preprocess_input(self, file_path):
        image = self.decode_image(file_path)
        label = self.get_label(file_path)
        return image, label
    
    
    def input_fn(self, data_dir, mode=None):
        list_ds = tf.data.Dataset.list_files(str(data_dir/'*'/'*'))
        labeled_ds = list_ds.map(self.preprocess_input, num_parallel_calls=AUTOTUNE)
        dataset = labeled_ds.cache().repeat()
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size = 10 * self.batch_size)
        dataset = dataset.batch(self.batch_size)
        return dataset
    
    
    @property
    def train_steps(self):
        image_count = len(list(self.train_data_dir.glob('*/*.jpg')))
        return np.ceil(image_count/self.batch_size)
    
    
    @property
    def val_steps(self):
        image_count = len(list(self.val_data_dir.glob('*/*.jpg')))
        return np.ceil(image_count/self.batch_size)
    
    
    def create_dataset(self):
        train_ds = self.input_fn(self.train_data_dir, 'train')
        val_ds = self.input_fn(self.val_data_dir)
        return train_ds, val_ds