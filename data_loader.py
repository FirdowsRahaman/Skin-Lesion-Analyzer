import os
import glob
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DatasetBuilder():
    def __init__(self, base_dir, csv_file):
        self.base_dir = base_dir
        self.csv_file = csv_file
        
    def transform_df(self, base_dir, csv_file):
        df = pd.read_csv(csv_file)
        image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x 
                       for x in glob.glob(os.path.join(base_dir, '*', '*.jpg'))}
        label_dict = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
                      'mel': 4, 'nv': 5, 'vasc': 6}
        df['image_path'] = df['image_id'].map(image_path_dict.get)
        df['label_id'] = df['dx'].map(label_dict.get)
        return df
    
    def split_df(self):
        dataframe = self.transform_df(self.base_dir, self.csv_file)
        train_df, val_df = train_test_split(dataframe, test_size=0.15)
        train_df, test_df = train_test_split(train_df, test_size=0.10)
        return train_df, val_df, test_df
    
    def get_labels(self, dataframe):
        label_list = dataframe.label_id.values
        labels = to_categorical(label_list, num_classes=7)
        return labels
    
    def decode_image(self, filename, label=None, image_size=(224, 224)):
        bits = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(bits, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_size)
        return image, label
    
    def input_fn(self, dataframe, batch_size=32, mode=None):
        image_list = dataframe.image_path.values
        labels = self.get_labels(dataframe)
        ds = (tf.data.Dataset     
                .from_tensor_slices((image_list, labels))
                .map(self.decode_image, num_parallel_calls=AUTOTUNE)
                .cache()
                .repeat()
                .shuffle(buffer_size = 10 * batch_size)
                .batch(batch_size)
                .prefetch(AUTOTUNE))
        return ds

    def create_dataset(self):
        train_df, val_df, test_df = self.split_df()
        train_ds = self.input_fn(train_df)
        val_ds = self.input_fn(val_df)
        return train_ds, val_ds
