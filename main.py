from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import datetime

from model import build_model
from data_loader import DatasetBuilder


def train_and_evaluate(params):
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    output_dir = params['output_dir']
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    savedmodel_dir = os.path.join(output_dir, 'savedmodel')
    model_export_path = os.path.join(savedmodel_dir, timestamp)
  
  if tf.io.gfile.exists(output_dir):
    tf.io.gfile.rmtree(output_dir)
    
    builder = DatasetBuilder(params)
    train_ds, val_ds = builder.create_dataset()
    
    class_names = builder.class_names
    num_classes = len(class_names)
    
    train_steps = builder.train_steps
    val_steps = builder.val_steps
    
    model = build_model(params, num_classes)
    history = model.fit(train_ds,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        validation_data=val_ds,
                        steps_per_epoch=train_steps,
                        validation_steps=val_steps)
    
    tf.saved_model.save(model, model_export_path)
    return history