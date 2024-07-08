import os
import random

import tensorflow as tf

from pathlib import Path

from .model import rhnet_v2
from .data_utils import balanced_dataset, read_schema


def cross_val(schema_path, data_path,
              graph_depth=5, head_dim=64, head_depth=1, dense_depth=1, l2=0,
              batch_size=16, buffer_size=100, epochs=100,
              steps_per_epoch=140, validation_steps=120,
              folds=5, idc_list=None):
    if idc_list:
        idc_list = [idc for idc in idc_list
                    if f'{idc}.tfrecord' in list(os.listdir(data_path))]
    else:
        idc_list = [Path(fname).stem for fname in os.listdir(data_path)
                    if fname.endswith('.tfrecord')]
    random.shuffle(idc_list)
    nidc = len(idc_list)
    graph_spec = read_schema(schema_path)

    history = []

    for i in range(folds):
        val_idc = \
            idc_list[i * nidc // folds: (i + 1) * nidc // folds]
        train_idc = \
            idc_list[0: i * nidc // folds] + idc_list[(i + 1) * nidc // folds:]

        train_data = \
            balanced_dataset(graph_spec, data_path,
                             shuffle_buffer=buffer_size, tfrecords=train_idc) \
            .batch(batch_size=batch_size)
        val_data = \
            balanced_dataset(graph_spec, data_path,
                             shuffle_buffer=0, tfrecords=val_idc) \
            .batch(batch_size=batch_size)

        model = rhnet_v2(graph_spec,
                         l2=l2,
                         graph_depth=graph_depth,
                         dense_depth=dense_depth,
                         head_dim=head_dim,
                         head_depth=head_depth)
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.MeanSquaredError(name='mse_loss'),
        metrics = [tf.keras.metrics.MeanAbsoluteError(name='mаe_loss')]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                      run_eagerly=False)

        history.append(model.fit(train_data, steps_per_epoch=steps_per_epoch,
                                 epochs=epochs, validation_data=val_data,
                                 validation_steps=validation_steps,
                                 verbose=0))

    return history


def model_fit(schema_path, train_data_path, val_data_path=None,
              verbose="auto", model_save_path=None,
              graph_depth=5, head_dim=64, head_depth=1, dense_depth=1, l2=0,
              batch_size=16, buffer_size=100, epochs=100,
              steps_per_epoch=140, validation_steps=120):
    graph_spec = read_schema(schema_path)

    train_data = \
        balanced_dataset(graph_spec, train_data_path,
                         shuffle_buffer=buffer_size) \
        .batch(batch_size=batch_size)
    if val_data_path:
        val_data = \
            balanced_dataset(graph_spec, val_data_path, shuffle_buffer=0) \
            .batch(batch_size=batch_size)
    else:
        val_data = None
    model = rhnet_v2(graph_spec,
                     l2=l2,
                     graph_depth=graph_depth,
                     dense_depth=dense_depth,
                     head_dim=head_dim,
                     head_depth=head_depth)
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError(name='mse_loss'),
    metrics = [tf.keras.metrics.MeanAbsoluteError(name='mаe_loss')]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                  run_eagerly=False)

    history = model.fit(train_data, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, validation_data=val_data,
                        validation_steps=validation_steps,
                        verbose=verbose)
    if model_save_path:
        model.save(os.path.join(model_save_path, 'saved_model'))
    return history
