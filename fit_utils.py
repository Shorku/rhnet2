import os

import tensorflow as tf

# from pathlib import Path

from .model import rhnet_v2
from .data_utils import balanced_dataset, read_schema


def model_fit(schema_path,
              train_data_path, train_idc_list=None,
              val_data_path=None, val_idc_list=None,
              compression_type=None,
              graph_depth=5,
              dense_depth=1,
              head_dim=64,
              head_depth=1,
              head_l2=0,
              head_dropout=0.0,
              gnn_l2=0,
              gnn_dropout=0.0,
              batch_size=16,
              buffer_size=100,
              epochs=100,
              steps_per_epoch=140,
              validation_steps=120,
              learning_rate=5e-6,
              verbose="auto",
              model_save_path=None):
    graph_spec = read_schema(schema_path)
    train_data = \
        balanced_dataset(graph_spec, train_data_path,
                         idc_list=train_idc_list,
                         shuffle_buffer=buffer_size,
                         compression_type=compression_type) \
        .batch(batch_size=batch_size)
    if val_data_path:
        val_data = \
            balanced_dataset(graph_spec, val_data_path,
                             idc_list=val_idc_list,
                             shuffle_buffer=0,
                             compression_type=compression_type) \
            .batch(batch_size=batch_size)
    else:
        val_data = None
    model = rhnet_v2(graph_spec,
                     graph_depth=graph_depth,
                     dense_depth=dense_depth,
                     head_dim=head_dim,
                     head_depth=head_depth,
                     head_l2=head_l2,
                     head_dropout=head_dropout,
                     gnn_l2=gnn_l2,
                     gnn_dropout=gnn_dropout)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError(name='loss'),
    metrics = [tf.keras.metrics.MeanSquaredError(name='mse_loss'),
               tf.keras.metrics.MeanAbsoluteError(name='mаe_loss')]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                  run_eagerly=False)
    history = model.fit(train_data, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, validation_data=val_data,
                        validation_steps=validation_steps,
                        verbose=verbose)
    if model_save_path:
        model.save(os.path.join(model_save_path, 'saved_model'))
    return history, model
