import os

import tensorflow as tf

from pathlib import Path
from tensorflow_gnn import runner

from model import neuro_mulliken
from data_utils import balanced_dataset, context_features


def model_fit(schema_path: str,
              train_data_path: str,
              train_records_list: dict = None,
              val_data_path: str = None,
              val_records_list: dict = None,
              activation: str = "elu",
              head_kernel_l2: float = 0.0,
              head_bias_l2: float = 0.0,
              head_dropout: float = 0.25,
              gnn_dropout: float = 0.0,
              gnn_kernel_l2: float = 0.0,
              gnn_bias_l2: float = 0.0,
              graph_depth: int = 1,
              gnn_dense_depth: int = 1,
              graph_pooling: str = "concat",
              nodes_to_pool: str = "atom",
              head_width: int = 64,
              head_depth: int = 1,
              multi_target: bool = False,
              single_head_dense: bool = True,
              noise_stddev: dict = None,
              shuffle_buffer: dict = None,
              compression_type: str = 'GZIP',
              batch_size: int = 16,
              epochs: int = 10,
              steps_per_epoch: int = 100,
              validation_steps: int = 100,
              learning_rate: float = 5e-6,
              learning_rate_decay_steps: int = 50,
              learning_decay_rate: float = 0.0,
              loss_weights: dict = None,
              verbose: str = "auto",
              model_save_path: str = None):
    targets = context_features(schema_path, multi_target=multi_target)
    train_data = balanced_dataset(schema_path,
                                  train_data_path,
                                  records_list=train_records_list,
                                  noise_stddev=noise_stddev,
                                  multi_target=multi_target,
                                  shuffle_buffer=shuffle_buffer,
                                  compression_type=compression_type) \
        .batch(batch_size=batch_size).prefetch(1)
    if val_data_path:
        val_data = balanced_dataset(schema_path,
                                    val_data_path,
                                    records_list=val_records_list,
                                    noise_stddev=None,
                                    multi_target=multi_target,
                                    shuffle_buffer=None,
                                    compression_type=compression_type) \
            .batch(batch_size=batch_size).prefetch(1)
    else:
        val_data = None
    model = neuro_mulliken(schema_path,
                           activation=activation,
                           head_kernel_l2=head_kernel_l2,
                           head_bias_l2=head_bias_l2,
                           head_dropout=head_dropout,
                           gnn_kernel_l2=gnn_kernel_l2,
                           gnn_bias_l2=gnn_bias_l2,
                           gnn_dropout=gnn_dropout,
                           graph_depth=graph_depth,
                           gnn_dense_depth=gnn_dense_depth,
                           graph_pooling=graph_pooling,
                           nodes_to_pool=nodes_to_pool,
                           head_width=head_width,
                           head_depth=head_depth,
                           multi_target=multi_target,
                           single_head_dense=single_head_dense)
    if learning_decay_rate:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=learning_rate_decay_steps,
            decay_rate=learning_decay_rate,
            staircase=False)
    else:
        lr_schedule = learning_rate
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
    loss = {target: tf.keras.losses.MeanSquaredError(name=f'{target}_loss')
            for target in targets}
    if loss_weights is None:
        loss_weights = {target: 1.0 for target in targets}

    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=None,
                  run_eagerly=False)

    history = model.fit(train_data,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=val_data,
                        validation_steps=validation_steps,
                        verbose=verbose,
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath=os.path.join(model_save_path,
                                                      'checkpoint'),
                                save_weights_only=True)]
                        if model_save_path else None)
    if model_save_path:
        serving_path = os.path.join(model_save_path, 'serving')
        runner.export_model(model, serving_path)

    return history, model


def loo_val(schema_path,
            train_data_path, train_idc_list=None,
            compression_type=None,
            graph_depth=5,
            dense_depth=1,
            head_dim=64,
            head_depth=1,
            head_l2=0,
            head_dropout=0.5,
            gnn_l2=0,
            gnn_dropout=0.5,
            batch_size=16,
            buffer_size=100,
            epochs=100,
            steps_per_epoch=140,
            validation_steps=120,
            learning_rate=5e-6,
            verbose="auto"):
    if train_idc_list:
        train_idc_list = \
            [tfrecord for tfrecord in train_idc_list
             if f'{tfrecord}.tfrecord' in list(os.listdir(train_data_path))]
    else:
        train_idc_list = \
            [Path(fname).stem for fname in os.listdir(train_data_path)
             if fname.endswith('.tfrecord')]
    history = []
    train_mse = []
    val_mse = []
    for idc in train_idc_list:
        val_list = [idc]
        train_list = [i for i in train_idc_list if i != idc]
        history.append(
            model_fit(schema_path, train_data_path,
                      train_idc_list=train_list,
                      val_data_path=train_data_path,
                      val_idc_list=val_list,
                      compression_type=compression_type,
                      graph_depth=graph_depth,
                      dense_depth=dense_depth,
                      head_dim=head_dim,
                      head_depth=head_depth,
                      head_l2=head_l2,
                      head_dropout=head_dropout,
                      gnn_l2=gnn_l2,
                      gnn_dropout=gnn_dropout,
                      batch_size=batch_size,
                      buffer_size=buffer_size,
                      epochs=epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_steps=validation_steps,
                      learning_rate=learning_rate,
                      verbose=verbose)[0].history)
        train_mse.append(history[-1]["mse_loss"][-1])
        val_mse.append(history[-1]["val_mse_loss"][-1])
        print(f'{idc}: '
              f'train mse: {train_mse[-1]}; '
              f'val mse: {val_mse[-1]}')
    print(f'avg train mse: {sum(train_mse) / len(train_mse)}; '
          f'avg val mse: {sum(val_mse) / len(val_mse)}')
    return history
