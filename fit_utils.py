import os

import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow_gnn import runner

from model import neurom
from data_utils import balanced_dataset, context_features
from data_utils import get_decode_fn, read_schema


def model_fit(schema_path: str,
              train_data_path: str,
              train_records_list: list = None,
              val_data_path: str = None,
              val_records_list: list = None,
              graph_kan: bool = False,
              head_kan: bool = False,
              weighting_kan: bool = False,
              activation: str = "elu",
              head_kernel_l2: float = 0.0,
              head_bias_l2: float = 0.0,
              head_dropout: float = 0.25,
              gnn_dropout: float = 0.0,
              gnn_kernel_l2: float = 0.0,
              gnn_bias_l2: float = 0.0,
              weighting_kernel_l2: float = 0.0,
              weighting_bias_l2: float = 0.0,
              graph_depth: int = 1,
              gnn_dense_depth: int = 1,
              graph_pooling: str = "concat",
              nodes_to_pool: str = "atom",
              head_width: int = 64,
              head_depth: int = 0,
              weighting_depth: int = 0,
              multi_target: bool = True,
              targets: list = None,
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
    targets = context_features(schema_path,
                               multi_target=multi_target,
                               requested_targets=targets)
    train_data = balanced_dataset(schema_path,
                                  train_data_path,
                                  records_list=train_records_list,
                                  noise_stddev=noise_stddev,
                                  multi_target=multi_target,
                                  targets=targets,
                                  shuffle_buffer=shuffle_buffer,
                                  compression_type=compression_type) \
        .batch(batch_size=batch_size).prefetch(1)
    if val_data_path:
        val_data = balanced_dataset(schema_path,
                                    val_data_path,
                                    records_list=val_records_list,
                                    noise_stddev=None,
                                    multi_target=multi_target,
                                    targets=targets,
                                    shuffle_buffer=None,
                                    compression_type=compression_type) \
            .batch(batch_size=batch_size).prefetch(1)
    else:
        val_data = None
    model = neurom(schema_path,
                   graph_kan=graph_kan,
                   head_kan=head_kan,
                   weighting_kan=weighting_kan,
                   activation=activation,
                   head_kernel_l2=head_kernel_l2,
                   head_bias_l2=head_bias_l2,
                   head_dropout=head_dropout,
                   gnn_kernel_l2=gnn_kernel_l2,
                   gnn_bias_l2=gnn_bias_l2,
                   gnn_dropout=gnn_dropout,
                   weighting_kernel_l2=weighting_kernel_l2,
                   weighting_bias_l2=weighting_bias_l2,
                   graph_depth=graph_depth,
                   gnn_dense_depth=gnn_dense_depth,
                   graph_pooling=graph_pooling,
                   nodes_to_pool=nodes_to_pool,
                   head_width=head_width,
                   head_depth=head_depth,
                   weighting_depth=weighting_depth,
                   multi_target=multi_target,
                   targets=targets,
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
    else:
        loss_weights = {loss: weight
                        for loss, weight in loss_weights.items()
                        if loss in targets}

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


def loo_val(schema_path: str,
            train_data_path: str,
            train_records_list: list = None,
            graph_kan: bool = False,
            head_kan: bool = False,
            weighting_kan: bool = False,
            activation: str = "elu",
            head_kernel_l2: float = 0.0,
            head_bias_l2: float = 0.0,
            head_dropout: float = 0.25,
            gnn_dropout: float = 0.0,
            gnn_kernel_l2: float = 0.0,
            gnn_bias_l2: float = 0.0,
            weighting_kernel_l2: float = 0.0,
            weighting_bias_l2: float = 0.0,
            graph_depth: int = 1,
            gnn_dense_depth: int = 1,
            graph_pooling: str = "concat",
            nodes_to_pool: str = "atom",
            head_width: int = 64,
            head_depth: int = 1,
            weighting_depth: int = 0,
            multi_target: bool = False,
            targets: list = None,
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
    if train_records_list:
        train_records_list = \
            [tfrecord for tfrecord in train_records_list
             if f'{tfrecord}.tfrecord' in list(os.listdir(train_data_path))]
    else:
        train_records_list = \
            [Path(fname).stem for fname in os.listdir(train_data_path)
             if fname.endswith('.tfrecord')]
    loo_history = {}
    for record in train_records_list:
        val_list = [record]
        train_list = [i for i in train_records_list if i != record]
        history = model_fit(
            schema_path=schema_path,
            train_data_path=train_data_path,
            train_records_list=train_list,
            val_data_path=train_data_path,
            val_records_list=val_list,
            graph_kan=graph_kan,
            head_kan=head_kan,
            weighting_kan=weighting_kan,
            activation=activation,
            head_kernel_l2=head_kernel_l2,
            head_bias_l2=head_bias_l2,
            head_dropout=head_dropout,
            gnn_dropout=gnn_dropout,
            gnn_kernel_l2=gnn_kernel_l2,
            gnn_bias_l2=gnn_bias_l2,
            weighting_kernel_l2=weighting_kernel_l2,
            weighting_bias_l2=weighting_bias_l2,
            graph_depth=graph_depth,
            gnn_dense_depth=gnn_dense_depth,
            graph_pooling=graph_pooling,
            nodes_to_pool=nodes_to_pool,
            head_width=head_width,
            head_depth=head_depth,
            weighting_depth=weighting_depth,
            multi_target=multi_target,
            targets=targets,
            single_head_dense=single_head_dense,
            noise_stddev=noise_stddev,
            shuffle_buffer=shuffle_buffer,
            compression_type=compression_type,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            learning_rate=learning_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            learning_decay_rate=learning_decay_rate,
            loss_weights=loss_weights,
            verbose=verbose,
            model_save_path=os.path.join(model_save_path, record)
            if model_save_path else None)[0].history
        for key, value in history.items():
            loo_history[f'{record}.{key}'] = value
        tf.keras.backend.clear_session()
    return loo_history


def predict_on_record(model: tf.keras.Model,
                      schema_path: str,
                      data_path: str,
                      log_path: str,
                      base_name: str,
                      multi_target: bool = False,
                      targets: list = None):
    decode_fn = get_decode_fn(read_schema(schema_path))
    targets = context_features(schema_path,
                               multi_target=multi_target,
                               requested_targets=targets)
    os.makedirs(os.path.join(log_path,
                             f'Predictions_{base_name}'), exist_ok=True)
    for record in os.listdir(data_path):
        if record.endswith('.tfrecord'):
            record_path = os.path.join(data_path, record)
            record_data = tf.data.TFRecordDataset(
                [record_path], compression_type="GZIP").map(decode_fn)
            eval_results = model.predict(record_data)
            pd.DataFrame(
                {target: list(eval_results[i].flatten())
                 for i, target in enumerate(targets)}).to_csv(
                os.path.join(log_path,
                             f'Predictions_{base_name}',
                             f'{Path(record).stem}_{base_name}.csv'),
                index=False)


def ensemble_fit(schema_path: str,
                 train_data_path: str,
                 log_path: str,
                 nmodels: int = 1,
                 run_id: str = '',
                 predict_on_train: bool = True,
                 predict_on_val: bool = True,
                 train_records_list: list = None,
                 val_data_path: str = None,
                 val_records_list: list = None,
                 graph_kan: bool = False,
                 head_kan: bool = False,
                 weighting_kan: bool = False,
                 activation: str = "elu",
                 head_kernel_l2: float = 0.0,
                 head_bias_l2: float = 0.0,
                 head_dropout: float = 0.25,
                 gnn_dropout: float = 0.0,
                 gnn_kernel_l2: float = 0.0,
                 gnn_bias_l2: float = 0.0,
                 weighting_kernel_l2: float = 0.0,
                 weighting_bias_l2: float = 0.0,
                 graph_depth: int = 1,
                 gnn_dense_depth: int = 1,
                 graph_pooling: str = "concat",
                 nodes_to_pool: str = "atom",
                 head_width: int = 64,
                 head_depth: int = 1,
                 weighting_depth: int = 0,
                 multi_target: bool = True,
                 targets: list = None,
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
                 verbose: str = "auto"):
    ensemble_history = {}
    base_name = (
        f'{run_id}_'
        f'G{graph_depth}D{gnn_dense_depth}Hd{head_depth}Hw{head_width}_'
        f'l2gnn{gnn_kernel_l2}l2head{head_kernel_l2}_'
        f'GNNdrop{gnn_dropout}Headdrop{head_dropout}_'
        f'batch{batch_size}_lr{learning_rate}_'
        f'activation{activation}_singleHead{single_head_dense}')
    log_path = os.path.join(log_path, base_name)
    os.makedirs(log_path, exist_ok=True)
    for n in range(nmodels):
        history, model = model_fit(
            schema_path=schema_path,
            train_data_path=train_data_path,
            train_records_list=train_records_list,
            val_data_path=val_data_path,
            val_records_list=val_records_list,
            graph_kan=graph_kan,
            head_kan=head_kan,
            weighting_kan=weighting_kan,
            activation=activation,
            head_kernel_l2=head_kernel_l2,
            head_bias_l2=head_bias_l2,
            head_dropout=head_dropout,
            gnn_dropout=gnn_dropout,
            gnn_kernel_l2=gnn_kernel_l2,
            gnn_bias_l2=gnn_bias_l2,
            weighting_kernel_l2=weighting_kernel_l2,
            weighting_bias_l2=weighting_bias_l2,
            graph_depth=graph_depth,
            gnn_dense_depth=gnn_dense_depth,
            graph_pooling=graph_pooling,
            nodes_to_pool=nodes_to_pool,
            head_width=head_width,
            head_depth=head_depth,
            weighting_depth=weighting_depth,
            multi_target=multi_target,
            targets=targets,
            single_head_dense=single_head_dense,
            noise_stddev=noise_stddev,
            shuffle_buffer=shuffle_buffer,
            compression_type=compression_type,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            learning_rate=learning_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            learning_decay_rate=learning_decay_rate,
            loss_weights=loss_weights,
            verbose=verbose,
            model_save_path=os.path.join(log_path,
                                         f'Models_{base_name}',
                                         f'model{n}'))
        for key, value in history.history.items():
            ensemble_history[f'model{n}.{key}'] = value
        pd.DataFrame(ensemble_history).to_csv(os.path.join(log_path,
                                                           f'{base_name}.csv'),
                                              index=False)
        if predict_on_train:
            predict_on_record(model=model,
                              schema_path=schema_path,
                              data_path=train_data_path,
                              log_path=log_path,
                              base_name=f'{base_name}_model{n}',
                              multi_target=multi_target,
                              targets=targets)
        if predict_on_val:
            predict_on_record(model=model,
                              schema_path=schema_path,
                              data_path=val_data_path,
                              log_path=log_path,
                              base_name=f'{base_name}_model{n}',
                              multi_target=multi_target,
                              targets=targets)
        tf.keras.backend.clear_session()
    return base_name
