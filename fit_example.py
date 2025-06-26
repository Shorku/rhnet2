import os
import pandas as pd
from fit_utils import model_fit


targets = ['target', 'logk', 'melt', 'vol', 'surf', 'dipole']
noise_stddev = {'target': 0.15,
                'logk': 0.15,
                'melt': 0.15,
                'vol': 0.1,
                'surf': 0.1,
                'dipole': 0.1}
loss_weights = {'target': 0.5,
                'logk': 0.5,
                'melt': 1.5,
                'vol': 1.5,
                'surf': 1.5,
                'dipole': 5.0}
shuffle_buffer = {record: 100 for record in os.listdir("data/train")}

history, model = model_fit(schema_path="data/schema_example.pbtxt",
                           train_data_path="data/some_train_data_folder",
                           val_data_path="data/some_val_data_folder",
                           head_kernel_l2=0.001,
                           head_dropout=0.0,
                           gnn_kernel_l2=0.001,
                           weighting_kernel_l2=0.001,
                           graph_depth=2,
                           gnn_dense_depth=2,
                           graph_pooling="sum|mean",
                           prepool_scaling=True,
                           weighting_depth=2,
                           multi_target=True,
                           targets=targets,
                           single_head_dense=True,
                           noise_stddev=noise_stddev,
                           shuffle_buffer=shuffle_buffer,
                           batch_size=8,
                           epochs=450,
                           steps_per_epoch=1000,
                           validation_steps=500,
                           learning_rate=1e-4,
                           learning_rate_decay_steps=13400,
                           learning_decay_rate=0.999,
                           loss_weights=loss_weights,
                           model_save_path="some_folder")

pd.DataFrame(
    {key: value for key, value in history.history.items()}).to_csv("some_csv")
