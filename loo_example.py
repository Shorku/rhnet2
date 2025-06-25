import sys
import pandas as pd
from fit_utils import loo_val


run_id = sys.argv[0][:-3]
targets = ['target', 'logk', 'melt', 'vol', 'surf', 'dipole']
noise_stddev = {'target': 0.35,
                'logk': 0.35,
                'melt': 0.35,
                'vol': 0.1,
                'surf': 0.1,
                'dipole': 0.1}
loss_weights = {'target': 0.5,
                'logk': 0.5,
                'melt': 1.5,
                'vol': 1.5,
                'surf': 1.5,
                'dipole': 5.0}
train_set_df = pd.read_csv('train_data.csv')
shuffle_buffer = {
    f'{cas}.tfrecord':
        train_set_df[train_set_df['cas'] == cas]['nconf'].sum() * 60
    for cas in train_set_df['cas'].unique()}

result = \
    loo_val(schema_path="data/schema_example.pbtxt",
            train_data_path="data/train",
            exclude_from_val=None,
            log_path="logs",
            run_id=run_id,
            graph_kan=True,
            head_kan=False,
            weighting_kan=False,
            kan_grid_size=3,
            kan_spline_order=2,
            head_kernel_l2=0.08,
            head_bias_l2=0.0,
            head_dropout=0.0,
            gnn_dropout=0.0,
            gnn_kernel_l2=0.0,
            gnn_bias_l2=0.0,
            weighting_kernel_l2=0.08,
            weighting_bias_l2=0.0,
            graph_depth=8,
            gnn_dense_depth=1,
            graph_pooling="sum|mean|max_no_inf",
            prepool_scaling=True,
            nodes_to_pool="atom",
            self_interaction=None,
            head_width=0,
            head_depth=0,
            weighting_depth=1,
            multi_target=True,
            targets=targets,
            single_head_dense=True,
            noise_stddev=noise_stddev,
            shuffle_buffer=shuffle_buffer,
            batch_size=8,
            epochs=20,
            steps_per_epoch=1000,
            validation_steps=1000,
            learning_rate=1e-4,
            learning_rate_decay_steps=13400,
            learning_decay_rate=0.999,
            loss_weights=loss_weights,
            models_save=True,
            test_data_path="data/test",
            maxiter=100)
