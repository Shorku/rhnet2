import tensorflow as tf
import tensorflow_gnn as tfgnn

from data_utils import read_schema, context_features
from data_utils import init_node_state, init_edge_state, drop_features
from custom_blocks import dense_block, ConvScaleByFeature


def neurom(schema_path: str,
           activation: str = "elu",
           head_kernel_l2: float = 0.0,
           head_bias_l2: float = 0.0,
           head_dropout: float = 0.25,
           gnn_kernel_l2: float = 0.0,
           gnn_bias_l2: float = 0.0,
           gnn_dropout: float = 0.0,
           graph_depth: int = 1,
           gnn_dense_depth: int = 1,
           graph_pooling: str = "concat",
           nodes_to_pool: str = "atom",
           head_width: int = 64,
           head_depth: int = 1,
           multi_target: bool = False,
           single_head_dense: bool = True):

    assert graph_pooling in ['mean',
                             'max',
                             'concat'], "Unrecognized graph pooling"
    graph_spec = read_schema(schema_path)
    targets = context_features(schema_path, multi_target=multi_target)
    input_graph = tf.keras.layers.Input(type_spec=graph_spec)
    graph = input_graph.merge_batch_to_components()
    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=init_node_state,
                                           edge_sets_fn=init_edge_state,
                                           context_fn=drop_features)(graph)
    atom_density_dim = \
        graph_spec.node_sets_spec['atom'].features_spec['density'].shape[1]
    link_density_dim = \
        graph_spec.node_sets_spec['link'].features_spec['density'].shape[1]

    for i in range(graph_depth):
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "link": tfgnn.keras.layers.NodeSetUpdate(
                    {"atom2link": tfgnn.keras.layers.SimpleConv(
                        message_fn=dense_block(units=link_density_dim,
                                               depth=gnn_dense_depth,
                                               activation=activation,
                                               kernel_l2=gnn_kernel_l2,
                                               bias_l2=gnn_bias_l2,
                                               dropout=gnn_dropout),
                        sender_edge_feature=tfgnn.HIDDEN_STATE,
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.SingleInputNextState())})(graph)
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "atom": tfgnn.keras.layers.NodeSetUpdate(
                    {"link2atom": ConvScaleByFeature(
                        message_fn=dense_block(units=link_density_dim,
                                               depth=gnn_dense_depth,
                                               activation=activation,
                                               kernel_l2=gnn_kernel_l2,
                                               bias_l2=gnn_bias_l2,
                                               dropout=gnn_dropout),
                        sender_edge_feature=tfgnn.HIDDEN_STATE,
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.NextStateFromConcat(
                        dense_block(units=atom_density_dim,
                                    depth=gnn_dense_depth,
                                    activation=activation,
                                    kernel_l2=gnn_kernel_l2,
                                    bias_l2=gnn_bias_l2,
                                    dropout=gnn_dropout)))})(graph)

    if graph_pooling == 'mean':
        output = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,
                                         'mean',
                                         node_set_name=nodes_to_pool)(graph)
    elif graph_pooling == 'max':
        output = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,
                                         'max_no_inf',
                                         node_set_name=nodes_to_pool)(graph)
    elif graph_pooling == 'concat':
        output1 = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,
                                          'mean',
                                          node_set_name=nodes_to_pool)(graph)
        output2 = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,
                                          'max_no_inf',
                                          node_set_name=nodes_to_pool)(graph)
        output = tf.keras.layers.Concatenate(axis=-1)([output1, output2])

    if single_head_dense:
        output = dense_block(units=head_width,
                             depth=head_depth,
                             activation=activation,
                             kernel_l2=head_kernel_l2,
                             bias_l2=head_bias_l2,
                             dropout=head_dropout)(output)
    outputs = []
    for target in targets:
        if single_head_dense:
            outputs.append(
                tf.keras.layers.Dense(
                    units=1,
                    activation=None,
                    name=target,
                    kernel_initializer=tf.keras.initializers.LecunNormal(),
                    bias_initializer='zeros')(output))
        else:
            outputs.append(
                dense_block(
                    units=head_width,
                    depth=head_depth,
                    activation=activation,
                    kernel_l2=head_kernel_l2,
                    bias_l2=head_bias_l2,
                    dropout=head_dropout)(output))
            outputs[-1] = tf.keras.layers.Dense(
                units=1,
                activation=None,
                name=target,
                kernel_initializer=tf.keras.initializers.LecunNormal(),
                bias_initializer='zeros')(outputs[-1])
    return tf.keras.Model(inputs=[input_graph], outputs=outputs)
