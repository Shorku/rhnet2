import tensorflow as tf
import tensorflow_gnn as tfgnn

from .data_utils import init_node_state, init_edge_state, drop_features
from .custom_blocks import graph_block, dense_block


def rhnet_v2(graph_spec,
             activation="elu",
             l2=0,
             graph_depth=1,
             dense_depth=1,
             graph_pooling="concat",
             nodes_to_pool="atom",
             head_dim=64,
             head_depth=1):
    assert graph_pooling in ['mean',
                             'max',
                             'concat'], "Unrecognized graph pooling"
    input_graph = tf.keras.layers.Input(type_spec=graph_spec)

    graph = input_graph.merge_batch_to_components()
    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=init_node_state,
                                           edge_sets_fn=init_edge_state,
                                           context_fn=drop_features)(graph)
    gnn_block = graph_block(graph_spec,
                            graph_depth=graph_depth,
                            dense_depth=dense_depth,
                            activation=activation,
                            l2=l2)
    for layer in gnn_block:
        graph = layer(graph)

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

    output = dense_block(head_dim,
                         depth=head_depth,
                         activation=activation,
                         l2=l2)(output)
    output = tf.keras.layers.Dense(1, activation=None)(output)

    return tf.keras.Model(inputs=[input_graph], outputs=[output])
