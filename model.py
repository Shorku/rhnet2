import tensorflow as tf
import tensorflow_gnn as tfgnn

from data_utils import read_schema, context_features
from data_utils import init_node_state, init_edge_state, drop_features
from custom_blocks import dense_block
from custom_blocks import ConvScaleByFeature, NodeSetScaler, OuterProduct


def neurom(schema_path: str,
           graph_kan: bool = False,
           head_kan: bool = False,
           weighting_kan: bool = False,
           kan_grid_size: int = 5,
           kan_spline_order: int = 3,
           activation: str = "elu",
           head_kernel_l2: float = 0.0,
           head_bias_l2: float = 0.0,
           head_dropout: float = 0.25,
           gnn_kernel_l2: float = 0.0,
           gnn_bias_l2: float = 0.0,
           gnn_dropout: float = 0.0,
           weighting_kernel_l2: float = 0.0,
           weighting_bias_l2: float = 0.0,
           weighting_dropout: float = 0.0,
           graph_depth: int = 1,
           gnn_dense_depth: int = 1,
           graph_pooling: str = "mean|max_no_inf",
           prepool_scaling: bool = False,
           nodes_to_pool: str = "atom",
           self_interaction: str = None,
           head_width: int = 64,
           head_depth: int = 0,
           weighting_depth: int = 2,
           multi_target: bool = False,
           targets: list = None,
           single_head_dense: bool = True):

    graph_spec = read_schema(schema_path)
    targets = context_features(schema_path,
                               multi_target=multi_target,
                               requested_targets=targets)
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
                                               use_kan=graph_kan,
                                               activation=activation,
                                               output_activation=activation,
                                               kernel_l2=gnn_kernel_l2,
                                               bias_l2=gnn_bias_l2,
                                               dropout=gnn_dropout,
                                               kan_grid_size=kan_grid_size,
                                               spline_order=kan_spline_order),
                        sender_edge_feature=tfgnn.HIDDEN_STATE,
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.SingleInputNextState())})(graph)
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "atom": tfgnn.keras.layers.NodeSetUpdate(
                    {"link2atom": ConvScaleByFeature(
                        message_fn=dense_block(units=link_density_dim,
                                               depth=gnn_dense_depth,
                                               use_kan=graph_kan,
                                               activation=activation,
                                               output_activation=activation,
                                               kernel_l2=gnn_kernel_l2,
                                               bias_l2=gnn_bias_l2,
                                               dropout=gnn_dropout,
                                               kan_grid_size=kan_grid_size,
                                               spline_order=kan_spline_order),
                        sender_edge_feature=tfgnn.HIDDEN_STATE,
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.NextStateFromConcat(
                        dense_block(units=atom_density_dim,
                                    depth=gnn_dense_depth,
                                    use_kan=graph_kan,
                                    activation=activation,
                                    output_activation=activation,
                                    kernel_l2=gnn_kernel_l2,
                                    bias_l2=gnn_bias_l2,
                                    dropout=gnn_dropout,
                                    kan_grid_size=kan_grid_size,
                                    spline_order=kan_spline_order)))})(graph)
    if prepool_scaling:
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "atom": NodeSetScaler(
                    dense_block(units=atom_density_dim,
                                depth=weighting_depth,
                                use_kan=weighting_kan,
                                activation=activation,
                                output_activation=None,
                                kernel_l2=weighting_kernel_l2,
                                bias_l2=weighting_bias_l2,
                                dropout=weighting_dropout,
                                kan_grid_size=kan_grid_size,
                                spline_order=kan_spline_order),
                    node_input_feature=tfgnn.HIDDEN_STATE
                )})(graph)

    output = tfgnn.keras.layers.Pool(tfgnn.CONTEXT, graph_pooling,
                                     node_set_name=nodes_to_pool)(graph)

    if self_interaction:
        if self_interaction == 'outer':
            second_order_out = OuterProduct()(output)
        else:
            second_order_out = tf.keras.layers.Multiply()([output, output])
        output = tf.keras.layers.Concatenate(axis=-1)([output,
                                                       second_order_out])

    if single_head_dense:
        output = dense_block(units=head_width,
                             depth=head_depth,
                             use_kan=head_kan,
                             activation=activation,
                             output_activation=activation,
                             kernel_l2=head_kernel_l2,
                             bias_l2=head_bias_l2,
                             dropout=head_dropout,
                             kan_grid_size=kan_grid_size,
                             spline_order=kan_spline_order)(output)
    outputs = []
    for target in targets:
        if single_head_dense:
            outputs.append(
                tf.keras.layers.Dense(
                    units=1,
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.l2(
                        head_kernel_l2) if head_kernel_l2 else None,
                    bias_regularizer=tf.keras.regularizers.l2(
                        head_bias_l2) if head_bias_l2 else None,
                    name=target,
                    kernel_initializer=tf.keras.initializers.LecunNormal(),
                    bias_initializer='zeros')(output))
        else:
            outputs.append(
                dense_block(
                    units=head_width,
                    depth=head_depth,
                    use_kan=head_kan,
                    activation=activation,
                    output_activation=activation,
                    kernel_l2=head_kernel_l2,
                    bias_l2=head_bias_l2,
                    dropout=head_dropout,
                    kan_grid_size=kan_grid_size,
                    spline_order=kan_spline_order)(output))
            outputs[-1] = tf.keras.layers.Dense(
                units=1,
                activation=None,
                name=target,
                kernel_initializer=tf.keras.initializers.LecunNormal(),
                bias_initializer='zeros')(outputs[-1])
    return tf.keras.Model(inputs=[input_graph], outputs=outputs)
