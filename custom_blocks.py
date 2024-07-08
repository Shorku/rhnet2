import tensorflow as tf
import tensorflow_gnn as tfgnn

from typing import Any, Callable, Optional
from tensorflow_gnn.keras.layers import convolution_base
from tensorflow_gnn.graph import graph_constants as const


@tf.keras.utils.register_keras_serializable()
class ConvScaleByFeature(convolution_base.AnyToAnyConvolutionBase):
    def __init__(self, message_fn: tf.keras.layers.Layer, *,
                 receiver_tag: const.IncidentNodeTag = const.TARGET,
                 receiver_feature: Optional[const.FieldName] =
                 const.HIDDEN_STATE,
                 sender_node_feature: Optional[const.FieldName] =
                 const.HIDDEN_STATE,
                 sender_edge_feature: Optional[const.FieldName] = None,
                 **kwargs):
        super().__init__(
            receiver_tag=receiver_tag,
            receiver_feature=receiver_feature,
            sender_node_feature=sender_node_feature,
            sender_edge_feature=sender_edge_feature,
            **kwargs)
        self._message_fn = message_fn

    def get_config(self):
        return super().get_config()

    def convolve(self, *,
                 sender_node_input: Optional[tf.Tensor],
                 sender_edge_input: Optional[tf.Tensor],
                 receiver_input: Optional[tf.Tensor],
                 broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
                 broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
                 pool_to_receiver: Callable[..., tf.Tensor],
                 extra_receiver_ops: Any = None,
                 training: bool) -> tf.Tensor:
        assert sender_edge_input is not None, \
            "Can not use ConvScaleByFeature with no edge feature specified"
        inputs = []
        if sender_node_input is not None:
            inputs.append(broadcast_from_sender_node(sender_node_input))
        if receiver_input is not None:
            inputs.append(broadcast_from_receiver(receiver_input))
        inputs = tf.concat(inputs, axis=-1)
        messages = self._message_fn(inputs)
        sender_edge_input.shape.assert_is_compatible_with(messages.shape)
        messages = tf.math.multiply(messages, sender_edge_input)
        messages = pool_to_receiver(messages, reduce_type="sum")
        return messages


def dense_block(units, depth=1, activation='elu', l2=0):

    return tf.keras.Sequential([tf.keras.layers.Dense(
        units,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(l2) if l2 else None,
        bias_regularizer=tf.keras.regularizers.l2(l2) if l2 else None)
        for i in range(depth)])


def graph_block(graph_spec,
                graph_depth=1, dense_depth=1,  activation='elu', l2=0):
    atom_msg_dim = \
        graph_spec.node_sets_spec['atom'].features_spec['density'].shape[1]
    atom_nxt_dim = \
        graph_spec.node_sets_spec['atom'].features_spec['density'].shape[1]
    link_msg_dim = \
        graph_spec.node_sets_spec['link'].features_spec['density'].shape[1]
    link_nxt_dim = \
        graph_spec.node_sets_spec['link'].features_spec['density'].shape[1]

    gnn = []
    for i in range(graph_depth):
        gnn.append(
            tfgnn.keras.layers.GraphUpdate(
                node_sets={
                    "link": tfgnn.keras.layers.NodeSetUpdate(
                        {"atom2link": tfgnn.keras.layers.SimpleConv(
                            message_fn=dense_block(atom_msg_dim,
                                                   depth=dense_depth,
                                                   activation=activation,
                                                   l2=l2),
                            sender_edge_feature=tfgnn.HIDDEN_STATE,
                            receiver_tag=tfgnn.TARGET)},
                        tfgnn.keras.layers.NextStateFromConcat(
                            dense_block(link_nxt_dim,
                                        depth=dense_depth,
                                        activation=activation,
                                        l2=l2)))}))
        gnn.append(
            tfgnn.keras.layers.GraphUpdate(
                node_sets={
                    "atom": tfgnn.keras.layers.NodeSetUpdate(
                        {"link2atom": ConvScaleByFeature(
                            message_fn=dense_block(link_msg_dim,
                                                   depth=dense_depth,
                                                   activation=activation,
                                                   l2=l2),
                            sender_edge_feature=tfgnn.HIDDEN_STATE,
                            receiver_tag=tfgnn.TARGET)},
                        tfgnn.keras.layers.NextStateFromConcat(
                            dense_block(atom_nxt_dim,
                                        depth=dense_depth,
                                        activation=activation,
                                        l2=l2)))}))
    return gnn
