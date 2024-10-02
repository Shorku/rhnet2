import tensorflow as tf

from typing import Any, Callable, Optional
from tensorflow_gnn.keras.layers import convolution_base
from tensorflow_gnn.graph import graph_constants as const


@tf.keras.utils.register_keras_serializable()
class ConvScaleByFeature(convolution_base.AnyToAnyConvolutionBase):
    def __init__(self, message_fn: tf.keras.layers.Layer, *,
                 receiver_tag: const.IncidentNodeTag = const.TARGET,
                 receiver_feature: Optional[
                     const.FieldName] = const.HIDDEN_STATE,
                 sender_node_feature: Optional[
                     const.FieldName] = const.HIDDEN_STATE,
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


def dense_block(units: int,
                depth: int = 1,
                activation: str = 'elu',
                kernel_l2: float = 0.0,
                bias_l2: float = 0.0,
                dropout: float = 0.0):
    layers = []
    for i in range(depth - 1):
        layers.append(tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.l2(
                kernel_l2) if kernel_l2 else None,
            bias_regularizer=tf.keras.regularizers.l2(
                bias_l2) if bias_l2 else None,
            kernel_initializer=tf.keras.initializers.LecunNormal(),
            bias_initializer='zeros'))
        if dropout:
            layers.append(tf.keras.layers.Dropout(rate=dropout))
    layers.append(tf.keras.layers.Dense(
        units,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(
            kernel_l2) if kernel_l2 else None,
        bias_regularizer=tf.keras.regularizers.l2(
            bias_l2) if bias_l2 else None,
        kernel_initializer=tf.keras.initializers.LecunNormal(),
        bias_initializer='zeros'))

    return tf.keras.Sequential(layers)
