import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Reshape, Lambda, \
    GlobalAveragePooling1D
from tensorflow.keras import backend as K

from models.layers import *


class RSquare(tf.keras.metrics.Metric):
    def __init__(self, name='r_square', **kwargs):
        super(RSquare, self).__init__(name=name, **kwargs)
        self.residuals = self.add_weight(name='residuals', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        residual = tf.reduce_sum(tf.square(y_true - y_pred))
        total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        self.residuals.assign_add(residual)
        self.total.assign_add(total)
        self.count.assign_add(1)

    def result(self):
        r2 = 1 - (self.residuals / (self.total + K.epsilon()))
        return r2

    def reset_states(self):
        self.residuals.assign(0.0)
        self.total.assign(0.0)
        self.count.assign(0.0)


loss_tracker = tf.keras.metrics.Mean(name="loss")
r2_tracker = RSquare(name="r2")
loss_age_tracker = tf.keras.metrics.Mean(name="loss_age")
loss_lc_tracker = tf.keras.metrics.Mean(name="loss_lc")
auc_age_tracker = tf.keras.metrics.AUC(name="auc_age")
auc_lc_tracker = tf.keras.metrics.AUC(name="auc_lc")


# def aggregate(x, number_of_nodes):
#     message, edge = x
#     merged_edges = tf.math.unsorted_segment_sum(message, edge[:, 1], number_of_nodes, )
#     # Probar a concatenar en ved de sumar según el paper 2021 Generalizable Machine Learning in Neuroscience using GNN
#     return merged_edges
# class Aggregate(tfk.layers.Layer):
#     def __init__(self,
#                  number_of_nodes,
#                  **kwargs):
#         super(Aggregate, self).__init__(**kwargs)
#         self.number_of_nodes = number_of_nodes
#     def call(self, x):
#         return aggregate(x, number_of_nodes=self.number_of_nodes)


def apply_encoder(dense_layer_dimensions, layer, name="node_ide"):
    for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)), dense_layer_dimensions
    ):
        layer = Dense(
            dense_layer_dimension,
            activation='gelu',
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5))(layer)
        layer = tf.keras.layers.LayerNormalization()(layer)
        layer = Dropout(0.1)(layer)
    return layer


def apply_decoder(dense_layer_dimensions, layer, name="node_idd"):
    for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)),
            reversed(dense_layer_dimensions),
    ):
        layer = Dense(
            dense_layer_dimension,
            activation='gelu',
            name=name + str(dense_layer_number + 1))(layer)
        layer = tf.keras.layers.LayerNormalization()(layer)
        layer = Dropout(0.1)(layer)
    return layer


def apply_graph_layer(base_layer_dimensions, activation, number_of_nodes, graph_layer, n_heads):
    if graph_layer[0].shape[1] == 121:
        for base_layer_number, base_layer_dimension in zip(
                range(len(base_layer_dimensions)), base_layer_dimensions
        ):
            if len(graph_layer) == 2:
                graph_layer_class = EdgeGraphLayer(base_layer_dimension, activation, number_of_nodes)
            else:
                graph_layer_class = NodeEdgeGlobalMessagePassing(base_layer_dimension, activation, number_of_nodes)
            graph_layer = graph_layer_class(graph_layer)
        return graph_layer
    else:
        att_weights_all = []
        for base_layer_number, base_layer_dimension in zip(
                range(len(base_layer_dimensions)), base_layer_dimensions
        ):
            if len(graph_layer) == 2:
                graph_layer_class = EdgeGraphLayer(base_layer_dimension, activation, number_of_nodes, n_heads=n_heads)
            elif graph_layer[0].shape[1] == 121:
                graph_layer_class = NodeEdgeGlobalMessagePassing(base_layer_dimension, activation, number_of_nodes,
                                                                 n_heads)
            else:
                graph_layer_class = NodeEdgeGraphLayer(base_layer_dimension, activation, number_of_nodes, n_heads)
            graph_layer, att_weights = graph_layer_class(graph_layer)
            att_weights_all.append(att_weights)
        return graph_layer, att_weights_all


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def MAE(y_true, y_pred, sample_weight=None):
    absolute_error = tf.abs(y_true - y_pred)
    if sample_weight is not None:
        absolute_error *= sample_weight
    return tf.reduce_mean(absolute_error, axis=-1)


class MultilayerPerceptron(tfk.Model):
    """

    """

    def __init__(self, number_of_edges, number_of_edge_features, attention=False, **kwargs):
        super(MultilayerPerceptron, self).__init__(**kwargs)
        self.number_of_edges = number_of_edges
        self.number_of_edge_features = number_of_edge_features
        self.dense_layers = [256, 512, 1024, 2048]
        self.attention = attention
        self.net = self.network()

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]

    def network(self):
        input_features, edges = (
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
        )
        x_global = input_features
        # x_global = tfk.layers.LayerNormalization()(input_features)
        for l in self.dense_layers:
            x_global = tfk.layers.Dense(l, activation='relu')(x_global)
            x_global = tfk.layers.Dropout(0.5)(x_global)
        for l_i in reversed(self.dense_layers):
            x_global = tfk.layers.Dense(l_i, activation='relu')(x_global)
            x_global = tfk.layers.Dropout(0.5)(x_global)
        if self.attention:
            x_global_dense = tfk.layers.Dense(32)(x_global)
            cls_layer = tf.keras.layers.Flatten()(x_global)
            cls_layer = tfk.layers.Dense(32)(cls_layer)
            cls_layer = tf.expand_dims(cls_layer, 1)
            attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=6, key_dim=32, dropout=0.1)
            updated_cls_layer, att_weights = attention_layer(query=cls_layer, value=x_global_dense, key=x_global_dense,
                                                             return_attention_scores=True)
            updated_cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer + cls_layer)
            feed_forward = tf.keras.Sequential(
                [
                    tfk.layers.Dense(16),
                    tfk.layers.Dense(32),
                    tfk.layers.Dropout(0.3),
                ]
            )
            updated_cls_layer_y = feed_forward(updated_cls_layer)
            x_global = tf.keras.layers.LayerNormalization()(updated_cls_layer_y + updated_cls_layer)
        else:
            cls_layer = tfk.layers.Dense(50)(x_global)
            att_weights = cls_layer
        x_global = GlobalAveragePooling1D()(x_global)
        x = tfk.layers.Dense(1, activation='sigmoid')(x_global)
        model = tf.keras.models.Model(
            inputs=[input_features, edges],
            outputs=[x, att_weights],
        )
        return model

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            outputs = self.net(images, training=True)
            loss = self.compiled_loss(labels, outputs[0])

        grads = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.net.trainable_weights))

        loss_tracker.update_state(loss)
        return {
            "loss": loss_tracker.result(),
        }

    def test_step(self, data):
        images, labels = data

        outputs = self.net(images, training=False)

        loss = self.compiled_loss(labels, outputs[0])

        loss_tracker.update_state(loss)
        return {
            "loss": loss_tracker.result(),
        }

    def call(self, inputs, training=None):
        outputs = self.net(inputs, training=training)
        return outputs


class AtrophyDeepBrain(tfk.Model):
    """
        AtrophyDeepBrain is a multilayer perceptron, to estimate
        the age from node properties.

        Parameters:
        -----------
        number_of_nodes: int
            Number of nodes in the graph.
        number_of_edges: int
            Number of edges in the graph.
        number_of_edge_features: int
            Number of edge features.
        dense_layer_dimensions: list of ints
            List of the number of units in each dense layer of the encoder and decoder. The
            number of layers is inferred from the length of this list.
        base_layer_dimensions: list of ints
            List of the latent dimensions of the graph blocks. The number of layers is
            inferred from the length of this list.
        activation: tensorflow keras function
            Activation function for the output node and edge layer.
        Returns:
        --------
        tf.keras.Model
            Keras model for the graph neural network.
    """

    def __init__(self,
                 number_of_nodes,
                 dense_layer_dimensions=(32, 64, 96),
                 activation=tf.keras.activations.gelu,
                 n_heads=6,
                 output_layer=tfk.layers.Dense(1, activation='sigmoid'),
                 attention=False,
                 **kwargs):
        super(AtrophyDeepBrain, self).__init__(**kwargs)
        self.attention = attention
        self.number_of_nodes = number_of_nodes
        self.dense_layer_dimensions = dense_layer_dimensions
        self.activation = activation
        self.n_heads = n_heads
        self.dense_to_combine = output_layer
        self.net = self.network()

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]

    def network(self):
        input_features = tf.keras.Input(shape=(1, self.number_of_nodes))

        x_global = tfk.layers.LayerNormalization()(input_features)
        for l in self.dense_layer_dimensions:
            x_global = tfk.layers.Dense(l, activation='relu')(x_global)
            x_global = tfk.layers.Dropout(0.1)(x_global)
        for l_i in reversed(self.dense_layer_dimensions):
            x_global = tfk.layers.Dense(l_i, activation='relu')(x_global)
            x_global = tfk.layers.Dropout(0.1)(x_global)
        if self.attention:
            x_global_dense = tfk.layers.Dense(8, activation=self.activation, name="gm_mlp_attention_block")(x_global)
            cls_layer = tf.keras.layers.Flatten()(x_global)
            cls_layer = tf.expand_dims(cls_layer, 1)
            cls_layer = tfk.layers.Dense(8, activation=self.activation, name="class_mlp_attention_block")(cls_layer)
            attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=self.n_heads, key_dim=2)
            updated_cls_layer, att_weights = attention_layer(cls_layer, x_global_dense, return_attention_scores=True)
            updated_cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer + cls_layer)
            feed_forward = tf.keras.Sequential(
                [
                    tfk.layers.Dense(8, activation=self.activation),
                    Dropout(0.5),
                    tfk.layers.Dense(8),
                    Dropout(0.5)
                ]
            )
            updated_cls_layer_y = feed_forward(updated_cls_layer)
            cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer_y + updated_cls_layer)
            cls_layer = tf.reduce_mean(cls_layer, axis=1)

        else:
            cls_layer = tfk.layers.Dense(8)(x_global)
            att_weights = cls_layer
            pass
        outputs = self.dense_to_combine(cls_layer)

        if self.attention:
            model = tf.keras.models.Model(
                inputs=input_features,
                outputs=[outputs, att_weights],
            )
        else:
            model = tf.keras.models.Model(
                inputs=input_features,
                outputs=[[outputs]],
            )
        return model

    def train_step(self, data):
        features, labels = data
        with tf.GradientTape() as tape:
            outputs = self.net(features, training=True)

            loss = self.compiled_loss(labels, outputs[0])

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights))
        loss_tracker.update_state(loss)
        return {
            "loss": loss_tracker.result(),
        }

    def test_step(self, data):
        features, labels = data
        outputs = self.net(features, training=False)
        loss = self.compiled_loss(labels, outputs[0])

        loss_tracker.update_state(loss)
        return {
            "loss": loss_tracker.result(),
        }

    def call(self, inputs, training=None):
        outputs = self.net(inputs, training=training)
        return outputs


class EdgeDeepBrain(tfk.Model):
    """
        EdgeDeepBrain is a message-passing Graph Neural Network with Class Token, to estimate
        the age from brain networks based on edge properties.

        Parameters:
        -----------
        number_of_nodes: int
            Number of nodes in the graph.
        number_of_edges: int
            Number of edges in the graph.
        number_of_edge_features: int
            Number of edge features.
        dense_layer_dimensions: list of ints
            List of the number of units in each dense layer of the encoder and decoder. The
            number of layers is inferred from the length of this list.
        base_layer_dimensions: list of ints
            List of the latent dimensions of the graph blocks. The number of layers is
            inferred from the length of this list.
        activation: tensorflow keras function
            Activation function for the output node and edge layer.
        Returns:
        --------
        tf.keras.Model
            Keras model for the graph neural network.
    """

    def __init__(self,
                 number_of_nodes,
                 number_of_edges,
                 number_of_edge_features,
                 dense_layer_dimensions=(32, 64, 96),
                 base_layer_dimensions=(96, 96),
                 activation=tf.keras.activations.gelu,
                 n_heads=6,
                 output_layer=tfk.layers.Dense(1, activation='sigmoid'),
                 edge_attention=True,
                 **kwargs):
        super(EdgeDeepBrain, self).__init__(**kwargs)
        self.edge_attention = edge_attention
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_edges
        self.number_of_edge_features = number_of_edge_features
        self.dense_layer_dimensions = dense_layer_dimensions
        self.base_layer_dimensions = base_layer_dimensions
        self.activation = activation
        self.n_heads = n_heads
        self.dense_to_combine = output_layer
        self.combine_layer = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))
        self.readout_block_edges = tf.keras.layers.Lambda(
            lambda x: tf.math.reduce_sum(x, axis=1), name="edges_readout"
        )
        # self.readout_block_nodes = layers.Lambda(
        #     lambda x: tf.math.reduce_sum(x, axis=1), name="nodes_readout"
        # )
        # self.readout_block_global = layers.Lambda(
        #     lambda x: tf.math.reduce_sum(x, axis=1), name="global_readout"
        # )
        self.readout_block_edges = GlobalAveragePooling1D()
        self.readout_block_nodes = GlobalAveragePooling1D()
        self.readout_block_global = GlobalAveragePooling1D()
        self.net = self.network()

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]

    def encode_block(self, edge_layer=None, node_layer=None, global_layer=None):
        if edge_layer != None:
            edge_layer = apply_encoder(self.dense_layer_dimensions, edge_layer, name="edge_ide")
        if node_layer != None:
            node_layer = apply_encoder(self.dense_layer_dimensions, node_layer, name="node_ide")
        if global_layer != None:
            global_layer = apply_encoder(self.dense_layer_dimensions, global_layer, name="global_ide")
        return edge_layer, node_layer, global_layer

    def graph_layer_block(self, graph_layer, only_edges=True, attention=False):
        att_weights_all = []
        for base_layer_number, base_layer_dimension in zip(
                range(len(self.base_layer_dimensions)), self.base_layer_dimensions
        ):
            if len(graph_layer) == 2:
                if only_edges == True:
                    graph_layer_class = EdgeGraphLayer(base_layer_dimension, self.activation,
                                                       self.number_of_nodes, self.n_heads,
                                                       attention=attention)
                else:
                    graph_layer_class = NodeGraphLayer(base_layer_dimension, self.activation,
                                                       self.number_of_nodes, self.n_heads,
                                                       attention=attention)

            else:
                graph_layer_class = NodeEdgeGraphLayer(base_layer_dimension, self.activation,
                                                       self.number_of_nodes, self.n_heads,
                                                       attention=attention)
            if attention:
                graph_layer, att_weights = graph_layer_class(graph_layer)
                att_weights_all.append(att_weights)
            else:
                graph_layer = graph_layer_class(graph_layer)
        if attention:
            return graph_layer, att_weights_all
        else:
            return graph_layer

    def get_graph_info(self, graph_layer):
        number_of_nodes = self.number_of_nodes
        messages, edges = graph_layer
        number_of_edge_features = messages.shape[-1]

        def aggregate(_, x):
            message, edge = x
            merged_edges = tf.math.unsorted_segment_sum(message, edge[:, 1], number_of_nodes, )
            # Probar a concatenar en ved de sumar según el paper 2021 Generalizable Machine Learning in Neuroscience using GNN
            return merged_edges

        aggregated = tf.scan(aggregate, (messages, edges),
                             initializer=tf.zeros((number_of_nodes, number_of_edge_features)), )
        updated_nodes = Dense(1)(aggregated)

        return updated_nodes, messages, edges

    def edge_attention_block_2(self, edge_layer, cls_layer, layer_name='', base_layer_dimension=None):
        if base_layer_dimension:
            dimensions = []
            for n in self.base_layer_dimensions:
                dimensions.append(base_layer_dimension)
        else:
            base_layer_dimension = self.base_layer_dimensions[0]
            dimensions = self.base_layer_dimensions
        cls_layer = tf.expand_dims(cls_layer, axis=1)
        for n, base_layer_dimension in enumerate(dimensions):
            cls_layer = Dense(
                base_layer_dimension,
                activation=self.activation,
                name="cls_mlp_edge_attention_block" + str(n))(cls_layer)
            edge_layer = Dense(
                base_layer_dimension,
                activation=self.activation,
                name="cls_edge_edge_attention_block" + str(n))(edge_layer)
            # cls_layer = tf.keras.layers.LayerNormalization()(cls_layer)
            attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=self.n_heads, key_dim=2)
            updated_cls_layer, att_weights = attention_layer(cls_layer, edge_layer, return_attention_scores=True)
            # attention_layer = MultiHeadGlobalGatedAttention(return_attention_weights=True, number_of_heads=self.n_heads)
            # updated_cls_layer, att_weights = attention_layer(edge_layer, cls_layer)
            # updated_cls_layer = tfk.layers.Dropout(0.3)(updated_cls_layer)
            updated_cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer + cls_layer)
            feed_forward = tf.keras.Sequential(
                [
                    tfk.layers.Dense(self.base_layer_dimensions[0], activation=self.activation),
                    tfk.layers.Dropout(0.1),
                    tfk.layers.Dense(self.base_layer_dimensions[0]),
                    tfk.layers.Dropout(0.1),
                ]
            )
            updated_cls_layer_y = feed_forward(updated_cls_layer)
            cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer_y + updated_cls_layer)
            # cls_layer = tf.reduce_mean(cls_layer, axis=1)
            # cls_layer = tf.keras.layers.GlobalMaxPool1D()(cls_layer)
            # cls_layer = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(cls_layer)

        return cls_layer, att_weights

    def edge_attention_block(self, edge_layer, cls_layer):
        base_layer_dimension = self.base_layer_dimensions[0]
        cls_layer = tf.expand_dims(cls_layer, axis=1)
        cls_layer = Dense(
            base_layer_dimension,
            activation=self.activation,
            name="cls_mlp_edge_attention_block")(cls_layer)
        edge_layer = Dense(
            base_layer_dimension,
            activation=self.activation,
            name="cls_edge_edge_attention_block")(edge_layer)
        cls_layer = tf.keras.layers.LayerNormalization()(cls_layer)
        attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=self.n_heads, key_dim=2)
        updated_cls_layer, att_weights = attention_layer(cls_layer, edge_layer, return_attention_scores=True)
        updated_cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer + cls_layer)
        feed_forward = tf.keras.Sequential(
            [
                tfk.layers.Dense(base_layer_dimension, activation=self.activation),
                Dropout(0.5),
                tfk.layers.Dense(base_layer_dimension),
                Dropout(0.5)
            ]
        )
        updated_cls_layer_y = feed_forward(updated_cls_layer)
        cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer_y + updated_cls_layer)
        cls_layer = tf.reduce_mean(cls_layer, axis=1)

        return cls_layer, att_weights

    def backbone(self, graph_layer, global_layer=None):
        edge_features, _ = graph_layer
        graph_layer = self.get_graph_info(graph_layer)
        # att_weights_nodes = tf.reduce_mean(att_weights_all[0], axis=1)
        # att_weights_nodes = tf.squeeze(tf.gather(att_weights_nodes, [0], axis=1), axis=1)

        # Decoder for node and edge features
        node_layer, edge_layer, _ = graph_layer
        # edge_output = self.readout_block_edges(edge_layer)
        # node_output = self.readout_block_nodes(node_layer)

        edge_output = Dense(
            1,
            name="edge_features_prediction",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5)
        )(edge_layer)
        node_output = Dense(
            1,
            name="node_features_prediction",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5)
        )(node_layer)
        edge_output = Flatten()(edge_output)
        node_output = Flatten()(node_output)
        if global_layer == None:
            # edge_output = Dense(120)(edge_output)
            # node_output = Dense(120)(node_output)
            cls_layer = tf.concat([node_output, edge_output], axis=-1)
        else:
            global_output = Dense(
                1,
                name="global_features_prediction",
            )(global_layer)
            global_output = Flatten()(global_output)
            cls_layer = tf.concat([global_output, node_output, edge_output], axis=-1)

        cls_layer = Dropout(0.5)(cls_layer)
        # updated_cls = Dense(16, activation=self.activation)(updated_cls)
        if self.edge_attention:
            updated_cls, att_weights = self.edge_attention_block(edge_layer, cls_layer)
            # updated_cls = Dense(16)(updated_cls)
            outputs = self.dense_to_combine(updated_cls)
            return outputs, att_weights
        else:
            updated_cls = Dense(16)(node_output)
            outputs = self.dense_to_combine(updated_cls)
            return outputs

    def network(self):
        edge_features, edges = (
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
        )
        edge_layer = edge_features

        # Encoder for node and edge features
        edge_layer, _, _ = self.encode_block(edge_layer=edge_layer)

        # Graph layer
        graph_layer = (edge_layer, edges)
        if self.edge_attention:
            outputs, att_weights = self.backbone(graph_layer)

            model = tf.keras.models.Model(
                inputs=[edge_features, edges],
                outputs=[outputs, att_weights],
            )
        else:
            outputs = self.backbone(graph_layer)

            model = tf.keras.models.Model(
                inputs=[edge_features, edges],
                outputs=[[outputs]],
            )
        return model

    def train_step(self, data):
        features, labels = data
        with tf.GradientTape() as tape:
            outputs = self.net(features, training=True)

            loss = self.compiled_loss(labels, outputs[0])

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights))
        loss_tracker.update_state(loss)
        return {
            "loss": loss_tracker.result(),
        }

    def test_step(self, data):
        features, labels = data
        outputs = self.net(features, training=False)
        loss = self.compiled_loss(labels, outputs[0])

        loss_tracker.update_state(loss)
        return {
            "loss": loss_tracker.result(),
        }

    def call(self, inputs, training=None):
        outputs = self.net(inputs, training=training)
        return outputs


class EdgeDeepBrainMultiModal(tfk.Model):
    """
        EdgeDeepBrain is a message-passing Graph Neural Network with Class Token, to estimate
        the age from brain networks based on edge properties.

        Parameters:
        -----------
        number_of_nodes: int
            Number of nodes in the graph.
        number_of_edges: int
            Number of edges in the graph.
        number_of_edge_features: int
            Number of edge features.
        dense_layer_dimensions: list of ints
            List of the number of units in each dense layer of the encoder and decoder. The
            number of layers is inferred from the length of this list.
        base_layer_dimensions: list of ints
            List of the latent dimensions of the graph blocks. The number of layers is
            inferred from the length of this list.
        activation: tensorflow keras function
            Activation function for the output node and edge layer.
        Returns:
        --------
        tf.keras.Model
            Keras model for the graph neural network.
    """

    def __init__(self,
                 number_of_nodes,
                 number_of_edges,
                 number_of_edge_features,
                 dense_layer_dimensions=(32, 64, 96),
                 base_layer_dimensions=(96, 96),
                 activation=tf.keras.activations.gelu,
                 n_heads=6,
                 edge_attention=False,
                 **kwargs):
        super(EdgeDeepBrainMultiModal, self).__init__(**kwargs)
        self.edge_attention = edge_attention
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_edges
        self.number_of_edge_features = number_of_edge_features
        self.dense_layer_dimensions = dense_layer_dimensions
        self.base_layer_dimensions = base_layer_dimensions
        self.activation = activation
        self.n_heads = n_heads
        self.combine_layer = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))
        self.readout_block_edges = tf.keras.layers.Lambda(
            lambda x: tf.math.reduce_sum(x, axis=1), name="edges_readout"
        )
        # self.readout_block_nodes = layers.Lambda(
        #     lambda x: tf.math.reduce_sum(x, axis=1), name="nodes_readout"
        # )
        # self.readout_block_global = layers.Lambda(
        #     lambda x: tf.math.reduce_sum(x, axis=1), name="global_readout"
        # )
        self.readout_block_edges = GlobalAveragePooling1D()
        self.readout_block_nodes = GlobalAveragePooling1D()
        self.readout_block_global = GlobalAveragePooling1D()
        self.combine_modalities = GlobalAveragePooling1D()
        self.net_multimodal = self.network_multimodal()

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]

    def encode_block(self, name, edge_layer=None, node_layer=None, global_layer=None):
        if edge_layer is not None:
            edge_layer = apply_encoder(self.dense_layer_dimensions, edge_layer, name="edge_ide" + name)
        if node_layer != None:
            node_layer = apply_encoder(self.dense_layer_dimensions, node_layer, name="node_ide" + name)
        if global_layer != None:
            global_layer = apply_encoder(self.dense_layer_dimensions, global_layer, name="global_ide" + name)
        return edge_layer, node_layer, global_layer

    def graph_layer_block(self, graph_layer, only_edges=True, attention=False):
        att_weights_all = []
        for base_layer_number, base_layer_dimension in zip(
                range(len(self.base_layer_dimensions)), self.base_layer_dimensions
        ):
            if len(graph_layer) == 2:
                if only_edges == True:
                    graph_layer_class = EdgeGraphLayer(base_layer_dimension, self.activation,
                                                       self.number_of_nodes, self.n_heads,
                                                       attention=attention)
                else:
                    graph_layer_class = NodeGraphLayer(base_layer_dimension, self.activation,
                                                       self.number_of_nodes, self.n_heads,
                                                       attention=attention)

            else:
                graph_layer_class = NodeEdgeGraphLayer(base_layer_dimension, self.activation,
                                                       self.number_of_nodes, self.n_heads,
                                                       attention=attention)
            if attention:
                graph_layer, att_weights = graph_layer_class(graph_layer)
                att_weights_all.append(att_weights)
            else:
                graph_layer = graph_layer_class(graph_layer)
        if attention:
            return graph_layer, att_weights_all
        else:
            return graph_layer

    def get_graph_info(self, graph_layer):
        number_of_nodes = self.number_of_nodes
        messages, edges = graph_layer
        number_of_edge_features = messages.shape[-1]

        def aggregate(_, x):
            message, edge = x
            merged_edges = tf.math.unsorted_segment_sum(message, edge[:, 1], number_of_nodes, )
            # Probar a concatenar en ved de sumar según el paper 2021 Generalizable Machine Learning in Neuroscience using GNN
            return merged_edges

        aggregated = tf.scan(aggregate, (messages, edges),
                             initializer=tf.zeros((number_of_nodes, number_of_edge_features)), )
        updated_nodes = Dense(1)(aggregated)

        return updated_nodes, messages, edges

    def edge_attention_block_2(self, edge_layer, cls_layer, layer_name='', base_layer_dimension=None):
        if base_layer_dimension:
            dimensions = []
            for n in self.base_layer_dimensions:
                dimensions.append(base_layer_dimension)
        else:
            base_layer_dimension = self.base_layer_dimensions[0]
            dimensions = self.base_layer_dimensions
        cls_layer = tf.expand_dims(cls_layer, axis=1)
        for n, base_layer_dimension in enumerate(dimensions):
            cls_layer = Dense(
                base_layer_dimension,
                activation=self.activation,
                name="cls_mlp_edge_attention_block" + str(n))(cls_layer)
            edge_layer = Dense(
                base_layer_dimension,
                activation=self.activation,
                name="cls_edge_edge_attention_block" + str(n))(edge_layer)
            # cls_layer = tf.keras.layers.LayerNormalization()(cls_layer)
            attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=self.n_heads, key_dim=2)
            updated_cls_layer, att_weights = attention_layer(cls_layer, edge_layer, return_attention_scores=True)
            # attention_layer = MultiHeadGlobalGatedAttention(return_attention_weights=True, number_of_heads=self.n_heads)
            # updated_cls_layer, att_weights = attention_layer(edge_layer, cls_layer)
            # updated_cls_layer = tfk.layers.Dropout(0.3)(updated_cls_layer)
            updated_cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer + cls_layer)
            feed_forward = tf.keras.Sequential(
                [
                    tfk.layers.Dense(self.base_layer_dimensions[0], activation=self.activation),
                    tfk.layers.Dropout(0.1),
                    tfk.layers.Dense(self.base_layer_dimensions[0]),
                    tfk.layers.Dropout(0.1),
                ]
            )
            updated_cls_layer_y = feed_forward(updated_cls_layer)
            cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer_y + updated_cls_layer)
            # cls_layer = tf.reduce_mean(cls_layer, axis=1)
            # cls_layer = tf.keras.layers.GlobalMaxPool1D()(cls_layer)
            # cls_layer = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(cls_layer)

        return cls_layer, att_weights

    def edge_attention_block(self, name, edge_layer, cls_layer):
        base_layer_dimension = self.base_layer_dimensions[0]
        cls_layer = tf.expand_dims(cls_layer, axis=1)
        cls_layer = Dense(
            base_layer_dimension,
            activation=self.activation,
            name="cls_mlp_edge_attention_block" + name)(cls_layer)
        edge_layer = Dense(
            base_layer_dimension,
            activation=self.activation,
            name="cls_edge_edge_attention_block" + name)(edge_layer)
        cls_layer = tf.keras.layers.LayerNormalization()(cls_layer)
        attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=self.n_heads, key_dim=2)
        updated_cls_layer, att_weights = attention_layer(cls_layer, edge_layer, return_attention_scores=True)
        updated_cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer + cls_layer)
        feed_forward = tf.keras.Sequential(
            [
                tfk.layers.Dense(base_layer_dimension, activation=self.activation),
                Dropout(0.5),
                tfk.layers.Dense(base_layer_dimension),
                Dropout(0.5)
            ]
        )
        updated_cls_layer_y = feed_forward(updated_cls_layer)
        cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer_y + updated_cls_layer)
        cls_layer = tf.reduce_mean(cls_layer, axis=1)

        return cls_layer, att_weights

    def backbone(self, name, graph_layer, global_layer=None):
        edge_features, _ = graph_layer
        graph_layer = self.get_graph_info(graph_layer)
        # att_weights_nodes = tf.reduce_mean(att_weights_all[0], axis=1)
        # att_weights_nodes = tf.squeeze(tf.gather(att_weights_nodes, [0], axis=1), axis=1)

        # Decoder for node and edge features
        node_layer, edge_layer, _ = graph_layer
        # edge_output = self.readout_block_edges(edge_layer)
        # node_output = self.readout_block_nodes(node_layer)

        edge_output = Dense(
            1,
            name="edge_features_prediction" + name,
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5)
        )(edge_layer)
        node_output = Dense(
            1,
            name="node_features_prediction" + name,
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5)
        )(node_layer)
        edge_output = Flatten()(edge_output)
        node_output = Flatten()(node_output)
        if global_layer == None:
            # edge_output = Dense(120)(edge_output)
            # node_output = Dense(120)(node_output)
            cls_layer = tf.concat([node_output, edge_output], axis=-1)
        else:
            global_output = Dense(
                1,
                name="global_features_prediction" + name,
            )(global_layer)
            global_output = Flatten()(global_output)
            cls_layer = tf.concat([global_output, node_output, edge_output], axis=-1)

        cls_layer = Dropout(0.5)(cls_layer)
        # updated_cls = Dense(16, activation=self.activation)(updated_cls)
        if self.edge_attention:
            updated_cls, att_weights = self.edge_attention_block(name, edge_layer, cls_layer)
            # updated_cls = Dense(16)(updated_cls)
            outputs = tfk.layers.Dense(1, activation='sigmoid')(updated_cls)
            return outputs, att_weights
        else:
            updated_cls = Dense(16)(node_output)
            outputs = tfk.layers.Dense(1, activation='sigmoid')(updated_cls)
            return outputs

    def network(self, edges, edge_features, name):

        edge_layer = edge_features

        # Encoder for node and edge features
        edge_layer, _, _ = self.encode_block(name, edge_layer=edge_layer)

        # Graph layer
        graph_layer = (edge_layer, edges)
        if self.edge_attention:
            outputs, att_weights = self.backbone(name, graph_layer)


        else:
            outputs = self.backbone(name, graph_layer)
            att_weights = []

        return outputs, att_weights

    def network_multimodal(self):
        edge_features_con, edges_con, edge_features_sc, edges_sc, edge_features_fc, edges_fc = (
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
        )

        outputs_con, att_weights_con = self.network(edges_con, edge_features_con, name='con')
        outputs_sc, att_weights_sc = self.network(edges_sc, edge_features_sc, name='sc')
        outputs_fc, att_weights_fc = self.network(edges_fc, edge_features_fc, name='fc')

        outputs_all = [outputs_con, outputs_sc, outputs_fc]
        # output = self.combine_layer(outputs_all)
        output = tf.keras.layers.Average()(outputs_all)
        output = tf.keras.layers.LayerNormalization()(output)

        if self.edge_attention:
            model = tf.keras.models.Model(
                inputs=[edge_features_con, edges_con, edge_features_sc, edges_sc, edge_features_fc, edges_fc],
                outputs=[output, att_weights_con, att_weights_sc, att_weights_fc],
            )
        else:
            model = tf.keras.models.Model(
                inputs=[edge_features_con, edges_con, edge_features_sc, edges_sc, edge_features_fc, edges_fc],
                outputs=[output],
            )
        return model

    def train_step(self, data):
        features, labels = data
        with tf.GradientTape() as tape:
            outputs = self.net_multimodal(features, training=True)

            loss = self.compiled_loss(labels, outputs[0])

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights))
        loss_tracker.update_state(loss)
        return {
            "loss": loss_tracker.result(),
        }

    def test_step(self, data):
        features, labels = data
        outputs = self.net_multimodal(features, training=False)
        loss = self.compiled_loss(labels, outputs[0])

        loss_tracker.update_state(loss)
        return {
            "loss": loss_tracker.result(),
        }

    def call(self, inputs, training=None):
        outputs = self.net_multimodal(inputs, training=training)
        return outputs


class EdgeDeepBrainMultiModalCombined(EdgeDeepBrain):
    """
        EdgeDeepBrain is a message-passing Graph Neural Network with Class Token, to estimate
        the age from brain networks based on edge properties.

        Parameters:
        -----------
        number_of_nodes: int
            Number of nodes in the graph.
        number_of_edges: int
            Number of edges in the graph.
        number_of_edge_features: int
            Number of edge features.
        dense_layer_dimensions: list of ints
            List of the number of units in each dense layer of the encoder and decoder. The
            number of layers is inferred from the length of this list.
        base_layer_dimensions: list of ints
            List of the latent dimensions of the graph blocks. The number of layers is
            inferred from the length of this list.
        activation: tensorflow keras function
            Activation function for the output node and edge layer.
        Returns:
        --------
        tf.keras.Model
            Keras model for the graph neural network.
    """

    def __init__(self,
                 number_of_nodes,
                 number_of_edges,
                 number_of_edge_features,
                 dense_layer_dimensions=(32, 64, 96),
                 base_layer_dimensions=(96, 96),
                 activation=tf.keras.activations.gelu,
                 n_heads=6,
                 output_layer=tfk.layers.Dense(1, activation='sigmoid', name='output_layer_multimodal'),
                 edge_attention=True,
                 **kwargs):
        super(EdgeDeepBrainMultiModalCombined, self).__init__(number_of_nodes,
                         number_of_edges,
                         number_of_edge_features,
                         dense_layer_dimensions=dense_layer_dimensions,
                         base_layer_dimensions=base_layer_dimensions,
                         activation=activation,
                        output_layer=output_layer,
                         **kwargs)
        self.edge_attention = edge_attention
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_edges
        self.number_of_edge_features = number_of_edge_features
        self.dense_layer_dimensions = dense_layer_dimensions
        self.base_layer_dimensions = base_layer_dimensions
        self.activation = activation
        self.n_heads = n_heads
        self.dense_to_combine = output_layer
        self.combine_layer = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))
        self.readout_block_edges = tf.keras.layers.Lambda(
            lambda x: tf.math.reduce_sum(x, axis=1), name="edges_readout"
        )
        # self.readout_block_nodes = layers.Lambda(
        #     lambda x: tf.math.reduce_sum(x, axis=1), name="nodes_readout"
        # )
        # self.readout_block_global = layers.Lambda(
        #     lambda x: tf.math.reduce_sum(x, axis=1), name="global_readout"
        # )
        self.readout_block_edges = GlobalAveragePooling1D()
        self.readout_block_nodes = GlobalAveragePooling1D()
        self.readout_block_global = GlobalAveragePooling1D()
        self.net = self.network()

    def get_graph_info(self, graph_layer):
        number_of_nodes = self.number_of_nodes
        messages, edges, edges_mask = graph_layer
        number_of_edge_features = messages.shape[-2]

        def aggregate(_, x):
            message, edge, edge_mask = x
            merged_edges = tf.math.unsorted_segment_sum(message, edge[:, 1], number_of_nodes, )
            # Probar a concatenar en ved de sumar según el paper 2021 Generalizable Machine Learning in Neuroscience using GNN
            return merged_edges

        aggregated = tf.scan(aggregate, (messages, edges, edges_mask),
                             initializer=tf.zeros((number_of_nodes, number_of_edge_features, messages.shape[-1])))
        updated_nodes = Dense(1)(aggregated)

        return updated_nodes, messages, edges, edges_mask

    def edge_attention_block(self, edge_layer, cls_layer, edges_mask):
        base_layer_dimension = self.base_layer_dimensions[0]
        cls_layer = tf.expand_dims(cls_layer, axis=1)
        cls_layer = Dense(
            base_layer_dimension,
            activation=self.activation,
            name="cls_mlp_edge_attention_block")(cls_layer)
        edge_layer = Dense(
            base_layer_dimension,
            activation=self.activation,
            name="cls_edge_edge_attention_block")(edge_layer)
        cls_layer = tf.keras.layers.LayerNormalization()(cls_layer)

        # edge_layer_con = edge_layer[:, :, 0, :]
        # edge_layer_sc = edge_layer[:, :, 1, :]
        # edge_layer_fc = edge_layer[:, :, 2, :]
        # edges_mask_con = tf.expand_dims(edges_mask[:, :, 0], 1)
        # edges_mask_sc = tf.expand_dims(edges_mask[:, :, 1], 1)
        # edges_mask_fc = tf.expand_dims(edges_mask[:, :, 2], 1)

        edge_layer_multi = tf.reduce_mean(edge_layer, 2)
        # attention_layer_con = tf.keras.layers.MultiHeadAttention(num_heads=self.n_heads, key_dim=2)
        # updated_cls_layer_con, att_weights_con = attention_layer_con(cls_layer, edge_layer_con, attention_mask=edges_mask_con, return_attention_scores=True)
        # attention_layer_sc = tf.keras.layers.MultiHeadAttention(num_heads=self.n_heads, key_dim=2)
        # updated_cls_layer_sc, att_weights_sc = attention_layer_sc(cls_layer, edge_layer_sc, attention_mask=edges_mask_sc, return_attention_scores=True)
        # attention_layer_fc = tf.keras.layers.MultiHeadAttention(num_heads=self.n_heads, key_dim=2)
        # updated_cls_layer_fc, att_weights_fc = attention_layer_fc(cls_layer, edge_layer_fc, attention_mask=edges_mask_fc, return_attention_scores=True)
        attention_layer_multi = tf.keras.layers.MultiHeadAttention(num_heads=self.n_heads, key_dim=2)
        updated_cls_layer_multi, att_weights_multi = attention_layer_multi(cls_layer, edge_layer_multi,
                                                                           return_attention_scores=True)
        # att_weights_con = tf.expand_dims(att_weights_con, axis=-1)
        # att_weights_sc = tf.expand_dims(att_weights_sc, axis=-1)
        # att_weights_fc = tf.expand_dims(att_weights_fc, axis=-1)
        att_weights_multi = tf.expand_dims(att_weights_multi, axis=-1)
        # att_weights = tf.keras.layers.Concatenate()([att_weights_con, att_weights_sc, att_weights_fc, att_weights_multi])
        att_weights = tf.keras.layers.Concatenate()([att_weights_multi])
        updated_cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer_multi + cls_layer)

        feed_forward = tf.keras.Sequential(
            [
                tfk.layers.Dense(base_layer_dimension, activation=self.activation),
                Dropout(0.5),
                tfk.layers.Dense(base_layer_dimension),
                Dropout(0.5)
            ]
        )
        updated_cls_layer_y = feed_forward(updated_cls_layer)
        cls_layer = tf.keras.layers.LayerNormalization()(updated_cls_layer_y + updated_cls_layer)
        cls_layer = tf.reduce_mean(cls_layer, axis=1)

        return cls_layer, att_weights

    def backbone(self, graph_layer, global_layer=None):
        edge_features, _, _ = graph_layer
        graph_layer = self.get_graph_info(graph_layer)
        # att_weights_nodes = tf.reduce_mean(att_weights_all[0], axis=1)
        # att_weights_nodes = tf.squeeze(tf.gather(att_weights_nodes, [0], axis=1), axis=1)

        # Decoder for node and edge features
        node_layer, edge_layer, _, edges_mask = graph_layer
        # edge_output = self.readout_block_edges(edge_layer)
        # node_output = self.readout_block_nodes(node_layer)

        edge_output = Dense(
            1,
            name="edge_features_prediction",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5)
        )(edge_layer)
        node_output = Dense(
            1,
            name="node_features_prediction",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5)
        )(node_layer)
        edge_output = Flatten()(edge_output)
        node_output = Flatten()(node_output)
        if global_layer == None:
            # edge_output = Dense(120)(edge_output)
            # node_output = Dense(120)(node_output)
            cls_layer = tf.concat([node_output, edge_output], axis=-1)
        else:
            global_output = Dense(
                1,
                name="global_features_prediction",
            )(global_layer)
            global_output = Flatten()(global_output)
            cls_layer = tf.concat([global_output, node_output, edge_output], axis=-1)

        cls_layer = Dropout(0.5)(cls_layer)
        # updated_cls = Dense(16, activation=self.activation)(updated_cls)
        if self.edge_attention:
            updated_cls, att_weights = self.edge_attention_block(edge_layer, cls_layer, edges_mask)
            # updated_cls = Dense(16)(updated_cls)
            outputs = self.dense_to_combine(updated_cls)
            return outputs, att_weights
        else:
            updated_cls = Dense(16)(node_output)
            outputs = self.dense_to_combine(updated_cls)
            return outputs

    def network(self):
        edge_features, edges, edges_mask = (
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features), dtype=tf.int32),
        )
        edge_features = tf.expand_dims(edge_features, -1)
        edge_layer = edge_features

        # Encoder for node and edge features
        edge_layer, _, _ = self.encode_block(edge_layer=edge_layer)

        # Graph layer
        graph_layer = (edge_layer, edges, edges_mask)
        if self.edge_attention:
            outputs, att_weights = self.backbone(graph_layer)

            model = tf.keras.models.Model(
                inputs=[edge_features, edges, edges_mask],
                outputs=[outputs, att_weights],
            )
        else:
            outputs = self.backbone(graph_layer)

            model = tf.keras.models.Model(
                inputs=[edge_features, edges, edges_mask],
                outputs=[[outputs]],
            )
        return model

    def train_step(self, data):
        features, labels = data
        with tf.GradientTape() as tape:
            outputs = self.net(features, training=True)

            loss = self.compiled_loss(labels, outputs[0])

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights))
        loss_tracker.update_state(loss)
        return {
            "loss": loss_tracker.result(),
        }

    def test_step(self, data):
        features, labels = data
        outputs = self.net(features, training=False)
        loss = self.compiled_loss(labels, outputs[0])

        loss_tracker.update_state(loss)
        return {
            "loss": loss_tracker.result(),
        }

    def call(self, inputs, training=None):
        outputs = self.net(inputs, training=training)
        return outputs


class NodeEdgeDeepBrain(EdgeDeepBrain):
    """
        GlobalNodeEdgeDeepBrain is a message-passing Graph Neural Network with Class Token, to estimate
        the age from brain networks based on edge and node properties.

        Parameters:
        -----------
        number_of_nodes: int
            Number of nodes in the graph.
        number_of_edges: int
            Number of edges in the graph.
        number_of_node_features: int
            Number of node features.
        number_of_edge_features: int
            Number of edge features.
        dense_layer_dimensions: list of ints
            List of the number of units in each dense layer of the encoder and decoder. The
            number of layers is inferred from the length of this list.
        base_layer_dimensions: list of ints
            List of the latent dimensions of the graph blocks. The number of layers is
            inferred from the length of this list.
        activation: tensorflow keras function
            Activation function for the output node and edge layer.
        Returns:
        --------
        tf.keras.Model
            Keras model for the graph neural network.
    """

    def __init__(self,
                 number_of_nodes,
                 number_of_edges,
                 number_of_node_features,
                 number_of_edge_features,
                 dense_layer_dimensions=(32, 64, 96),
                 base_layer_dimensions=(96, 96),
                 activation=tf.keras.activations.gelu,
                 **kwargs):
        self.number_of_node_features = number_of_node_features
        super().__init__(number_of_nodes,
                         number_of_edges,
                         number_of_edge_features,
                         dense_layer_dimensions=dense_layer_dimensions,
                         base_layer_dimensions=base_layer_dimensions,
                         activation=activation,
                         **kwargs)

    def network(self):
        node_features, edge_features, edges = (
            tf.keras.Input(shape=(self.number_of_nodes, self.number_of_node_features)),
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
        )
        node_layer = node_features
        edge_layer = edge_features

        # Encoder for node and edge features
        edge_layer, node_layer, _ = self.encode_block(edge_layer=edge_layer, node_layer=node_layer)

        # Graph layer
        graph_layer = (
            node_layer,
            edge_layer,
            edges,
        )
        outputs, att_weights_nodes, att_weights = self.backbone(graph_layer)

        model = tf.keras.models.Model(
            inputs=[node_features, edge_features, edges],
            outputs=[outputs, att_weights_nodes, att_weights],
        )

        # model.summary()
        return model


class NodeDeepBrain(EdgeDeepBrain):
    """
        GlobalNodeEdgeDeepBrain is a message-passing Graph Neural Network with Class Token, to estimate
        the age from brain networks based on edge and node properties.

        Parameters:
        -----------
        number_of_nodes: int
            Number of nodes in the graph.
        number_of_edges: int
            Number of edges in the graph.
        number_of_node_features: int
            Number of node features.
        number_of_edge_features: int
            Number of edge features.
        dense_layer_dimensions: list of ints
            List of the number of units in each dense layer of the encoder and decoder. The
            number of layers is inferred from the length of this list.
        base_layer_dimensions: list of ints
            List of the latent dimensions of the graph blocks. The number of layers is
            inferred from the length of this list.
        activation: tensorflow keras function
            Activation function for the output node and edge layer.
        Returns:
        --------
        tf.keras.Model
            Keras model for the graph neural network.
    """

    def __init__(self,
                 number_of_nodes,
                 number_of_edges,
                 number_of_node_features,
                 number_of_edge_features,
                 dense_layer_dimensions=(32, 64, 96),
                 base_layer_dimensions=(96, 96),
                 activation=tf.keras.activations.gelu,
                 n_heads=6,
                 output_layer=False,
                 **kwargs):
        self.number_of_node_features = number_of_node_features
        super().__init__(number_of_nodes,
                         number_of_edges,
                         number_of_edge_features,
                         dense_layer_dimensions=dense_layer_dimensions,
                         base_layer_dimensions=base_layer_dimensions,
                         activation=activation,
                         n_heads=n_heads,
                         output_layer=output_layer,
                         **kwargs)

    def backbone(self, graph_layer, global_layer=None):
        graph_layer, att_weights_all = self.graph_layer_block(graph_layer, only_edges=False)
        att_weights_nodes = tf.reduce_mean(att_weights_all[0], axis=1)
        att_weights_nodes = tf.squeeze(tf.gather(att_weights_nodes, [0], axis=1), axis=1)

        # Decoder for node and edge features
        node_layer, *_ = graph_layer
        # edge_layer, node_layer = self.decoder_block(edge_layer, node_layer)

        cls_layer = Dense(
            1,
            name="node_features_prediction",
        )(node_layer)

        if global_layer == None:
            cls_layer = Flatten()(cls_layer)
        else:
            global_output = Dense(
                1,
                name="global_features_prediction",
            )(global_layer)
            global_output = Flatten()(global_output)
            cls_layer = tf.concat([global_output, cls_layer], axis=-1)

        updated_cls = Dense(16)(cls_layer)
        outputs = self.dense_to_combine(updated_cls)
        return outputs, att_weights_nodes, att_weights_nodes

    def network(self):
        node_features, edges = (
            tf.keras.Input(shape=(self.number_of_nodes, self.number_of_node_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
        )
        node_layer = node_features

        # Encoder for node and edge features
        _, node_layer, _ = self.encode_block(node_layer=node_layer)

        # Graph layer
        graph_layer = (
            node_layer,
            edges,
        )
        outputs, att_weights_nodes, att_weights = self.backbone(graph_layer)

        model = tf.keras.models.Model(
            inputs=[node_features, edges],
            outputs=[outputs, att_weights_nodes, att_weights],
        )
        return model


class GlobalEdgeDeepBrain(EdgeDeepBrain):
    """
        GlobalNodeEdgeDeepBrain is a message-passing Graph Neural Network with Class Token, to estimate
        the age from brain networks based on edge and global properties.

        Parameters:
        -----------
        number_of_nodes: int
            Number of nodes in the graph.
        number_of_edges: int
            Number of edges in the graph.
        number_of_edge_features: int
            Number of edge features.
        number_of_global_features: int
            Number of global features.
        dense_layer_dimensions: list of ints
            List of the number of units in each dense layer of the encoder and decoder. The
            number of layers is inferred from the length of this list.
        base_layer_dimensions: list of ints
            List of the latent dimensions of the graph blocks. The number of layers is
            inferred from the length of this list.
        activation: tensorflow keras function
            Activation function for the output node and edge layer.
        Returns:
        --------
        tf.keras.Model
            Keras model for the graph neural network.
    """

    def __init__(self,
                 number_of_nodes,
                 number_of_edges,
                 number_of_edge_features,
                 number_of_global_features,
                 dense_layer_dimensions=(32, 64, 96),
                 base_layer_dimensions=(96, 96),
                 activation=tf.keras.activations.gelu,
                 edge_attention=False,
                 **kwargs):
        self.number_of_global_features = number_of_global_features
        super().__init__(number_of_nodes,
                         number_of_edges,
                         number_of_edge_features,
                         dense_layer_dimensions=dense_layer_dimensions,
                         base_layer_dimensions=base_layer_dimensions,
                         activation=activation,
                         edge_attention=edge_attention,
                         **kwargs)

    def network(self):
        global_features, edge_features, edges = (
            tf.keras.Input(shape=(1, self.number_of_global_features)),
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
        )
        edge_layer = edge_features
        global_layer = global_features

        # Encoder for node and edge features
        edge_layer, node_layer, global_layer = self.encode_block(edge_layer=edge_layer,
                                                                 node_layer=None,
                                                                 global_layer=global_layer)

        # Graph layer
        graph_layer = (edge_layer, edges)
        if self.edge_attention:
            outputs, att_weights = self.backbone(graph_layer, global_layer)

            model = tf.keras.models.Model(
                inputs=[global_features, edge_features, edges],
                outputs=[outputs, att_weights],
            )
        else:
            outputs = self.backbone(graph_layer, global_layer)

            model = tf.keras.models.Model(
                inputs=[global_features, edge_features, edges],
                outputs=[[outputs]],
            )
        return model


class GlobalNodeEdgeDeepBrain(EdgeDeepBrain):
    """
        GlobalNodeEdgeDeepBrain is a message-passing Graph Neural Network with Class Token, to estimate
        the age from brain networks based on edge, node and global properties.

        Parameters:
        -----------
        number_of_nodes: int
            Number of nodes in the graph.
        number_of_edges: int
            Number of edges in the graph.
        number_of_node_features: int
            Number of node features.
        number_of_edge_features: int
            Number of edge features.
        number_of_global_features: int
            Number of global features.
        dense_layer_dimensions: list of ints
            List of the number of units in each dense layer of the encoder and decoder. The
            number of layers is inferred from the length of this list.
        base_layer_dimensions: list of ints
            List of the latent dimensions of the graph blocks. The number of layers is
            inferred from the length of this list.
        activation: tensorflow keras function
            Activation function for the output node and edge layer.
        Returns:
        --------
        tf.keras.Model
            Keras model for the graph neural network.
    """

    def __init__(self,
                 number_of_nodes,
                 number_of_edges,
                 number_of_node_features,
                 number_of_edge_features,
                 number_of_global_features,
                 dense_layer_dimensions=(32, 64, 96),
                 base_layer_dimensions=(96, 96),
                 activation=tf.keras.activations.gelu,
                 **kwargs):
        self.number_of_node_features = number_of_node_features
        self.number_of_global_features = number_of_global_features
        super().__init__(number_of_nodes,
                         number_of_edges,
                         number_of_edge_features,
                         dense_layer_dimensions=dense_layer_dimensions,
                         base_layer_dimensions=base_layer_dimensions,
                         activation=activation,
                         **kwargs)

    def network(self):
        global_features, node_features, edge_features, edges = (
            tf.keras.Input(shape=(1, self.number_of_global_features)),
            tf.keras.Input(shape=(self.number_of_nodes, self.number_of_node_features)),
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
        )
        node_layer = node_features
        edge_layer = edge_features
        global_layer = global_features

        # Encoder for node and edge features
        edge_layer, node_layer, global_layer = self.encode_block(edge_layer=edge_layer,
                                                                 node_layer=node_layer,
                                                                 global_layer=global_layer)

        # Graph layer
        graph_layer = (
            node_layer,
            edge_layer,
            edges,
        )
        outputs, att_weights_nodes, att_weights = self.backbone(graph_layer, global_layer)

        model = tf.keras.models.Model(
            inputs=[global_features, node_features, edge_features, edges],
            outputs=[outputs, att_weights_nodes, att_weights],
        )
        # model.summary()
        return model


regularizer = tf.keras.regularizers.l1_l2(l1=0.1, l2=0.1)


class EdgeDeepBrainLC(EdgeDeepBrain):
    """
        EdgeDeepBrainLC is a message-passing Graph Neural Network with Class Token, to estimate
        the age from brain networks based on edge properties.

        Parameters:
        -----------
        number_of_nodes: int
            Number of nodes in the graph.
        number_of_edges: int
            Number of edges in the graph.
        number_of_edge_features: int
            Number of edge features.
        dense_layer_dimensions: list of ints
            List of the number of units in each dense layer of the encoder and decoder. The
            number of layers is inferred from the length of this list.
        base_layer_dimensions: list of ints
            List of the latent dimensions of the graph blocks. The number of layers is
            inferred from the length of this list.
        activation: tensorflow keras function
            Activation function for the output node and edge layer.
        Returns:
        --------
        tf.keras.Model
            Keras model for the graph neural network.
    """

    def __init__(self,
                 number_of_nodes,
                 number_of_edges,
                 number_of_edge_features,
                 dense_layer_dimensions=(32, 64, 96),
                 base_layer_dimensions=(96, 96),
                 activation=tf.keras.activations.gelu,
                 n_heads=6,
                 output_layer=tfk.layers.Dense(1, activation='sigmoid'),
                 **kwargs):
        super().__init__(number_of_nodes,
                         number_of_edges,
                         number_of_edge_features,
                         dense_layer_dimensions=dense_layer_dimensions,
                         base_layer_dimensions=base_layer_dimensions,
                         activation=activation,
                         n_heads=n_heads,
                         output_layer=output_layer,
                         **kwargs)

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, loss_age_tracker, loss_lc_tracker, auc_age_tracker, auc_lc_tracker]

    def graph_layer_block(self, graph_layer, only_edges=True):
        for base_layer_number, base_layer_dimension in zip(
                range(len(self.base_layer_dimensions)), self.base_layer_dimensions
        ):
            if len(graph_layer) == 2:
                graph_layer_class = EdgeGraphLayer(base_layer_dimension, self.activation,
                                                   self.number_of_nodes, self.n_heads,
                                                   attention=False)

            else:
                graph_layer_class = NodeEdgeGraphLayer(base_layer_dimension, self.activation,
                                                       self.number_of_nodes, self.n_heads,
                                                       attention=False)
            graph_layer = graph_layer_class(graph_layer)
        return graph_layer

    def get_graph_info(self, graph_layer):
        number_of_nodes = self.number_of_nodes
        messages, edges = graph_layer
        number_of_edge_features = messages.shape[-1]

        def aggregate(_, x):
            message, edge = x
            merged_edges = tf.math.unsorted_segment_sum(message, edge[:, 1], number_of_nodes, )
            return merged_edges

        aggregated = tf.scan(aggregate, (messages, edges),
                             initializer=tf.zeros((number_of_nodes, number_of_edge_features)), )
        updated_nodes = Dense(1)(aggregated)

        return updated_nodes, messages

    def backbone(self, graph_layer, global_layer=None):
        graph_layer = self.graph_layer_block(graph_layer)
        # att_weights_nodes = tf.reduce_mean(att_weights_all[0], axis=1)
        # att_weights_nodes = tf.squeeze(tf.gather(att_weights_nodes, [0], axis=1), axis=1)

        # Decoder for node and edge features
        node_layer, edge_layer, *_ = graph_layer
        # node_output = self.readout_block_nodes(node_layer)
        # edge_output = self.readout_block_edges(edge_layer)
        # node_output = self.readout_block_nodes(node_layer)

        edge_output = Dense(
            1,
            name="edge_features_prediction",
            activation=self.activation,
            kernel_regularizer=regularizer,
        )(edge_layer)
        node_output = Dense(
            1,
            name="node_features_prediction",
            activation=self.activation,
            kernel_regularizer=regularizer,
        )(node_layer)
        edge_output = Flatten()(edge_output)
        node_output = Flatten()(node_output)
        if global_layer == None:
            # edge_output = Dense(120)(edge_output)
            # node_output = Dense(120)(node_output)
            cls_layer = tf.concat([node_output, edge_output], axis=-1)
        else:
            global_output = Dense(
                1,
                name="global_features_prediction",
            )(global_layer)
            global_output = Flatten()(global_output)
            cls_layer = tf.concat([global_output, node_output, edge_output], axis=-1)

        cls_layer = Dropout(0.5)(cls_layer)
        cls_layer_age = Dense(2, kernel_regularizer=regularizer)(cls_layer)
        cls_layer_lc = Dense(2, kernel_regularizer=regularizer)(cls_layer)
        updated_cls_age, att_weights_age = self.edge_attention_block(edge_layer, cls_layer_age, layer_name='_age')
        updated_cls_lc, att_weights_lc = self.edge_attention_block(edge_layer, cls_layer_lc, layer_name='_lc')

        # updated_cls = Dense(16, activation=self.activation)(updated_cls)
        outputs_age = Dense(1, activation='sigmoid')(cls_layer_age)
        outputs_lc = Dense(1, activation='sigmoid')(updated_cls_lc)

        return outputs_age, outputs_lc, att_weights_age, att_weights_lc

    def network(self):
        edge_features, edges = (
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
        )
        edge_layer = edge_features

        # Encoder for node and edge features
        edge_layer, _, _ = self.encode_block(edge_layer=edge_layer)

        # Graph layer
        graph_layer = (edge_layer, edges)
        outputs_age, outputs_lc, att_weights_age, att_weights_lc = self.backbone(graph_layer)

        model = tf.keras.models.Model(
            inputs=[edge_features, edges],
            outputs=[[outputs_age, outputs_lc], att_weights_age, att_weights_lc],
        )
        return model

    def train_step(self, data):
        features, labels_age, labels_lc = data
        label_all = tf.stack([labels_age, labels_lc], axis=1)
        with tf.GradientTape() as tape:
            outputs = self.net(features, training=True)

            loss_age = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy()(labels_age, outputs[0][0]))
            loss_lc = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy()(labels_lc, outputs[0][1]))
            loss = loss_age + loss_lc
            # output_all = tf.stack([outputs[0][0], outputs[0][1]], axis=1)
            # loss = tf.keras.losses.CategoricalHinge()(label_all, output_all)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights))

        loss_tracker.update_state(loss)
        loss_age_tracker.update_state(loss_age)
        loss_lc_tracker.update_state(loss_lc)
        auc_age_tracker.update_state(labels_age, outputs[0][0])
        auc_lc_tracker.update_state(labels_lc, outputs[0][1])

        return {
            "loss": loss_tracker.result(),
            "loss_age": loss_age_tracker.result(),
            "loss_lc": loss_lc_tracker.result(),
            "auc_age": auc_age_tracker.result(),
            "auc_lc": auc_lc_tracker.result(),

        }

    def test_step(self, data):
        features, labels_age, labels_lc = data
        label_all = tf.stack([labels_age, labels_lc], axis=1)
        outputs = self.net(features, training=False)
        loss_age = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy()(labels_age, outputs[0][0]))
        loss_lc = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy()(labels_lc, outputs[0][1]))
        loss = loss_age + loss_lc
        # output_all = tf.stack([outputs[0][0], outputs[0][1]], axis=1)
        # loss = tf.keras.losses.CategoricalHinge()(label_all, output_all)

        loss_tracker.update_state(loss)
        loss_age_tracker.update_state(loss_age)
        loss_lc_tracker.update_state(loss_lc)
        auc_age_tracker.update_state(labels_age, outputs[0][0])
        auc_lc_tracker.update_state(labels_lc, outputs[0][1])

        return {
            "loss": loss_tracker.result(),
            "loss_age": loss_age_tracker.result(),
            "loss_lc": loss_lc_tracker.result(),
            "auc_age": auc_age_tracker.result(),
            "auc_lc": auc_lc_tracker.result(),

        }

    def call(self, inputs, training=None):
        outputs = self.net(inputs, training=training)
        return outputs


class EdgeDeepBrainLC_reg(EdgeDeepBrain):
    """
        EdgeDeepBrainLC is a message-passing Graph Neural Network with Class Token, to estimate
        the age from brain networks based on edge properties.

        Parameters:
        -----------
        number_of_nodes: int
            Number of nodes in the graph.
        number_of_edges: int
            Number of edges in the graph.
        number_of_edge_features: int
            Number of edge features.
        dense_layer_dimensions: list of ints
            List of the number of units in each dense layer of the encoder and decoder. The
            number of layers is inferred from the length of this list.
        base_layer_dimensions: list of ints
            List of the latent dimensions of the graph blocks. The number of layers is
            inferred from the length of this list.
        activation: tensorflow keras function
            Activation function for the output node and edge layer.
        Returns:
        --------
        tf.keras.Model
            Keras model for the graph neural network.
    """

    def __init__(self,
                 number_of_nodes,
                 number_of_edges,
                 number_of_edge_features,
                 dense_layer_dimensions=(32, 64, 96),
                 base_layer_dimensions=(96, 96),
                 activation=tf.keras.activations.gelu,
                 n_heads=6,
                 output_layer=tfk.layers.Dense(1, activation='sigmoid'),
                 **kwargs):
        super().__init__(number_of_nodes,
                         number_of_edges,
                         number_of_edge_features,
                         dense_layer_dimensions=dense_layer_dimensions,
                         base_layer_dimensions=base_layer_dimensions,
                         activation=activation,
                         n_heads=n_heads,
                         output_layer=output_layer,
                         **kwargs)

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, loss_age_tracker, loss_lc_tracker]

    def graph_layer_block(self, graph_layer, only_edges=True):
        for base_layer_number, base_layer_dimension in zip(
                range(len(self.base_layer_dimensions)), self.base_layer_dimensions
        ):
            if len(graph_layer) == 2:
                graph_layer_class = EdgeGraphLayer(base_layer_dimension, self.activation,
                                                   self.number_of_nodes, self.n_heads,
                                                   attention=False)

            else:
                graph_layer_class = NodeEdgeGraphLayer(base_layer_dimension, self.activation,
                                                       self.number_of_nodes, self.n_heads,
                                                       attention=False)
            graph_layer = graph_layer_class(graph_layer)
        return graph_layer

    def get_graph_info(self, graph_layer):
        number_of_nodes = self.number_of_nodes
        messages, edges = graph_layer
        number_of_edge_features = messages.shape[-1]

        def aggregate(_, x):
            message, edge = x
            merged_edges = tf.math.unsorted_segment_sum(message, edge[:, 1], number_of_nodes, )
            return merged_edges

        aggregated = tf.scan(aggregate, (messages, edges),
                             initializer=tf.zeros((number_of_nodes, number_of_edge_features)), )
        updated_nodes = Dense(1)(aggregated)

        return updated_nodes, messages

    def backbone(self, graph_layer, global_layer=None):
        graph_layer = self.graph_layer_block(graph_layer)
        # att_weights_nodes = tf.reduce_mean(att_weights_all[0], axis=1)
        # att_weights_nodes = tf.squeeze(tf.gather(att_weights_nodes, [0], axis=1), axis=1)

        # Decoder for node and edge features
        node_layer, edge_layer, *_ = graph_layer
        # node_output = self.readout_block_nodes(node_layer)
        # edge_output = self.readout_block_edges(edge_layer)
        # node_output = self.readout_block_nodes(node_layer)

        edge_output = Dense(
            36,
            name="edge_features_prediction",
            activation=self.activation,
            kernel_regularizer=regularizer,
        )(edge_layer)
        node_output = Dense(
            36,
            name="node_features_prediction",
            activation=self.activation,
            kernel_regularizer=regularizer,
        )(node_layer)
        edge_output = Flatten()(edge_output)
        node_output = Flatten()(node_output)
        if global_layer == None:
            # edge_output = Dense(120)(edge_output)
            # node_output = Dense(120)(node_output)
            cls_layer = tf.concat([node_output, edge_output], axis=-1)
        else:
            global_output = Dense(
                1,
                name="global_features_prediction",
            )(global_layer)
            global_output = Flatten()(global_output)
            cls_layer = tf.concat([global_output, node_output, edge_output], axis=-1)

        # cls_layer = Dropout(0.1)(cls_layer)
        cls_layer_age = Dense(64, kernel_regularizer=regularizer)(cls_layer)
        cls_layer_lc = Dense(36, kernel_regularizer=regularizer)(cls_layer)
        updated_cls_age, att_weights_age = self.edge_attention_block(edge_layer, cls_layer_age, layer_name='_age')
        updated_cls_lc, att_weights_lc = self.edge_attention_block(edge_layer, cls_layer_lc, layer_name='_lc')
        # updated_cls = Dense(16, activation=self.activation)(updated_cls)
        outputs_age = Dense(1, activation='sigmoid')(updated_cls_age)
        outputs_lc = Dense(1, activation='sigmoid')(updated_cls_lc)

        return outputs_age, outputs_lc, att_weights_age, att_weights_lc

    def network(self):
        edge_features, edges = (
            tf.keras.Input(shape=(self.number_of_edges, self.number_of_edge_features)),
            tf.keras.Input(shape=(self.number_of_edges, 2), dtype=tf.int32),
        )
        edge_layer = edge_features

        # Encoder for node and edge features
        edge_layer, _, _ = self.encode_block(edge_layer=edge_layer)

        # Graph layer
        graph_layer = (edge_layer, edges)
        outputs_age, outputs_lc, att_weights_age, att_weights_lc = self.backbone(graph_layer)

        model = tf.keras.models.Model(
            inputs=[edge_features, edges],
            outputs=[[outputs_age, outputs_lc], att_weights_age, att_weights_lc],
        )
        return model

    def train_step(self, data):
        features, labels_age, labels_lc = data
        labels_age = tf.expand_dims(labels_age, axis=1)
        labels_lc = tf.expand_dims(labels_lc, axis=1)
        sample_weight = tf.divide(((tf.divide(labels_age, 100) + 0.5) + tf.multiply(labels_lc, 10)), 2)
        with tf.GradientTape() as tape:
            outputs = self.net(features, training=True)

            loss_age = MAE(labels_age, outputs[0][0])
            loss_lc = MAE(labels_lc, outputs[0][1])
            loss = loss_age + loss_lc
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights))

        loss_tracker.update_state(loss)
        loss_age_tracker.update_state(loss_age)
        loss_lc_tracker.update_state(loss_lc)

        return {
            "loss": loss_tracker.result(),
            "loss_age": loss_age_tracker.result(),
            "loss_lc": loss_lc_tracker.result(),

        }

    def test_step(self, data):
        features, labels_age, labels_lc = data
        outputs = self.net(features, training=False)
        labels_age = tf.expand_dims(labels_age, axis=1)
        labels_lc = tf.expand_dims(labels_lc, axis=1)
        sample_weight = tf.divide(((tf.divide(labels_age, 100) + 0.5) + tf.multiply(labels_lc, 10)), 2)
        loss_age = MAE(labels_age, outputs[0][0])
        loss_lc = MAE(labels_lc, outputs[0][1])
        loss = loss_age + loss_lc
        loss_tracker.update_state(loss)
        loss_age_tracker.update_state(loss_age)
        loss_lc_tracker.update_state(loss_lc)

        return {
            "loss": loss_tracker.result(),
            "loss_age": loss_age_tracker.result(),
            "loss_lc": loss_lc_tracker.result(),

        }

    def call(self, inputs, training=None):
        outputs = self.net(inputs, training=training)
        return outputs
