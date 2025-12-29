""" Standardized layers implemented in keras.
"""

import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention layer.
    Parameters
    ----------
    number_of_heads : int
        Number of attention heads.
    use_bias : bool
        Whether to use bias in attention layer.
    return_attention_weights : bool
        Whether to return the attention weights for visualization.
    clip_scores_by_value: tuple of float (Optional)
        Clipping values for attention scores.
    kwargs
        Other arguments for the keras.layers.Layer
    """

    def __init__(
            self,
            number_of_heads=4,
            use_bias=True,
            return_attention_weights=False,
            clip_scores_by_value: tuple = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.number_of_heads = number_of_heads
        self.use_bias = use_bias
        self.return_attention_weights = return_attention_weights
        self.clip_scores_by_value = clip_scores_by_value

    def build(self, input_shape):
        try:
            filters = input_shape[1][-1]
        except TypeError:
            filters = input_shape[-1]
        if filters % self.number_of_heads != 0:
            raise ValueError(
                f"embedding dimension = {filters} should be divisible by number of heads = {self.number_of_heads}"
            )
        self.filters = filters
        self.projection_dim = filters // self.number_of_heads

        self.query_dense = layers.Dense(filters, use_bias=self.use_bias)
        self.key_dense = layers.Dense(filters, use_bias=self.use_bias)
        self.value_dense = layers.Dense(filters, use_bias=self.use_bias)

        self.combine_dense = layers.Dense(filters, use_bias=self.use_bias)

    def compute_attention_mask(self, x, edges, batch_size=None, **kwargs):
        """
        Computes the attention mask. The mask prevents
        attention to certain positions.
        Parameters
        ----------
        edges : tf.Tensor
            The edges of the graph.
        Returns
        -------
        tf.Tensor
            The attention mask.
        """
        number_of_edges = tf.shape(edges)[1]

        batch_dims = tf.range(batch_size)
        batch_dims = tf.repeat(batch_dims, number_of_edges)
        batch_dims = tf.reshape(
            batch_dims, shape=(batch_size, number_of_edges, 1)
        )
        indices = tf.concat(
            [batch_dims, tf.zeros_like(batch_dims), edges], axis=-1
        )

        mask = tf.tensor_scatter_nd_update(
            x, indices, tf.ones((batch_size, number_of_edges))
        )

        return -10e9 * (1.0 - mask)

    def softmax(self, x, axis=-1):
        exp = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))

        if self.clip_scores_by_value:
            exp = tf.clip_by_value(exp, *self.clip_scores_by_value)

        return tf.math.divide(exp, tf.reduce_sum(exp, axis=-1, keepdims=True))

    def SingleAttention(
            self, query, key, value, gate=None, edges=None, **kwargs
    ):
        """
        Single attention layer.
        Parameters
        ----------
        query : tf.Tensor
            Query tensor.
        key : tf.Tensor
            Key tensor.
        value : tf.Tensor
            Value tensor.
        gate : tf.Tensor (optional). If provided, the attention gate is applied.
            Gate tensor.
        """
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)

        if edges is not None:
            scaled_score += self.compute_attention_mask(
                tf.zeros_like(scaled_score[:, 0:1]),
                edges,
                **kwargs,
            )

        weights = self.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)

        if gate is not None:
            output = tf.math.multiply(output, gate)

        return output, weights

    def separate_heads(self, x, batch_size):
        """
        Parameters
        ----------
        x : tf.Tensor
            Input tensor.
        batch_size : int
            Batch size.
        projection_dim : int
            Projection dimension.
        """
        x = tf.reshape(
            x, (batch_size, -1, self.number_of_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def compute_attention(self, x, **kwargs):
        """
        Parameters
        ----------
        x : tf.Tensor
            Input tensor.
        kwargs
            Other arguments to pass to SingleAttention.
        """
        if not isinstance(x, list):
            x = [x]

        x = tf.concat(x, axis=-1)
        batch_size = tf.shape(x)[0]

        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        return (
            self.SingleAttention(
                query, key, value, batch_size=batch_size, **kwargs
            ),
            batch_size,
        )

    def call(self, x, **kwargs):
        """
        Parameters
        ----------
        x : tuple of tf.Tensors
            Input tensors.
        """
        (attention, self.att_weights), batch_size = self.compute_attention(
            x, **kwargs
        )
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.filters)
        )
        output = self.combine_dense(concat_attention)

        if self.return_attention_weights:
            return output, self.att_weights
        else:
            return output


class MultiHeadGatedSelfAttention(MultiHeadSelfAttention):
    def build(self, input_shape):
        """
        Build the layer.
        """
        try:
            filters = input_shape[1][-1]
        except TypeError:
            filters = input_shape[-1]

        if filters % self.number_of_heads != 0:
            raise ValueError(
                f"embedding dimension = {filters} should be divisible by number of heads = {self.number_of_heads}"
            )
        self.filters = filters
        self.projection_dim = filters // self.number_of_heads

        self.query_dense = layers.Dense(filters, use_bias=self.use_bias)
        self.key_dense = layers.Dense(filters, use_bias=self.use_bias)
        self.value_dense = layers.Dense(filters, use_bias=self.use_bias)
        self.gate_dense = layers.Dense(
            filters, use_bias=self.use_bias, activation="sigmoid"
        )

        self.combine_dense = layers.Dense(filters)

    def compute_attention(self, x, **kwargs):
        """
        Compute attention.
        Parameters
        ----------
        x : tf.Tensor
            Input tensor.
        kwargs
            Other arguments to pass to SingleAttention.
        """
        if not isinstance(x, list):
            x = [x]

        x = tf.concat(x, axis=-1)
        batch_size = tf.shape(x)[0]

        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)
        gate = self.gate_dense(x)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        gate = self.separate_heads(gate, batch_size)

        return (
            self.SingleAttention(
                query, key, value, gate=gate, batch_size=batch_size, **kwargs
            ),
            batch_size,
        )

class NodeEdgeGraphLayer(tf.keras.layers.Layer):
    """
    Message-passing Graph Layer.

    ----------
    dense_filters : int
        Number of filters.
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    number_of_nodes: int
        Number of nodes in the graph
    edges: None or list
        Whether to use edges in the attention
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def __init__(
            self,
            dense_filters,
            activation,
            number_of_nodes,
            n_heads,
            attention=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = attention
        self.dense_filters = dense_filters
        self.activation = activation
        self.number_of_nodes = number_of_nodes
        self.combine_layer = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))
        self.dense_to_combine = layers.Dense(self.dense_filters)
        if self.attention:
            self.update_layer = MultiHeadGatedSelfAttention(return_attention_weights=True, number_of_heads=n_heads)
            self.update_norm = tf.keras.Sequential(
                [
                    layers.Activation(self.activation),
                    tf.keras.layers.LayerNormalization(),
                ]
            )
        else:
            self.update_layer = tf.keras.layers.Dense(self.dense_filters)

        self.message_layer = tf.keras.Sequential(
            [
                layers.Dense(self.dense_filters, activation=self.activation),
                tf.keras.layers.LayerNormalization()
            ]
        )

    def update_node_features(self, nodes, aggregated, learnable_embs, edges):
        Combined = self.dense_to_combine(self.combine_layer([nodes, aggregated]))
        if self.attention:
            updated_nodes, att_weights = self.update_layer(Combined, edges=edges)
            updated_nodes = self.update_norm(updated_nodes)
            return updated_nodes, att_weights
        else:
            updated_nodes = self.update_layer(Combined)
            return updated_nodes

    def call(self, inputs):
        nodes, edge_features, edges = inputs
        number_of_nodes = self.number_of_nodes
        number_of_edges = tf.shape(edges)[1]
        number_of_node_features = nodes.shape[-1]
        batch_size = tf.shape(nodes)[0]
        # Get neighbors node features, shape = (batch, nOfedges, 2, nOffeatures)
        message_inputs = tf.gather(nodes, edges, batch_dims=1)
        # Concatenate nodes features with edge features,
        # shape = (batch, nOfedges, 2*nOffeatures + nOffedgefeatures)
        messages = tf.reshape(
            message_inputs,
            (
                batch_size,
                number_of_edges,
                2 * number_of_node_features,
            ),
        )
        reshaped = tf.concat(
            [
                messages,
                edge_features,
            ],
            -1,
        )
        messages = self.message_layer(reshaped)

        # Merge repeated edges, shape = (batch, nOfedges (before augmentation), filters)
        def aggregate(_, x):
            message, edge = x
            merged_edges = tf.math.unsorted_segment_sum(
                message,
                edge[:, 1],
                number_of_nodes,
            )

            return merged_edges

        # Aggregate messages, shape = (batch, nOfnodes, filters)
        aggregated = tf.scan(
            aggregate,
            (messages, edges),
            initializer=tf.zeros((number_of_nodes, number_of_node_features)),
        )

        # Update node features, (nOfnode, filters)
        if self.attention:
            updated_nodes, att_weights = self.update_node_features(
                nodes, aggregated, None, edges
            )

            updated_nodes = tf.reshape(
                updated_nodes,
                (
                    batch_size,
                    number_of_nodes,
                    self.dense_filters,
                ),
            )
            return (
                       updated_nodes,
                       messages,
                       edges,
                   ), att_weights
        else:
            updated_nodes = self.update_node_features(
                nodes, aggregated, None, edges
            )

            return (
                updated_nodes,
                messages,
                edges,
            )


class NodeGraphLayer(tf.keras.layers.Layer):
    """
    Message-passing Graph Layer.

    ----------
    dense_filters : int
        Number of filters.
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    number_of_nodes: int
        Number of nodes in the graph
    edges: None or list
        Whether to use edges in the attention
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def __init__(
            self,
            dense_filters,
            activation,
            number_of_nodes,
            n_heads,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dense_filters = dense_filters
        self.activation = activation
        self.number_of_nodes = number_of_nodes
        self.combine_layer = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))
        self.dense_to_combine = layers.Dense(self.dense_filters)
        self.update_layer = MultiHeadGatedSelfAttention(return_attention_weights=True, number_of_heads=n_heads)
        self.update_norm = tf.keras.Sequential(
            [
                layers.Activation(self.activation),
                tf.keras.layers.LayerNormalization(),
            ]
        )
        self.message_layer = tf.keras.Sequential(
            [
                layers.Dense(self.dense_filters, activation=self.activation),
                tf.keras.layers.LayerNormalization()
            ]
        )

    def update_node_features(self, nodes, aggregated, learnable_embs, edges):
        Combined = self.dense_to_combine(self.combine_layer([nodes, aggregated]))
        updated_nodes, att_weights = self.update_layer(Combined, edges=edges)
        updated_nodes = self.update_norm(updated_nodes)
        return updated_nodes, att_weights

    def call(self, inputs):
        nodes, edges = inputs
        number_of_nodes = self.number_of_nodes
        number_of_edges = tf.shape(edges)[1]
        number_of_node_features = nodes.shape[-1]
        batch_size = tf.shape(nodes)[0]
        # Get neighbors node features, shape = (batch, nOfedges, 2, nOffeatures)
        message_inputs = tf.gather(nodes, edges, batch_dims=1)
        # Concatenate nodes features with edge features,
        # shape = (batch, nOfedges, 2*nOffeatures + nOffedgefeatures)
        messages = tf.reshape(
            message_inputs,
            (
                batch_size,
                number_of_edges,
                2 * number_of_node_features,
            ),
        )

        messages = self.message_layer(messages)

        # Merge repeated edges, shape = (batch, nOfedges (before augmentation), filters)
        def aggregate(_, x):
            message, edge = x
            merged_edges = tf.math.unsorted_segment_sum(
                message,
                edge[:, 1],
                number_of_nodes,
            )

            return merged_edges

        # Aggregate messages, shape = (batch, nOfnodes, filters)
        aggregated = tf.scan(
            aggregate,
            (messages, edges),
            initializer=tf.zeros((number_of_nodes, number_of_node_features)),
        )

        # Update node features, (nOfnode, filters)
        updated_nodes, att_weights = self.update_node_features(
            nodes, aggregated, None, edges
        )

        updated_nodes = tf.reshape(
            updated_nodes,
            (
                batch_size,
                number_of_nodes,
                self.dense_filters,
            ),
        )
        return (
                   updated_nodes,
                   edges,
               ), att_weights


class EdgeGraphLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            dense_filters,
            activation,
            number_of_nodes,
            n_heads,
            attention=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = attention
        self.dense_filters = dense_filters
        self.activation = activation
        self.number_of_nodes = number_of_nodes
        self.message_layer = tf.keras.Sequential(
            [
                layers.Dense(self.dense_filters, activation=self.activation),
                tf.keras.layers.LayerNormalization()
            ]
        )
        if attention == True:
            self.update_layer = MultiHeadGatedSelfAttention(return_attention_weights=True, number_of_heads=n_heads)
            self.update_norm = tf.keras.Sequential(
                [
                    layers.Activation(self.activation),
                    tf.keras.layers.LayerNormalization(),
                ]
            )
        else:
            self.update_layer = tf.keras.layers.Dense(self.dense_filters)


    def update_edge_features(self, aggregated, edges):
        if self.attention:
            updated_edges, att_weights = self.update_layer(aggregated, edges=edges)
            updated_edges = self.update_norm(updated_edges)
            return updated_edges, att_weights
        else:
            updated_edges = self.update_layer(aggregated)
            return updated_edges

    def call(self, inputs):
        edge_features, edges = inputs
        number_of_nodes = self.number_of_nodes
        number_of_edge_features = edge_features.shape[-1]
        batch_size = tf.shape(edge_features)[0]

        messages = self.message_layer(edge_features)

        def aggregate(_, x):
            message, edge = x
            merged_edges = tf.math.unsorted_segment_sum(
                message,
                edge[:, 1],
                number_of_nodes,
            )

            return merged_edges

        # Aggregate messages, shape = (batch, nOfnodes, filters)
        aggregated = tf.scan(
            aggregate,
            (messages, edges),
            initializer=tf.zeros((number_of_nodes, number_of_edge_features)),
        )

        # Update edge features, (nOfedge, filters)'
        if self.attention:
            updated_edges, att_weights = self.update_edge_features(aggregated, edges)
            updated_edges = tf.reshape(
                updated_edges,
                (
                    batch_size,
                    number_of_nodes,
                    self.dense_filters,
                ),
            )
            return (
                       updated_edges,
                       messages,
                       edges,
                   ), att_weights
        else:
            updated_edges = self.update_edge_features(aggregated, edges)
            return (
                updated_edges,
                messages,
                edges,
            )


class NodeEdgeGlobalMessagePassing(tf.keras.layers.Layer):
    """
    Message-passing Graph Layer.

    ----------
    dense_filters : int
        Number of filters.
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    number_of_nodes: int
        Number of nodes in the graph
    edges: None or list
        Whether to use edges in the attention
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def __init__(
            self,
            dense_filters,
            activation,
            number_of_nodes,
            n_heads,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dense_filters = dense_filters
        self.activation = activation
        self.number_of_nodes = number_of_nodes
        self.combine_layer = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))
        self.dense_to_combine = layers.Dense(self.dense_filters)
        self.update_layer = MultiHeadGatedSelfAttention(return_attention_weights=True, number_of_heads=n_heads)
        self.update_norm = tf.keras.Sequential(
            [
                layers.Activation(self.activation),
                tf.keras.layers.LayerNormalization(),
            ]
        )
        self.message_layer = tf.keras.Sequential(
            [
                layers.Dense(self.dense_filters, activation=self.activation),
                tf.keras.layers.LayerNormalization()
            ]
        )
        self.process_class_token = layers.Dense(self.dense_filters)

    def update_node_features(self, nodes, aggregated, learnable_embs, edges):
        Combined = tf.concat(
            [
                self.process_class_token(learnable_embs),
                self.dense_to_combine(self.combine_layer([nodes, aggregated])),
            ],
            axis=1,
        )
        updated_nodes, att_weights = self.update_layer(Combined, edges=edges)
        updated_nodes = self.update_norm(updated_nodes)
        return updated_nodes, att_weights

    def call(self, inputs):
        nodes, edge_features, edges = inputs
        nodes, learnable_embs = nodes[:, 1:, :], nodes[:, 0:1, :]
        number_of_nodes = self.number_of_nodes
        number_of_edges = tf.shape(edges)[1]
        number_of_node_features = nodes.shape[-1]
        batch_size = tf.shape(nodes)[0]
        # Get neighbors node features, shape = (batch, nOfedges, 2, nOffeatures)
        message_inputs = tf.gather(nodes, edges, batch_dims=1)
        # Concatenate nodes features with edge features,
        # shape = (batch, nOfedges, 2*nOffeatures + nOffedgefeatures)
        messages = tf.reshape(
            message_inputs,
            (
                batch_size,
                number_of_edges,
                2 * number_of_node_features,
            ),
        )
        reshaped = tf.concat(
            [
                messages,
                edge_features,
            ],
            -1,
        )
        messages = self.message_layer(reshaped)

        # Merge repeated edges, shape = (batch, nOfedges (before augmentation), filters)
        def aggregate(_, x):
            message, edge = x
            merged_edges = tf.math.unsorted_segment_sum(
                message,
                edge[:, 1],
                number_of_nodes,
            )

            return merged_edges

        # Aggregate messages, shape = (batch, nOfnodes, filters)
        aggregated = tf.scan(
            aggregate,
            (messages, edges),
            initializer=tf.zeros((number_of_nodes, number_of_node_features)),
        )

        # Update node features, (nOfnode, filters)
        updated_nodes, att_weights = self.update_node_features(
            nodes, aggregated, learnable_embs, edges
        )

        # updated_nodes = tf.reshape(
        #     updated_nodes,
        #     (
        #         batch_size,
        #         number_of_nodes,
        #         self.dense_filters,
        #     ),
        # )
        return (
                   updated_nodes,
                   messages,
                   edges,
               ), att_weights
