from tensorflow.keras import layers, models


def as_activation(x):
    """Converts a string or a callable to a keras activation function."""
    if x is None:
        return layers.Layer()
    elif isinstance(x, str):
        return layers.Activation(x)
    elif isinstance(x, layers.Layer):
        return x
    else:
        return layers.Layer(x)


def as_normalization(x):
    """Converts a string or a callable to a keras normalization function."""
    if x is None:
        return layers.Layer()
    elif isinstance(x, str):
        return _get_norm_by_name(x)
    elif isinstance(x, layers.Layer) or callable(x):
        return x
    else:
        return layers.Layer(x)


def single_layer_call(
    x, layer, activation, normalization, norm_kwargs, activation_first=True, **kwargs
):
    """Calls a layer with activation and normalization."""
    assert isinstance(norm_kwargs, dict), "norm_kwargs must be a dict. Got {0}".format(
        type(norm_kwargs)
    )

    n = (
        lambda x: as_normalization(normalization)(**norm_kwargs)(x)
        if normalization
        else x
    )
    a = lambda x: as_activation(activation)(x) if activation else x
    fs = [(layer, kwargs)]
    fs = fs + [(a, {}), (n, {})] if activation_first else fs + [(n, {}), (a, {})]

    return reduce(lambda x, f: f[0](x, **f[1]), fs, x)