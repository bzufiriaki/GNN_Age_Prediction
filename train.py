import os.path

import matplotlib.pyplot as plt
import numpy as np

from utils import *
from datasets.utils import *
from models.models import *
from tensorflow.keras import callbacks


# from clr import *


def split_train_val_test_mask(train_index, test_index, edges, edge_features, edge_mask, ages, filenames):
    if train_index is not None and test_index is not None:
        edges_train, edges_test = edges[train_index], edges[test_index]
        edge_features_train, edge_features_test = edge_features[train_index], edge_features[test_index]
        edge_mask_train, edge_mask_test = edge_mask[train_index], edge_mask[test_index]

        ages_train, ages_test = ages[train_index], ages[test_index]
        filenames_train, filenames_test = filenames[train_index], filenames[test_index]
    else:
        if TrainConfig.split == "stratified":
            edges_train, edges_test, \
                edge_features_train, edge_features_test, \
                ages_train, ages_test, \
                filenames_train, filenames_test = split_stratified(edges, edge_features,
                                                                   ages, filenames)

        else:
            edges_train, edges_test, \
                edge_features_train, edge_features_test, \
                edge_mask_train, edge_mask_test, \
                ages_train, ages_test, \
                filenames_train, filenames_test = train_test_split(edges, edge_features, edge_mask, ages,
                                                                   filenames, test_size=0.1, random_state=42)

    if TrainConfig.split == "stratified":
        edges_train, edges_val, \
            edge_features_train, edge_features_val, \
            ages_train, ages_val, \
            filenames_train, filenames_val = split_stratified(edges_train, edge_features_train, ages_train,
                                                              filenames_train)

    else:
        edges_train, edges_val, \
            edge_features_train, edge_features_val, \
            edge_mask_train, edge_mask_val, \
            ages_train, ages_val, \
            filenames_train, filenames_val = train_test_split(edges_train, edge_features_train,
                                                              edge_mask_train,
                                                              ages_train,
                                                              filenames_train,
                                                              test_size=0.1, random_state=False)
    return edges_train, edges_val, edges_test, edge_features_train, edge_features_val, edge_features_test, edge_mask_train, edge_mask_val, edge_mask_test, ages_train, ages_val, ages_test, filenames_train, filenames_val, filenames_test


def split_train_val_test(train_index, test_index, edges, edge_features, ages, filenames):
    if train_index is not None and test_index is not None:
        edges_train, edges_test = edges[train_index], edges[test_index]
        edge_features_train, edge_features_test = edge_features[train_index], edge_features[test_index]

        ages_train, ages_test = ages[train_index], ages[test_index]
        filenames_train, filenames_test = filenames[train_index], filenames[test_index]
    else:
        if TrainConfig.split == "stratified":
            edges_train, edges_test, \
                edge_features_train, edge_features_test, \
                ages_train, ages_test, \
                filenames_train, filenames_test = split_stratified(edges, edge_features,
                                                                   ages, filenames)

        else:
            edges_train, edges_test, \
                edge_features_train, edge_features_test, \
                ages_train, ages_test, \
                filenames_train, filenames_test = train_test_split(edges, edge_features, ages,
                                                                   filenames, test_size=0.1, random_state=42)

    if TrainConfig.split == "stratified":
        edges_train, edges_val, \
            edge_features_train, edge_features_val, \
            ages_train, ages_val, \
            filenames_train, filenames_val = split_stratified(edges_train, edge_features_train, ages_train,
                                                              filenames_train)

    else:
        edges_train, edges_val, \
            edge_features_train, edge_features_val, \
            ages_train, ages_val, \
            filenames_train, filenames_val = train_test_split(edges_train, edge_features_train,
                                                              ages_train,
                                                              filenames_train,
                                                              test_size=0.1, random_state=False)
    return edges_train, edges_val, edges_test, edge_features_train, edge_features_val, edge_features_test, ages_train, ages_val, ages_test, filenames_train, filenames_val, filenames_test

def split_train_val_test_atrophy(train_index, test_index, gm_features, ages, filenames):
    if train_index is not None and test_index is not None:
        gm_features_train, gm_features_test = gm_features[train_index], gm_features[test_index]

        ages_train, ages_test = ages[train_index], ages[test_index]
        filenames_train, filenames_test = filenames[train_index], filenames[test_index]
    else:
        if TrainConfig.split == "stratified":
                gm_features_train, gm_features_test, \
                ages_train, ages_test, \
                filenames_train, filenames_test = split_stratified_atrophy(gm_features,
                                                                   ages, filenames)

        else:
                gm_features_train, gm_features_test, \
                ages_train, ages_test, \
                filenames_train, filenames_test = train_test_split(gm_features, ages,
                                                                   filenames, test_size=0.1, random_state=42)

    if TrainConfig.split == "stratified":
        gm_features_train, gm_features_val,\
            ages_train, ages_val, \
            filenames_train, filenames_val = split_stratified_atrophy(gm_features_train, ages_train,
                                                              filenames_train)

    else:
        gm_features_train, gm_features_val,\
            ages_train, ages_val, \
            filenames_train, filenames_val = train_test_split(gm_features_train,
                                                              ages_train,
                                                              filenames_train,
                                                              test_size=0.1, random_state=False)
    return gm_features_train, gm_features_val, gm_features_test, ages_train, ages_val, ages_test, filenames_train, filenames_val, filenames_test


def train_model_cv(output,
                   atlas_coords,
                   edges,
                   edge_features,
                   ages,
                   filenames,
                   train_index=None,
                   test_index=None):
    if os.path.exists(output):
        pass
    else:
        os.mkdir(output)

    # Split data into train, val and test
    edges_train, edges_val, edges_test, \
        edge_features_train, edge_features_val, edge_features_test, \
        ages_train, ages_val, ages_test, \
        filenames_train, filenames_val, filenames_test = split_train_val_test(train_index, test_index, edges,
                                                                              edge_features, ages, filenames)

    # Normalize data
    edge_features_train, edge_features_val, edge_features_test = normalize_data(edge_features_train, edge_features_val,
                                                                                edge_features_test)

    try:
        loss = TrainConfig.loss.name
    except:
        loss = TrainConfig.loss.__name__
    if loss == 'binary_crossentropy':
        output_layer = tfk.layers.Dense(1, activation='sigmoid')
        test_model = TestModelBinary(output,
                                     filenames_test,
                                     ages_test)
    elif loss == 'categorical_crossentropy':
        output_layer = tfk.layers.Dense(ages_test.shape[1], activation='softmax')
        test_model = TestModelCategorical(output,
                                          filenames_test,
                                          ages_test)
    else:
        output_layer = tfk.layers.Dense(1)
        test_model = TestModel(output,
                               filenames_test,
                               ages_test)

    model_name = TrainConfig.model
    # To train only with edge features
    graph_train = tf.data.Dataset.from_tensor_slices(((edge_features_train, edges_train), ages_train))
    graph_test = tf.data.Dataset.from_tensor_slices(((edge_features_test, edges_test), ages_test))
    graph_val = tf.data.Dataset.from_tensor_slices(((edge_features_val, edges_val), ages_val))
    if loss == 'binary_crossentropy':
        graph_train = balance_dataset(graph_train)
    else:
        pass
    n_training = len(list(graph_train))
    n_validation = len(list(graph_val))
    batches_per_epoch = math.ceil(n_training / TrainConfig.batch_size)
    validation_steps = math.ceil(n_validation / TrainConfig.batch_size)
    graph_train = graph_train.repeat()
    if TrainConfig.augmentation:
        graph_train = augment_data(graph_train)
    else:
        pass
    graph_train = graph_train.batch(TrainConfig.batch_size)
    graph_train = graph_train.shuffle(50, reshuffle_each_iteration=True)
    graph_val = graph_val.batch(TrainConfig.batch_size)
    model = EdgeDeepBrain(
        number_of_nodes=GeneralConfig.atlas_coords,
        number_of_edges=edges_train.shape[1],
        number_of_edge_features=edge_features_train.shape[2],
        dense_layer_dimensions=TrainConfig.dense_layer_dimensions,
        base_layer_dimensions=TrainConfig.base_layer_dimensions,
        n_heads=TrainConfig.n_heads,
        output_layer=output_layer,
        edge_attention=TrainConfig.edge_attention,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TrainConfig.learning_rate),
        loss=TrainConfig.loss,
        metrics=TrainConfig.metrics,
    )

    if GeneralConfig.run == 'train':
        callbacks_model = [
            callbacks.ModelCheckpoint(os.path.join(output, 'mymodel_train.tf'), monitor=TrainConfig.monitor,
                                      save_weights_only=True,
                                      save_best_only=True, mode=TrainConfig.monitor_mode)
        ]
        hist = model.fit(graph_train, validation_data=graph_val,
                         steps_per_epoch=batches_per_epoch,
                         validation_steps=validation_steps,
                         epochs=TrainConfig.epochs,
                         verbose=1, callbacks=[callbacks_model], shuffle=True)
        plot_metrics(hist, output, metrics=TrainConfig.metrics, title='Metrics')
    else:
        pass

    model.load_weights(os.path.join(output, 'mymodel_train.tf')).expect_partial()

    prediction, metrics = test_model.test(model, graph_test)
    n_test = len(list(graph_test))
    with open(os.path.join(output, 'output.txt'), 'a') as fp:
        fp.write('number of validation subjects:' + str(n_validation) + "\n")
        fp.write('number of training subjects:' + str(n_training) + "\n")
        fp.write('number of test subjects:' + str(n_test) + "\n")

    edges_pd = pd.DataFrame(edges_test[0])
    edges_pd.to_csv(os.path.join(output, 'edges.csv'))

    if TrainConfig.edge_attention:
        prediction_reduced = pd.DataFrame(tf.reduce_mean(prediction[1], axis=0)[:, 0, :])
        edges_predicted = np.array(transform_edges_to_matrix(edges_test, prediction[1]))
        prediction_reduced.to_csv(os.path.join(output, 'edges_attention.csv'))
        attention_heatmap = average_explanations(edges_predicted, output, atlas_coords,
                                                 name='_' + TrainConfig.target)

    else:
        pass

    model.compiled_metrics.reset_state()
    model.compiled_loss.reset_state()
    model.reset_states()
    model.reset_metrics()
    model.weights.clear()

    return metrics, model


def train_model(output,
                edges,
                edge_features,
                ages,
                filenames,
                train_index=None,
                test_index=None):
    # Split data into train, val and test
    edges_train, edges_val, edges_test, \
        edge_features_train, edge_features_val, edge_features_test, \
        ages_train, ages_val, ages_test, \
        filenames_train, filenames_val, filenames_test = split_train_val_test(train_index, test_index, edges,
                                                                              edge_features, ages, filenames)

    # Normalize data
    edge_features_train, edge_features_val, edge_features_test = normalize_data(edge_features_train, edge_features_val,
                                                                                edge_features_test)

    try:
        loss = TrainConfig.loss.name
    except:
        loss = TrainConfig.loss.__name__
    if loss == 'binary_crossentropy':
        output_layer = tfk.layers.Dense(1, activation='sigmoid')
    elif loss == 'categorical_crossentropy':
        output_layer = tfk.layers.Dense(ages_test.shape[1], activation='softmax')
    else:
        output_layer = tfk.layers.Dense(1)

    model_name = TrainConfig.model
    # To train only with edge features
    graph_train = tf.data.Dataset.from_tensor_slices(((edge_features_train, edges_train), ages_train))
    graph_test = tf.data.Dataset.from_tensor_slices(((edge_features_test, edges_test), ages_test))
    graph_val = tf.data.Dataset.from_tensor_slices(((edge_features_val, edges_val), ages_val))
    if loss == 'binary_crossentropy':
        graph_train = balance_dataset(graph_train)
    else:
        pass
    n_training = len(list(graph_train))
    n_validation = len(list(graph_val))
    batches_per_epoch = math.ceil(n_training / TrainConfig.batch_size)
    validation_steps = math.ceil(n_validation / TrainConfig.batch_size)
    graph_train = graph_train.repeat()
    if TrainConfig.augmentation:
        graph_train = augment_data(graph_train)
    else:
        pass
    graph_train = graph_train.batch(TrainConfig.batch_size)
    graph_train = graph_train.shuffle(50, reshuffle_each_iteration=True)
    graph_val = graph_val.batch(TrainConfig.batch_size)
    model = EdgeDeepBrain(
        number_of_nodes=GeneralConfig.atlas_coords,
        number_of_edges=edges_train.shape[1],
        number_of_edge_features=edge_features_train.shape[2],
        dense_layer_dimensions=TrainConfig.dense_layer_dimensions,
        base_layer_dimensions=TrainConfig.base_layer_dimensions,
        n_heads=TrainConfig.n_heads,
        output_layer=output_layer,
        edge_attention=TrainConfig.edge_attention,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TrainConfig.learning_rate),
        loss=TrainConfig.loss,
        metrics=TrainConfig.metrics,
    )

    if GeneralConfig.run == 'train':
        hist = model.fit(graph_train, validation_data=graph_val,
                         steps_per_epoch=batches_per_epoch,
                         validation_steps=validation_steps,
                         epochs=TrainConfig.epochs,
                         verbose=0, shuffle=True)
    else:
        pass

    dataset_test_batch = graph_test.batch(TrainConfig.batch_size)
    prediction = model.predict(dataset_test_batch)
    dataset_test = list(graph_test)
    y_test_np = np.squeeze(np.array([el[1] for el in dataset_test]))
    output_np = np.squeeze(np.array(prediction[0]))

    if loss == 'binary_crossentropy':
        auc, best_threshold = calculate_ROC(y_test_np, output_np)
        acc, error_rate, sensitivity, specificity = calculate_confusion_matrix(y_test_np, output_np,
                                                                               th=best_threshold)
        metrics = [auc, acc, sensitivity, specificity]
    else:
        r2_s = (r2_score(y_test_np, output_np))
        mae = mean_absolute_error(y_test_np, output_np)
        mse = mean_squared_error(y_test_np, output_np)
        r = scipy.stats.pearsonr(y_test_np, output_np)
        metrics = [r2_s, mae, mse, r.statistic]

    model.compiled_metrics.reset_state()
    model.compiled_loss.reset_state()
    model.reset_states()
    model.reset_metrics()
    model.weights.clear()

    return metrics, model, edges_test, prediction, y_test_np, filenames_test, hist

def train_model_atrophy(output,
                        gm_features,
                        ages,
                        filenames,
                        train_index=None,
                        test_index=None):
    # Split data into train, val and test
    gm_features_train, gm_features_val, gm_features_test, ages_train, ages_val, ages_test, filenames_train, filenames_val, filenames_test = split_train_val_test_atrophy(train_index, test_index,
                                                                                      gm_features, ages, filenames)

    # Normalize data
    gm_features_train, gm_features_val, gm_features_test = normalize_data(gm_features_train, gm_features_val, gm_features_test)

    try:
        loss = TrainConfigAtrophy.loss.name
    except:
        loss = TrainConfigAtrophy.loss.__name__
    if loss == 'binary_crossentropy':
        output_layer = tfk.layers.Dense(1, activation='sigmoid')
    elif loss == 'categorical_crossentropy':
        output_layer = tfk.layers.Dense(ages_test.shape[1], activation='softmax')
    else:
        output_layer = tfk.layers.Dense(1)

    # To train only with edge features
    graph_train = tf.data.Dataset.from_tensor_slices((gm_features_train, ages_train))
    graph_test = tf.data.Dataset.from_tensor_slices((gm_features_test, ages_test))
    graph_val = tf.data.Dataset.from_tensor_slices((gm_features_val, ages_val))
    if loss == 'binary_crossentropy':
        graph_train = balance_dataset(graph_train)
    else:
        pass
    n_training = len(list(graph_train))
    n_validation = len(list(graph_val))
    batches_per_epoch = math.ceil(n_training / TrainConfigAtrophy.batch_size)
    validation_steps = math.ceil(n_validation / TrainConfigAtrophy.batch_size)
    graph_train = graph_train.repeat()
    if TrainConfigAtrophy.augmentation:
        graph_train = augment_data(graph_train)
    else:
        pass
    graph_train = graph_train.batch(TrainConfigAtrophy.batch_size)
    graph_train = graph_train.shuffle(50, reshuffle_each_iteration=True)
    graph_val = graph_val.batch(TrainConfigAtrophy.batch_size)
    model = AtrophyDeepBrain(
        number_of_nodes=GeneralConfigAtrophy.atlas_coords,
        dense_layer_dimensions=TrainConfigAtrophy.dense_layer_dimensions,
        n_heads=TrainConfigAtrophy.n_heads,
        output_layer=output_layer,
        attention=TrainConfigAtrophy.attention,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TrainConfigAtrophy.learning_rate),
        loss=TrainConfigAtrophy.loss,
        metrics=TrainConfigAtrophy.metrics,
    )

    if GeneralConfig.run == 'train':
        hist = model.fit(graph_train, validation_data=graph_val,
                         steps_per_epoch=batches_per_epoch,
                         validation_steps=validation_steps,
                         epochs=TrainConfigAtrophy.epochs,
                         verbose=0, shuffle=True)
    else:
        pass

    dataset_test_batch = graph_test.batch(TrainConfigAtrophy.batch_size)
    prediction = model.predict(dataset_test_batch)
    dataset_test = list(graph_test)
    y_test_np = np.squeeze(np.array([el[1] for el in dataset_test]))
    output_np = np.squeeze(np.array(prediction[0]))

    if loss == 'binary_crossentropy':
        auc, best_threshold = calculate_ROC(y_test_np, output_np)
        acc, error_rate, sensitivity, specificity = calculate_confusion_matrix(y_test_np, output_np,
                                                                               th=best_threshold)
        metrics = [auc, acc, sensitivity, specificity]
    else:
        r2_s = (r2_score(y_test_np, output_np))
        mae = mean_absolute_error(y_test_np, output_np)
        mse = mean_squared_error(y_test_np, output_np)
        r = scipy.stats.pearsonr(y_test_np, output_np)
        metrics = [r2_s, mae, mse, r.statistic]

    model.compiled_metrics.reset_state()
    model.compiled_loss.reset_state()
    model.reset_states()
    model.reset_metrics()
    model.weights.clear()

    return metrics, model, prediction, y_test_np, filenames_test, hist

def prepare_data_for_training_multimodal(ages_train, ages_val, ages_test, edges_train_con, edges_val_con,
                                         edges_test_con, edge_features_train_con, edge_features_val_con,
                                         edge_features_test_con,
                                         edges_train_sc, edges_val_sc, edges_test_sc, edge_features_train_sc,
                                         edge_features_val_sc, edge_features_test_sc,
                                         edges_train_fc, edges_val_fc, edges_test_fc, edge_features_train_fc,
                                         edge_features_val_fc, edge_features_test_fc):
    # To train only with edge features
    graph_train = tf.data.Dataset.from_tensor_slices(((edge_features_train_con, edges_train_con, edge_features_train_sc,
                                                       edges_train_sc, edge_features_train_fc, edges_train_fc),
                                                      ages_train))
    graph_test = tf.data.Dataset.from_tensor_slices(((edge_features_test_con, edges_test_con, edge_features_test_sc,
                                                      edges_test_sc, edge_features_test_fc, edges_test_fc), ages_test))
    graph_val = tf.data.Dataset.from_tensor_slices(((edge_features_val_con, edges_val_con, edge_features_val_sc,
                                                     edges_val_sc, edge_features_val_fc, edges_val_fc), ages_val))

    n_training = len(list(graph_train))
    n_validation = len(list(graph_val))
    batches_per_epoch = math.ceil(n_training / TrainConfig.batch_size)
    validation_steps = math.ceil(n_validation / TrainConfig.batch_size)
    graph_train = graph_train.repeat()
    if TrainConfig.augmentation:
        graph_train = augment_data_multimodal(graph_train)
    else:
        pass
    graph_train = graph_train.batch(TrainConfig.batch_size)
    graph_train = graph_train.shuffle(50, reshuffle_each_iteration=True)
    graph_val = graph_val.batch(TrainConfig.batch_size)

    return graph_train, graph_val, graph_test, batches_per_epoch, validation_steps, n_training, n_validation


def train_model_multimodel_cv(output,
                              atlas_coords,
                              edges_con,
                              edge_features_con,
                              edges_sc,
                              edge_features_sc,
                              edges_fc,
                              edge_features_fc,
                              ages,
                              filenames,
                              train_index=None,
                              test_index=None):
    if os.path.exists(output):
        pass
    else:
        os.mkdir(output)

    # Split data into train, val and test
    edges_train_con, edges_val_con, edges_test_con, \
        edge_features_train_con, edge_features_val_con, edge_features_test_con, \
        ages_train, ages_val, ages_test, \
        filenames_train, filenames_val, filenames_test = split_train_val_test(train_index, test_index, edges_con,
                                                                              edge_features_con, ages, filenames)

    edges_train_sc, edges_val_sc, edges_test_sc, \
        edge_features_train_sc, edge_features_val_sc, edge_features_test_sc, \
        ages_train, ages_val, ages_test, \
        filenames_train, filenames_val, filenames_test = split_train_val_test(train_index, test_index, edges_sc,
                                                                              edge_features_sc, ages, filenames)

    edges_train_fc, edges_val_fc, edges_test_fc, \
        edge_features_train_fc, edge_features_val_fc, edge_features_test_fc, \
        ages_train, ages_val, ages_test, \
        filenames_train, filenames_val, filenames_test = split_train_val_test(train_index, test_index, edges_fc,
                                                                              edge_features_fc, ages, filenames)

    # Normalize data
    edge_features_train_con, edge_features_val_con, edge_features_test_con = normalize_data(edge_features_train_con,
                                                                                            edge_features_val_con,
                                                                                            edge_features_test_con)
    edge_features_train_sc, edge_features_val_sc, edge_features_test_sc = normalize_data(edge_features_train_sc,
                                                                                         edge_features_val_sc,
                                                                                         edge_features_test_sc)
    edge_features_train_fc, edge_features_val_fc, edge_features_test_fc = normalize_data(edge_features_train_fc,
                                                                                         edge_features_val_fc,
                                                                                         edge_features_test_fc)

    graph_train, graph_val, graph_test, batches_per_epoch, validation_steps, n_training, n_validation = prepare_data_for_training_multimodal(
        ages_train, ages_val, ages_test, edges_train_con, edges_val_con, edges_test_con, edge_features_train_con,
        edge_features_val_con, edge_features_test_con,
        edges_train_sc, edges_val_sc, edges_test_sc, edge_features_train_sc, edge_features_val_sc,
        edge_features_test_sc,
        edges_train_fc, edges_val_fc, edges_test_fc, edge_features_train_fc, edge_features_val_fc,
        edge_features_test_fc)

    try:
        loss = TrainConfig.loss.name
    except:
        loss = TrainConfig.loss.__name__
    if loss == 'binary_crossentropy':
        output_layer = tfk.layers.Dense(1, activation='sigmoid')
        test_model = TestModelBinary(output,
                                     filenames_test,
                                     ages_test)
    elif loss == 'categorical_crossentropy':
        test_model = TestModelCategorical(output,
                                          filenames_test,
                                          ages_test)
    else:
        test_model = TestModel(output,
                               filenames_test,
                               ages_test)

    model = EdgeDeepBrainMultiModal(
        number_of_nodes=GeneralConfig.atlas_coords,
        number_of_edges=edges_train_con.shape[1],
        number_of_edge_features=edge_features_train_con.shape[2],
        dense_layer_dimensions=TrainConfig.dense_layer_dimensions,
        base_layer_dimensions=TrainConfig.base_layer_dimensions,
        n_heads=TrainConfig.n_heads,
        edge_attention=TrainConfig.edge_attention,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TrainConfig.learning_rate),
        loss=TrainConfig.loss,
        metrics=TrainConfig.metrics,
    )

    if GeneralConfig.run == 'train':
        callbacks_model = [
            callbacks.ModelCheckpoint(os.path.join(output, 'mymodel_train.tf'), monitor=TrainConfig.monitor,
                                      save_weights_only=True,
                                      save_best_only=True, mode=TrainConfig.monitor_mode)
        ]
        hist = model.fit(graph_train, validation_data=graph_val,
                         steps_per_epoch=batches_per_epoch,
                         validation_steps=validation_steps,
                         epochs=TrainConfig.epochs,
                         verbose=1, callbacks=[callbacks_model], shuffle=True)
        plot_metrics(hist, output, metrics=TrainConfig.metrics, title='Metrics')
    else:
        pass

    model.load_weights(os.path.join(output, 'mymodel_train.tf')).expect_partial()

    prediction, metrics = test_model.test(model, graph_test)
    n_test = len(list(graph_test))
    with open(os.path.join(output, 'output.txt'), 'a') as fp:
        fp.write('number of validation subjects:' + str(n_validation) + "\n")
        fp.write('number of training subjects:' + str(n_training) + "\n")
        fp.write('number of test subjects:' + str(n_test) + "\n")

    edges_pd_con = pd.DataFrame(edges_test_con[0])
    edges_pd_con.to_csv(os.path.join(output, 'edges_con.csv'))

    edges_pd_sc = pd.DataFrame(edges_test_sc[0])
    edges_pd_sc.to_csv(os.path.join(output, 'edges_sc.csv'))

    edges_pd_fc = pd.DataFrame(edges_test_fc[0])
    edges_pd_fc.to_csv(os.path.join(output, 'edges_fc.csv'))

    if TrainConfig.edge_attention:
        prediction_attention_con = prediction[1]
        prediction_attention_sc = prediction[2]
        prediction_attention_fc = prediction[3]
        prediction_reduced_con = pd.DataFrame(tf.reduce_mean(prediction_attention_con, axis=0)[:, 0, :])
        prediction_reduced_sc = pd.DataFrame(tf.reduce_mean(prediction_attention_con, axis=0)[:, 0, :])
        prediction_reduced_fc = pd.DataFrame(tf.reduce_mean(prediction_attention_con, axis=0)[:, 0, :])

        edges_predicted_con = np.array(transform_edges_to_matrix(edges_test_con, prediction[1]))
        edges_predicted_sc = np.array(transform_edges_to_matrix(edges_test_sc, prediction[1]))
        edges_predicted_fc = np.array(transform_edges_to_matrix(edges_test_fc, prediction[1]))

        prediction_reduced_con.to_csv(os.path.join(output, 'edges_attention_con.csv'))
        prediction_reduced_sc.to_csv(os.path.join(output, 'edges_attention_sc.csv'))
        prediction_reduced_fc.to_csv(os.path.join(output, 'edges_attention_fc.csv'))

        attention_heatmap_con = average_explanations(edges_predicted_con, output, atlas_coords,
                                                     name='_con_' + TrainConfig.target)
        attention_heatmap_sc = average_explanations(edges_predicted_sc, output, atlas_coords,
                                                    name='_sc_' + TrainConfig.target)
        attention_heatmap_fc = average_explanations(edges_predicted_fc, output, atlas_coords,
                                                    name='_fc_' + TrainConfig.target)

    else:
        pass

    model.compiled_metrics.reset_state()
    model.compiled_loss.reset_state()
    model.reset_states()
    model.reset_metrics()
    model.weights.clear()

    return metrics, model


def train_model_multimodel_combined_cv(output,
                                       atlas_coords,
                                       edges_con,
                                       edge_features_con,
                                       edges_sc,
                                       edge_features_sc,
                                       edges_fc,
                                       edge_features_fc,
                                       ages,
                                       filenames,
                                       train_index=None,
                                       test_index=None):
    if os.path.exists(output):
        pass
    else:
        os.mkdir(output)

    edges, edge_features, edge_mask_all = combine_edges_multimodal(edges_con,
                                                                   edge_features_con,
                                                                   edges_sc,
                                                                   edge_features_sc,
                                                                   edges_fc,
                                                                   edge_features_fc)

    # Split data into train, val and test
    edges_train, edges_val, edges_test, \
        edge_features_train, edge_features_val, edge_features_test, \
        edge_mask_all_train, edge_mask_all_val, edge_mask_all_test, \
        ages_train, ages_val, ages_test, \
        filenames_train, filenames_val, filenames_test = split_train_val_test_mask(train_index, test_index, edges,
                                                                                   edge_features, edge_mask_all, ages,
                                                                                   filenames)

    # Normalize data
    edge_features_train, edge_features_val, edge_features_test = normalize_data_combined(edge_features_train,
                                                                                         edge_features_val,
                                                                                         edge_features_test)

    # To train only with edge features
    graph_train = tf.data.Dataset.from_tensor_slices(
        ((edge_features_train, edges_train, edge_mask_all_train), ages_train))
    graph_test = tf.data.Dataset.from_tensor_slices(((edge_features_test, edges_test, edge_mask_all_test), ages_test))
    graph_val = tf.data.Dataset.from_tensor_slices(((edge_features_val, edges_val, edge_mask_all_val), ages_val))

    n_training = len(list(graph_train))
    n_validation = len(list(graph_val))
    batches_per_epoch = math.ceil(n_training / TrainConfig.batch_size)
    validation_steps = math.ceil(n_validation / TrainConfig.batch_size)
    graph_train = graph_train.repeat()
    if TrainConfig.augmentation:
        graph_train = augment_data(graph_train)
    else:
        pass
    graph_train = graph_train.batch(TrainConfig.batch_size)
    graph_train = graph_train.shuffle(50, reshuffle_each_iteration=True)
    graph_val = graph_val.batch(TrainConfig.batch_size)

    try:
        loss = TrainConfig.loss.name
    except:
        loss = TrainConfig.loss.__name__
    if loss == 'binary_crossentropy':
        output_layer = tfk.layers.Dense(1, activation='sigmoid')
        test_model = TestModelBinary(output,
                                     filenames_test,
                                     ages_test)
    elif loss == 'categorical_crossentropy':
        test_model = TestModelCategorical(output,
                                          filenames_test,
                                          ages_test)
    else:
        test_model = TestModel(output,
                               filenames_test,
                               ages_test)

    output_layer = tfk.layers.Dense(1)
    model = EdgeDeepBrainMultiModalCombined(
        number_of_nodes=GeneralConfig.atlas_coords,
        number_of_edges=edges_train.shape[1],
        number_of_edge_features=edge_features_train.shape[2],
        dense_layer_dimensions=TrainConfig.dense_layer_dimensions,
        base_layer_dimensions=TrainConfig.base_layer_dimensions,
        n_heads=TrainConfig.n_heads,
        edge_attention=TrainConfig.edge_attention,
        output_layer=output_layer
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TrainConfig.learning_rate),
        loss=TrainConfig.loss,
        metrics=TrainConfig.metrics,
    )

    if GeneralConfig.run == 'train':
        callbacks_model = [
            callbacks.ModelCheckpoint(os.path.join(output, 'mymodel_train.tf'), monitor=TrainConfig.monitor,
                                      save_weights_only=True,
                                      save_best_only=True, mode=TrainConfig.monitor_mode)
        ]
        hist = model.fit(graph_train, validation_data=graph_val,
                         steps_per_epoch=batches_per_epoch,
                         validation_steps=validation_steps,
                         epochs=TrainConfig.epochs,
                         verbose=1, callbacks=[callbacks_model], shuffle=True)
        plot_metrics(hist, output, metrics=TrainConfig.metrics, title='Metrics')
    else:
        pass

    model.load_weights(os.path.join(output, 'mymodel_train.tf')).expect_partial()

    prediction, metrics = test_model.test(model, graph_test)
    n_test = len(list(graph_test))
    with open(os.path.join(output, 'output.txt'), 'a') as fp:
        fp.write('number of validation subjects:' + str(n_validation) + "\n")
        fp.write('number of training subjects:' + str(n_training) + "\n")
        fp.write('number of test subjects:' + str(n_test) + "\n")

    edges_pd_con = pd.DataFrame(edges_test[0])
    edges_pd_con.to_csv(os.path.join(output, 'edges_all.csv'))

    if TrainConfig.edge_attention:
        prediction_attention = prediction[1]
        # prediction_reduced_con = pd.DataFrame(tf.reduce_mean(prediction_attention[:, :, :, :, 1], axis=0)[:, 0, :])
        # prediction_reduced_sc = pd.DataFrame(tf.reduce_mean(prediction_attention[:, :, :, :, 2], axis=0)[:, 0, :])
        # prediction_reduced_fc = pd.DataFrame(tf.reduce_mean(prediction_attention[:, :, :, :, 3], axis=0)[:, 0, :])
        prediction_reduced_multi = pd.DataFrame(tf.reduce_mean(prediction_attention[:, :, :, :, 0], axis=0)[:, 0, :])

        # edges_predicted_con = np.array(transform_edges_to_matrix(edges_test, prediction_attention[:, :, :, :, 1]))
        # edges_predicted_sc = np.array(transform_edges_to_matrix(edges_test, prediction_attention[:, :, :, :, 2]))
        # edges_predicted_fc = np.array(transform_edges_to_matrix(edges_test, prediction_attention[:, :, :, :, 3]))
        edges_predicted_multi = np.array(transform_edges_to_matrix(edges_test, prediction_attention[:, :, :, :, 0]))

        # prediction_reduced_con.to_csv(os.path.join(output, 'edges_attention_con.csv'))
        # prediction_reduced_sc.to_csv(os.path.join(output, 'edges_attention_sc.csv'))
        # prediction_reduced_fc.to_csv(os.path.join(output, 'edges_attention_fc.csv'))
        prediction_reduced_multi.to_csv(os.path.join(output, 'edges_attention_multi.csv'))

        # attention_heatmap = average_explanations(edges_predicted_con, output, atlas_coords,
        #                                          name='_con_' + TrainConfig.target)
        # attention_heatmap = average_explanations(edges_predicted_sc, output, atlas_coords,
        #                                          name='_sc_' + TrainConfig.target)
        # attention_heatmap = average_explanations(edges_predicted_fc, output, atlas_coords,
        #                                          name='_fc_' + TrainConfig.target)
        attention_heatmap = average_explanations(edges_predicted_multi, output, atlas_coords,
                                                 name='_multi_' + TrainConfig.target)

    else:
        pass

    model.compiled_metrics.reset_state()
    model.compiled_loss.reset_state()
    model.reset_states()
    model.reset_metrics()
    model.weights.clear()

    return metrics, model


def train_model_multimodal(edges_con,
                           edge_features_con,
                           edges_sc,
                           edge_features_sc,
                           edges_fc,
                           edge_features_fc,
                           ages,
                           filenames,
                           train_index=None,
                           test_index=None):
    # Split data into train, val and test
    edges_train_con, edges_val_con, edges_test_con, \
        edge_features_train_con, edge_features_val_con, edge_features_test_con, \
        ages_train, ages_val, ages_test, \
        filenames_train, filenames_val, filenames_test = split_train_val_test(train_index, test_index, edges_con,
                                                                              edge_features_con, ages, filenames)

    edges_train_sc, edges_val_sc, edges_test_sc, \
        edge_features_train_sc, edge_features_val_sc, edge_features_test_sc, \
        ages_train, ages_val, ages_test, \
        filenames_train, filenames_val, filenames_test = split_train_val_test(train_index, test_index, edges_sc,
                                                                              edge_features_sc, ages, filenames)

    edges_train_fc, edges_val_fc, edges_test_fc, \
        edge_features_train_fc, edge_features_val_fc, edge_features_test_fc, \
        ages_train, ages_val, ages_test, \
        filenames_train, filenames_val, filenames_test = split_train_val_test(train_index, test_index, edges_fc,
                                                                              edge_features_fc, ages, filenames)

    # Normalize data
    edge_features_train_con, edge_features_val_con, edge_features_test_con = normalize_data(edge_features_train_con,
                                                                                            edge_features_val_con,
                                                                                            edge_features_test_con)
    edge_features_train_sc, edge_features_val_sc, edge_features_test_sc = normalize_data(edge_features_train_sc,
                                                                                         edge_features_val_sc,
                                                                                         edge_features_test_sc)
    edge_features_train_fc, edge_features_val_fc, edge_features_test_fc = normalize_data(edge_features_train_fc,
                                                                                         edge_features_val_fc,
                                                                                         edge_features_test_fc)

    graph_train, graph_val, graph_test, batches_per_epoch, validation_steps, n_training, n_validation = prepare_data_for_training_multimodal(
        ages_train, ages_val, ages_test, edges_train_con, edges_val_con, edges_test_con, edge_features_train_con,
        edge_features_val_con, edge_features_test_con,
        edges_train_sc, edges_val_sc, edges_test_sc, edge_features_train_sc, edge_features_val_sc,
        edge_features_test_sc,
        edges_train_fc, edges_val_fc, edges_test_fc, edge_features_train_fc, edge_features_val_fc,
        edge_features_test_fc)

    model = EdgeDeepBrainMultiModal(
        number_of_nodes=GeneralConfig.atlas_coords,
        number_of_edges=edges_train_con.shape[1],
        number_of_edge_features=edge_features_train_con.shape[2],
        dense_layer_dimensions=TrainConfig.dense_layer_dimensions,
        base_layer_dimensions=TrainConfig.base_layer_dimensions,
        n_heads=TrainConfig.n_heads,
        edge_attention=TrainConfig.edge_attention,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TrainConfig.learning_rate),
        loss=TrainConfig.loss,
        metrics=TrainConfig.metrics,
    )

    if GeneralConfig.run == 'train':
        hist = model.fit(graph_train, validation_data=graph_val,
                         steps_per_epoch=batches_per_epoch,
                         validation_steps=validation_steps,
                         epochs=TrainConfig.epochs,
                         verbose=0, shuffle=True)
    else:
        pass

    dataset_test_batch = graph_test.batch(TrainConfig.batch_size)
    prediction = model.predict(dataset_test_batch)
    dataset_test = list(graph_test)
    y_test_np = np.squeeze(np.array([el[1] for el in dataset_test]))
    output_np = np.squeeze(np.array(prediction[0]))
    r2_s = (r2_score(y_test_np, output_np))
    mae = mean_absolute_error(y_test_np, output_np)
    mse = mean_squared_error(y_test_np, output_np)
    r = scipy.stats.pearsonr(y_test_np, output_np)
    metrics = [r2_s, mae, mse, r.statistic]

    model.compiled_metrics.reset_state()
    model.compiled_loss.reset_state()
    model.reset_states()
    model.reset_metrics()
    model.weights.clear()

    return metrics, model


def combine_edges_multimodal(edges_con,
                             edge_features_con,
                             edges_sc,
                             edge_features_sc,
                             edges_fc,
                             edge_features_fc):
    edges_all = np.concatenate([edges_con, edges_sc, edges_fc], 1)
    edges_all_arr, uniq_cnt = np.unique(edges_all[0, :, :], axis=0, return_counts=True)
    edge_features_all = np.zeros((edges_all.shape[0], len(edges_all_arr), 3))
    edge_mask_all = np.zeros((len(edges_all_arr), 3))
    for e in range(len(edges_all_arr)):
        e_selected = edges_all_arr[e, :]
        e_con = np.argwhere(np.logical_and(edges_con[0, :, 0] == e_selected[0], edges_con[0, :, 1] == e_selected[1]))
        e_sc = np.argwhere(np.logical_and(edges_sc[0, :, 0] == e_selected[0], edges_sc[0, :, 1] == e_selected[1]))
        e_fc = np.argwhere(np.logical_and(edges_fc[0, :, 0] == e_selected[0], edges_fc[0, :, 1] == e_selected[1]))
        if e_con.size == 1:
            edge_features_all[:, e, 0] = edge_features_con[:, e_con[0][0]]
            edge_mask_all[e, 0] = 1
        else:
            edge_features_all[:, e, 0] = np.zeros(edges_all.shape[0])
            edge_mask_all[e, 0] = 0
        if e_sc.size == 1:
            edge_features_all[:, e, 1] = edge_features_sc[:, e_sc[0][0]]
            edge_mask_all[e, 1] = 1
        else:
            edge_features_all[:, e, 1] = np.zeros(edges_all.shape[0])
            edge_mask_all[e, 1] = 0
        if e_fc.size == 1:
            edge_features_all[:, e, 2] = edge_features_fc[:, e_fc[0][0]]
            edge_mask_all[e, 2] = 1
        else:
            edge_features_all[:, e, 2] = np.zeros(edges_all.shape[0])
            edge_mask_all[e, 2] = 0

    edges_all_arr = np.expand_dims(edges_all_arr, 0)
    edges_all_arr = np.repeat(edges_all_arr, edges_all.shape[0], axis=0)

    edge_mask_all = np.expand_dims(edge_mask_all, 0)
    edge_mask_all = np.repeat(edge_mask_all, edges_all.shape[0], axis=0)

    return edges_all_arr, edge_features_all, edge_mask_all


def normalize_data_combined(edge_features_train, edge_features_val, edge_features_test):
    edge_features_train_con = edge_features_train[:, :, 0]
    edge_features_train_sc = edge_features_train[:, :, 1]
    edge_features_train_fc = edge_features_train[:, :, 2]

    edge_features_val_con = edge_features_val[:, :, 0]
    edge_features_val_sc = edge_features_val[:, :, 1]
    edge_features_val_fc = edge_features_val[:, :, 2]

    edge_features_test_con = edge_features_test[:, :, 0]
    edge_features_test_sc = edge_features_test[:, :, 1]
    edge_features_test_fc = edge_features_test[:, :, 2]

    edge_features_train_con, edge_features_val_con, edge_features_test_con = normalize_data(edge_features_train_con,
                                                                                            edge_features_val_con,
                                                                                            edge_features_test_con)
    edge_features_train_sc, edge_features_val_sc, edge_features_test_sc = normalize_data(edge_features_train_sc,
                                                                                         edge_features_val_sc,
                                                                                         edge_features_test_sc)
    edge_features_train_fc, edge_features_val_fc, edge_features_test_fc = normalize_data(edge_features_train_fc,
                                                                                         edge_features_val_fc,
                                                                                         edge_features_test_fc)

    edge_features_train = np.concatenate([edge_features_train_con, edge_features_train_sc, edge_features_train_fc],
                                         axis=2)
    edge_features_val = np.concatenate([edge_features_val_con, edge_features_val_sc, edge_features_val_fc], axis=2)
    edge_features_test = np.concatenate([edge_features_test_con, edge_features_test_sc, edge_features_test_fc], axis=2)

    edge_features_train, edge_features_val, edge_features_test = normalize_data(edge_features_train, edge_features_val,
                                                                                edge_features_test)
    return edge_features_train, edge_features_val, edge_features_test


def train_model_multimodal_combined(output,
                                    edges_con,
                                    edge_features_con,
                                    edges_sc,
                                    edge_features_sc,
                                    edges_fc,
                                    edge_features_fc,
                                    ages,
                                    filenames,
                                    train_index=None,
                                    test_index=None):
    edges, edge_features, edge_mask_all = combine_edges_multimodal(edges_con,
                                                                   edge_features_con,
                                                                   edges_sc,
                                                                   edge_features_sc,
                                                                   edges_fc,
                                                                   edge_features_fc)

    # Split data into train, val and test
    edges_train, edges_val, edges_test, \
        edge_features_train, edge_features_val, edge_features_test, \
        edge_mask_all_train, edge_mask_all_val, edge_mask_all_test, \
        ages_train, ages_val, ages_test, \
        filenames_train, filenames_val, filenames_test = split_train_val_test_mask(train_index, test_index, edges,
                                                                                   edge_features, edge_mask_all, ages,
                                                                                   filenames)

    # Normalize data
    edge_features_train, edge_features_val, edge_features_test = normalize_data_combined(edge_features_train,
                                                                                         edge_features_val,
                                                                                         edge_features_test)

    # To train only with edge features
    graph_train = tf.data.Dataset.from_tensor_slices(
        ((edge_features_train, edges_train, edge_mask_all_train), ages_train))
    graph_test = tf.data.Dataset.from_tensor_slices(((edge_features_test, edges_test, edge_mask_all_test), ages_test))
    graph_val = tf.data.Dataset.from_tensor_slices(((edge_features_val, edges_val, edge_mask_all_val), ages_val))

    n_training = len(list(graph_train))
    n_validation = len(list(graph_val))
    batches_per_epoch = math.ceil(n_training / TrainConfig.batch_size)
    validation_steps = math.ceil(n_validation / TrainConfig.batch_size)
    graph_train = graph_train.repeat()
    if TrainConfig.augmentation:
        graph_train = augment_data(graph_train)
    else:
        pass
    graph_train = graph_train.batch(TrainConfig.batch_size)
    graph_train = graph_train.shuffle(50, reshuffle_each_iteration=True)
    graph_val = graph_val.batch(TrainConfig.batch_size)

    try:
        loss = TrainConfig.loss.name
    except:
        loss = TrainConfig.loss.__name__
    if loss == 'binary_crossentropy':
        output_layer = tfk.layers.Dense(1, activation='sigmoid')
        test_model = TestModelBinary(output,
                                     filenames_test,
                                     ages_test)
    elif loss == 'categorical_crossentropy':
        test_model = TestModelCategorical(output,
                                          filenames_test,
                                          ages_test)
    else:
        test_model = TestModel(output,
                               filenames_test,
                               ages_test)

    if loss == 'binary_crossentropy':
        output_layer = tfk.layers.Dense(1, activation='sigmoid')
    elif loss == 'categorical_crossentropy':
        output_layer = tfk.layers.Dense(ages_test.shape[1], activation='softmax', name='output_layer_multimodal')
    else:
        output_layer = tfk.layers.Dense(1, name='output_layer_multimodal')

    model = EdgeDeepBrainMultiModalCombined(
        number_of_nodes=GeneralConfig.atlas_coords,
        number_of_edges=edges_train.shape[1],
        number_of_edge_features=edge_features_train.shape[2],
        dense_layer_dimensions=TrainConfig.dense_layer_dimensions,
        base_layer_dimensions=TrainConfig.base_layer_dimensions,
        n_heads=TrainConfig.n_heads,
        edge_attention=TrainConfig.edge_attention,
        output_layer=output_layer
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TrainConfig.learning_rate),
        loss=TrainConfig.loss,
        metrics=TrainConfig.metrics,
    )

    if GeneralConfig.run == 'train':
        hist = model.fit(graph_train, validation_data=graph_val,
                         steps_per_epoch=batches_per_epoch,
                         validation_steps=validation_steps,
                         epochs=TrainConfig.epochs,
                         verbose=0, shuffle=True)
    else:
        pass

    dataset_test_batch = graph_test.batch(TrainConfig.batch_size)
    prediction = model.predict(dataset_test_batch)
    dataset_test = list(graph_test)
    y_test_np = np.squeeze(np.array([el[1] for el in dataset_test]))
    output_np = np.squeeze(np.array(prediction[0]))

    if loss == 'binary_crossentropy':
        auc, best_threshold = calculate_ROC(y_test_np, output_np)
        acc, error_rate, sensitivity, specificity = calculate_confusion_matrix(y_test_np, output_np,
                                                                                        th=best_threshold)
        metrics = [auc, acc, sensitivity, specificity]
    else:
        r2_s = (r2_score(y_test_np, output_np))
        mae = mean_absolute_error(y_test_np, output_np)
        mse = mean_squared_error(y_test_np, output_np)
        r = scipy.stats.pearsonr(y_test_np, output_np)
        metrics = [r2_s, mae, mse, r.statistic]

    model.compiled_metrics.reset_state()
    model.compiled_loss.reset_state()
    model.reset_states()
    model.reset_metrics()
    model.weights.clear()

    return metrics, model, edges_test, prediction, y_test_np, filenames_test, hist
