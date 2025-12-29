import os.path

import matplotlib.pyplot as plt

from utils import *
from datasets.utils import *
from models.models import *
from tensorflow.keras import callbacks
# from clr import *
from keras.utils.np_utils import to_categorical


def evaluate_model(model, output,
                   atlas_coords,
                   node_features,
                   edges,
                   edge_features,
                   global_features,
                   target,
                   extra_target,
                   ages,
                   lcs,
                   lcs_values,
                   filenames,
                   edges_features_test_out,
                   edges_test_out,
                   target_test_out,
                   filenames_out,
                    lcs_values_binary_out,
                   train_index=None,
                   test_index=None):
    if os.path.exists(output):
        pass
    else:
        os.mkdir(output)
    if train_index is not None and test_index is not None:
        node_features_train, node_features_test = node_features[train_index], node_features[test_index]
        edges_train, edges_test = edges[train_index], edges[test_index]
        edge_features_train, edge_features_test = edge_features[train_index], edge_features[test_index]
        global_features_train, global_features_test = global_features[train_index], global_features[test_index]
        ages_train, ages_test = ages[train_index], ages[test_index]
        filenames_train, filenames_test = filenames[train_index], filenames[test_index]
        target_train, target_test = target[train_index], target[test_index]
        extra_target_train, extra_target_test = extra_target[train_index], extra_target[test_index]
        lcs_values_train, lcs_values_test = lcs_values[train_index], lcs_values[test_index]
        lcs_train, lcs_test = lcs[train_index], lcs[test_index]
    else:
        if TrainConfig.split == "stratified":
            node_features_train, node_features_test, \
                edges_train, edges_test, \
                edge_features_train, edge_features_test, \
                global_features_train, global_features_test, \
                ages_train, ages_test, \
                filenames_train, filenames_test, \
                target_train, target_test, \
                extra_target_train, extra_target_test, \
                lcs_values_train, lcs_values_test, \
                lcs_train, lcs_test = split_stratified(node_features, edges, edge_features, global_features, target,
                                                       extra_target, ages, lcs, lcs_values, filenames)

        else:
            node_features_train, node_features_test, \
                edges_train, edges_test, \
                edge_features_train, edge_features_test, \
                global_features_train, global_features_test, \
                target_train, target_test, \
                extra_target_train, extra_target_test, \
                ages_train, ages_test, \
                filenames_train, filenames_test, \
                lcs_values_train, lcs_values_test, \
                lcs_train, lcs_test \
                = train_test_split(node_features, edges, edge_features, global_features, target, extra_target, ages,
                                   filenames, lcs_values, lcs,
                                   test_size=0.1, random_state=42)

    if TrainConfig.split == "stratified":
        node_features_train, node_features_val, \
            edges_train, edges_val, \
            edge_features_train, edge_features_val, \
            global_features_train, global_features_val, \
            ages_train, ages_val, \
            filenames_train, filenames_val, \
            target_train, target_val, \
            extra_target_train, extra_target_val, \
            lcs_values_train, lcs_values_val_val, \
            lcs_train, lcs_val = split_stratified(node_features_train, edges_train, edge_features_train,
                                                  global_features_train,
                                                  target_train,
                                                  extra_target_train, ages_train, lcs_train, lcs_values_train,
                                                  filenames_train)

        plt.hist(np.array([lcs_train, lcs_val, lcs_test]), bins=len(np.unique(lcs_train)), histtype='bar',
                 label=['train', 'val', 'test'])
        plt.title('Datasets distribution according to LC value')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(output, 'data_distribution.png'))
        plt.close()
        plt.cla()
        plt.clf()
    else:
        node_features_train, node_features_val, \
            edges_train, edges_val, \
            edge_features_train, edge_features_val, \
            global_features_train, global_features_val, \
            ages_train, ages_val, \
            filenames_train, filenames_val, \
            target_train, target_val, \
            extra_target_train, extra_target_val, \
            lcs_values_train, lcs_values_val_val, \
            lcs_train, lcs_val \
            = train_test_split(node_features_train, edges_train, edge_features_train,
                               global_features_train,
                               ages_train,
                               filenames_train,
                               target_train,
                               extra_target_train, lcs_values_train, lcs_train,
                               test_size=0.1, random_state=False)


    edge_features_train, edge_features_val, edge_features_test = normalize_data(edge_features_train, edge_features_val,
                                                                                edges_features_test_out)


    test_model = TestModel(output,
                           filenames_out,
                           lc_values=lcs_values_binary_out)


    target_train, target_test_out, target_val = normalize_MinMaxScaler(target_train,
                                                                   target_test_out,
                                                                   target_val,
                                                                   expand_dim=False, range=(0, 100))


    # To train only with edge features
    graph_test = tf.data.Dataset.from_tensor_slices(((edge_features_test, edges_test_out), target_test_out))


    prediction, metrics = test_model.test(model, graph_test)
    n_test = len(list(graph_test))
    with open(os.path.join(output, 'output.txt'), 'a') as fp:
        fp.write('number of test subjects:' + str(n_test) + "\n")

    edges_pd = pd.DataFrame(edges_test[0])
    edges_pd.to_csv(os.path.join(output, 'edges.csv'))


    prediction_3 = pd.DataFrame(tf.reduce_mean(prediction[1], axis=0)[:, 0, :])
    edges_predicted = np.array(transform_edges_to_matrix(edges_test, prediction[1]))
    prediction_3.to_csv(os.path.join(output, 'edges_attention.csv'))
    attention_heatmap = average_explanations(edges_predicted, output, atlas_coords, name='_' + TrainConfig.target)


    return metrics
