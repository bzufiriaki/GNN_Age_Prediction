from tensorflow import keras
# Import TensorFlow:
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import itertools
import pandas as pd
# import seaborn as sns
from scipy import stats
import os
import scipy
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, auc, \
    roc_curve, roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix, \
    multilabel_confusion_matrix
from nilearn import plotting
from Config import TrainConfig, GeneralConfig
from Config_atrophy import TrainConfigAtrophy, GeneralConfigAtrophy
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler
from itertools import cycle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, VarianceThreshold, GenericUnivariateSelect, SelectPercentile
from sklearn.feature_selection import mutual_info_regression, f_regression, chi2, r_regression, \
    SelectFdr, f_classif, mutual_info_classif


# from statannot import add_stat_annotation


def stratify_division_lc(lcs):
    lcs_list = [0.05, 0.1, 0.15, 0.2]
    lcs_stratified = []
    for l in lcs:
        closest = min(lcs_list, key=lambda x: abs(x - l))
        lcs_stratified.append(closest)
    return lcs_stratified


def get_stratify_division_ages(ages):
    ages_strat = []
    for elem in ages:
        if elem == 21:
            ages_strat.append(20)
        elif elem == 88:
            ages_strat.append(87)
        else:
            ages_strat.append(int(elem))
    stratify_label = np.expand_dims(np.array(ages_strat), 1)
    return stratify_label


def get_stratify_division_class(elements):
    elements_strat = []
    for elem in elements:
        if elem == 21:
            elements_strat.append(20)
        elif elem == 88:
            aelements_strat.append(87)
        else:
            elements_strat.append(int(elem))
    stratify_label = np.expand_dims(np.array(elements_strat), 1)
    return stratify_label


def plot_metrics(hist, out, metrics=["loss", "mse"], title='Metrics'):
    plt.clf()
    plt.cla()
    if len(metrics) == 1:
        try:
            plt.plot(hist.history[metrics[0]], 'steelblue')
            plt.plot(hist.history["val_" + metrics[0]], 'lightsalmon')
            plt.ylabel(metrics[0])
            plt.legend([metrics[0], "val_" + metrics[0]])
        except:
            plt.plot(hist.history[metrics[0].name], 'steelblue')
            plt.plot(hist.history["val_" + metrics[0].name], 'lightsalmon')
            plt.ylabel(metrics[0].name)
            plt.legend([metrics[0].name, "val_" + metrics[0].name])
        plt.xlabel('Number of Training Epochs')
        plt.title(title)
        plt.savefig(os.path.join(out, title + '.png'))
        plt.close()
    else:
        f, ax = plt.subplots(nrows=1, ncols=len(metrics), figsize=(12, 4))
        for i in range(len(metrics)):
            try:
                ax[i].plot(hist.history[metrics[i]], 'steelblue')
                ax[i].plot(hist.history["val_" + metrics[i]], 'lightsalmon')
                ax[i].set_ylabel(metrics[i])
                ax[i].legend([metrics[i], "val_" + metrics[i]])
            except:
                ax[i].plot(hist.history[metrics[i].name], 'steelblue')
                ax[i].plot(hist.history["val_" + metrics[i].name], 'lightsalmon')
                ax[i].set_ylabel(metrics[i].name)
                ax[i].legend([metrics[i].name, "val_" + metrics[i].name])
            ax[i].set_xlabel('Number of Training Epochs')
        plt.title(title)
        plt.savefig(os.path.join(out, title + '.png'))
        plt.close()


def plot_predictions(y_test, output, output_folder, colors=None, name='', ):
    plt.clf()
    plt.cla()
    plt.scatter(y_test, output, c=colors)
    coef = np.polyfit(y_test, output, 1)
    coef_gt = np.polyfit(y_test, y_test, 1)
    poly1d_fn = np.poly1d(coef)
    poly1d_fn_gt = np.poly1d(coef_gt)
    plt.plot(y_test, poly1d_fn(y_test), '--k')
    plt.plot(y_test, poly1d_fn_gt(y_test), color='green', linestyle='dashed')
    plt.ylabel('Predicted')
    plt.xlabel('True')
    plt.savefig(os.path.join(output_folder, 'predictions' + name + '.png'))
    plt.close()


class TestModelBinary():
    def __init__(
            self,
            output_folder, filenames_test, age_values,
    ):
        super(TestModelBinary, self).__init__()
        self.output_folder = output_folder
        self.filenames_test = filenames_test
        self.age_values = age_values

    def calculate_prediction(self, model, dataset_test):
        dataset_test_batch = dataset_test.batch(TrainConfig.batch_size)
        prediction = model.predict(dataset_test_batch)
        dataset_test = list(dataset_test)
        y_test_np = np.squeeze(np.array([el[1] for el in dataset_test]))
        output_np = np.squeeze(np.array(prediction[0]))
        return prediction, y_test_np, output_np

    def write_output(self):
        with open(os.path.join(self.output_folder, 'output.txt'), 'w') as fp:
            fp.write("Config: " + "\n")
            fp.write("Test: \n")
            if TrainConfig.edge_features == 'fc':
                fp.write('thresholded_mean_matrix:' + str(GeneralConfig.thresholded_mean_matrix_fc) + "\n")
            elif TrainConfig.edge_features == 'sc':
                fp.write('thresholded_mean_matrix:' + str(GeneralConfig.thresholded_mean_matrix_sc) + "\n")
            elif TrainConfig.edge_features == 'con':
                fp.write('thresholded_mean_matrix:' + str(GeneralConfig.thresholded_mean_matrix_con) + "\n")
            else:
                fp.write('thresholded_mean_matrix:' + "\n")
            fp.write('epochs:' + str(TrainConfig.epochs) + "\n")
            fp.write('learning_rate:' + str(TrainConfig.learning_rate) + "\n")
            fp.write('batch_size:' + str(TrainConfig.batch_size) + "\n")
            fp.write('optimizer:' + str(TrainConfig.optimizer) + "\n")
            fp.write('loss:' + str(TrainConfig.loss) + "\n")
            fp.write('n_heads:' + str(TrainConfig.n_heads) + "\n")
            fp.write('base_layer_dimensions:' + str(TrainConfig.base_layer_dimensions) + "\n")
            fp.write('dense_layer_dimensions:' + str(TrainConfig.dense_layer_dimensions) + "\n")
            fp.write('normalization:' + str(TrainConfig.normalization) + "\n")

        fp.close()

    def test(self, model, dataset_test):
        prediction, y_test_np, output_np = self.calculate_prediction(model, dataset_test)
        #
        data = pd.DataFrame({"filenames": self.filenames_test, "gt": y_test_np, "pred": output_np})
        data.to_csv(os.path.join(self.output_folder, "predictions.csv"))

        auc, best_threshold = calculate_ROC_and_save(y_test_np, output_np, self.output_folder)
        acc, error_rate, sensitivity, specificity = calculate_confusion_matrix_and_save(y_test_np, output_np, self.output_folder,
                                                                                        th=best_threshold)
        self.write_output()
        metrics = [auc, sensitivity, specificity]
        with open(os.path.join(self.output_folder, 'output.txt'), 'a') as fp:
            fp.write('acc:' + str(acc) + "\n")
            fp.write('auc:' + str(auc) + "\n")
            fp.write('error_rate:' + str(error_rate) + "\n")
            fp.write('sensitivity:' + str(sensitivity) + "\n")
            fp.write('specificity:' + str(specificity) + "\n")
        return prediction, metrics


class TestModelCategorical():
    def __init__(
            self,
            output_folder, filenames_test, age_values,
    ):
        super(TestModelCategorical, self).__init__()
        self.output_folder = output_folder
        self.filenames_test = filenames_test
        self.age_values = age_values

    def calculate_prediction(self, model, dataset_test):
        dataset_test_batch = dataset_test.batch(TrainConfig.batch_size)
        prediction = model.predict(dataset_test_batch)
        dataset_test = list(dataset_test)
        y_test_np = np.squeeze(np.array([el[1] for el in dataset_test]))
        output_np = np.squeeze(np.array(prediction[0]))
        return prediction, y_test_np, output_np

    def write_output(self):
        with open(os.path.join(self.output_folder, 'output.txt'), 'w') as fp:
            fp.write("Config: " + "\n")
            fp.write("Test: \n")
            if TrainConfig.edge_features == 'fc':
                fp.write('thresholded_mean_matrix:' + str(GeneralConfig.thresholded_mean_matrix_fc) + "\n")
            elif TrainConfig.edge_features == 'sc':
                fp.write('thresholded_mean_matrix:' + str(GeneralConfig.thresholded_mean_matrix_sc) + "\n")
            elif TrainConfig.edge_features == 'con':
                fp.write('thresholded_mean_matrix:' + str(GeneralConfig.thresholded_mean_matrix_con) + "\n")
            else:
                fp.write('thresholded_mean_matrix:' + "\n")
            fp.write('epochs:' + str(TrainConfig.epochs) + "\n")
            fp.write('learning_rate:' + str(TrainConfig.learning_rate) + "\n")
            fp.write('batch_size:' + str(TrainConfig.batch_size) + "\n")
            fp.write('optimizer:' + str(TrainConfig.optimizer) + "\n")
            fp.write('loss:' + str(TrainConfig.loss) + "\n")
            fp.write('n_heads:' + str(TrainConfig.n_heads) + "\n")
            fp.write('base_layer_dimensions:' + str(TrainConfig.base_layer_dimensions) + "\n")
            fp.write('dense_layer_dimensions:' + str(TrainConfig.dense_layer_dimensions) + "\n")
            fp.write('normalization:' + str(TrainConfig.normalization) + "\n")

        fp.close()

    def test(self, model, dataset_test):
        prediction, y_test_np, output_np = self.calculate_prediction(model, dataset_test)
        #
        # data = pd.DataFrame({"filenames": self.filenames_test, "gt": y_test_np, "pred": output_np})
        # data.to_csv(os.path.join(self.output_folder, "predictions.csv"))

        calculate_ROC_categories(y_test_np, output_np, self.output_folder)
        ouput_np_one_hot = np.eye(output_np.shape[1])[output_np.argmax(1)]
        report = classification_report(y_test_np, ouput_np_one_hot,
                                       target_names=list(map(str, list(range(y_test_np.shape[1])))))
        acc, error_rate, sensitivity, specificity = calculate_confusion_matrix_categories(y_test_np, ouput_np_one_hot,
                                                                                          self.output_folder)

        self.write_output()
        metrics = [error_rate, sensitivity, specificity]
        with open(os.path.join(self.output_folder, 'output.txt'), 'a') as fp:
            fp.write('acc:' + str(acc) + "\n")
            fp.write('classification report:' + "\n" + report + "\n")
            fp.write('error_rate:' + str(error_rate) + "\n")
            fp.write('sensitivity:' + str(sensitivity) + "\n")
            fp.write('specificity:' + str(specificity) + "\n")
        return prediction, metrics


class TestModel(TestModelBinary):
    def __init__(self, output_folder, filenames_test, age_values):
        super().__init__(output_folder, filenames_test, age_values)

    def test(self, model, dataset_test):
        prediction, y_test_np, output_np = self.calculate_prediction(model, dataset_test)

        data = pd.DataFrame({"filenames": self.filenames_test, "gt": y_test_np, "pred": output_np})
        data.to_csv(os.path.join(self.output_folder, "predictions.csv"))
        plot_predictions(y_test_np, output_np, self.output_folder)
        r2_s = (r2_score(y_test_np, output_np))
        mae = mean_absolute_error(y_test_np, output_np)
        mse = mean_squared_error(y_test_np, output_np)
        r = scipy.stats.pearsonr(y_test_np, output_np)
        self.write_output()
        metrics = [r2_s, mae, mse, r.statistic]
        print('r2 score', r2_s)
        print('Pearson r:', r.statistic)
        print('mean absolute error', mae)
        print('mean square error', mse)
        with open(os.path.join(self.output_folder, 'output.txt'), 'a') as fp:
            fp.write('mean absolute error:' + str(mae) + "\n")
            fp.write('mean square error:' + str(mse) + "\n")
            fp.write('r2 score:' + str(r2_s) + "\n")
            fp.write('Pearson r:' + str(r.statistic) + "\n")
        return prediction, metrics


def test_model(model, output_folder, dataset_test, filenames_test, lc_values=None):
    prediction = model.predict(dataset_test[0])

    y_test_np = np.squeeze(np.array(dataset_test[1]))
    output_np = np.squeeze(np.array(prediction[0]))

    try:
        loss = TrainConfig.loss.name
    except:
        loss = TrainConfig.loss.__name__

    if loss == 'binary_crossentropy':
        data = pd.DataFrame({"filenames": filenames_test, "gt": y_test_np, "pred": output_np})
        data.to_csv(os.path.join(output_folder, "predictions.csv"))
        auc, best_threshold = calculate_ROC_and_save(y_test_np, output_np, output_folder)
        acc, error_rate, sensitivity, specificity = calculate_confusion_matrix_and_save(y_test_np, output_np, output_folder,
                                                                                        th=best_threshold)
    elif loss == 'categorical_crossentropy':
        calculate_ROC_categories(y_test_np, output_np, output_folder)
        ouput_np_one_hot = np.eye(output_np.shape[1])[output_np.argmax(1)]
        report = classification_report(y_test_np, ouput_np_one_hot,
                                       target_names=list(map(str, list(range(y_test_np.shape[1])))))
        acc, error_rate, sensitivity, specificity = calculate_confusion_matrix_categories(y_test_np, ouput_np_one_hot,
                                                                                          output_folder)
    else:
        data = pd.DataFrame({"filenames": filenames_test, "gt": y_test_np, "pred": output_np})
        data.to_csv(os.path.join(output_folder, "predictions.csv"))
        plot_predictions(y_test_np, output_np, output_folder, colors=lc_values)
        r2_s = (r2_score(y_test_np, output_np))
        r = scipy.stats.pearsonr(y_test_np, output_np)
        mae = mean_absolute_error(y_test_np, output_np)
        mse = mean_squared_error(y_test_np, output_np)
        print('r2 score', r2_s)
        print('Pearson r:', r.statistic)
        print('mean absolute error', mae)
        print('mean square error', mse)

    with open(os.path.join(output_folder, 'output.txt'), 'w') as fp:
        pass
        fp.write("Config: " + "\n")
        fp.write("Test: \n")
        fp.write('thresholded_mean_matrix:' + str(GeneralConfig.thresholded_mean_matrix) + "\n")
        fp.write('epochs:' + str(TrainConfig.epochs) + "\n")
        fp.write('learning_rate:' + str(TrainConfig.learning_rate) + "\n")
        fp.write('batch_size:' + str(TrainConfig.batch_size) + "\n")
        fp.write('optimizer:' + str(TrainConfig.optimizer) + "\n")
        fp.write('loss:' + str(TrainConfig.loss) + "\n")
        fp.write('n_heads:' + str(TrainConfig.n_heads) + "\n")
        fp.write('base_layer_dimensions:' + str(TrainConfig.base_layer_dimensions) + "\n")
        fp.write('dense_layer_dimensions:' + str(TrainConfig.dense_layer_dimensions) + "\n")
        fp.write('number of test subjects:' + str(dataset_test[0][0].shape[0]) + "\n")
        if loss == 'binary_crossentropy':
            fp.write('acc:' + str(acc) + "\n")
            fp.write('auc:' + str(auc) + "\n")
            fp.write('error_rate:' + str(error_rate) + "\n")
            fp.write('sensitivity:' + str(sensitivity) + "\n")
            fp.write('specificity:' + str(specificity) + "\n")
        elif loss == 'categorical_crossentropy':
            fp.write('classification report:' + "\n" + report + "\n")
            fp.write('acc:' + str(acc) + "\n")
            fp.write('error_rate:' + str(error_rate) + "\n")
            fp.write('sensitivity:' + str(sensitivity) + "\n")
            fp.write('specificity:' + str(specificity) + "\n")
        else:
            fp.write('mean absolute error:' + str(mae) + "\n")
            fp.write('mean square error:' + str(mse) + "\n")
            fp.write('r2 score:' + str(r2_s) + "\n")
            fp.write('Pearson r:' + str(scipy.stats.pearsonr(y_test_np, output_np)) + "\n")

    return prediction


def average_explanations(heatmaps, y_test, output_folder, atlas_coords, name='0'):
    age_ranges = [28, 38, 48, 58, 68, 78, 88]
    f, ax = plt.subplots(nrows=1, ncols=len(age_ranges), figsize=(20, 4))
    attention_heatmaps_by_age_range = []
    for n, i in enumerate(age_ranges):
        all_heatmap_age_range = []
        for heatmap_age_range, y_test_age_range in zip(heatmaps, y_test):
            if i - 10 <= y_test_age_range < i:
                all_heatmap_age_range.append(heatmap_age_range)
        heatmap_age_range_average = np.average(all_heatmap_age_range, axis=0)
        if len(heatmaps.shape) == 4:
            heatmap_age_range_average = np.average(heatmap_age_range_average, axis=0)
        else:
            pass
        attention_heatmaps_by_age_range.append(heatmap_age_range_average)
        ax[n].imshow(heatmap_age_range_average)

    f.tight_layout()
    f.savefig(os.path.join(output_folder, 'average_heatmaps_' + name + '.png'))

    for heatmap_age_range_average, age in zip(attention_heatmaps_by_age_range, age_ranges):
        plotting.plot_connectome(heatmap_age_range_average.T + heatmap_age_range_average,
                                 atlas_coords, edge_threshold="99.1%",
                                 node_size=10, node_color='auto')
        plt.savefig(os.path.join(output_folder, 'average_connectomes_' + name + str(age) + '.png'))
        plt.clf()
        plt.cla()
        plt.close()
        plotting.plot_markers(np.average(heatmap_age_range_average.T + heatmap_age_range_average, axis=0), atlas_coords,
                              node_size=20)
        plt.savefig(os.path.join(output_folder, 'average_node_importance_from_edges_' + name + str(age) + '.png'))
        plt.clf()
        plt.cla()
        plt.close()

    return attention_heatmaps_by_age_range


def explanation_from_heatmaps(heatmaps, y_test, age_ranges):
    attention_heatmaps_by_age_range = []
    for n, i in enumerate(age_ranges):
        all_heatmap_age_range = []
        for heatmap_age_range, y_test_age_range in zip(heatmaps, y_test):
            if i - 10 <= y_test_age_range < i:
                all_heatmap_age_range.append(heatmap_age_range)
        heatmap_age_range_average = np.average(all_heatmap_age_range, axis=0)
        if len(heatmaps.shape) == 4:
            heatmap_age_range_average = np.average(heatmap_age_range_average, axis=0)
        else:
            pass
        attention_heatmaps_by_age_range.append(heatmap_age_range_average)
    return attention_heatmaps_by_age_range


def plot_heatmaps(heatmaps_list, age_ranges, output_folder, name):
    f, ax = plt.subplots(nrows=1, ncols=len(age_ranges), figsize=(20, 4))
    for n, i in enumerate(age_ranges):
        ax[n].imshow(heatmaps_list[n])

    f.tight_layout()
    f.savefig(os.path.join(output_folder, 'average_heatmaps_' + name + '.png'))


def plot_connectomes(heatmaps, age_ranges, atlas_coords, output_folder, name):
    for heatmap_age_range_average, age in zip(heatmaps, age_ranges):
        plotting.plot_connectome(heatmap_age_range_average.T + heatmap_age_range_average,
                                 atlas_coords, edge_threshold="99.1%",
                                 node_size=10, node_color='auto')
        plt.savefig(os.path.join(output_folder, 'average_connectomes_' + name + '_' + str(age) + '.png'))
        plt.clf()
        plt.cla()
        plt.close()
        plotting.plot_markers(np.average(heatmap_age_range_average.T + heatmap_age_range_average, axis=0), atlas_coords,
                              node_size=20)
        plt.savefig(os.path.join(output_folder, 'average_node_importance_from_edges_' + name + '_' + str(age) + '.png'))
        plt.clf()
        plt.cla()
        plt.close()
        # plotting.plot_markers(heatmap_age_range_average_nodes, atlas_coords, node_size=20)
        # plt.savefig(os.path.join(output_folder, 'average_node_importance_from_nodes_' + name + str(age) + '.png'))
        # plt.clf()
        # plt.cla()
        # plt.close()


def average_explanations_with_nodes_by_age_range(heatmaps, y_test, ages_test, output_folder, atlas_coords,
                                                 heatmaps_nodes):
    age_ranges = [28, 38, 48, 58, 68, 78, 88]
    attention_heatmaps_by_age_range_edges = []
    for n, h in enumerate(heatmaps):
        attention_heatmaps_by_age_range = explanation_from_heatmaps(h, ages_test, age_ranges)
        attention_heatmaps_by_age_range_edges.append(attention_heatmaps_by_age_range)
        plot_heatmaps(attention_heatmaps_by_age_range, age_ranges, output_folder, name='from_edges_' + str(n))
        plot_connectomes(attention_heatmaps_by_age_range, age_ranges, atlas_coords, output_folder, name='_' + str(n))
    attention_heatmaps_by_age_range_nodes = explanation_from_heatmaps(heatmaps_nodes, y_test, age_ranges)

    return attention_heatmaps_by_age_range_edges


def average_explanations(heatmaps_edges, output_folder, atlas_coords, name=''):
    heatmap_edges_average = np.average(heatmaps_edges, axis=0)
    for n, head in enumerate(heatmaps_edges):
        plt.imshow(np.average(head, axis=0))
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'average_heatmaps_from_edges_head' + str(n) + '.png'))
        plt.clf()
        plt.cla()
        plt.close()
        plotting.plot_connectome(np.average(head, axis=0).T + np.average(head, axis=0),
                                 atlas_coords, edge_threshold="99.1%",
                                 node_size=10, node_color='auto')
        plt.savefig(os.path.join(output_folder, 'average_connectomes_from_edges_head' + str(n) + '.png'))
        plt.clf()
        plt.cla()
        plt.close()
    if len(heatmaps_edges.shape) == 4:
        heatmap_edges_average = np.average(heatmap_edges_average, axis=0)
    else:
        pass
    plt.imshow(heatmap_edges_average)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'average_heatmaps_from_edges' + name + '.png'))
    plt.clf()
    plt.cla()
    plt.close()
    plotting.plot_connectome(heatmap_edges_average.T + heatmap_edges_average,
                             atlas_coords, edge_threshold="99.1%",
                             node_size=10, node_color='auto')
    plt.savefig(os.path.join(output_folder, 'average_connectomes_from_edges' + name + '.png'))
    plt.clf()
    plt.cla()
    plt.close()
    plotting.plot_markers(np.average(heatmap_edges_average + heatmap_edges_average.T, axis=0), atlas_coords,
                          node_size=20)
    plt.savefig(os.path.join(output_folder, 'average_node_importance_from_edges_' + name + '.png'))
    plt.clf()
    plt.cla()
    plt.close()
    return heatmap_edges_average


def transform_edges_to_matrix(edges_test, prediction):
    matrices_heads = []
    for p in range(prediction.shape[1]):
        matrices = []
        for e_s, v_s in zip(edges_test, prediction[:, p, 0, :]):
            matrix = np.zeros((GeneralConfig.atlas_coords, GeneralConfig.atlas_coords))
            for e, v in zip(e_s, v_s):
                matrix[tuple(e)] = v
            matrices.append(matrix)
        matrices_heads.append(matrices)
    return matrices_heads


def transform_edges_to_matrix_evaluation(edges_x, edges_y, edges_attention_head, atlas_n_coords):
    matrix = np.zeros((atlas_n_coords, atlas_n_coords))
    for ex, ey, v in zip(edges_x, edges_y, edges_attention_head):
        matrix[tuple([ex, ey])] = v
    return matrix


def get_upper_triangle_coordinates(edges, n_nodes):
    n_subjects = edges.shape[0]
    upper_edges = []
    row, col = np.triu_indices(n_nodes, 1)
    for e in edges:
        for r, c in zip(row, col):
            if (np.array([r, c]) == e).all():
                upper_edges.append(e)
    upper_edges = np.array(upper_edges)
    upper_edges = np.expand_dims(upper_edges, 0)
    upper_edges = np.repeat(upper_edges, n_subjects, 0)

    return upper_edges


def normalize_std(dataset_train, dataset_test):
    mean = np.mean(dataset_train)
    stdv = np.std(dataset_train)
    dataset_train = (dataset_train - mean) / stdv
    dataset_test = (dataset_test - mean) / stdv
    return dataset_train, dataset_test


def normalize_minmax(dataset_train, dataset_test):
    max = np.max(dataset_train)
    dataset_train = dataset_train / max
    dataset_test = dataset_test / max
    return dataset_train, dataset_test


def normalize_StandardScaler(dataset_train, dataset_test, dataset_val):
    scaler_edges = StandardScaler()
    original_shape = len(dataset_train.shape)
    if original_shape == 3:
        nx = dataset_train.shape[1]
        ny = dataset_train.shape[2]
        dataset_train = dataset_train.reshape((dataset_train.shape[0], nx * ny))
        dataset_test = dataset_test.reshape((dataset_test.shape[0], nx * ny))
        dataset_val = dataset_val.reshape((dataset_val.shape[0], nx * ny))
    else:
        pass
    dataset_train = scaler_edges.fit_transform(dataset_train)
    dataset_test = scaler_edges.transform(dataset_test)
    dataset_val = scaler_edges.transform(dataset_val)
    if original_shape == 3:
        dataset_train = dataset_train.reshape((dataset_train.shape[0], nx, ny))
        dataset_test = dataset_test.reshape((dataset_test.shape[0], nx, ny))
        dataset_val = dataset_val.reshape((dataset_val.shape[0], nx, ny))
    else:
        dataset_train = np.expand_dims(dataset_train, -1)  # node features
        dataset_test = np.expand_dims(dataset_test, -1)
        dataset_val = np.expand_dims(dataset_val, -1)
    return dataset_train, dataset_test, dataset_val


def normalize_MinMaxScaler_StandardScaler(dataset_train, dataset_test, dataset_val):
    scaler_edges = MinMaxScaler(feature_range=(0, 1))
    dataset_train = scaler_edges.fit_transform(dataset_train)
    dataset_test = scaler_edges.transform(dataset_test)
    dataset_val = scaler_edges.transform(dataset_val)
    scaler_edges = StandardScaler()
    dataset_train = scaler_edges.fit_transform(dataset_train)
    dataset_test = scaler_edges.transform(dataset_test)
    dataset_train = np.expand_dims(dataset_train, 2)  # node features
    dataset_test = np.expand_dims(dataset_test, 2)
    dataset_val = scaler_edges.transform(dataset_val)
    dataset_val = np.expand_dims(dataset_val, 2)
    return dataset_train, dataset_test, dataset_val


def normalize_MinMaxScaler(dataset_train, dataset_test, dataset_val, expand_dim=True, range=(0, 1)):
    scaler_edges = MinMaxScaler(feature_range=range)
    original_shape = len(dataset_train.shape)
    if original_shape == 3:
        nx = dataset_train.shape[1]
        ny = dataset_train.shape[2]
        dataset_train = dataset_train.reshape((dataset_train.shape[0], nx * ny))
        dataset_test = dataset_test.reshape((dataset_test.shape[0], nx * ny))
        dataset_val = dataset_val.reshape((dataset_val.shape[0], nx * ny))
    else:
        pass
    dataset_train = scaler_edges.fit_transform(dataset_train)
    dataset_test = scaler_edges.transform(dataset_test)
    dataset_val = scaler_edges.transform(dataset_val)
    if original_shape == 3:
        dataset_train = dataset_train.reshape((dataset_train.shape[0], nx, ny))
        dataset_test = dataset_test.reshape((dataset_test.shape[0], nx, ny))
        dataset_val = dataset_val.reshape((dataset_val.shape[0], nx, ny))
    else:
        pass
        dataset_train = np.expand_dims(dataset_train, -1)  # node features
        dataset_test = np.expand_dims(dataset_test, -1)
        dataset_val = np.expand_dims(dataset_val, -1)
    return dataset_train, dataset_test, dataset_val


def normalize_RobustScaler(dataset_train, dataset_test, dataset_val):
    scaler_edges = RobustScaler()
    dataset_train = scaler_edges.fit_transform(dataset_train)
    dataset_test = scaler_edges.transform(dataset_test)
    dataset_train = np.expand_dims(dataset_train, 2)  # node features
    dataset_test = np.expand_dims(dataset_test, 2)
    dataset_val = scaler_edges.transform(dataset_val)
    dataset_val = np.expand_dims(dataset_val, 2)
    return dataset_train, dataset_test, dataset_val


def normalize_data(edge_features_train, edge_features_val, edge_features_test):
    if TrainConfig.normalization == "MinMax":
        edge_features_train, edge_features_test, edge_features_val = normalize_MinMaxScaler(edge_features_train,
                                                                                            edge_features_test,
                                                                                            edge_features_val)
    elif TrainConfig.normalization == "Standard":
        edge_features_train, edge_features_test, edge_features_val = normalize_StandardScaler(edge_features_train,
                                                                                              edge_features_test,
                                                                                              edge_features_val)
    elif TrainConfig.normalization == "MinMax_Standard":
        edge_features_train, edge_features_test, edge_features_val = normalize_MinMaxScaler_StandardScaler(
            edge_features_train,
            edge_features_test,
            edge_features_val)
    elif TrainConfig.normalization == "manualStd":
        edge_features_test = (edge_features_test - np.mean(edge_features_train)) / np.std(edge_features_train)
        edge_features_val = (edge_features_val - np.mean(edge_features_train)) / np.std(edge_features_train)
        edge_features_train = (edge_features_train - np.mean(edge_features_train)) / np.std(edge_features_train)
        edge_features_train = np.expand_dims(edge_features_train, 2)  # node features
        edge_features_test = np.expand_dims(edge_features_test, 2)
        edge_features_val = np.expand_dims(edge_features_val, 2)
    else:
        edge_features_train = np.expand_dims(edge_features_train, 2)  # node features
        edge_features_test = np.expand_dims(edge_features_test, 2)
        edge_features_val = np.expand_dims(edge_features_val, 2)
    return edge_features_train, edge_features_val, edge_features_test


def calculate_ROC_and_save(trues, preds, output, name=''):
    ns_probs = [0 for _ in range(len(trues))]
    #
    i_auc = roc_auc_score(trues, trues)
    ns_auc = roc_auc_score(trues, ns_probs)
    lr_auc = roc_auc_score(trues, preds)
    # summarize scores
    print('Ideal: ROC AUC=%.3f' % (i_auc))
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    i_fpr, i_tpr, _ = roc_curve(trues, trues)
    ns_fpr, ns_tpr, _ = roc_curve(trues, ns_probs)
    lr_fpr, lr_tpr, threshold = roc_curve(trues, preds)

    i = np.arange(len(lr_tpr))
    roc = pd.DataFrame({'tf': pd.Series(lr_tpr - (1 - lr_fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    best_threshold = list(roc_t['threshold'])[0]

    # plot the roc curve for the networks
    plt.plot(i_fpr, i_tpr, linestyle='-.', label='Ideal')
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    # plt.show()
    plt.savefig(output + 'ROC' + name + '.png')
    plt.clf()
    plt.cla()

    auc = roc_auc_score(trues, preds)
    return auc, best_threshold

def calculate_ROC(trues, preds):
    ns_probs = [0 for _ in range(len(trues))]
    #
    i_auc = roc_auc_score(trues, trues)
    ns_auc = roc_auc_score(trues, ns_probs)
    lr_auc = roc_auc_score(trues, preds)
    # calculate roc curves
    i_fpr, i_tpr, _ = roc_curve(trues, trues)
    ns_fpr, ns_tpr, _ = roc_curve(trues, ns_probs)
    lr_fpr, lr_tpr, threshold = roc_curve(trues, preds)

    i = np.arange(len(lr_tpr))
    roc = pd.DataFrame({'tf': pd.Series(lr_tpr - (1 - lr_fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    best_threshold = list(roc_t['threshold'])[0]

    auc = roc_auc_score(trues, preds)
    return auc, best_threshold


def plot_confusion_matrix(compare, output, name=''):
    plt.imshow(compare, cmap=plt.cm.Blues)
    fmt = 'd'
    # write the number of predictions in each bucket
    thresh = compare.max() / 2.
    for i, j in itertools.product(range(compare.shape[0]), range(compare.shape[1])):
        # if background is dark, use a white number, and vice-versa
        plt.text(j, i, format(compare[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if compare[i, j] > thresh else "black")
    plt.colorbar()
    plt.title(name)
    plt.xlabel('Classes prediction')
    plt.ylabel('True classes')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.savefig(output + 'confussion_matrix' + name + '.png')
    plt.clf()
    plt.cla()


def plot_confusion_matrix_categorical(compare, output, classes):
    plt.imshow(compare, cmap=plt.cm.Blues)
    fmt = 'd'
    # write the number of predictions in each bucket
    thresh = compare.max() / 2.
    for i, j in itertools.product(range(compare.shape[0]), range(compare.shape[1])):
        # if background is dark, use a white number, and vice-versa
        plt.text(j, i, format(compare[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if compare[i, j] > thresh else "black")
    plt.colorbar()
    plt.xlabel('Classes prediction')
    plt.ylabel('True classes')
    plt.xticks(list(range(len(classes))), classes)
    plt.yticks(list(range(len(classes))), classes)
    plt.savefig(output + 'confussion_matrix_categorical.png')
    plt.clf()
    plt.cla()


def calculate_classification_metrics(compare):
    # Accuracy
    tn = compare[0, 0]
    tp = compare[1, 1]
    fp = compare[0, 1]
    fn = compare[1, 0]
    acc = (tn + tp) / (tn + tp + fp + fn)
    error_rate = (fp + fn) / (tn + tp + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return acc, error_rate, sensitivity, specificity


def calculate_confusion_matrix_and_save(trues, preds, output, name='', th=0.5):
    preds_int = [1 if x > th else 0 for x in preds]
    compare = confusion_matrix(trues, preds_int)
    plot_confusion_matrix(compare, output, name=name)
    acc, error_rate, sensitivity, specificity = calculate_classification_metrics(compare)

    return acc, error_rate, sensitivity, specificity

def calculate_confusion_matrix(trues, preds, th=0.5):
    preds_int = [1 if x > th else 0 for x in preds]
    compare = confusion_matrix(trues, preds_int)
    acc, error_rate, sensitivity, specificity = calculate_classification_metrics(compare)

    return acc, error_rate, sensitivity, specificity

def calculate_ROC_categories(y_test, output, output_folder):
    classes = list(map(str, list(range(y_test.shape[1]))))
    n_classes = y_test.shape[1]
    plt.figure()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), output.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['lightgreen', 'darkgreen', 'navajowhite', 'darkorange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_folder, 'roc_categories.png'))
    plt.close()
    plt.cla()
    plt.clf()


def calculate_confusion_matrix_categories(y_test, output, output_folder):
    classes = list(map(str, list(range(y_test.shape[1]))))
    plt.clf()
    plt.cla()
    confusion = multilabel_confusion_matrix(y_test, output)
    accs = []
    error_rates = []
    sensitivities = []
    specificities = []
    for compare, c in zip(confusion, classes):
        plot_confusion_matrix(compare, output_folder + str(c) + '_')
        acc, error_rate, sensitivity, specificity = calculate_classification_metrics(compare)
        accs.append(acc)
        error_rates.append(error_rate)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    output_label = np.argmax(output, axis=1)
    y_test_label = np.argmax(y_test, axis=1)
    compare_categorical = confusion_matrix(y_test_label, output_label)
    plot_confusion_matrix_categorical(compare_categorical, output_folder, classes)
    return accs, error_rates, sensitivities, specificities


def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    -> param y_true: list of true values
    -> param y_pred: list of predicted values
    -> return: accuracy score

    """

    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == yp:
            correct_predictions += 1

    # returns accuracy
    return correct_predictions / len(y_true)


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def balance_dataset(dataset):
    # Filter by label
    train_dataset_neg, train_dataset_pos = filter_by_label(dataset)
    number_training_data_negative = len(list(train_dataset_neg))
    number_training_data_positive = len(list(train_dataset_pos))

    # Sapling data to equal number of positive and negative samples
    train_dataset_pos = train_dataset_pos.repeat(
        math.ceil(number_training_data_negative / number_training_data_positive))
    train_dataset_pos = train_dataset_pos.take(number_training_data_negative)
    joint_dataset = [train_dataset_neg, train_dataset_pos]
    choice_dataset = tf.data.Dataset.range(2).repeat(number_training_data_negative)
    train_dataset = tf.data.experimental.choose_from_datasets(joint_dataset, choice_dataset)
    # train_dataset = train_dataset.repeat()
    return train_dataset


def generate_random_graphs(x):
    edge_features, edges = x

    # num_centroids = edge_features.shape[0] - int(edge_features.shape[0]*0.9)
    # selected_centroids = int(np.random.uniform(1, edge_features.shape[0]))
    # features = edge_features[selected_centroids-num_centroids:selected_centroids, :]
    # features += np.random.randn(*features.shape) * np.random.rand() * 0.1
    # edge_features[selected_centroids-num_centroids:selected_centroids, :] = features

    noise = np.random.rand() * 0.1
    edge_features = tf.keras.layers.GaussianNoise(stddev=noise)(edge_features, training=True)

    return (edge_features, edges)


def generate_random_graphs_multimodal(x):
    edge_features_con, edges_con, edge_features_sc, edges_sc, edge_features_fc, edges_fc = x

    # num_centroids = edge_features.shape[0] - int(edge_features.shape[0]*0.9)
    # selected_centroids = int(np.random.uniform(1, edge_features.shape[0]))
    # features = edge_features[selected_centroids-num_centroids:selected_centroids, :]
    # features += np.random.randn(*features.shape) * np.random.rand() * 0.1
    # edge_features[selected_centroids-num_centroids:selected_centroids, :] = features

    noise = np.random.rand() * 0.1
    edge_features_con = tf.keras.layers.GaussianNoise(stddev=noise)(edge_features_con, training=True)
    edge_features_sc = tf.keras.layers.GaussianNoise(stddev=noise)(edge_features_sc, training=True)
    edge_features_fc = tf.keras.layers.GaussianNoise(stddev=noise)(edge_features_fc, training=True)

    return (edge_features_con, edges_con, edge_features_sc, edges_sc, edge_features_fc, edges_fc)


def augment_data(train_dataset):
    train_dataset = train_dataset.map(
        lambda x, y_lc: (generate_random_graphs(x), y_lc))
    return train_dataset


def augment_data_multimodal(train_dataset):
    train_dataset = train_dataset.map(
        lambda x, y_lc: (generate_random_graphs_multimodal(x), y_lc))
    return train_dataset


def filter_by_label(dataset):
    dataset_neg = dataset.filter(lambda x, y_lc: tf.math.equal(y_lc[0], 0))
    dataset_pos = dataset.filter(lambda x, y_lc: tf.math.equal(y_lc[0], 1))
    return dataset_neg, dataset_pos


def get_common_indices(filenames_sc, filenames_fc):
    both = set(list(filenames_sc)).intersection(list(filenames_fc))
    return both


def combine_features(filenames_sc, filenames_fc, features_sc, features_fc):
    filenames_common = get_common_indices(filenames_sc, filenames_fc)
    features_common = []
    for file in filenames_common:
        idx_sc = filenames_sc.index(file)
        idx_fc = filenames_fc.index(file)
        features_sc_idx = np.expand_dims(features_sc[idx_sc], 1)
        features_fc_idx = np.expand_dims(features_fc[idx_fc], 1)
        features_common.append(np.concatenate([features_sc_idx, features_fc_idx], axis=1))
    return features_common


def get_common_values(filenames_sc, filenames_fc, features_sc):
    if len(features_sc) == 0:
        return features_sc
    else:
        filenames_common = get_common_indices(filenames_sc, filenames_fc)
        features_common = []
        for file in filenames_common:
            idx_sc = filenames_sc.index(file)
            features_common.append(features_sc[idx_sc])
        return features_common


def split_stratified(edges,
                     edge_features,
                     target,
                     filenames, test_size=0.14):
    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=test_size, random_state=42)
    for i, (train_index, test_index) in enumerate(sss.split(edge_features, target)):
        edges_train, edges_test = edges[train_index], edges[test_index]
        edge_features_train, edge_features_test = edge_features[train_index], edge_features[test_index]
        filenames_train, filenames_test = filenames[train_index], filenames[test_index]
        target_train, target_test = target[train_index], target[test_index]

    return edges_train, edges_test, edge_features_train, edge_features_test, target_train, target_test, filenames_train, filenames_test

def split_stratified_atrophy(gm_features,
                     target,
                     filenames, test_size=0.14):
    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=test_size, random_state=42)
    for i, (train_index, test_index) in enumerate(sss.split(gm_features, target)):
        edge_features_train, edge_features_test = gm_features[train_index], gm_features[test_index]
        filenames_train, filenames_test = filenames[train_index], filenames[test_index]
        target_train, target_test = target[train_index], target[test_index]

    return edge_features_train, edge_features_test, target_train, target_test, filenames_train, filenames_test


def spearman_corr(edge_features, edges, target, name, output):
    if TrainConfig.edge_features == 'combined':
        edge_features_value = np.sum(edge_features, axis=2)
    else:
        edge_features_value = edge_features
    edges = edges[0]
    corrs = []
    p_values = []
    selected_edge_features = []
    selected_edges = []
    p_values_selected = []
    corrs_selected = []
    for edge in range(edge_features_value.shape[1]):
        if np.sum(edge_features_value[:, edge]) > 0:
            corr = scipy.stats.spearmanr(edge_features_value[:, edge], target)
            corrs.append(corr.correlation)
            p_values.append(corr.pvalue)
            if corr.pvalue <= 0.001:
                if TrainConfig.edge_features == 'combined':
                    selected_edge_features.append(edge_features[:, edge, :])
                else:
                    selected_edge_features.append(edge_features[:, edge])
                selected_edges.append(edges[edge])
                p_values_selected.append(corr.pvalue)
                corrs_selected.append(corr.correlation)
            else:
                pass
        else:
            pass
    corrected, array = sm.stats.fdrcorrection(p_values, alpha=0.01, method='indep', is_sorted=False)
    corrected_edge_features = []
    corrected_edges = []
    for ef, e, c in zip(selected_edge_features, selected_edges, corrected):
        if c == True:
            corrected_edge_features.append(ef)
            corrected_edges.append(e)
    # survive = sum(bool(x) for x in list(corrected))
    if TrainConfig.edge_features == 'combined':
        selected_edge_features_array = np.array(corrected_edge_features).transpose(1, 0, 2)
    else:
        selected_edge_features_array = np.array(corrected_edge_features).T
    selected_edge_array = np.array(corrected_edges)
    selected_edge_array = np.expand_dims(selected_edge_array, 0)
    selected_edge_array = np.repeat(selected_edge_array, selected_edge_features_array.shape[0], 0)
    plt.scatter(list(range(len(corrs_selected))), corrs_selected)
    plt.xlabel('Edges')
    plt.ylabel(name)
    plt.title('Spearman r between ' + name + ' and the different edges')
    plt.savefig(os.path.join(output, 'correlations.png'))
    plt.close()
    plt.clf()
    plt.cla()
    matrix = np.zeros((GeneralConfig.atlas_coords, GeneralConfig.atlas_coords))
    edge_features_mean = np.mean(edge_features_value, axis=0)
    for n, e in enumerate(selected_edge_array[0]):
        # matrix[list(e)[0], list(e)[1]] = edge_features_mean[n]
        matrix[list(e)[0], list(e)[1]] = 1
    plt.imshow(matrix)
    plt.savefig(os.path.join(output, 'selected_edges.png'))
    return corrs, selected_edge_features_array, selected_edge_array


def spearman_corr_for_training(edge_features_train, edge_features_val, edge_features_test, edges_train, target, output,
                               lcs_val=None, lcs_test=None):
    edges_train = edges_train[0]
    corrs = []
    p_values = []
    selected_edge_features_train = []
    selected_edges_train = []
    selected_edge_features_test = []
    selected_edge_features_val = []
    p_values_selected = []
    corrs_selected = []
    edges_features_all = np.concatenate([edge_features_train, edge_features_val])
    # target = np.concatenate([target, lcs_val])
    for edge in range(edge_features_train.shape[1]):
        if np.sum(edge_features_train[:, edge]) > 0:
            corr = scipy.stats.spearmanr(edge_features_train[:, edge], target)
            corrs.append(corr.correlation)
            p_values.append(corr.pvalue)
            if corr.pvalue <= 0.001:
                selected_edge_features_train.append(edge_features_train[:, edge])
                selected_edges_train.append(edges_train[edge])
                selected_edge_features_test.append(edge_features_test[:, edge])
                selected_edge_features_val.append(edge_features_val[:, edge])
                p_values_selected.append(corr.pvalue)
                corrs_selected.append(corr.correlation)
            else:
                pass
        else:
            pass
    corrected, array = sm.stats.fdrcorrection(p_values, alpha=0.05, method='indep', is_sorted=False)
    corrected_edge_features_train = []
    corrected_edges_train = []
    corrected_edge_features_test = []
    corrected_edges_test = []
    corrected_edge_features_val = []
    corrected_edges_val = []
    for ef_train, e_train, ef_val, ef_test, c in zip(selected_edge_features_train, selected_edges_train,
                                                     selected_edge_features_val, selected_edge_features_test,
                                                     corrected):
        if c == True:
            corrected_edge_features_train.append(ef_train)
            corrected_edges_train.append(e_train)
            corrected_edge_features_test.append(ef_test)
            corrected_edges_test.append(e_train)
            corrected_edge_features_val.append(ef_val)
            corrected_edges_val.append(e_train)
    # survive = sum(bool(x) for x in list(corrected))
    selected_edge_features_array_train = np.array(corrected_edge_features_train).T
    selected_edge_array_train = np.array(corrected_edges_train)
    selected_edge_features_array_test = np.array(corrected_edge_features_test).T
    selected_edge_array_test = np.array(corrected_edges_test)
    selected_edge_features_array_val = np.array(corrected_edge_features_val).T
    selected_edge_array_val = np.array(corrected_edges_val)
    selected_edge_array_train = np.expand_dims(selected_edge_array_train, 0)
    selected_edge_array_train = np.repeat(selected_edge_array_train, selected_edge_features_array_train.shape[0], 0)
    selected_edge_array_test = np.expand_dims(selected_edge_array_test, 0)
    selected_edge_array_test = np.repeat(selected_edge_array_test, selected_edge_features_array_test.shape[0], 0)
    selected_edge_array_val = np.expand_dims(selected_edge_array_val, 0)
    selected_edge_array_val = np.repeat(selected_edge_array_val, selected_edge_features_array_val.shape[0], 0)
    matrix = np.zeros((GeneralConfig.atlas_coords, GeneralConfig.atlas_coords))
    edge_features_mean = np.mean(edge_features_train, axis=0)
    for n, e in enumerate(selected_edge_array_train[0]):
        # matrix[list(e)[0], list(e)[1]] = edge_features_mean[n]
        matrix[list(e)[0], list(e)[1]] = 1
    plt.imshow(matrix)
    plt.savefig(os.path.join(output, 'selected_edges_cv.png'))
    return corrs, selected_edge_features_array_train, selected_edge_array_train, selected_edge_features_array_val, selected_edge_array_val, selected_edge_features_array_test, selected_edge_array_test


def pearson_corr(edges, target):
    corrs = []
    p_values = []
    for edge in range(edges.shape[1]):
        corr = scipy.stats.pearsonr(edges[:, edge], target)
        corrs.append(corr.statistic[0])
        p_values.append(corr.pvalue)
    plt.scatter(list(range(edges.shape[1])), corrs)
    plt.xlabel('Edges')
    plt.ylabel('LC')
    plt.title('Pearson r between LC and the different edges')
    plt.show()
    return corrs


def apply_select_features_function(fs, X_train, X_val, X_test):
    X_train_fs = np.expand_dims(fs.transform(X_train), axis=-1)
    X_test_fs = np.expand_dims(fs.transform(X_test), axis=-1)
    X_val_fs = np.expand_dims(fs.transform(X_val), axis=-1)

    return X_train_fs, X_val_fs, X_test_fs


def apply_select_features_function_edges(fs, X_train, X_val, X_test):
    X_train_fs_2_0 = np.expand_dims(fs.transform(X_train[:, :, 0]), axis=-1)
    X_test_fs_2_0 = np.expand_dims(fs.transform(X_test[:, :, 0]), axis=-1)
    X_val_fs_2_0 = np.expand_dims(fs.transform(X_val[:, :, 0]), axis=-1)

    X_train_fs_2_1 = np.expand_dims(fs.transform(X_train[:, :, 1]), axis=-1)
    X_test_fs_2_1 = np.expand_dims(fs.transform(X_test[:, :, 1]), axis=-1)
    X_val_fs_2_1 = np.expand_dims(fs.transform(X_val[:, :, 1]), axis=-1)

    X_train_fs_2 = np.concatenate([X_train_fs_2_0, X_train_fs_2_1], axis=-1)
    X_test_fs_2 = np.concatenate([X_test_fs_2_0, X_test_fs_2_1], axis=-1)
    X_val_fs_2 = np.concatenate([X_val_fs_2_0, X_val_fs_2_1], axis=-1)

    return X_train_fs_2, X_val_fs_2, X_test_fs_2


def combine_sc_fc(X_fs, X_fs_2, X_fs_sc, X_fs_fc, X_fs_2_sc, X_fs_2_fc):
    sc_list = [i.tolist() for i in X_fs_2_sc[0]]
    fc_list = [i.tolist() for i in X_fs_2_fc[0]]
    fc_sc_all_list = [i.tolist() for i in X_fs_2[0]]
    s_count = 0
    sc_fc_list_edges = []
    sc_fc_list_edge_features = []
    for s_element in sc_list:
        if s_element in fc_list:
            indx_fc = fc_list.index(s_element)
            sc_fc_list_edges.append(s_element)
            sc_fc_list_edge_features.append([X_fs_sc[:, s_count, 0], X_fs_fc[:, indx_fc, 0]])
        else:
            indx_fc = fc_sc_all_list.index(s_element)
            sc_fc_list_edges.append(s_element)
            sc_fc_list_edge_features.append([X_fs_sc[:, s_count, 0], X_fs[:, indx_fc, 1]])
        s_count += 1
    f_count = 0
    for f_element in fc_list:
        if f_element not in sc_list:
            indx_sc = fc_sc_all_list.index(f_element)
            sc_fc_list_edges.append(f_element)
            sc_fc_list_edge_features.append([X_fs[:, indx_sc, 0], X_fs_fc[:, f_count, 0]])
        else:
            pass
        f_count += 1

    sc_fc_list_edge_features_array = np.array(sc_fc_list_edge_features)
    sc_fc_list_edge_features_array = np.array(sc_fc_list_edge_features_array).transpose(2, 0, 1)
    sc_fc_list_edges_array = np.array(sc_fc_list_edges)
    sc_fc_list_edges_array = np.repeat(np.expand_dims(sc_fc_list_edges_array, 0),
                                       sc_fc_list_edge_features_array.shape[0], 0)

    return sc_fc_list_edge_features_array, sc_fc_list_edges_array


def plot_selected_edges(X_train_fs_2, output, name='_'):
    matrix = np.zeros((GeneralConfig.atlas_coords, GeneralConfig.atlas_coords))
    for n, e in enumerate(X_train_fs_2[0]):
        matrix[list(e)[0], list(e)[1]] = 1
    plt.imshow(matrix)
    plt.savefig(os.path.join(output, name + 'selected_edges.png'))
    plt.cla()
    plt.close()
    plt.clf()


# feature selection
def select_features(X_train, y_train, y_val, X_val, X_test, X_train_2, X_val_2, X_test_2, output, combined=False):
    # configure to select all features
    # fs = SelectKBest(score_func=r_regression, k=400)
    try:
        loss = TrainConfig.loss.name
    except:
        loss = TrainConfig.loss.__name__
    if loss == 'binary_crossentropy':
        selected_function = f_classif
    else:
        selected_function = f_regression
    fs = SelectFdr(score_func=selected_function, alpha=TrainConfig.fdr)
    # fs = SelectKBest(score_func=selected_function, k=100)
    # fs = SelectPercentile(score_func=mutual_info_classif, percentile=5)
    # fs = GenericUnivariateSelect(f_classif, mode='fdr', param=0.05)
    # fs = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # learn relationship from training data
    if combined:
        fs_sc = SelectFdr(score_func=selected_function, alpha=0.05)
        fs_sc.fit(X_train[:, :, 0], y_train)

        fs_fc = SelectFdr(score_func=selected_function, alpha=0.05)
        fs_fc.fit(X_train[:, :, 1], y_train)

        X_train_fs_sc, X_val_fs_sc, X_test_fs_sc = apply_select_features_function(fs_sc, X_train[:, :, 0],
                                                                                  X_val[:, :, 0], X_test[:, :, 0])
        X_train_fs_fc, X_val_fs_fc, X_test_fs_fc = apply_select_features_function(fs_fc, X_train[:, :, 1],
                                                                                  X_val[:, :, 1], X_test[:, :, 1])

        X_train_fs_2_sc, X_val_fs_2_sc, X_test_fs_2_sc = apply_select_features_function_edges(fs_sc, X_train_2, X_val_2,
                                                                                              X_test_2)
        X_train_fs_2_fc, X_val_fs_2_fc, X_test_fs_2_fc = apply_select_features_function_edges(fs_fc, X_train_2, X_val_2,
                                                                                              X_test_2)

        plot_selected_edges(X_train_fs_2_sc, output, name='structural')
        plot_selected_edges(X_train_fs_2_fc, output, name='functional')

        X_train_fs, X_train_fs_2 = combine_sc_fc(X_train, X_train_2, X_train_fs_sc, X_train_fs_fc, X_train_fs_2_sc,
                                                 X_train_fs_2_fc)
        X_val_fs, X_val_fs_2 = combine_sc_fc(X_val, X_val_2, X_val_fs_sc, X_val_fs_fc, X_val_fs_2_sc, X_val_fs_2_fc)
        X_test_fs, X_test_fs_2 = combine_sc_fc(X_test, X_test_2, X_test_fs_sc, X_test_fs_fc, X_test_fs_2_sc,
                                               X_test_fs_2_fc)

    else:
        # X_train_val = np.concatenate([X_train, X_val])
        # y_train_val = np.concatenate([y_train, y_val])
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        X_val_fs = fs.transform(X_val)

        X_train_fs_2_0 = np.expand_dims(fs.transform(X_train_2[:, :, 0]), axis=-1)
        X_test_fs_2_0 = np.expand_dims(fs.transform(X_test_2[:, :, 0]), axis=-1)
        X_val_fs_2_0 = np.expand_dims(fs.transform(X_val_2[:, :, 0]), axis=-1)

        X_train_fs_2_1 = np.expand_dims(fs.transform(X_train_2[:, :, 1]), axis=-1)
        X_test_fs_2_1 = np.expand_dims(fs.transform(X_test_2[:, :, 1]), axis=-1)
        X_val_fs_2_1 = np.expand_dims(fs.transform(X_val_2[:, :, 1]), axis=-1)

        X_train_fs_2 = np.concatenate([X_train_fs_2_0, X_train_fs_2_1], axis=-1)
        X_test_fs_2 = np.concatenate([X_test_fs_2_0, X_test_fs_2_1], axis=-1)
        X_val_fs_2 = np.concatenate([X_val_fs_2_0, X_val_fs_2_1], axis=-1)

    plot_selected_edges(X_train_fs_2, output)

    return X_train_fs, X_val_fs, X_test_fs, X_train_fs_2, X_val_fs_2, X_test_fs_2, fs


def select_features_variance(X_train, X_train_2):
    # configure to select all features
    fs = VarianceThreshold(threshold=0.01)
    # fs = GenericUnivariateSelect(f_regression, mode='fdr', param=0.05)
    # learn relationship from training data
    fs.fit(X_train)

    X_train_fs = fs.transform(X_train)

    X_train_fs_2_0 = np.expand_dims(fs.transform(X_train_2[:, :, 0]), axis=-1)

    X_train_fs_2_1 = np.expand_dims(fs.transform(X_train_2[:, :, 1]), axis=-1)

    X_train_fs_2 = np.concatenate([X_train_fs_2_0, X_train_fs_2_1], axis=-1)

    return X_train_fs, X_train_fs_2, fs


def mean_attention_from_heads(edges_attention, edges_x, edges_y, atlas_n_coords):
    edges_attention_heads = []
    matrices_attention = []
    for n_heads in range(edges_attention.shape[0]):
        edges_attention_heads.append(list(edges_attention.iloc[n_heads])[1:])
        matrices_attention.append(
            transform_edges_to_matrix_evaluation(edges_x, edges_y, list(edges_attention.iloc[n_heads])[1:],
                                                 atlas_n_coords))
    edges_attention_head = np.mean(edges_attention_heads, axis=0)
    matrix_attention = transform_edges_to_matrix_evaluation(edges_x, edges_y, edges_attention_head, atlas_n_coords)
    return edges_attention_head, edges_attention_heads, matrix_attention, matrices_attention


def generate_mean_attention_matrix(edges, attention_file):
    edges_x = list(edges[edges.columns[1]])
    edges_y = list(edges[edges.columns[2]])
    edges_attention_head_all = []
    edges_attention_heads_all = []
    matrix_attention_all = []
    matrices_attention_all = []
    matrix_attention_mean = np.zeros((GeneralConfig.atlas_coords, GeneralConfig.atlas_coords))
    for i in range(TrainConfig.n_splits):
        edges_attention = pd.read_csv(os.path.join(TrainConfig.output, str(i), attention_file))
        # get from attention
        edges_attention_head, edges_attention_heads, matrix_attention, matrices_attention = mean_attention_from_heads(
            edges_attention, edges_x, edges_y, GeneralConfig.atlas_coords)
        edges_attention_head_all.append(edges_attention_head)
        edges_attention_heads_all.append(edges_attention_heads)
        matrices_attention_all.append(matrices_attention)
        matrix_attention_mean = np.mean([matrix_attention_mean, matrix_attention], axis=0)
    return matrix_attention_mean


def get_attention_importances_atlas_names(matrix_attention_mean, atlas_names, atlas_coords, atlas_node_names,
                                          atlas_hemi, name=''):
    indices = np.where(matrix_attention_mean > 0)
    b = [indices[0], indices[1]]
    matrix_ones = np.zeros((GeneralConfig.atlas_coords, GeneralConfig.atlas_coords))
    edges_importance_names_x = []
    edges_importance_names_y = []
    value_importance = []
    for ex, ey in zip(b[0], b[1]):
        matrix_ones[tuple([ex, ey])] = 1
        value_importance.append(matrix_attention_mean[ex, ey])
        edges_importance_names_x.append(atlas_names[ex])
        edges_importance_names_y.append(atlas_names[ey])

    importances = {'edges_importance_names_x': edges_importance_names_x,
                   'edges_importance_names_y': edges_importance_names_y,
                   'Values': value_importance}
    importances = pd.DataFrame(importances)
    importances.to_csv(os.path.join(TrainConfig.output, 'importances_edges'+name+'.csv'), sep=';', index=None)
    matrix_edge_importance = matrix_attention_mean
    plotting.plot_connectome(matrix_edge_importance + matrix_edge_importance.T, atlas_coords, node_size=10,
                             edge_threshold="99.1%", node_color='auto')
    plt.savefig(os.path.join(TrainConfig.output, 'average_edges_importance'+name+'.png'))
    plt.clf()
    plt.cla()
    plt.close()
    matrix_edge_importance_T = matrix_edge_importance + matrix_edge_importance.T
    counts = [np.where(matrix_edge_importance_T_row > 0)[0].shape[0] for matrix_edge_importance_T_row in
              matrix_edge_importance_T]
    node_importance = np.mean(matrix_edge_importance_T, axis=0)
    node_importance = np.nan_to_num(node_importance)
    node_importance = node_importance / np.max(node_importance)
    node_importance_positive_average = np.sum(matrix_edge_importance_T, axis=0) / counts
    node_importance_positive_average = np.nan_to_num(node_importance_positive_average)
    node_importance_positive_average = node_importance_positive_average / np.max(node_importance_positive_average)
    node_importance_positive = np.sum(matrix_edge_importance_T, axis=0)
    node_importance_positive = np.nan_to_num(node_importance_positive)
    node_importance_positive = node_importance_positive / np.max(node_importance_positive)
    importances_nodes = {'nodes': atlas_names,
                         'node_names': atlas_node_names,
                         'values': node_importance,
                         'hemi': atlas_hemi,
                         'values_positive_average': node_importance_positive_average,
                         'values_positive': node_importance_positive}
    cut_value = np.mean(node_importance) + np.std(node_importance)
    node_importance[node_importance <= cut_value] = 0
    importances_nodes['values_selected'] = node_importance
    np.savetxt(os.path.join(TrainConfig.output, 'importances_nodes_selected'+ name+'.txt'), node_importance, fmt='%.18f')
    importances_nodes = pd.DataFrame(importances_nodes)
    importances_nodes.to_csv(os.path.join(TrainConfig.output, 'importances_nodes'+name+ '.csv'), sep=';', index=None)

    plotting.plot_markers(node_importance, atlas_coords, node_size=20,
                          node_threshold=np.mean(node_importance) + np.std(node_importance))
    plt.savefig(os.path.join(TrainConfig.output, 'average_nodes_importance'+ name+'.png'))
    plt.clf()
    plt.cla()
    plt.close()
    plotting.plot_markers(node_importance_positive, atlas_coords, node_size=20,
                          node_threshold=np.mean(node_importance_positive) + np.std(node_importance_positive))
    plt.savefig(os.path.join(TrainConfig.output, 'average_nodes_importance_positive'+ name+'.png'))
    plt.clf()
    plt.cla()
    plt.close()
    plotting.plot_markers(node_importance_positive_average, atlas_coords, node_size=20,
                          node_threshold=np.mean(node_importance_positive_average) + np.std(
                              node_importance_positive_average))
    plt.savefig(os.path.join(TrainConfig.output, 'average_nodes_importance_positive_average'+ name+ '.png'))
    plt.clf()
    plt.cla()
    plt.close()


def explore_attention_importances(edges, attention_file, atlas_names, atlas_coords, atlas_node_names, atlas_hemi, name=''):
    matrix_attention_mean = generate_mean_attention_matrix(edges, attention_file)
    get_attention_importances_atlas_names(matrix_attention_mean, atlas_names, atlas_coords, atlas_node_names,
                                          atlas_hemi, name=name)


def get_predictions(p, results, modality):
    df = pd.DataFrame(columns=['participant_id',
                               'age',
                               'gt', 'pred',
                               'classified',
                               'modality'])
    count = 0
    abs_err = []
    for d in os.listdir(results):
        s = os.path.join(results, d)
        if os.path.isdir(s):
            predictions = pd.read_csv(os.path.join(s, 'predictions.csv'), sep=',')
            for participant in list(predictions.filenames):
                participant_predictions = predictions[predictions.filenames == participant]
                participant_covariates = p[p.participant_id == participant]
                abs_err.append(np.abs(list(participant_predictions['pred'] - participant_predictions['gt'])))
                df.loc[count] = {'participant_id': participant,
                                 'age': float(participant_covariates['age'].iloc[0]),
                                 'gt': float(participant_predictions['gt'].iloc[0]),
                                 'pred': float(participant_predictions['pred'].iloc[0]),
                                 'modality': modality}
                count += 1
    mean_abs_error = np.mean(abs_err)
    mean_abs_error = 8.0
    classified_classes = []
    for n, part in enumerate(list(df.participant_id)):
        if abs_err[n][0] > mean_abs_error:
            classified_class = 0.0
        else:
            classified_class = 1.0
        classified_classes.append(classified_class)
    df['classified'] = classified_classes
    return df


def create_results_dataset(p, vascular, cognitive_tests, results, selected_covariate, selected_dataset, drop_outliers,
                           results_folder, modality):
    df = pd.DataFrame(columns=['participant_id',
                               'age', 'Age Group',
                               'gt', 'pred', 'error', 'classified', 'Absolute Error', 'diff_gr',
                               'modality',
                               selected_covariate])
    gt = []
    pred = []
    abs_diff = []
    diff = []
    age = []
    age_group = []
    diff_group = []
    selected_covariate_list = []
    count = 0
    for d in os.listdir(results):
        s = os.path.join(results, d)
        if os.path.isdir(s):
            predictions = pd.read_csv(os.path.join(s, 'predictions.csv'), sep=',')
            for participant in list(predictions.filenames):
                participant_predictions = predictions[predictions.filenames == participant]
                participant_covariates = p[p.participant_id == participant]
                participant_vascular = vascular[vascular.CCID == participant]
                participant_cog = cognitive_tests[cognitive_tests.ccid == participant.split('-')[1]]
                try:
                    if selected_dataset == 'vascular':
                        participant_selected_covariate = float(participant_vascular[selected_covariate].iloc[0])
                    elif selected_dataset == 'cognition':
                        participant_selected_covariate = float(participant_cog[selected_covariate].iloc[0])
                    elif selected_dataset == 'covariates':
                        participant_selected_covariate = float(participant_covariates[selected_covariate].iloc[0])
                    else:
                        pass
                    if str(participant_selected_covariate) == 'nan':
                        pass
                    else:
                        covariates = [float(participant_predictions['gt'].iloc[0]),
                                      float(participant_predictions['pred'].iloc[0]),
                                      np.abs(float(participant_predictions['gt'].iloc[0]) - float(
                                          participant_predictions['pred'].iloc[0])),
                                      float(participant_predictions['gt'].iloc[0]) - float(
                                          participant_predictions['pred'].iloc[0]),
                                      float(participant_covariates['age'].iloc[0]),
                                      participant_selected_covariate]

                        gt.append(covariates[0])
                        pred.append(covariates[1])
                        abs_diff.append(covariates[2])
                        diff.append(covariates[3])
                        if covariates[3] >= 0:
                            gr = 1
                            diff_group.append(1)
                        else:
                            gr = 0
                            diff_group.append(0)
                        if covariates[4] >= 55:
                            age_gr = 'old'
                            age_group.append('old')
                        else:
                            age_gr = 'young'
                            age_group.append('young')

                        age.append(covariates[4])
                        selected_covariate_list.append(covariates[5])
                        df.loc[count] = {'participant_id': participant,
                                         'age': covariates[4],
                                         'Age Group': age_gr,
                                         'gt': covariates[0],
                                         'pred': covariates[1],
                                         'error': covariates[3],
                                         'Absolute Error': covariates[2],
                                         'diff_gr': gr,
                                         'modality': modality,
                                         selected_covariate: covariates[5]}
                        count += 1
                except:
                    pass
        else:
            pass

    mean_abs_error = np.median(list(df['Absolute Error']))
    classified_classes = []
    for n, part in enumerate(list(df.participant_id)):
        part_s = df[df.participant_id == part]
        if float(part_s['Absolute Error'].iloc[0]) > mean_abs_error:
            classified_class = 'HEP'
        else:
            classified_class = 'LEP'
        classified_classes.append(classified_class)
    df['classified'] = classified_classes

    if drop_outliers:
        th_above = np.mean(df[selected_covariate]) + 2 * np.std(df[selected_covariate])
        th_below = np.mean(df[selected_covariate]) - 2 * np.std(df[selected_covariate])
        df_drop = df.drop(df[df[selected_covariate] > th_above].index)
        df_drop = df_drop.drop(df_drop[df_drop[selected_covariate] < th_below].index)
    else:
        df_drop = df

    if np.unique(df_drop[selected_covariate]).shape[0] > 2:
        df_drop_0 = df_drop[df_drop["Age Group"] == 'young']
        df_drop_1 = df_drop[df_drop["Age Group"] == 'old']
        slope_0, intercept_0, r_value_0, p_value_0, std_err_0 = stats.linregress(df_drop_0[selected_covariate],
                                                                                 df_drop_0['Absolute Error'])
        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(df_drop_1[selected_covariate],
                                                                                 df_drop_1['Absolute Error'])
        print('young', r_value_0, p_value_0)
        print('old', r_value_1, p_value_1)
        # plt.title('Absolute Error vs ' + selected_covariate)
        if p_value_0 < 0.05 or p_value_1 < 0.05:
            p = sns.lmplot(x=selected_covariate, y="Absolute Error",
                           hue="Age Group", data=df_drop, facet_kws=dict(sharex=False, sharey=False))
            if p_value_0 < 0.05:
                plt.text(0, np.max(df_drop["Absolute Error"]),
                         "young: r=" + str(r_value_0.round(decimals=3)) + ", p=" + str(p_value_0.round(decimals=3)))
            if p_value_1 < 0.05:
                plt.text(0, np.max(df_drop["Absolute Error"]) - 1,
                         "old: r=" + str(r_value_1.round(decimals=3)) + ", p=" + str(p_value_1.round(decimals=3)))

            plt.savefig(os.path.join(results_folder, selected_covariate + '_' + modality + '.png'))
            plt.clf()
            plt.cla()
            plt.close()

        # p = sns.lmplot(x=selected_covariate, y="Absolute Error",
        #                data=df_drop, facet_kws=dict(sharex=False, sharey=False))
        # slope_0, intercept_0, r_value_0, p_value_0, std_err_0 = stats.linregress(df_drop[selected_covariate],
        #                                                                          df_drop['Absolute Error'])
        # print('all', r_value_0, p_value_0)
        # plt.show()
    else:
        ax = sns.violinplot(x=selected_covariate, y="Absolute Error", hue="Age Group",
                            data=df_drop, density_norm="count", split=True, gap=.1, inner="quart", fill=False)
        # ax = sns.boxplot(x=selected_covariate, y="Absolute Error", hue="Age Group",
        #                     data=df_drop, notch=True, fill=False)

        plt.savefig(os.path.join(results_folder, selected_covariate + '.png'))
        plt.clf()
        plt.cla()
        plt.close()
    return df


def create_results_all_modalities(p, vascular, cognitive_tests, results_dti, results_fc, results_sc, selected_covariate,
                                  selected_dataset,
                                  drop_outliers, results_folder):
    df_dti = create_results_dataset(p, vascular, cognitive_tests, results_dti, selected_covariate, selected_dataset,
                                    drop_outliers, results_folder, modality='DTI')
    df_fc = create_results_dataset(p, vascular, cognitive_tests, results_fc, selected_covariate, selected_dataset,
                                   drop_outliers, results_folder, modality='FC')
    df_sc = create_results_dataset(p, vascular, cognitive_tests, results_sc, selected_covariate, selected_dataset,
                                   drop_outliers, results_folder, modality='SC')

    if drop_outliers:
        th_above = np.mean(df_dti[selected_covariate]) + 2 * np.std(df_dti[selected_covariate])
        th_below = np.mean(df_dti[selected_covariate]) - 2 * np.std(df_dti[selected_covariate])
        df_drop_dti = df_dti.drop(df_dti[df_dti[selected_covariate] > th_above].index)
        df_drop_dti = df_drop_dti.drop(df_drop_dti[df_drop_dti[selected_covariate] < th_below].index)
        df_drop_fc = df_fc.drop(df_fc[df_fc[selected_covariate] > th_above].index)
        df_drop_fc = df_drop_fc.drop(df_drop_fc[df_drop_fc[selected_covariate] < th_below].index)
        df_drop_sc = df_sc.drop(df_sc[df_sc[selected_covariate] > th_above].index)
        df_drop_sc = df_drop_sc.drop(df_drop_sc[df_drop_sc[selected_covariate] < th_below].index)
        df_drop = pd.concat([df_drop_dti, df_drop_fc, df_drop_sc])
    else:
        df_drop = pd.concat([df_dti, df_fc, df_sc])

    df_drop_dti = df_drop[df_drop["modality"] == 'DTI']
    df_drop_fc = df_drop[df_drop["modality"] == 'FC']
    df_drop_sc = df_drop[df_drop["modality"] == 'SC']
    slope_0, intercept_0, r_value_0, p_value_0, std_err_0 = stats.linregress(df_drop_dti[selected_covariate],
                                                                             df_drop_dti['Absolute Error'])
    slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(df_drop_fc[selected_covariate],
                                                                             df_drop_fc['Absolute Error'])
    slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(df_drop_sc[selected_covariate],
                                                                             df_drop_sc['Absolute Error'])
    print('dti', r_value_0, p_value_0)
    print('fc', r_value_1, p_value_1)
    print('sc', r_value_2, p_value_2)
    psns = sns.lmplot(x=selected_covariate, y="Absolute Error",
                      hue="modality", data=df_drop, facet_kws=dict(sharex=False, sharey=False))
    plt.text(2, np.max(df_drop['Absolute Error']),
             "dti: r=" + str(r_value_0.round(decimals=3)) + ", p=" + str(p_value_0.round(decimals=3)))
    plt.text(2, np.max(df_drop['Absolute Error']) - 1,
             "fc: r=" + str(r_value_1.round(decimals=3)) + ", p=" + str(p_value_1.round(decimals=3)))
    plt.text(2, np.max(df_drop['Absolute Error']) - 2,
             "sc: r=" + str(r_value_2.round(decimals=3)) + ", p=" + str(p_value_2.round(decimals=3)))
    if p_value_0 < 0.01:
        psns.legend.legendHandles[0].set_edgecolor('red')
        psns.legend.legendHandles[0].set_linewidth(15)
    if p_value_1 < 0.01:
        psns.legend.legendHandles[1].set_edgecolor('red')
        psns.legend.legendHandles[1].set_linewidth(15)
    if p_value_2 < 0.01:
        psns.legend.legendHandles[2].set_edgecolor('red')
        psns.legend.legendHandles[2].set_linewidth(15)
    plt.savefig(os.path.join(results_folder, selected_covariate + '_all.png'))
    plt.clf()
    plt.cla()
    plt.close()

    sns.boxplot(x="classified", y=selected_covariate, hue="Age Group",
                data=df_drop_dti, fill=False, notch=True, showcaps=False)
    plt.xlabel('Participants')
    plt.savefig(os.path.join(results_folder, selected_covariate + '_diff_by_age_DTI.png'))
    plt.clf()
    plt.cla()
    plt.close()

    sns.boxplot(x="classified", y=selected_covariate, hue="Age Group",
                data=df_drop_fc, fill=False, notch=True, showcaps=False)
    plt.xlabel('Participants')
    plt.savefig(os.path.join(results_folder, selected_covariate + '_diff_by_age_FC.png'))
    plt.clf()
    plt.cla()
    plt.close()

    sns.boxplot(x="classified", y=selected_covariate, hue="Age Group",
                data=df_drop_sc, fill=False, notch=True, showcaps=False)
    plt.xlabel('Participants')
    plt.savefig(os.path.join(results_folder, selected_covariate + '_diff_by_age_SC.png'))
    plt.clf()
    plt.cla()
    plt.close()

    return df_dti, df_fc, df_sc


def fill_dataframe(c, df_dti, df_fc, df_sc, df, selected_covariate, count, classified):
    df_dti_common = df_dti[df_dti.participant_id == c]
    df_fc_common = df_fc[df_fc.participant_id == c]
    df_sc_common = df_sc[df_sc.participant_id == c]
    df.loc[count] = {'participant_id': c,
                     'age': float(df_dti_common['age'].iloc[0]),
                     'Age Group': df_dti_common['Age Group'].iloc[0],
                     'gt': float(df_dti_common['gt'].iloc[0]),
                     'pred': float(df_dti_common['pred'].iloc[0]),
                     'Absolute Error': np.abs(
                         float(df_dti_common['gt'].iloc[0]) - float(df_dti_common['pred'].iloc[0])),
                     'modality': 'DTI',
                     'common_classified': classified,
                     selected_covariate: float(df_dti_common[selected_covariate].iloc[0])}
    count += 1
    df.loc[count] = {'participant_id': c,
                     'age': float(df_fc_common['age'].iloc[0]),
                     'Age Group': df_fc_common['Age Group'].iloc[0],
                     'gt': float(df_fc_common['gt'].iloc[0]),
                     'pred': float(df_fc_common['pred'].iloc[0]),
                     'Absolute Error': np.abs(float(df_fc_common['gt'].iloc[0]) - float(df_fc_common['pred'].iloc[0])),
                     'modality': 'FC',
                     'common_classified': classified,
                     selected_covariate: float(df_fc_common[selected_covariate].iloc[0])}
    count += 1
    df.loc[count] = {'participant_id': c,
                     'age': float(df_sc_common['age'].iloc[0]),
                     'Age Group': df_sc_common['Age Group'].iloc[0],
                     'gt': float(df_sc_common['gt'].iloc[0]),
                     'pred': float(df_sc_common['pred'].iloc[0]),
                     'Absolute Error': np.abs(float(df_sc_common['gt'].iloc[0]) - float(df_sc_common['pred'].iloc[0])),
                     'modality': 'SC',
                     'common_classified': classified,
                     selected_covariate: float(df_sc_common[selected_covariate].iloc[0])}
    count += 1
    return df, count


def create_results_missclassified(common_elements_missclassified, common_elements_classified, selected_covariate,
                                  df_dti, df_fc, df_sc, results_folder):
    df = pd.DataFrame(columns=['participant_id',
                               'age', 'Age Group', 'gt',
                               'pred', 'Absolute Error', 'modality', 'common_classified',
                               selected_covariate])
    count = 0
    for c in common_elements_missclassified:
        df, count = fill_dataframe(c, df_dti, df_fc, df_sc, df, selected_covariate, count, classified='MCP')
    for c in common_elements_classified:
        df, count = fill_dataframe(c, df_dti, df_fc, df_sc, df, selected_covariate, count, classified='CCP')

    df_correct = df[df['common_classified'] == 'CCP']
    psns = sns.lmplot(x=selected_covariate, y="age",
                      data=df_correct, facet_kws=dict(sharex=False, sharey=False))
    slope_0, intercept_0, r_value_0, p_value_0, std_err_0 = stats.linregress(df_correct[selected_covariate],
                                                                             df_correct['age'])

    plt.text(np.min(df[selected_covariate]), 80,
             "r=" + str(r_value_0.round(decimals=3)) + ", p=" + str(p_value_0.round(decimals=3)))

    plt.savefig(os.path.join(results_folder, selected_covariate + '_correct.png'))
    plt.clf()
    plt.cla()
    plt.close()

    df_correct = df[df['common_classified'] == 'MCP']
    psns = sns.lmplot(x=selected_covariate, y="age",
                      data=df_correct, facet_kws=dict(sharex=False, sharey=False))
    slope_0, intercept_0, r_value_0, p_value_0, std_err_0 = stats.linregress(df_correct[selected_covariate],
                                                                             df_correct['age'])
    plt.text(np.min(df[selected_covariate]), 80,
             "r=" + str(r_value_0.round(decimals=3)) + ", p=" + str(p_value_0.round(decimals=3)))

    plt.savefig(os.path.join(results_folder, selected_covariate + '_incorrect.png'))
    plt.clf()
    plt.cla()
    plt.close()

    # plt.scatter(error_dti, selected_covariate_common, label='dti')
    # plt.scatter(error_fc, selected_covariate_common, label='fc')
    # plt.scatter(error_sc, selected_covariate_common, label='sc')
    # plt.legend()
    # plt.show()

    # sns.boxplot(x="common_classified", y=selected_covariate,
    #                     data=df, fill=False,notch=True, showcaps=False)
    # plt.xlabel('Participants')
    # plt.savefig(os.path.join(results_folder, selected_covariate + '_diff.png'))
    # plt.clf()
    # plt.cla()
    # plt.close()
    #
    # sns.boxplot(x="common_classified", y=selected_covariate, hue="Age Group",
    #                     data=df, fill=False,notch=True, showcaps=False)
    # plt.xlabel('Participants')
    # plt.savefig(os.path.join(results_folder, selected_covariate + '_diff_by_age.png'))
    # plt.clf()
    # plt.cla()
    # plt.close()
    return df
