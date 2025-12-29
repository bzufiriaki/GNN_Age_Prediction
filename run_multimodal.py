import datetime

import matplotlib.pyplot as plt
import numpy as np

from utils import *
from datasets.generate_datasets import *
from datasets.utils import *
from models.models import *
from train import train_model, train_model_cv, train_model_multimodal, \
    train_model_multimodel_cv, train_model_multimodal_combined, train_model_multimodel_combined_cv
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.test.gpu_device_name():
    print('Default GPU device correct')
else:
    print('Please install GPU version of TF')

init_time = datetime.datetime.now()

if os.path.exists(TrainConfig.output):
    pass
else:
    os.mkdir(TrainConfig.output)

if os.path.exists(TrainConfig.logs):
    pass
else:
    os.mkdir(TrainConfig.logs)

# Atlas
atlas_coords, atlas_names, atlas_node_names, atlas_hemi, atlas_node_names_hemi, atlas_ggseg_names = load_atlas(
    GeneralConfig.atlas)

# Input data
dataset_dir = GeneralConfig.dataset
if GeneralConfig.dataset_type == 'LEMON':
    age_range = TrainConfig.age_range_lemon
elif GeneralConfig.dataset_type == 'HABS':
    age_range = TrainConfig.age_range_habs
else:
    age_range = TrainConfig.age_range

participants_details_common = pd.read_csv(os.path.join(dataset_dir, GeneralConfig.dataset_details), sep=';')

threshold_by_region = GeneralConfig.threshold_by_region

file_type_conn = GeneralConfig.file_type_conn
dir_conn = os.path.join(dataset_dir, GeneralConfig.connectivity_matrices_con)
thresholded_mean_matrix_conn = GeneralConfig.thresholded_mean_matrix_con
file_type_fc = GeneralConfig.file_type_fc
dir_fc = os.path.join(dataset_dir, GeneralConfig.connectivity_matrices_fc)
thresholded_mean_matrix_fc = GeneralConfig.thresholded_mean_matrix_fc
file_type_sc = GeneralConfig.file_type_sc
dir_sc = os.path.join(dataset_dir, GeneralConfig.connectivity_matrices_sc)
thresholded_mean_matrix_sc = GeneralConfig.thresholded_mean_matrix_sc

# Generate simulated data
print('Generating Data')
matrices_con, \
    edges_con, \
    edge_values_con, \
    ages_con, \
    filenames_con = get_dataset_array(dir_conn,
                                      participants_details_common,
                                      threshold=TrainConfig.threshold,
                                      threshold_by_matrix=thresholded_mean_matrix_conn,
                                      threshold_by_region=threshold_by_region,
                                      age_range=age_range,
                                      file_type=file_type_conn)
matrices_fc, \
    edges_fc, \
    edge_values_fc, \
    ages_fc, \
    filenames_fc = get_dataset_array(dir_fc,
                                     participants_details_common,
                                     threshold=TrainConfig.threshold,
                                     threshold_by_matrix=thresholded_mean_matrix_fc,
                                     threshold_by_region=threshold_by_region,
                                     age_range=age_range,
                                     file_type=file_type_fc)

matrices_sc, \
    edges_sc, \
    edge_values_sc, \
    ages_sc, \
    filenames_sc = get_dataset_array(dir_sc,
                                     participants_details_common,
                                     threshold=TrainConfig.threshold,
                                     threshold_by_matrix=thresholded_mean_matrix_sc,
                                     threshold_by_region=threshold_by_region,
                                     age_range=age_range,
                                     file_type=file_type_sc)

edge_features_con = np.array(edge_values_con)
edges_con = np.array(edges_con)

edge_features_fc = np.array(edge_values_fc)
edges_fc = np.array(edges_fc)

edge_features_sc = np.array(edge_values_sc)
edges_sc = np.array(edges_sc)

print('Training with density SC:', (edge_features_sc.shape[1] * 2 / (GeneralConfig.atlas_coords ** 2)) * 100)
print('Training with density FC:', (edge_features_fc.shape[1] * 2 / (GeneralConfig.atlas_coords ** 2)) * 100)
print('Training with density CON:', (edge_features_con.shape[1] * 2 / (GeneralConfig.atlas_coords ** 2)) * 100)

ages = np.expand_dims(np.array(ages_sc), 1)
ages = ages.astype('float32')

try:
    loss_name = TrainConfig.loss.name
except:
    loss_name = TrainConfig.loss.__name__
if loss_name == 'binary_crossentropy':
    metric_names = ['auc', 'acc', 'sensitivity', 'specificity']
else:
    metric_names = ['r2', 'mae', 'mse', 'r']


tf.keras.backend.clear_session()
c = 1
average_saliencies = []
indices_pd_train = pd.DataFrame()
indices_pd_test = pd.DataFrame()
metrics_all = [[], [], [], []]

r2_all = list()
r_all = list()
mae_all = list()
mse_all = list()
prev_r2_score = 0.0
best_iteration = -1


for r in range(100):
    count = 0
    sss = ShuffleSplit(n_splits=TrainConfig.n_splits, test_size=0.1, random_state=r)
    splits = KFold(TrainConfig.n_splits, shuffle=True, random_state=r).split(edge_features_con)
    if r == 1000:
        for i, (train_index, test_index) in enumerate(splits):
            print('Cross Validation Iteration ', str(c))
            metrics, model = train_model_multimodel_combined_cv(os.path.join(TrainConfig.output, str(c) + '/'),
                                                                atlas_coords,
                                                                edges_con,
                                                                edge_features_con,
                                                                edges_sc,
                                                                edge_features_sc,
                                                                edges_fc,
                                                                edge_features_fc,
                                                                ages,
                                                                filenames_con,
                                                                train_index,
                                                                test_index)
            c += 1
            metrics_all[0].append(metrics[0])
            metrics_all[1].append(metrics[1])
            metrics_all[2].append(metrics[2])
            metrics_all[3].append(metrics[3])
    else:
        r2_iteration = []
        edges_test_iteration = []
        prediction_iteration = []
        y_test_np_iteration = []
        filenames_test_iteration = []
        hist_iteration = []
        indices_pd_train = pd.DataFrame()
        indices_pd_test = pd.DataFrame()
        metrics_all = [[], [], [], []]
        c = 1
        for i, (train_index, test_index) in enumerate(splits):
            print('Cross Validation Iteration ', str(c))
            metrics, model, edges_test, prediction, y_test_np, filenames_test, hist = train_model_multimodal_combined(os.path.join(TrainConfig.output, str(c) + '/'),
                                                                                           edges_con,
                                                                                           edge_features_con,
                                                                                           edges_sc,
                                                                                           edge_features_sc,
                                                                                           edges_fc,
                                                                                           edge_features_fc,
                                                                                           ages,
                                                                                           filenames_con,
                                                                                           train_index,
                                                                                           test_index)
            hist_iteration.append(hist)
            edges_test_iteration.append(edges_test)
            prediction_iteration.append(prediction)
            y_test_np_iteration.append(y_test_np)
            filenames_test_iteration.append(filenames_test)
            r2_iteration.append(metrics[0])
            r2_all.append(metrics[0])
            mae_all.append(metrics[1])
            mse_all.append(metrics[2])
            r_all.append(metrics[3])
            c += 1
            metrics_all[0].append(metrics[0])
            metrics_all[1].append(metrics[1])
            metrics_all[2].append(metrics[2])
            metrics_all[3].append(metrics[3])
        if np.mean(r2_iteration) > prev_r2_score:
            best_iteration = r
            prev_r2_score = np.mean(r2_iteration)
            for it in range(10):
                output = os.path.join(TrainConfig.output, str(it) + '/')
                if os.path.exists(output):
                    pass
                else:
                    os.mkdir(output)
                edges_test = edges_test_iteration[it]
                prediction = prediction_iteration[it]
                y_test_np = y_test_np_iteration[it]
                filenames_test = filenames_test_iteration[it]
                output_np = np.squeeze(np.array(prediction[0]))
                edges_pd = pd.DataFrame(edges_test[0])
                edges_pd.to_csv(os.path.join(output, 'edges_all.csv'))
                plot_metrics(hist_iteration[it], output, metrics=TrainConfig.metrics, title='Metrics')
                data = pd.DataFrame({"filenames": filenames_test, "gt": y_test_np, "pred": output_np})
                data.to_csv(os.path.join(output, "predictions.csv"))

                if TrainConfig.edge_attention:
                    prediction_attention = prediction[1]

                    prediction_reduced_multi = pd.DataFrame(
                        tf.reduce_mean(prediction_attention[:, :, :, :, 0], axis=0)[:, 0, :])
                    edges_predicted_multi = np.array(
                        transform_edges_to_matrix(edges_test, prediction_attention[:, :, :, :, 0]))
                    prediction_reduced_multi.to_csv(os.path.join(output, 'edges_attention_multi.csv'))
                    attention_heatmap = average_explanations(edges_predicted_multi, output, atlas_coords,
                                                             name='_multi_' + TrainConfig.target)


            with open(os.path.join(TrainConfig.output, 'output.txt'), 'w') as fp:
                fp.write("Cross validation" + "\n")
                fp.write("Best iteration" + ": " + str(best_iteration) + "\n")
                fp.write("thresholded_mean_matrix" + ": " + str(thresholded_mean_matrix_conn) + "\n")
                fp.write("number of edges" + ": " + str(edges_con.shape) + "\n")
                fp.write(metric_names[0] + ": " + str(metrics_all[0]) + "\n")
                fp.write(metric_names[1] + ": " + str(metrics_all[1]) + "\n")
                fp.write(metric_names[2] + ": " + str(metrics_all[2]) + "\n")
                fp.write(metric_names[3] + ": " + str(metrics_all[3]) + "\n")

                fp.write(metric_names[0] + " mean: " + str(np.mean(metrics_all[0])) + "\n")
                fp.write(metric_names[1] + "mean: " + str(np.mean(metrics_all[1])) + "\n")
                fp.write(metric_names[2] + "mean: " + str(np.mean(metrics_all[2])) + "\n")
                fp.write(metric_names[3] + "mean: " + str(np.mean(metrics_all[3])) + "\n")

                fp.write(metric_names[0] + " std: " + str(np.std(metrics_all[0])) + "\n")
                fp.write(metric_names[1] + "std: " + str(np.std(metrics_all[1])) + "\n")
                fp.write(metric_names[2] + "std: " + str(np.std(metrics_all[2])) + "\n")
                fp.write(metric_names[3] + "std: " + str(np.std(metrics_all[3])) + "\n")

with open(os.path.join(TrainConfig.output, 'output.txt'), 'a') as fp:

    fp.write("density: " + str((edge_features_con.shape[1] * 2 / (GeneralConfig.atlas_coords ** 2)) * 100) + "\n")
    fp.write("Best iteration: " + str(best_iteration) + "\n")
    fp.write('epochs:' + str(TrainConfig.epochs) + "\n")
    fp.write('learning_rate:' + str(TrainConfig.learning_rate) + "\n")
    fp.write('batch_size:' + str(TrainConfig.batch_size) + "\n")
    fp.write('optimizer:' + str(TrainConfig.optimizer) + "\n")
    fp.write('loss:' + str(TrainConfig.loss) + "\n")
    fp.write('n_heads:' + str(TrainConfig.n_heads) + "\n")
    fp.write('base_layer_dimensions:' + str(TrainConfig.base_layer_dimensions) + "\n")
    fp.write('dense_layer_dimensions:' + str(TrainConfig.dense_layer_dimensions) + "\n")
    fp.write('normalization:' + str(TrainConfig.normalization) + "\n")
    fp.write("All iterations" + "\n")
    fp.write(metric_names[0] + ": " + str(np.mean(r2_all)) + " + " + str(np.std(r2_all)) + "\n")
    fp.write(metric_names[1] + ": " + str(np.mean(mae_all)) + " + " + str(np.std(mae_all)) + "\n")
    fp.write(metric_names[2] + ": " + str(np.mean(mse_all)) + " + " + str(np.std(mse_all)) + "\n")
    fp.write(metric_names[3] + ": " + str(np.mean(r_all)) + " + " + str(np.std(r_all)) + "\n")
    fp.write(metric_names[0] + ": " + str(r2_all) + "\n")
    fp.write(metric_names[1] + ": " + str(mae_all) + "\n")
    fp.write(metric_names[2] + ": " + str(mse_all) +  "\n")
    fp.write(metric_names[3] + ": " + str(r_all) +  "\n")

if TrainConfig.explore_attentions:
    edges = pd.read_csv(os.path.join(TrainConfig.output, "1", 'edges_all.csv'))
    edges_x = list(edges[edges.columns[1]])
    edges_y = list(edges[edges.columns[2]])
    all_edges = []
    for e in range(len(edges)):
        e_selected = [edges_x[e], edges_y[e]]
        e_con = np.argwhere(np.logical_and(edges_con[0, :, 0] == e_selected[0], edges_con[0, :, 1] == e_selected[1]))
        e_sc = np.argwhere(np.logical_and(edges_sc[0, :, 0] == e_selected[0], edges_sc[0, :, 1] == e_selected[1]))
        e_fc = np.argwhere(np.logical_and(edges_fc[0, :, 0] == e_selected[0], edges_fc[0, :, 1] == e_selected[1]))
        all_edges_selected = []
        if e_con.size == 1:
            all_edges_selected.append('con')
        else:
            pass
        if e_sc.size == 1:
            all_edges_selected.append('sc')
        else:
            pass
        if e_fc.size == 1:
            all_edges_selected.append('fc')
        else:
            pass
        all_edges.append(all_edges_selected)

    attention_file = "edges_attention_multi.csv"
    explore_attention_importances(edges, attention_file, atlas_ggseg_names, atlas_coords, atlas_node_names, atlas_hemi,
                                  name='_multi')

    importances_edges_multi = pd.read_csv(os.path.join(TrainConfig.output, 'importances_edges_multi.csv'), sep=';')
    importances_edges_multi['Modality'] = all_edges
    importances_edges_multi.to_csv(os.path.join(TrainConfig.output, 'importances_edges_multi.csv'), sep=';', index=None)
    unique, counts = np.unique(all_edges, return_counts=True)
    th_all_edges = np.mean(importances_edges_multi.Values) + np.std(importances_edges_multi.Values)
    importances_edges_multi_th = importances_edges_multi[importances_edges_multi.Values >= th_all_edges]
    unique_th_multi, counts_th_multi = np.unique(importances_edges_multi_th.Modality, return_counts=True)
    densities = [0.01, 0.05, 0.1, 0.2, 0.3]
    sorted_importances_edges_multi = importances_edges_multi.sort_values(by='Values', ascending=False)
    counts_d = []
    for d in densities:
        sorted_importances_edges_multi_d = sorted_importances_edges_multi[
                                           0:int(sorted_importances_edges_multi.shape[0] * d)]
        unique_th_multi, counts_th_multi = np.unique(sorted_importances_edges_multi_d.Modality, return_counts=True)
        count_th_match_multi = []
        for u in unique:
            loc = [index for (index, item) in enumerate(unique_th_multi) if item == u]
            if len(loc) == 1:
                count_th_match_multi.append(counts_th_multi[loc][0])
            else:
                count_th_match_multi.append(0)
        counts_d.append(count_th_match_multi)

    unique_th_multi, counts_th_multi = np.unique(importances_edges_multi_th.Modality, return_counts=True)
    count_th_match_multi = []
    for u in unique:
        loc = [index for (index, item) in enumerate(unique_th_multi) if item == u]
        if len(loc) == 1:
            count_th_match_multi.append(counts_th_multi[loc][0])
        else:
            count_th_match_multi.append(0)

    normalized_data = np.expand_dims(
        np.array(sorted_importances_edges_multi.Values / np.max(sorted_importances_edges_multi.Values)),
        axis=1)

    summary = {'unique': unique,
               'counts': counts,
               'counts_th': count_th_match_multi,
               'counts_multi_0.01': counts_d[0],
               'counts_multi_0.05': counts_d[1],
               'counts_multi_0.1': counts_d[2],
               'counts_multi_0.2': counts_d[3]}
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(TrainConfig.output, 'summary.csv'), sep=';', index=None)

    plt.close()
    plt.cla()
    plt.clf()
    range_low = 0
    range_high = np.sum(counts)
    selected_data = normalized_data[range_low:range_high]
    plt.fill_between(list(range(range_low, range_high)), np.mean(selected_data, axis=1), where=(
                np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['con', 'sc', 'fc']"),
                     color='darkblue', alpha=0.5, label=str(['con', 'sc', 'fc']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1), where=(
                np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['sc', 'fc']"),
                     color='black', alpha=0.5, label=str(['sc', 'fc']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1), where=(
                np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['con', 'sc']"),
                     color='darkred', alpha=0.5, label=str(['con', 'sc']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1), where=(
                np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['con', 'fc']"),
                     color='lightblue', alpha=0.5, label=str(['con', 'fc']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1),
                     where=(np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['con']"),
                     color='orange', alpha=0.5, label=str(['con']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1),
                     where=(np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['sc']"),
                     color='green', alpha=0.5, label=str(['sc']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1),
                     where=(np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['fc']"),
                     color='pink', alpha=0.5, label=str(['fc']), step='pre', linewidth=0)
    plt.legend()
    plt.savefig(os.path.join(TrainConfig.output, 'multimoda_edge_feature_importance_all.svg'))

    plt.close()
    plt.cla()
    plt.clf()
    range_low = 0
    range_high = 400
    selected_data = normalized_data[range_low:range_high]
    plt.fill_between(list(range(range_low, range_high)), np.mean(selected_data, axis=1), where=(
                np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['con', 'sc', 'fc']"),
                     color='darkblue', alpha=0.5, label=str(['con', 'sc', 'fc']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1), where=(
                np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['sc', 'fc']"),
                     color='black', alpha=0.5, label=str(['sc', 'fc']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1), where=(
                np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['con', 'sc']"),
                     color='darkred', alpha=0.5, label=str(['con', 'sc']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1), where=(
                np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['con', 'fc']"),
                     color='lightblue', alpha=0.5, label=str(['con', 'fc']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1),
                     where=(np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['con']"),
                     color='orange', alpha=0.5, label=str(['con']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1),
                     where=(np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['sc']"),
                     color='green', alpha=0.5, label=str(['sc']), step='pre', linewidth=0)
    plt.fill_between(list(range(range_high - range_low)), np.mean(selected_data, axis=1),
                     where=(np.array(list(sorted_importances_edges_multi.Modality[range_low:range_high])) == "['fc']"),
                     color='pink', alpha=0.5, label=str(['fc']), step='pre', linewidth=0)
    plt.legend()
    plt.savefig(os.path.join(TrainConfig.output, 'multimoda_edge_feature_importance_from44.svg'))

    matrix_edge_importance = np.zeros((len(atlas_names), len(atlas_names)))
    for e in range(len(edges)):
        if importances_edges_multi.Modality[e] ==  "['con', 'sc', 'fc']":
            matrix_edge_importance[edges_x[e], edges_y[e]] = importances_edges_multi.Values[e]
    plt.close()
    plt.cla()
    plt.clf()
    plotting.plot_connectome(matrix_edge_importance + matrix_edge_importance.T, atlas_coords, node_size=2,
                            node_color='black', edge_cmap='Purples')
    plt.savefig(os.path.join(TrainConfig.output, 'average_edges_importance_multi_con_sc_fc2.svg'))