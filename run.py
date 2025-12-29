import datetime

import numpy as np

from utils import *
from datasets.generate_datasets import *
from datasets.utils import *
from models.models import *
from train import train_model, train_model_cv
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold
from keras.utils.np_utils import to_categorical

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

participants_details = pd.read_csv(os.path.join(dataset_dir, GeneralConfig.dataset_details), sep=';')

threshold_by_region = GeneralConfig.threshold_by_region
if TrainConfig.edge_features == 'fc':
    file_type = GeneralConfig.file_type_fc
    dir = os.path.join(dataset_dir, GeneralConfig.connectivity_matrices_fc)
    thresholded_mean_matrix = GeneralConfig.thresholded_mean_matrix_fc
elif TrainConfig.edge_features == 'con':
    file_type = GeneralConfig.file_type_conn
    dir = os.path.join(dataset_dir, GeneralConfig.connectivity_matrices_con)
    thresholded_mean_matrix = GeneralConfig.thresholded_mean_matrix_con
elif TrainConfig.edge_features == 'sc':
    file_type = GeneralConfig.file_type_sc
    dir = os.path.join(dataset_dir, GeneralConfig.connectivity_matrices_sc)
    thresholded_mean_matrix = GeneralConfig.thresholded_mean_matrix_sc
elif TrainConfig.edge_features == 'combined':
    file_type_conn = GeneralConfig.file_type_conn
    dir_conn = os.path.join(dataset_dir, GeneralConfig.connectivity_matrices_con)
    thresholded_mean_matrix_conn = GeneralConfig.thresholded_mean_matrix_fc
    file_type_fc = GeneralConfig.file_type_fc
    dir_fc = os.path.join(dataset_dir, GeneralConfig.connectivity_matrices_fc)
    thresholded_mean_matrix_fc = GeneralConfig.thresholded_mean_matrix_con
else:
    file_type = GeneralConfig.file_type_conn
    dir = os.path.join(dataset_dir, GeneralConfig.connectivity_matrices_con)
    thresholded_mean_matrix = GeneralConfig.thresholded_mean_matrix

# Generate simulated data
print('Generating Data')
matrices, \
    edges, \
    edge_values, \
    ages, \
    filenames = get_dataset_array(dir,
                                  participants_details,
                                  threshold=TrainConfig.threshold,
                                  threshold_by_matrix=thresholded_mean_matrix,
                                  threshold_by_region=threshold_by_region,
                                  age_range=age_range,
                                  file_type=file_type)

edge_features = np.array(edge_values)
edges = np.array(edges)
print('Training with density:', (edge_features.shape[1] * 2 / (GeneralConfig.atlas_coords ** 2)) * 100)

try:
    loss_name = TrainConfig.loss.name
except:
    loss_name = TrainConfig.loss.__name__
if loss_name == 'binary_crossentropy':
    metric_names = ['auc', 'acc', 'sensitivity', 'specificity']
    ages = np.expand_dims(np.array(ages), 1)
    ages = ages.astype('float32')
elif loss_name == 'categorical_crossentropy':
    ages = to_categorical(ages, num_classes=None)
else:
    ages = np.expand_dims(np.array(ages), 1)
    ages = ages.astype('float32')
    metric_names = ['r2', 'mae', 'mse', 'r']

tf.keras.backend.clear_session()
c = 1
average_saliencies = []
indices_pd_train = pd.DataFrame()
indices_pd_test = pd.DataFrame()
metrics_all = [[], [], [], []]
if TrainConfig.split == "stratified":
    if GeneralConfig.dataset_type == "HABS":
        stratify_label = ages
    else:
        stratify_label = get_stratify_division_ages(ages)
    sss = StratifiedShuffleSplit(n_splits=5,
                                 test_size=0.1, random_state=42)
    sss.get_n_splits(edge_features, stratify_label)
    for i, (train_index, test_index) in enumerate(sss.split(edge_features, stratify_label)):
        print('Cross Validation Iteration ', str(c))
        indices_pd_train["Fold " + str(c)] = train_index
        indices_pd_test["Fold " + str(c)] = test_index
        metrics, model = train_model(os.path.join(TrainConfig.output, str(c) + '/'),
                                     atlas_coords,
                                     edges,
                                     edge_features,
                                     ages,
                                     filenames,
                                     train_index,
                                     test_index)
        metrics_all[0].append(metrics[0])
        metrics_all[1].append(metrics[1])
        metrics_all[2].append(metrics[2])
        metrics_all[3].append(metrics[3])
        c += 1
else:

    r2_all = list()
    r_all = list()
    mae_all = list()
    mse_all = list()
    prev_r2_score = 0.0

    for r in range(100):
        print('Iteration ', str(r))
        count = 0
        #sss = ShuffleSplit(n_splits=TrainConfig.n_splits, test_size=0.1, random_state=r)
        splits = KFold(TrainConfig.n_splits, shuffle=True, random_state=r).split(edge_features)
        if r == 1000:
            for i, (train_index, test_index) in enumerate(splits):
                print('Cross Validation Iteration ', str(c))
                indices_pd_train["Fold " + str(c)] = train_index
                indices_pd_test["Fold " + str(c)] = test_index
                metrics, model = train_model_cv(os.path.join(TrainConfig.output, str(c) + '/'),
                                                atlas_coords,
                                                edges,
                                                edge_features,
                                                ages,
                                                filenames,
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
                indices_pd_train["Fold " + str(c)] = train_index
                indices_pd_test["Fold " + str(c)] = test_index
                metrics, model, edges_test, prediction, y_test_np, filenames_test, hist = train_model(
                    os.path.join(TrainConfig.output, str(c) + '/'),
                    edges,
                    edge_features,
                    ages,
                    filenames,
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
                edges_pd.to_csv(os.path.join(output, 'edges.csv'))
                plot_metrics(hist_iteration[it], output, metrics=TrainConfig.metrics, title='Metrics')
                data = pd.DataFrame({"filenames": filenames_test, "gt": y_test_np, "pred": output_np})
                data.to_csv(os.path.join(output, "predictions.csv"))

                if TrainConfig.edge_attention:
                    prediction_reduced = pd.DataFrame(tf.reduce_mean(prediction[1], axis=0)[:, 0, :])
                    edges_predicted = np.array(transform_edges_to_matrix(edges_test, prediction[1]))
                    prediction_reduced.to_csv(os.path.join(output, 'edges_attention.csv'))
                    attention_heatmap = average_explanations(edges_predicted, output, atlas_coords,
                                                             name='_' + TrainConfig.target)

            indices_pd_train.to_csv(os.path.join(TrainConfig.output, 'indices_train.csv'), sep=';')
            indices_pd_test.to_csv(os.path.join(TrainConfig.output, 'indices_test.csv'), sep=';')

            with open(os.path.join(TrainConfig.output, 'output.txt'), 'w') as fp:
                fp.write("Cross validation" + "\n")
                fp.write("Best iteration" + str(r) + "\n")
                fp.write("thresholded_mean_matrix" + ": " + str(thresholded_mean_matrix) + "\n")
                fp.write("number of edges" + ": " + str(edges.shape) + "\n")
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
    fp.write("density: " + str((edge_features.shape[1] * 2 / (GeneralConfig.atlas_coords ** 2)) * 100) + "\n")
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
    fp.write(metric_names[2] + ": " + str(mse_all) + "\n")
    fp.write(metric_names[3] + ": " + str(r_all) + "\n")

all_metrics = {metric_names[0]: r2_all,
               metric_names[1]: mae_all,
               metric_names[2]: mse_all,
               metric_names[3]: r_all}
all_metrics = pd.DataFrame(all_metrics)
all_metrics.to_csv(os.path.join(TrainConfig.output, 'all_metrics' + '.csv'), sep=';', index=None)

if TrainConfig.explore_attentions:
    attention_file = "edges_attention.csv"
    edges = pd.read_csv(os.path.join(TrainConfig.output, "1", 'edges.csv'))
    explore_attention_importances(edges, attention_file, atlas_ggseg_names, atlas_coords, atlas_node_names, atlas_hemi)

# selected_edges = save_selected_edges_by_subject(data, b, aal2_node_names_hemi, out_path)
# combine_edges_and_cognitive(selected_edges, cognitive, out_path)
