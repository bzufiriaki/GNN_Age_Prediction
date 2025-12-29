import datetime

import numpy as np

from utils import *
from datasets.generate_datasets import *
from datasets.utils import *
from models.models import *
from train import train_model_atrophy
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold
from keras.utils.np_utils import to_categorical

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.test.gpu_device_name():
    print('Default GPU device correct')
else:
    print('Please install GPU version of TF')

init_time = datetime.datetime.now()

if os.path.exists(TrainConfigAtrophy.output):
    pass
else:
    os.mkdir(TrainConfigAtrophy.output)

if os.path.exists(TrainConfigAtrophy.logs):
    pass
else:
    os.mkdir(TrainConfigAtrophy.logs)

# Atlas
atlas_coords, atlas_names, atlas_node_names, atlas_hemi, atlas_node_names_hemi, atlas_ggseg_names = load_atlas(
    GeneralConfigAtrophy.atlas)

# Input data
dataset_dir = GeneralConfigAtrophy.dataset
if GeneralConfigAtrophy.dataset_type == 'LEMON':
    age_range = TrainConfigAtrophy.age_range_lemon
elif GeneralConfigAtrophy.dataset_type == 'HABS':
    age_range = TrainConfigAtrophy.age_range_habs
else:
    age_range = TrainConfigAtrophy.age_range

participants_details = pd.read_csv(os.path.join(dataset_dir, GeneralConfigAtrophy.dataset_details), sep=';')
file_type = GeneralConfigAtrophy.file_type_sc
dir_matrices = os.path.join(dataset_dir, GeneralConfigAtrophy.connectivity_matrices_sc)
# Generate simulated data
print('Generating Data')
matrices, \
    ages, \
    filenames = get_dataset_atrophy_array(dir_matrices, GeneralConfigAtrophy.gm_values,
                                  participants_details,file_type,
                                  age_range=age_range)
gm_features = np.array(matrices)
try:
    loss_name = TrainConfigAtrophy.loss.name
except:
    loss_name = TrainConfigAtrophy.loss.__name__
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
if TrainConfigAtrophy.split == "stratified":
    if GeneralConfigAtrophy.dataset_type == "HABS":
        stratify_label = ages
    else:
        stratify_label = get_stratify_division_ages(ages)
    sss = StratifiedShuffleSplit(n_splits=5,
                                 test_size=0.1, random_state=42)
    sss.get_n_splits(gm_features, stratify_label)
    for i, (train_index, test_index) in enumerate(sss.split(gm_features, stratify_label)):
        print('Cross Validation Iteration ', str(c))
        indices_pd_train["Fold " + str(c)] = train_index
        indices_pd_test["Fold " + str(c)] = test_index
        metrics, model = train_model_atrophy(os.path.join(TrainConfigAtrophy.output, str(c) + '/'),
                                             gm_features,
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
    best_iteration = 1
    for r in range(2):
        print('Iteration ', str(r))
        count = 0
        #sss = ShuffleSplit(n_splits=TrainConfigAtrophy.n_splits, test_size=0.1, random_state=r)
        splits = KFold(TrainConfigAtrophy.n_splits, shuffle=True, random_state=r).split(gm_features)
        r2_iteration = []
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
            metrics, model, prediction, y_test_np, filenames_test, hist = train_model_atrophy(
                os.path.join(TrainConfigAtrophy.output, str(c) + '/'),
                gm_features,
                ages,
                filenames,
                train_index,
                test_index)
            hist_iteration.append(hist)
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
                output = os.path.join(TrainConfigAtrophy.output, str(it) + '/')
                if os.path.exists(output):
                    pass
                else:
                    os.mkdir(output)
                prediction = prediction_iteration[it]
                y_test_np = y_test_np_iteration[it]
                filenames_test = filenames_test_iteration[it]
                output_np = np.squeeze(np.array(prediction[0]))
                plot_metrics(hist_iteration[it], output, metrics=TrainConfigAtrophy.metrics, title='Metrics')
                data = pd.DataFrame({"filenames": filenames_test, "gt": y_test_np, "pred": output_np})
                data.to_csv(os.path.join(output, "predictions.csv"))

                if TrainConfigAtrophy.explore_attentions:
                    prediction_reduced = pd.DataFrame(tf.reduce_mean(prediction[1], axis=0)[:, 0, :])
                    prediction_reduced.to_csv(os.path.join(output, 'nodes_attention.csv'),sep=';', index=None)

            indices_pd_train.to_csv(os.path.join(TrainConfigAtrophy.output, 'indices_train.csv'), sep=';')
            indices_pd_test.to_csv(os.path.join(TrainConfigAtrophy.output, 'indices_test.csv'), sep=';')

            with open(os.path.join(TrainConfigAtrophy.output, 'output.txt'), 'w') as fp:
                fp.write("Cross validation" + "\n")
                fp.write("Best iteration" + str(r) + "\n")
                fp.write("number of edges" + ": " + str(gm_features.shape) + "\n")
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

with open(os.path.join(TrainConfigAtrophy.output, 'output.txt'), 'a') as fp:
    fp.write("Best iteration: " + str(best_iteration) + "\n")
    fp.write('epochs:' + str(TrainConfigAtrophy.epochs) + "\n")
    fp.write('learning_rate:' + str(TrainConfigAtrophy.learning_rate) + "\n")
    fp.write('batch_size:' + str(TrainConfigAtrophy.batch_size) + "\n")
    fp.write('optimizer:' + str(TrainConfigAtrophy.optimizer) + "\n")
    fp.write('loss:' + str(TrainConfigAtrophy.loss) + "\n")
    fp.write('n_heads:' + str(TrainConfigAtrophy.n_heads) + "\n")
    fp.write('dense_layer_dimensions:' + str(TrainConfigAtrophy.dense_layer_dimensions) + "\n")
    fp.write('normalization:' + str(TrainConfigAtrophy.normalization) + "\n")
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
               metric_names[2]: r_all}
all_metrics = pd.DataFrame(all_metrics)
all_metrics.to_csv(os.path.join(TrainConfigAtrophy.output, 'all_metrics' + '.csv'), sep=';', index=None)

if TrainConfigAtrophy.explore_attentions:
    attention_file = "nodes_attention.csv"
    gm_attention_all = []
    for i in range(TrainConfigAtrophy.n_splits):
        gm_attention = pd.read_csv(os.path.join(TrainConfigAtrophy.output, str(i), attention_file))
        # get from attention
        gm_attention_heads = []
        for n_heads in range(gm_attention.shape[0]):
            gm_attention_heads.append(list(gm_attention.iloc[n_heads])[1:])
        gm_attention_all.append(np.mean(gm_attention_heads, axis=0))
    node_importance_positive = np.sum(gm_attention_all, axis=0)
    node_importance_positive = np.nan_to_num(node_importance_positive)
    node_importance_positive = node_importance_positive / np.max(node_importance_positive)
    importances_nodes = {'nodes': atlas_names,
                         'node_names': atlas_node_names,
                         'hemi': atlas_hemi,
                         'values_positive': node_importance_positive}
    importances_nodes = pd.DataFrame(importances_nodes)
    importances_nodes.to_csv(os.path.join(TrainConfigAtrophy.output, 'importances_nodes.csv'), sep=';', index=None)

# selected_edges = save_selected_edges_by_subject(data, b, aal2_node_names_hemi, out_path)
# combine_edges_and_cognitive(selected_edges, cognitive, out_path)
