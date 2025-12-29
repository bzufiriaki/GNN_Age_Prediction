import tensorflow as tf
import os


class GeneralConfigAtrophy(object):
    dataset = r"...\Brainnetome"
    dataset_type = "CamCAN"  # os.path.basename(os.path.normpath(dataset))
    atlas = os.path.join(dataset, 'bna_atlas.xlsx')
    atlas_coords = 246
    gm_values = os.path.join(dataset, ".\gm_value_bn.xlsx")
    connectivity_matrices_con = "conn_mats_con"
    connectivity_matrices_fc = "conn_mats_fun"
    connectivity_matrices_sc = ".\conn_mats_gm"
    if dataset_type == "LEMON":
        dataset_details = "Participants_LEMON.csv"
    else:
        dataset_details = "Participants_CamCAN.csv"
    run = 'train'  # 'test'
    file_type_fc = '.xlsx'
    file_type_conn = '.xlsx'
    file_type_sc = '.xlsx'

class TrainConfigAtrophy(object):
    model = "EdgeDeepBrain"
    target = 'age'
    dense_layer_dimensions = (8, 16)
    # normalization = "Standard"
    split = "fold"  # "stratified" "fold"
    stratify_label = 'age'
    augmentation = False
    normalization = "MinMax"
    n_heads = 2
    loss = tf.keras.losses.mae  # tf.keras.losses.mae # tf.keras.losses.binary_crossentropy
    epochs = 100
    batch_size = 8
    learning_rate = 0.005
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  #
    logs = './logs/'  # folder to save tensorboard logs
    output = r'.\output_atrophy'  # folder to save output models
    threshold = 1.0
    attention = True
    age_range_lemon = ['20-25']
    age_range = [30, 90]
    age_range_habs = [60, 95]
    cutoff_age_eval = 55
    fdr = 0.05
    n_splits = 10
    # metrics = ['mae', 'mse']
    # metrics = ['binary_crossentropy', 'mse']
    if model == "EdgeDeepBrainLC_reg":
        metrics = ['loss', 'loss_age', 'loss_lc']
        monitor = 'val_loss_lc'
    elif model == "EdgeDeepBrainLC":
        metrics = ['loss', 'loss_age', 'loss_lc', 'auc_age', 'auc_lc']
        monitor = 'val_loss_lc'
    elif model == "EdgeDeepBrain" or model == "GlobalEdgeDeepBrain" or model == 'MLP':
        try:
            loss_name = loss.name
        except:
            loss_name = loss.__name__
        if loss_name == 'binary_crossentropy':
            # metrics = ['binary_crossentropy', 'mse']
            metrics = ['loss']
            monitor = 'val_loss'
        else:
            # metrics = ['mae', 'mse']
            metrics = ['loss']
            monitor = 'val_loss'
    else:
        metrics = ['loss', 'loss_age', 'loss_lc']
        monitor = 'val_loss'
    monitor_mode = 'min'
    # metrics = ['binary_crossentropy', tf.keras.metrics.AUC(name='auc')]
    cross_validation = True
    explore_attentions = False


class EvalConfigAtrophy(object):
    save_path = './output/out_att/'
