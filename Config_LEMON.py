import tensorflow as tf
import os

class GeneralConfig(object):
    dataset = r"./LEMON/"
    dataset_type = os.path.basename(os.path.normpath(dataset))
    atlas = os.path.join(dataset, 'bna_atlas.xlsx')
    atlas_coords = 246
    connectivity_matrices_con = "conn_mats_dti"
    connectivity_matrices_fc = "conn_mats_fun"
    connectivity_matrices_sc = "conn_mats_sc_gm"
    #connectivity_matrices_sc = "conn_mats_t1gmISBN"
    connectivity_matrices_con_mask = "conn_mats_dti_mask"
    connectivity_matrices_fc_mask = "conn_mats_fun_mask"
    connectivity_matrices_sc_mask = "conn_mats_sc_gm_mask"
    #connectivity_matrices_sc_mask = "conn_mats_t1gmISBN_mask"
    if dataset_type == "LEMON":
      dataset_details = "Participants_LEMON.csv"
    else:
      dataset_details = "Participants_CamCAN.csv"
    threshold_by_region = None
    thresholded_mean_matrix_con = os.path.join(dataset, connectivity_matrices_con_mask, "percent_30.xlsx")
    thresholded_mean_matrix_fc = os.path.join(dataset, connectivity_matrices_fc_mask, "percent_30.xlsx")
    thresholded_mean_matrix_sc = os.path.join(dataset, connectivity_matrices_sc_mask, "percent_30.xlsx")
    #thresholded_mean_matrix_sc = os.path.join(dataset, connectivity_matrices_sc_mask, "thresholded_02_brainnectome_t1gmISBN.xlsx")
    #thresholded_mean_matrix_sc = None
    run = 'train'  # 'test'
 
    file_type_fc = '.xlsx'
    file_type_conn = '.xlsx'
    file_type_sc = '.xlsx'
    #file_type_sc = '.mat'

class TrainConfig(object):
    model = "EdgeDeepBrain"
    target = 'age'
    edge_features = 'fc'
    dense_layer_dimensions = (8, 16)
    base_layer_dimensions = (16, 16, 16)
    #normalization = "Standard"
    split = "fold" #"stratified" "fold"
    stratify_label = 'age'
    augmentation = False
    normalization = "MinMax"
    n_heads = 2
    loss =  tf.keras.losses.binary_crossentropy #tf.keras.losses.mae # tf.keras.losses.binary_crossentropy
    epochs = 200
    batch_size = 64
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  #
    logs = './logs/'  # folder to save tensorboard logs
    output = '/output'  # folder to save output models
    threshold = 1.0
    edge_attention = True
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
    explore_attentions = True


class EvalConfig(object):
    save_path = './output/out_att/'
