import os
import numpy as np
import pandas as pd
import scipy
from Config import *
from Config_atrophy import *

import random


def get_edges(connectivity_matrix):
    connectivity_matrix_up = np.triu(connectivity_matrix, k=0)
    b = np.where(connectivity_matrix_up > 0)
    edges = np.array([(x, y) for x, y in zip(b[0], b[1])])
    isolated_nodes = np.where(np.sum(connectivity_matrix_up, axis=0) == 0)
    return edges, isolated_nodes


def get_data(dir, file, filenames, participant_details, file_type,
             threshold_by_matrix, threshold_by_region, threshold,
             matrices, edge_values_all, edges_all, ages
             ):
    filenames.append(file.split('.')[0])
    if file_type == '.txt':
        with open(os.path.join(dir, file)) as f:
            fc = f.readlines()
            fc_float = []
            for fc_node in fc:
                fc_float.append([float(x) for x in fc_node.split('\n')[0].split('\t')])
            fc_array = np.array([np.array(xi) for xi in fc_float])
    elif file_type == '.xlsx':
        fc = pd.read_excel(
            os.path.join(dir, file + file_type),
            header=None)
        fc_array = np.array(fc)
    else:
        mat = scipy.io.loadmat(os.path.join(dir, file + file_type))
        try:
            fc_array = mat['connectivity']
        except:
            fc_array = mat['data']
    if threshold_by_matrix:
        threshold_matrix = np.array(pd.read_excel(threshold_by_matrix, header=None))
        threshold_matrix[threshold_matrix > 0] = 1
        edge_mask, isolated_nodes = get_edges(threshold_matrix)
        fc_array = fc_array * threshold_matrix
        fc_array[np.isnan(fc_array)] = 0
        #fc_array[fc_array < 0] = 0
    elif threshold_by_region:
        threshold_by_region_matrix = np.zeros([GeneralConfig.atlas_coords, GeneralConfig.atlas_coords])
        threshold_by_region_matrix[threshold_by_region[0]:threshold_by_region[1], :] = 1
        edge_mask, isolated_nodes = get_edges(threshold_by_region_matrix)
        fc_array = fc_array * threshold_by_region_matrix
        fc_array[np.isnan(fc_array)] = 0
        #fc_array[fc_array < 0] = 0
    else:
        edge_mask, isolated_nodes = get_edges(np.ones([GeneralConfig.atlas_coords, GeneralConfig.atlas_coords]))

    if threshold != 1.0:
        num_largest = int((fc_array.shape[0] * fc_array.shape[1]) * threshold)
        if num_largest % 2 == 0:
            pass  # Even
        else:
            num_largest = num_largest + 1
        indices = fc_array.argpartition(fc_array.size - num_largest, axis=None)[-num_largest:]
        x, y = np.unravel_index(indices, fc_array.shape)
        mask_array = np.zeros(fc_array.shape)
        for x_i, y_i in zip(x, y):
            mask_array[x_i, y_i] = 1
        fc_array = fc_array * mask_array
        edge_mask, isolated_nodes = get_edges(mask_array)
    else:
        pass
    edge_values = [fc_array[r, c] for r, c in edge_mask]

    matrices.append(fc_array)
    edge_values_all.append(edge_values)
    edges_all.append(edge_mask)
    try:
        loss = TrainConfig.loss.name
    except:
        loss = TrainConfig.loss.__name__
    if GeneralConfig.dataset_type == 'LEMON' and loss == 'binary_crossentropy':
        if participant_details.age.iloc[0] == '20-25' or participant_details.age.iloc[0] == '25-30' or \
                participant_details.age.iloc[0] == '30-35' or participant_details.age.iloc[0] == '35-40':
            ages.append(0.0)
        else:
            ages.append(1.0)
    elif GeneralConfig.dataset_type == 'LEMON' and loss == 'categorical_crossentropy':
        if participant_details.age.iloc[0] == '20-25':
            ages.append(0.0)
        elif participant_details.age.iloc[0] == '25-30':
            ages.append(1.0)
        elif participant_details.age.iloc[0] == '30-35' or participant_details.age.iloc[0] == '35-40':
            ages.append(2.0)
        elif participant_details.age.iloc[0] == '55-60' or participant_details.age.iloc[0] == '60-65':
            ages.append(3.0)
        elif participant_details.age.iloc[0] == '65-70' or participant_details.age.iloc[0] == '35-40':
            ages.append(4.0)
        elif participant_details.age.iloc[0] == '70-75' or participant_details.age.iloc[0] == '75-80':
            ages.append(5.0)
        else:
            ages.append(6.0)
    else:
        age = float(participant_details.age.iloc[0])
        ages.append(age)

    return filenames, ages, matrices, edge_values_all, fc_array, edges_all

def get_data_atrophy(participant_gm, file,participant_details,filenames,
                     matrices,ages):
    filenames.append(file.split('.')[0])
    gm_array = np.array(participant_gm)
    gm_array = gm_array[:,1:247]
    matrices.append(gm_array)

    try:
        loss = TrainConfigAtrophy.loss.name
    except:
        loss = TrainConfigAtrophy.loss.__name__
    if GeneralConfigAtrophy.dataset_type == 'LEMON' and loss == 'binary_crossentropy':
        if participant_details.age.iloc[0] == '20-25' or participant_details.age.iloc[0] == '25-30' or \
                participant_details.age.iloc[0] == '30-35' or participant_details.age.iloc[0] == '35-40':
            ages.append(0.0)
        else:
            ages.append(1.0)
    elif GeneralConfigAtrophy.dataset_type == 'LEMON' and loss == 'categorical_crossentropy':
        if participant_details.age.iloc[0] == '20-25':
            ages.append(0.0)
        elif participant_details.age.iloc[0] == '25-30':
            ages.append(1.0)
        elif participant_details.age.iloc[0] == '30-35' or participant_details.age.iloc[0] == '35-40':
            ages.append(2.0)
        elif participant_details.age.iloc[0] == '55-60' or participant_details.age.iloc[0] == '60-65':
            ages.append(3.0)
        elif participant_details.age.iloc[0] == '65-70' or participant_details.age.iloc[0] == '35-40':
            ages.append(4.0)
        elif participant_details.age.iloc[0] == '70-75' or participant_details.age.iloc[0] == '75-80':
            ages.append(5.0)
        else:
            ages.append(6.0)
    else:
        age = float(participant_details.age.iloc[0])
        ages.append(age)

    return filenames, ages, matrices, gm_array
def get_common_subjects(dir_con, dir_fc, dir_sc):
    files_dti = os.listdir(dir_con)
    files_dti_names = []
    for f in files_dti:
        file = f.split('.')[0]
        files_dti_names.append(file)

    files_sc = os.listdir(dir_sc)
    files_sc_names = []
    for f in files_sc:
        file = f.split('.')[0]
        files_sc_names.append(file)

    files_fc = os.listdir(dir_fc)
    files_fc_names = []
    for f in files_fc:
        file = f.split('.')[0]
        files_fc_names.append(file)
    common_elements = set(files_dti_names) & set(files_sc_names) & set(files_fc_names)
    return common_elements


def get_dataset_array(dir, participants_details, threshold=0.1, threshold_by_matrix=None,
                      threshold_by_region=None, age_range=[18, 88],
                      file_type='.txt', train=True):
    files = os.listdir(dir)
    print('Number of subjects in folder: ' + str(len(files)))
    files_names = []
    if GeneralConfig.dataset_type == 'LEMON':
        files_names = []
        for file in files:
            participant_id = file.split(file_type)[0]
            files_names.append(participant_id)
    else:
        for file in files:
            participant_id = file.split(file_type)[0]
            files_names.append(participant_id)
    print('Number of filenames: ' + str(len(files_names)))
    matrices = []
    ages = []
    edge_values_all = []
    edges_all = []
    filenames = []
    count = 0
    data_dict = dict()
    list_participants = list(participants_details.participant_id)

    list_participants = get_common_subjects(
        os.path.join(GeneralConfig.dataset, GeneralConfig.connectivity_matrices_con),
        os.path.join(GeneralConfig.dataset, GeneralConfig.connectivity_matrices_sc),
        os.path.join(GeneralConfig.dataset, GeneralConfig.connectivity_matrices_fc))
    print('Number of subjects in file: ' + str(len(list_participants)))

    # random.Random(4).shuffle(list_participants)
    list_participants = list(list_participants)
    for file in list_participants[0:20]:
        pass_subject = False
        participant_details = participants_details[participants_details.participant_id == str(file)]
        if file in files_names:
            if GeneralConfig.dataset_type == 'LEMON' or GeneralConfig.dataset_type == 'CamCAN':
                try:
                    if float(participant_details.age.iloc[0]) < age_range[0] or float(participant_details.age.iloc[0]) > \
                            age_range[1]:
                        pass_subject = True
                except:
                    if participant_details.age.iloc[0] == age_range[0]:
                        pass_subject = False
            else:
                if float(participant_details.age.iloc[0]) < age_range[0] or float(participant_details.age.iloc[0]) > \
                        age_range[1]:
                    pass_subject = True
                else:
                    if list(participant_details.sub_sess)[0].split('_')[1][0] == file.split('_')[3][0]:
                        pass_subject = False
                    else:
                        pass_subject = True
            if pass_subject:
                pass
            else:

                filenames, ages, matrices, edge_values_all, fc_array, edges_all = get_data(dir, file, filenames,
                                                                                           participant_details,
                                                                                           file_type,
                                                                                           threshold_by_matrix,
                                                                                           threshold_by_region,
                                                                                           threshold,
                                                                                           matrices,
                                                                                           edge_values_all,
                                                                                           edges_all, ages)

                data_dict[file] = [participant_details.age.iloc[0],
                                   fc_array]
                count += 1
        else:
            pass
    print('Number of subjects: ' + str(count))
    filenames = np.array(filenames)
    if train:
        return matrices, edges_all, edge_values_all, ages, filenames
    else:
        return data_dict

def get_dataset_atrophy_array(dir_matrices, gm_values, participants_details, file_type, age_range=[18, 88], train=True):
    gm_values_read = pd.read_excel(gm_values)
    files = os.listdir(dir_matrices)
    print('Number of subjects in folder: ' + str(len(files)))
    files_names = []
    if GeneralConfigAtrophy.dataset_type == 'LEMON':
        files_names = []
        for file in files:
            participant_id = file.split(file_type)[0]
            files_names.append(participant_id)
    else:
        for file in files:
            participant_id = file.split(file_type)[0]
            files_names.append(participant_id)
    matrices = []
    ages = []
    filenames = []
    count = 0
    data_dict = dict()
    list_participants = list(participants_details.participant_id)

    list_participants = get_common_subjects(
        os.path.join(GeneralConfigAtrophy.dataset, GeneralConfigAtrophy.connectivity_matrices_con),
        os.path.join(GeneralConfigAtrophy.dataset, GeneralConfigAtrophy.connectivity_matrices_sc),
        os.path.join(GeneralConfigAtrophy.dataset, GeneralConfigAtrophy.connectivity_matrices_fc))
    print('Number of subjects in file: ' + str(len(list_participants)))

    # random.Random(4).shuffle(list_participants)
    list_participants = list(list_participants)
    for file in list_participants:
        pass_subject = False
        participant_details = participants_details[participants_details.participant_id == str(file)]
        participant_gm = gm_values_read[gm_values_read.subjects == str(file)]
        if file in files_names:
            if GeneralConfigAtrophy.dataset_type == 'LEMON' or GeneralConfigAtrophy.dataset_type == 'CamCAN':
                try:
                    if float(participant_details.age.iloc[0]) < age_range[0] or float(participant_details.age.iloc[0]) > \
                            age_range[1]:
                        pass_subject = True
                except:
                    if participant_details.age.iloc[0] == age_range[0]:
                        pass_subject = False
            else:
                if float(participant_details.age.iloc[0]) < age_range[0] or float(participant_details.age.iloc[0]) > \
                        age_range[1]:
                    pass_subject = True
                else:
                    if list(participant_details.sub_sess)[0].split('_')[1][0] == file.split('_')[3][0]:
                        pass_subject = False
                    else:
                        pass_subject = True
            if pass_subject:
                pass
            else:

                filenames, ages, matrices, gm_array = get_data_atrophy(participant_gm,
                                                                       file,
                                                                       participant_details,
                                                                       filenames,
                                                                       matrices,
                                                                       ages)
                data_dict[file] = [participant_details.age.iloc[0],
                                   gm_array]
                count += 1
        else:
            pass
    print('Number of subjects: ' + str(count))
    filenames = np.array(filenames)
    if train:
        return matrices, ages, filenames
    else:
        return data_dict
