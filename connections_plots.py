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


def get_edges_from_mask(mask_file):
    threshold_matrix = np.array(pd.read_excel(mask_file, header=None))
    threshold_matrix[threshold_matrix > 0] = 1
    edge_mask, isolated_nodes = get_edges(threshold_matrix)
    return edge_mask, isolated_nodes

def load_atlas_new(path):
    aal2 = pd.ExcelFile(path)
    aal2 = {sheet_name: aal2.parse(sheet_name)
            for sheet_name in aal2.sheet_names}
    aal2_coords = [[v[1], v[2], v[3]] for v in aal2.get('Sheet1').values]
    aal2_names = [v[0] for v in aal2.get('Sheet1').values]
    aal2_node_names = [v[5] for v in aal2.get('Sheet1').values]
    aal2_node_names_hemi = [v[6] for v in aal2.get('Sheet1').values]
    aal2_hemi = [v[4] for v in aal2.get('Sheet1').values]
    aal2_ggseg_names = [v[7] for v in aal2.get('Sheet1').values]
    node_names_gyrus = [v[8] for v in aal2.get('Sheet1').values]
    gyrus = [v[9] for v in aal2.get('Sheet1').values]
    node_names_area = [v[10] for v in aal2.get('Sheet1').values]
    cyto_group = [v[11] for v in aal2.get('Sheet1').values]
    return aal2_coords, aal2_names, aal2_node_names, aal2_hemi, aal2_node_names_hemi, aal2_ggseg_names, node_names_gyrus, gyrus, node_names_area, cyto_group

output = r'.\output'

# Edges
edges_con, isolated_nodes_con = get_edges_from_mask(GeneralConfig.thresholded_mean_matrix_con)
edges_fc, isolated_nodes_fc = get_edges_from_mask(GeneralConfig.thresholded_mean_matrix_fc)
edges_sc, isolated_nodes_sc = get_edges_from_mask(GeneralConfig.thresholded_mean_matrix_sc)


# Atlas
atlas_coords, atlas_names, atlas_node_names, atlas_hemi, atlas_node_names_hemi, atlas_ggseg_names, node_names_gyrus, gyrus, node_names_area, cyto_group = load_atlas_new(
    GeneralConfig.atlas)

edges = pd.read_csv(os.path.join(output, "1", 'edges_all.csv'))
edges_x = list(edges[edges.columns[1]])
edges_y = list(edges[edges.columns[2]])

all_edges = []
all_edges_nodes_gyrus = []
all_edges_gyrus = []
all_edges_area = []
for e in range(len(edges)):
    e_selected = [edges_x[e], edges_y[e]]
    e_con = np.argwhere(np.logical_and(edges_con[:, 0] == e_selected[0], edges_con[:, 1] == e_selected[1]))
    e_sc = np.argwhere(np.logical_and(edges_sc[:, 0] == e_selected[0], edges_sc[:, 1] == e_selected[1]))
    e_fc = np.argwhere(np.logical_and(edges_fc[:, 0] == e_selected[0], edges_fc[:, 1] == e_selected[1]))
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

importances_edges_multi = pd.read_csv(os.path.join(output, 'importances_edges_multi.csv'), sep=';')
importances_edges_multi['Modality'] = all_edges

edges_importance_names_x_list = list(importances_edges_multi.edges_importance_names_x)
edges_importance_names_y_list = list(importances_edges_multi.edges_importance_names_y)
edges_importance_names_area_x = np.array([None] * len(edges_importance_names_x_list))
edges_importance_names_gyrus_x = np.array([None] * len(edges_importance_names_x_list))
edges_importance_names_node_gyrus_x = np.array([None] * len(edges_importance_names_x_list))
edges_importance_names_cyto_x = np.array([None] * len(edges_importance_names_x_list))
edges_importance_names_area_y = np.array([None] * len(edges_importance_names_y_list))
edges_importance_names_gyrus_y = np.array([None] * len(edges_importance_names_y_list))
edges_importance_names_node_gyrus_y = np.array([None] * len(edges_importance_names_y_list))
edges_importance_names_cyto_y = np.array([None] * len(edges_importance_names_y_list))
for i in range(len(atlas_ggseg_names)):
    edge_x = atlas_ggseg_names[i]
    name_gyrus = gyrus[i]
    edge_node_names_area = node_names_area[i]
    edge_node_gyrus = node_names_gyrus[i]
    edge_cyto = cyto_group[i]
    matches_x = [i for i, x in enumerate(edges_importance_names_x_list) if x == edge_x]
    matches_y = [i for i, x in enumerate(edges_importance_names_y_list) if x == edge_x]
    edges_importance_names_node_gyrus_x[matches_x] = edge_node_gyrus
    edges_importance_names_cyto_x[matches_x] = edge_cyto
    edges_importance_names_area_x[matches_x] = edge_node_names_area
    edges_importance_names_gyrus_x[matches_x] = name_gyrus
    edges_importance_names_node_gyrus_y[matches_y] = edge_node_gyrus
    edges_importance_names_area_y[matches_y] = edge_node_names_area
    edges_importance_names_gyrus_y[matches_y] = name_gyrus
    edges_importance_names_cyto_y[matches_y] = edge_cyto

importances_edges_multi['edges_importance_names_node_gyrus_x'] = edges_importance_names_node_gyrus_x
importances_edges_multi['edges_importance_names_area_x'] = edges_importance_names_area_x
importances_edges_multi['edges_importance_names_gyrus_x'] = edges_importance_names_gyrus_x
importances_edges_multi['edges_importance_names_node_gyrus_y'] = edges_importance_names_node_gyrus_y
importances_edges_multi['edges_importance_names_area_y'] = edges_importance_names_area_y
importances_edges_multi['edges_importance_names_gyrus_y'] = edges_importance_names_gyrus_y
importances_edges_multi['edges_importance_names_cyto_x'] = edges_importance_names_cyto_x
importances_edges_multi['edges_importance_names_cyto_y'] = edges_importance_names_cyto_y
#importances_edges_multi.to_csv(r'C:\Users\blazuf\Documents\PROJECTS\DeepBrain\Analyze\brainnetome\final\Cam-CAN\conn_mats_multi_100_final\importances_edges_multi_node_names.csv', sep=';')

unique, counts = np.unique(importances_edges_multi.Modality, return_counts=True)
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
plt.savefig(os.path.join(output, 'multimoda_edge_feature_importance_all.svg'))

plt.close()
plt.cla()
plt.clf()
range_low = 0
range_high = 200
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
plt.savefig(os.path.join(output, 'multimoda_edge_feature_importance_from44.svg'))

matrix_edge_importance = np.zeros((len(atlas_names), len(atlas_names)))
importances_modality = importances_edges_multi.Modality
for e in range(len(edges)):
    if importances_modality[e] == ['con', 'sc', 'fc']:
        e_x = list(importances_edges_multi.edges_importance_names_x)[e]
        e_y = list(importances_edges_multi.edges_importance_names_y)[e]
        matrix_edge_importance[edges_x[e], edges_y[e]] = importances_edges_multi.Values[e]
plt.close()
plt.cla()
plt.clf()
plotting.plot_connectome(matrix_edge_importance + matrix_edge_importance.T, atlas_coords, node_size=2,
                         node_color='black', edge_cmap='Purples')
plt.savefig(os.path.join(output, 'average_edges_importance_multi_con_sc_fc_100.svg'))

sorted_importances_edges_multi_100 = sorted_importances_edges_multi[0:100]
sorted_modality_100 = list(sorted_importances_edges_multi_100.Modality)
matrix_edge_importance = np.zeros((len(atlas_names), len(atlas_names)))
for e in range(len(sorted_importances_edges_multi_100)):
    if sorted_modality_100[e] == ['sc', 'fc']:
        e_x = list(sorted_importances_edges_multi_100.edges_importance_names_x)[e]
        e_y = list(sorted_importances_edges_multi_100.edges_importance_names_y)[e]
        matrix_edge_importance[atlas_ggseg_names.index(e_x), atlas_ggseg_names.index(e_y)] = \
            importances_edges_multi.Values[e]
plt.close()
plt.cla()
plt.clf()
plotting.plot_connectome(matrix_edge_importance + matrix_edge_importance.T, atlas_coords, node_size=2,
                         node_color='black', edge_cmap='Reds')
plt.savefig(os.path.join(output, 'average_edges_importance_multi_sc_fc_100.svg'))


sorted_importances_edges_multi_100 = sorted_importances_edges_multi
sorted_modality_100 = list(sorted_importances_edges_multi_100.Modality)
matrix_edge_importance = np.zeros((len(atlas_names), len(atlas_names)))
for e in range(len(sorted_importances_edges_multi_100)):
    if sorted_modality_100[e] == ['sc', 'fc']:
        e_x = list(sorted_importances_edges_multi_100.edges_importance_names_x)[e]
        e_y = list(sorted_importances_edges_multi_100.edges_importance_names_y)[e]
        matrix_edge_importance[atlas_ggseg_names.index(e_x), atlas_ggseg_names.index(e_y)] = \
            importances_edges_multi.Values[e]
plt.close()
plt.cla()
plt.clf()
plotting.plot_connectome(matrix_edge_importance + matrix_edge_importance.T, atlas_coords, node_size=2,
                         node_color='black', edge_cmap='Reds')
plt.savefig(os.path.join(output, 'average_edges_importance_multi_sc_fc_all.svg'))

sorted_importances_edges_multi_100 = sorted_importances_edges_multi[0:100]
sorted_modality_100 = list(sorted_importances_edges_multi_100.Modality)
matrix_edge_importance = np.zeros((len(atlas_names), len(atlas_names)))
for e in range(len(sorted_importances_edges_multi_100)):
    if sorted_modality_100[e] == ['con', 'fc']:
        e_x = list(sorted_importances_edges_multi_100.edges_importance_names_x)[e]
        e_y = list(sorted_importances_edges_multi_100.edges_importance_names_y)[e]
        matrix_edge_importance[atlas_ggseg_names.index(e_x), atlas_ggseg_names.index(e_y)] = \
            importances_edges_multi.Values[e]
plt.close()
plt.cla()
plt.clf()
plotting.plot_connectome(matrix_edge_importance + matrix_edge_importance.T, atlas_coords, node_size=2,
                         node_color='black', edge_cmap='Greys')
plt.savefig(os.path.join(output, 'average_edges_importance_multi_con_fc_100.svg'))

sorted_importances_edges_multi_100 = sorted_importances_edges_multi
sorted_modality_100 = list(sorted_importances_edges_multi_100.Modality)
matrix_edge_importance = np.zeros((len(atlas_names), len(atlas_names)))
for e in range(len(sorted_importances_edges_multi_100)):
    if sorted_modality_100[e] == ['con', 'fc']:
        e_x = list(sorted_importances_edges_multi_100.edges_importance_names_x)[e]
        e_y = list(sorted_importances_edges_multi_100.edges_importance_names_y)[e]
        matrix_edge_importance[atlas_ggseg_names.index(e_x), atlas_ggseg_names.index(e_y)] = \
            importances_edges_multi.Values[e]
plt.close()
plt.cla()
plt.clf()
plotting.plot_connectome(matrix_edge_importance + matrix_edge_importance.T, atlas_coords, node_size=2,
                         node_color='black', edge_cmap='Greys')
plt.savefig(os.path.join(output, 'average_edges_importance_multi_con_fc_all.svg'))

sorted_importances_edges_multi_100 = sorted_importances_edges_multi[0:100]
sorted_modality_100 = list(sorted_importances_edges_multi_100.Modality)
matrix_edge_importance = np.zeros((len(atlas_names), len(atlas_names)))
for e in range(len(sorted_importances_edges_multi_100)):
    if sorted_modality_100[e] == ['con', 'sc']:
        e_x = list(sorted_importances_edges_multi_100.edges_importance_names_x)[e]
        e_y = list(sorted_importances_edges_multi_100.edges_importance_names_y)[e]
        matrix_edge_importance[atlas_ggseg_names.index(e_x), atlas_ggseg_names.index(e_y)] = \
            importances_edges_multi.Values[e]
plt.close()
plt.cla()
plt.clf()
plotting.plot_connectome(matrix_edge_importance + matrix_edge_importance.T, atlas_coords, node_size=2,
                         node_color='black', edge_cmap='Greens')
plt.savefig(os.path.join(output, 'average_edges_importance_multi_con_sc_100.svg'))


sorted_importances_edges_multi_100 = sorted_importances_edges_multi
sorted_modality_100 = list(sorted_importances_edges_multi_100.Modality)
matrix_edge_importance = np.zeros((len(atlas_names), len(atlas_names)))
for e in range(len(sorted_importances_edges_multi_100)):
    if sorted_modality_100[e] == ['con', 'sc']:
        e_x = list(sorted_importances_edges_multi_100.edges_importance_names_x)[e]
        e_y = list(sorted_importances_edges_multi_100.edges_importance_names_y)[e]
        matrix_edge_importance[atlas_ggseg_names.index(e_x), atlas_ggseg_names.index(e_y)] = \
            importances_edges_multi.Values[e]
plt.close()
plt.cla()
plt.clf()
plotting.plot_connectome(matrix_edge_importance + matrix_edge_importance.T, atlas_coords, node_size=2,
                         node_color='black', edge_cmap='Greens')
plt.savefig(os.path.join(output, 'average_edges_importance_multi_con_sc_all.svg'))