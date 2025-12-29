import matplotlib.pyplot as plt
from nilearn import plotting
import os
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import numpy as np


def load_atlas(path):
    aal2 = pd.ExcelFile(path)
    aal2 = {sheet_name: aal2.parse(sheet_name)
            for sheet_name in aal2.sheet_names}
    aal2_coords = [[v[1], v[2], v[3]] for v in aal2.get('Sheet1').values]
    aal2_names = [v[0] for v in aal2.get('Sheet1').values]
    aal2_node_names = [v[5] for v in aal2.get('Sheet1').values]
    aal2_node_names_hemi = [v[6] for v in aal2.get('Sheet1').values]
    aal2_hemi = [v[4] for v in aal2.get('Sheet1').values]
    aal2_ggseg_names = [v[7] for v in aal2.get('Sheet1').values]
    return aal2_coords, aal2_names, aal2_node_names, aal2_hemi, aal2_node_names_hemi, aal2_ggseg_names

def read_atlas(atlas_path):
    aal2 = pd.ExcelFile(atlas_path)
    aal2 = {sheet_name: aal2.parse(sheet_name)
            for sheet_name in aal2.sheet_names}
    aal2_coords = [[v[1], v[2], v[3]] for v in aal2.get('Sheet1').values]
    aal2_names = [v[0] for v in aal2.get('Sheet1').values]
    return aal2_coords, aal2_names

def plot_connectome(connectivity_matrix, atlas_path, out_path, title):
    aal2_coords, aal2_names = read_atlas(atlas_path)

    display = plotting.plot_connectome(connectivity_matrix.T + connectivity_matrix, aal2_coords, node_size=5,
                                       title=title,
                                       edge_threshold=None, node_color='auto', colorbar=True)
    if os.path.exists(os.path.join(out_path, 'connectomes')):
        pass
    else:
        os.mkdir(os.path.join(out_path, 'connectomes'))
    plt.savefig(os.path.join(out_path, 'connectomes', title + '.png'))

def statistic(x, y, axis=0):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def transform_edges_to_matrix_dataset(edges_x, edges_y, edges_attention_head, atlas_n_coords):
    matrix = np.zeros((atlas_n_coords, atlas_n_coords))
    for ex, ey, v in zip(edges_x, edges_y, edges_attention_head):
        matrix[tuple([ex, ey])] = v
    return matrix
