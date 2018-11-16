import pandas as pd
import numpy as np
import copy
import os
import time

from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import directed_hausdorff
from sklearn.preprocessing import StandardScaler

from msci.utils import utils
from msci.analysis.networks import *
import msci.utils.plot as pfun

from tqdm import tqdm

from msci.analysis.complexity import matrix_correlation

matplotlib.style.use('ggplot')

data_import = True


def position_dictionary(signal_df, list_type=True):
    signal_df = signal_df.sort_values('date_time')
    macs = signal_df.sort_values('mac_address').mac_address.drop_duplicates().tolist()
    grouped_df = signal_df.groupby('mac_address')
    groups = [grouped_df.get_group(i) for i in macs]
    if list_type:
        positions = [list(zip(groups[i].x.tolist(), groups[i].y.tolist())) for i in range(len(groups))]
        return positions
    else:
        positions = {macs[i]: list(zip(groups[i].x.tolist(), groups[i].y.tolist())) for i in range(len(groups))}
        return positions


def pairwise_hausdorff(positions, macs, undirected=True):
    ph = np.zeros((len(macs), len(macs)))
    if undirected:
        for i in tqdm(range(len(macs))):
            for j in range(i, len(macs)):
                hd = undirected_hausdorff(positions[i], positions[j])
                ph[i][j] = hd
                ph[j][i] = hd
    else:
        for i in tqdm(range(len(macs))):
            for j in range(len(macs)):
                hd = directed_hausdorff(positions[i], positions[j])[0]
                ph[i][j] = hd
    return ph


def partitioned_pairwise_hausdorff(positions, macs, undirected=True, filename=None):
    if filename is not None:
        with open(filename, "w") as f:
            f.write('')

    if undirected:
        for i in tqdm(range(len(macs))):
            ph_row = np.zeros(len(macs))
            for j in tqdm(range(i, len(macs))):
                hd = undirected_hausdorff(positions[i], positions[j])
                ph_row[j] = hd
            if filename is not None:
                with open(filename, "a") as f:
                    f.write((' '.join(['%10.6f ']*ph_row.size)+'\n\n') % tuple(ph_row))
    else:
        for i in tqdm(range(len(macs))):
            ph_row = np.zeros(len(macs))
            for j in range(len(macs)):
                hd = directed_hausdorff(positions[i], positions[j])[0]
                ph_row[j] = hd
            if filename is not None:
                with open(filename, "a") as f:
                    f.write((' '.join(['%10.6f ']*ph_row.size)+'\n\n') % tuple(ph_row))


def undirected_hausdorff(path_a, path_b):
    hab = directed_hausdorff(path_a, path_b)[0]
    hba = directed_hausdorff(path_b, path_a)[0]
    return np.amax([hab, hba])


def dbscan_cluster(haussdorf_matrix, resolution, samples):
    X = StandardScaler().fit_transform(haussdorf_matrix)
    db = DBSCAN(eps=resolution, min_samples=samples).fit(X)
    labels = db.labels_
    return labels


def plot_community_path(db_labels, macs, signal_df):
    clusters = [np.array(macs)[np.where(db_labels == i)] for i in list(set(db_labels))]
    for c in clusters:
        pfun.plot_path(signal_df, c.tolist(), scatter=False)
    return clusters


def plot_3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r']
    X = data[0]
    Y = data[1]
    Z = data[2]
    ax.scatter(X, Y, Z)
    ax.set_xlabel('Radius of Gyration')
    ax.set_ylabel('Speed')
    ax.set_zlabel('Length of Stay')
    fig.show()


def sampled_hausdorff(pos_dict, macs, N, samples):
    t0 = time.time()
    mac_samples = []
    hausdorffs = []
    for i in range(samples):
        subset_macs = np.random.choice(macs, N)
        mac_samples.append(subset_macs)
        ph = pairwise_haussdorf(pos_dict, subset_macs)
        hausdorffs.append(ph)
    t = time.time() - t0
    return mac_samples, hausdorffs, t


if data_import:
    signal_df = utils.import_signals('Mall of Mauritius', version=4)

    mac_address_df = utils.import_mac_addresses(version=4)
    shopper_df = mac_address_df[mac_address_df.bayesian_label == 'Shopper']
    cleaner_signal_df = signal_df[signal_df.mac_address.isin(shopper_df.mac_address)]

    # analysis_mac_addresses = sorted(cleaner_signal_df.mac_address.unique().tolist()[:100] + cleaner_signal_df.mac_address.unique().tolist()[5000:5200])

    shopper_macs = sorted(
        cleaner_signal_df.mac_address.unique().tolist()  # [:200]
        # + cleaner_signal_df.mac_address.unique().tolist()[600:800]
        # + cleaner_signal_df.mac_address.unique().tolist()[1000:1200]
    )

    pos_dict = position_dictionary(
        cleaner_signal_df[cleaner_signal_df.mac_address.isin(shopper_macs)].sort_values('mac_address'),
        list_type=True
    )
