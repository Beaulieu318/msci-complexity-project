import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import msci.utils.plot as pfun

from tqdm import tqdm_notebook as tqdm


def store_dictionary(mm_df, store_only=False):
    stores = mm_df.store_id.drop_duplicates().tolist()
    if store_only:
        stores = [i for i in stores if i[0] == 'B']
    grouped = mm_df.groupby('store_id')
    groups = [grouped.get_group(i) for i in stores]
    xs = [group.x.tolist() for group in groups]
    ys = [group.y.tolist() for group in groups]
    pos = {stores[i]: list(zip(xs[i], ys[i])) for i in range(len(stores))}
    return groups, pos


def location_outliers(store_pos, method='density', plot=True):
    """
    NOTE: Must hash plt.figure() lines in pfun.plot_points_on_map
    :param store_pos:
    :param plot:
    :return:
    """
    if method is 'density':
        db = DBSCAN(eps=0.3, min_samples=10).fit_predict(store_pos)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
        class_member_mask = (labels == k)

        xy = store_pos[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = store_pos[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    if method is 'km':
        kmeans = KMeans(n_clusters=2, random_state=0).fit(store_pos)
        clusters = KMeans(n_clusters=2, random_state=0).fit_predict(store_pos).astype('bool')
    if plot:
        fig = plt.figure()
        x = [store_pos[i][0] for i in range(len(store_pos)) if clusters[i]]
        y = [store_pos[i][1] for i in range(len(store_pos)) if clusters[i]]
        x_out = [store_pos[i][0] for i in range(len(store_pos)) if clusters[i] == False]
        y_out = [store_pos[i][1] for i in range(len(store_pos)) if clusters[i] == False]
        pfun.plot_points_on_map(x, y)
        pfun.plot_points_on_map(x_out, y_out, c='b')
        fig.show()
    return kmeans, clusters


def kth_nearest_neighbours(positions, k=5, tolerance=1, plot=True):
    """
    NOTE: Must hash plt.figure() lines in pfun.plot_points_on_map
    :param positions: (list) list of x,y coordinates for store signals
    :param k:
    :param plot:
    :param tolerance:
    :return:
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    distances = np.array([d[k-1] for d in distances])
    in_store_indices = np.where(distances <= tolerance)[0]
    out_store_indices = np.where(distances > tolerance)[0]
    in_store = np.array(positions)[in_store_indices]
    out_store = np.array(positions)[out_store_indices]
    if plot:
        fig = plt.figure()
        pfun.plot_points_on_map([i[0] for i in in_store], [i[1] for i in in_store])
        pfun.plot_points_on_map([i[0] for i in out_store], [i[1] for i in out_store], c='b')
        fig.show()
    return in_store_indices


def clean_store_id(signal_df, store_only=False):
    signal_df.store_id[signal_df.store_id.isnull()] = 'H'
    stores = signal_df.store_id.drop_duplicates().tolist()
    if store_only:
        stores = [i for i in stores if i[0] == 'B']
        sd = store_dictionary(signal_df, store_only=True)
    else:
        sd = store_dictionary(signal_df)
    groups = sd[0]
    clean_groups = []
    for i in tqdm(range(len(stores)), 'Store ID'):
        if stores[i] == 'H':
            clean_groups.append(groups[i])
        else:
            in_store = kth_nearest_neighbours(sd[1][stores[i]], plot=False)
            clean_group = groups[i].iloc[in_store]
            clean_groups.append(clean_group)

    clean_signal_df = pd.concat(clean_groups)
    clean_signal_df.store_id[clean_signal_df.store_id == 'H'] = np.nan

    return clean_signal_df
