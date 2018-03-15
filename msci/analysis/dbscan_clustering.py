from sklearn.preprocessing import scale

from msci.cleaning.features import *

import pandas as pd
import numpy as np
import copy

from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

import seaborn as sns

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import directed_hausdorff
from sklearn.preprocessing import StandardScaler

from msci.utils import utils
from msci.analysis.networks import *
import msci.utils.plot as pfun

from msci.utils.utils import data_path

matplotlib.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D

data_import = False

FEATURE_LIST_ = [
    'length_of_stay',
    'radius_of_gyration',
    'count_density_variance',
    'av_speed',
    'av_turning_angle',
    'av_path_length',
    'av_straightness',
    'turning_angle_density',
]

FEATURE_LIST = [
    'length_of_stay',
    'radius_of_gyration',
    'av_speed',
    'av_turning_angle',
    'av_path_length',
]


def scale_dataframe(mac_address_df):
    scaled_df = mac_address_df.copy()
    for feature in FEATURE_LIST:
        scaled_df[feature] = scale(mac_address_df[feature])
    samples = scaled_df.as_matrix(
        columns=FEATURE_LIST
    )
    return samples


def db_clust(data, resolution, samples):
    db = DBSCAN(eps=resolution, min_samples=samples)
    db.fit(data)
    labels = db.labels_
    dbp = labels[labels > -1]
    print('Successively classified', str(len(dbp)), 'paths')
    print('Failed to classify', str(len(labels) - len(dbp)), 'paths')
    print('Clustered into', str(len(np.bincount(dbp))), 'Communties')
    return labels, dbp, np.bincount(dbp)


def plot_community_path(db_labels, macs, signal_df, cluster='all'):
    clusters = [np.array(macs)[np.where(db_labels == i)] for i in list(set(db_labels))]
    if cluster == 'all':
        for c in clusters:
            pfun.plot_path(signal_df, c.tolist(), scatter=False)
    else:
        pfun.plot_path(signal_df, clusters[cluster].tolist(), scatter=False)
    return clusters


def plot_cluster_distributions(shopper_df, labels, number_of_clusters):
    cluster_df = shopper_df.copy()
    cluster_df['db_cluster'] = labels
    clusters_to_plot = np.argsort(np.bincount(labels[labels > -1]))[::-1][:number_of_clusters]
    print(clusters_to_plot)
    clustered_df = cluster_df[cluster_df.db_cluster.isin(clusters_to_plot)]
    g = sns.pairplot(
        clustered_df,
        vars=FEATURE_LIST,
        hue="db_cluster", diag_kind="hist", dropna=True
    )
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()


if data_import:
    signal_df = utils.import_signals('Mall of Mauritius', version=4)

    dbclean_df = utils.import_mac_addresses(version=3)
    bayes_df = pd.read_csv(data_path + 'bayesian_label.csv')
    shopper_df = dbclean_df[dbclean_df.dbscan_label == 'Shopper'].append(bayes_df[bayes_df.bayesian_label == 'Shopper'])

    shopper_macs = shopper_df.mac_address.tolist()

    cleaner_signal_df = signal_df[signal_df.mac_address.isin(shopper_df.mac_address)]
