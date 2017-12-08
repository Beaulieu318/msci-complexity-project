import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import scale

from msci.utils import utils

FEATURE_LIST = [
    'frequency',
    'length_of_stay',
    'radius_of_gyration',
    'count_density_variance',
    'av_speed',
    'av_turning_angle',
    'total_turning_angle',
    'av_turning_angle_velocity',
    'av_path_length',
    'total_path_length',
    'av_straightness',
]


def get_clean_mac_address(feature_list=FEATURE_LIST):
    """
    Creates a new mac address feature dataframe by removing the mac addresses with small count numbers.
    Drops all mac addresses which have features with nan (these can't be used for clustering).

    :param feature_list: (list) A list of features which will be used for clustering
    :return: (pd.DataFrame) The mac address features dataframe which can be used in the unsupervised clustering algorithms
    """
    mac_address_df = utils.import_mac_addresses()
    mac_address_clean_df = mac_address_df[mac_address_df.frequency > 10].dropna(subset=feature_list)
    return mac_address_clean_df


def get_samples(mac_address_clean_df, feature_list=FEATURE_LIST, scaled=True):
    """
    Creates an array out of the features and scales them to be gaussian around zero if scaled=True

    :param mac_address_clean_df: (pd.DataFrame) Contains the mac addresses and features
    :param feature_list: (list) A list of the features which will be used for clustering
    :param scaled: (boolean) Whether the features should be gaussian scaled
    :return: (array) An array containing the features for each mac address (mac_address * feature)
    """
    mac_address_scaled_df = mac_address_clean_df.copy()
    for feature in feature_list:
        mac_address_scaled_df[feature] = scale(mac_address_clean_df[feature])

    if scaled:
        samples = mac_address_scaled_df.as_matrix(columns=feature_list)
    else:
        samples = mac_address_clean_df.as_matrix(columns=feature_list)

    return samples


def kmeans(mac_address_clean_df, samples, n_clusters):
    """
    Finds the clusters using the K means algorithm and adds the column kmneas_label to mac_address_clean_df

    :param mac_address_clean_df: (pd.DataFrame) Contains the mac addresses and features
    :param samples: (array) The scaled and finite values features for each mac address
    :param n_clusters: (int) The number of clusters which the algorithm will create
    """
    model = KMeans(n_clusters=n_clusters)
    model.fit(samples)
    labels = model.predict(samples)
    mac_address_clean_df['kmeans_label'] = labels


def mixture(mac_address_clean_df, samples, n_components):
    """
    Finds the clusters using the Bayesian Gaussian Mixture algorithm and adds the column mixture_label to mac_address_clean_df

    :param mac_address_clean_df: (pd.DataFrame) Contains the mac addresses and features
    :param samples: (array) The scaled and finite values features for each mac address
    :param n_components: (int) The number of components (clusters) which the algorithm will create
    """
    model = BayesianGaussianMixture(n_components=n_components)
    model.fit(samples)
    labels = model.predict(samples)
    mac_address_clean_df['mixture_label'] = labels
