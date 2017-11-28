import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
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


def get_clean_mac_address():
    mac_address_df = utils.import_mac_addresses()
    mac_address_clean_df = mac_address_df[mac_address_df.frequency > 10].dropna(subset=FEATURE_LIST)
    return mac_address_clean_df


def get_samples(mac_address_clean_df, scaled=True):
    mac_address_scaled_df = mac_address_clean_df.copy()
    for feature in FEATURE_LIST:
        mac_address_scaled_df[feature] = scale(mac_address_clean_df[feature])

    if scaled:
        samples = mac_address_scaled_df.as_matrix(columns=FEATURE_LIST)
    else:
        samples = mac_address_clean_df.as_matrix(columns=FEATURE_LIST)

    return samples


def kmeans(mac_address_clean_df, samples):
    n_clusters = 2
    model = KMeans(n_clusters=n_clusters)
    model.fit(samples)
    labels = model.predict(samples)
    mac_address_clean_df['kmeans_label'] = labels


def mixture(mac_address_clean_df, samples):
    model = BayesianGaussianMixture(n_components=3)
    model.fit(samples)
    labels = model.predict(samples)
    mac_address_clean_df['mixture_label'] = labels
