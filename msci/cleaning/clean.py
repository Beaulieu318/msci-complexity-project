"""
This script cleans the raw data, extracts features from the mac addresses and clusters them
"""

import os
import pandas as pd

from tqdm import tqdm_notebook as tqdm

from msci.cleaning import duplicates
from msci.cleaning import features
from msci.cleaning import bayesian_inference
from msci.cleaning import store_ids

from msci.utils import utils

dir_path = os.path.dirname(os.path.realpath(__file__))

FEATURE_LIST = [
    'frequency',
    'length_of_stay',
    'radius_of_gyration',
    'count_density_variance',
    'av_speed',
    'av_turning_angle',
    'av_turning_angle_velocity',
    'av_path_length',
    'av_straightness',
]

CONFIDENCE_LEVEL = 0.05


def find_label(x):
    if x > (1-CONFIDENCE_LEVEL):
        return 1
    elif x < CONFIDENCE_LEVEL:
        return 0
    else:
        return 0.5


def clean_signals(largest_separation=15, save=False):
    """
    Deduplicate all the mall sequentially using a maximum separation of `largest_separation`.
    This imports the data using utils.import_signals

    :param largest_separation: (int) The largest error between points that is allowed
    :return: (pd.DataFrame) A cleaned concatenated signals dataframe containing all the malls
    """
    malls = ['Mall of Mauritius', 'Phoenix Mall', 'Home & Leisure']
    new_dfs = []
    for mall in tqdm(malls, desc='Clean Signals'):
        df = utils.import_signals(mall, version=1)

        clean_df = duplicates.duplicate_fill(df, largest_separation)

        if mall == 'Mall of Mauritius':
            clean_df = store_ids.clean_store_id(clean_df, store_only=False)

        new_dfs.append(clean_df)

    signal_df = pd.concat(new_dfs)
    signal_df.reset_index(inplace=True, drop=True)

    if save:
        signal_df.to_csv(dir_path + '/../data/bag_mus_12-22-2016v4.csv', index=False)

    return signal_df


def feature_extraction(signal_df, save=False):
    malls = [
        {'title': 'Mall of Mauritius', 'file_name': 'mauritius'},
        {'title': 'Phoenix Mall', 'file_name': 'phoenix'},
        {'title': 'Home & Leisure', 'file_name': 'home_and_leisure'},
    ]

    for mall in tqdm(malls, desc='Feature Extraction'):
        mall_mac_address_df = features.create_mac_address_features(
            signal_df=signal_df[signal_df.location == mall['title']]
        )

        if save:
            mall_mac_address_df.to_csv(dir_path + '/../data/{}_featuresv2.csv'.format(mall['file_name']), index=False)

        mall['df'] = mall_mac_address_df

    return malls


def clustering(save=False):
    malls = [
        {'title': 'Mall of Mauritius', 'file_name': 'mauritius'},
        {'title': 'Phoenix Mall', 'file_name': 'phoenix'},
        {'title': 'Home & Leisure', 'file_name': 'home_and_leisure'},
    ]

    for mall in tqdm(malls, desc='Clustering'):

        mall_mac_address_df = pd.read_csv(dir_path + '/../data/{}_features.csv'.format(mall['file_name']))

        mall_mac_address_df = mall_mac_address_df[mall_mac_address_df.frequency > 10]
        mall_mac_address_df = mall_mac_address_df.dropna()

        mac_address_probabilities = bayesian_inference.sequential(
            prior=0.5,
            feature_df=mall_mac_address_df,
            feature_list=FEATURE_LIST,
        )

        mall_mac_address_df['shopper_probability'] = mac_address_probabilities[-1][1]
        mall_mac_address_df['shopper_label'] = mall_mac_address_df['shopper_probability'].apply(find_label)

        if save:
            mall_mac_address_df.to_csv(dir_path + '/../data/{}_features_clustered.csv'.format(mall['file_name']), index=False)

        mall['df'] = mall_mac_address_df

    return malls


def clean_signals_and_create_mac_address_features():
    """
    Full clean of the signals data and find the probability of each device being a shopper.
    The cleaned signals data is output in the data folder and is called `bag_mus_12-22-2016v3.csv`.
    The mac addresses have been labelled with value of 1, 0.5, and 0 given a 5% confidence interval
    that they are shoppers.
    All mac address dataframe are saved to separate csv for each mall.

    :return: (list(dict)) each mall and then the title, file_name and df
    """
    signal_df = clean_signals(largest_separation=15)
    malls = feature_extraction(signal_df)
    malls = clustering()

    return malls
