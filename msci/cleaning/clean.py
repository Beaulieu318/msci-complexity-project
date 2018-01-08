"""
This script cleans the raw data
"""

import os

from msci.cleaning import duplicates
from msci.cleaning import features
from msci.cleaning import bayesian_inference

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

confidence_level = 0.05


def find_label(x):
    if x > (1-confidence_level):
        return 1
    elif x < confidence_level:
        return 0
    else:
        return 0.5


def get_clean_data():
    signal_df = duplicates.all_mall_duplicate_fill(largest_separation=15)
    signal_df.to_csv(dir_path + '/../data/bag_mus_12-22-2016v3.csv', index=False)

    malls = [
        {'title': 'Mall of Mauritius', 'file_name': 'mauritius'},
        {'title': 'Phoenix Mall', 'file_name': 'phoenix'},
        {'title': 'Home & Leisure', 'file_name': 'home_and_leisure'},
    ]

    for mall in malls:

        mall_mac_address_df = features.create_mac_address_features(
            signal_df=signal_df[signal_df.location == mall['title']]
        )

        mall_mac_address_clean_df = mall_mac_address_df[mall_mac_address_df.frequency > 10]
        mall_mac_address_clean_df = mall_mac_address_clean_df.dropna()

        mac_address_probabilities = bayesian_inference.sequential(
            prior=0.5,
            feature_df=mall_mac_address_clean_df,
            feature_list=FEATURE_LIST,
        )

        mall_mac_address_clean_df['shopper_probability'] = mac_address_probabilities[-1][1]
        mall_mac_address_clean_df['shopper_label'] = mall_mac_address_clean_df['shopper_probability'].apply(find_label)

        mall_mac_address_clean_df.to_csv(dir_path + '/../data/{}_features.csv'.format(mall['file_name']), index=False)
        mall['df'] = mall_mac_address_clean_df

    return malls
