import os
import numpy as np
import pandas as pd

COLUMNS_TO_IMPORT = ['mac_address', 'date_time', 'location', 'store_id', 'x', 'y', 'wifi_type', 'email']

dir_path = os.path.dirname(os.path.realpath(__file__))


def import_signals(mall='Mall of Mauritius'):
    shopper_df = pd.read_csv(dir_path + '/../data/bag_mus_12-22-2016v2.csv', usecols=COLUMNS_TO_IMPORT)
    shopper_df.date_time = shopper_df.date_time.astype('datetime64[ns]')
    signal_df = shopper_df[shopper_df['location'] == mall]
    signal_df = signal_df.sort_values('date_time')
    return signal_df


def import_mac_addresses():
    mac_address_df = pd.read_csv(dir_path + '/../data/mac_address_features.csv')
    return mac_address_df


def df_to_csv(df, name, sort=False):
    if sort:
        time_sort = df.sort_values('date_time')
        mac_group = time_sort.groupby('mac_address')
        mac_group.to_csv(path_or_buf=dir_path + '/../data/clean_data_' + name + '.csv', columns=COLUMNS_TO_IMPORT, index=False)
    else:
        df.to_csv(path_or_buf=dir_path + '/../data/clean_data_' + name + '.csv', columns=COLUMNS_TO_IMPORT, index=False)


def reduce_df(df, mac_list):
    return df[df.mac_address.isin(mac_list)]


def euclidean_distance(xy1, xy2):
    """
    Returns euclidean distance between points xy1 and xy2

    :param xy1: (tuple) 1st position in (x,y)
    :param xy2: (tuple) 2nd position in (x,y)
    :return: (float) euclidean distance
    """
    return np.sqrt((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)


def time_difference(t0, t1):
    """
    time difference between two timedelta objects
    :param t0: (timedelta object) first timestamp
    :param t1: (timedelta object) second timestamp
    :return: (float) number of seconds elapsed between t0, t1
    """
    tdelta = t1 - t0
    return tdelta.seconds


def add_manufacturer_to_signal(signal_df):
    """
    Returns a list containing the manufacturer for each signal in the signal_df

    :param signal_df: (pd.DataFrame) Contains the signals
    :return: (list) containing the manufacturer
    """
    mac_cross_reference_df = pd.read_csv(dir_path + '/../data/mac_address_cross_reference.csv')
    signal_df2 = signal_df.copy()
    signal_df2['mac_address_short'] = signal_df2.mac_address.str.replace(':', '').str.upper().str[:6]
    signal_df2 = signal_df2.merge(mac_cross_reference_df, how='left', left_on='mac_address_short', right_on='Assignment')
    return signal_df2['Organization Name'].tolist()


def bayes_bool(result, likelihood, prior):
    """

    :param result: (np.array) The boolean result (1 or 0) of result of the function
    :param likelihood: (float) The probability that the result is going to be true
    :param prior: (np.array) The initial probability estimate
    :return: (np.array) The posterior which is the updated estimate of the probability given the result of the function
    """
    func_probabilities = np.array([likelihood[0], likelihood[1]])
    macs_likelihood = func_probabilities[result]
    posterior = (macs_likelihood * prior)
    return posterior
