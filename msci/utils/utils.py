import os
import numpy as np
import pandas as pd

COLUMNS_TO_IMPORT = ['mac_address', 'date_time', 'location', 'store_id', 'x', 'y', 'wifi_type', 'email']

data_path = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
images_path = os.path.dirname(os.path.realpath(__file__)) + '/../images/'


def import_signals(mall='Mall of Mauritius', version=3, signal_type=None):
    """
    Imports the signals of all the devices from a particular mall.
    The data can be downloaded from google drive:
    https://drive.google.com/drive/u/0/folders/0B4BnJuo0Kb2NR2lrNHZscTkzUGs

    :param mall: (str) 'Mall of Mauritius', 'Phoenix Mall' or 'Home & Leisure'
    :param version: (int) 1: raw signals, 2: raw mall of mauritius, 3: all mall with no duplicates (cleaned)
    :param signal_type: (int) 1: shopper, 0: non-shopper, 0.5: not classified (there is a 0.05 confidence level)
    :return: (pd.DataFrame) the signals
    """
    malls = {
        'Mall of Mauritius': 'mauritius',
        'Phoenix Mall': 'phoenix',
        'Home & Leisure': 'home_and_leisure',
    }

    if version == 1:
        signal_df = pd.read_csv(data_path + 'bag_mus_12-22-2016.csv', usecols=COLUMNS_TO_IMPORT)
        signal_df.date_time = signal_df.date_time.astype('datetime64[ns]')
        signal_df = signal_df[signal_df['location'] == mall]
        signal_df = signal_df.sort_values('date_time')
    elif version == 2:
        signal_df = pd.read_csv(data_path + 'bag_mus_12-22-2016v2.csv', usecols=COLUMNS_TO_IMPORT)
        signal_df.date_time = signal_df.date_time.astype('datetime64[ns]')
        signal_df = signal_df[signal_df['location'] == mall]
        signal_df = signal_df.sort_values('date_time')
    elif version == 3:
        mac_address_df = pd.read_csv(
            data_path + '{}_features.csv'.format(malls[mall])
        )

        if signal_type == 1:
            mac_address_df = mac_address_df[mac_address_df.shopper_label == 1]
        elif signal_type == 0.5:
            mac_address_df = mac_address_df[mac_address_df.shopper_label == 0.5]
        elif signal_type == 0:
            mac_address_df = mac_address_df[mac_address_df.shopper_label == 0]

        signal_df = pd.read_csv(data_path + 'bag_mus_12-22-2016v3.csv', usecols=COLUMNS_TO_IMPORT)
        signal_df.date_time = signal_df.date_time.astype('datetime64[ns]')
        signal_df = signal_df[
            (signal_df['location'] == mall) &
            (signal_df.mac_address.isin(mac_address_df.mac_address.tolist()))
        ]
        signal_df = signal_df.sort_values('date_time')
    elif version == 4:
        signal_df = pd.read_csv(data_path + 'bag_mus_12-22-2016v4.csv', usecols=COLUMNS_TO_IMPORT)
        signal_df.date_time = signal_df.date_time.astype('datetime64[ns]')
        signal_df = signal_df[
            (signal_df['location'] == mall)
        ]
        signal_df = signal_df.sort_values('date_time')
    else:
        raise Exception('Version of signals is not defined! Please choose 1,2 or 3')

    return signal_df


def import_mac_addresses(mall='Mall of Mauritius', version=1, signal_type=None):
    """
    Imports the mac addresses for the chosen mall with all the mac addresses' features.
    The data can be downloaded from google drive:
    https://drive.google.com/drive/u/0/folders/0B4BnJuo0Kb2NR2lrNHZscTkzUGs

    :param mall: (str) 'Mall of Mauritius', 'Phoenix Mall' or 'Home & Leisure'
    :param signal_type: (int) 1: shopper, 0: non-shopper, 0.5: not classified (there is a 0.05 confidence level)
    :return: (pd.DataFrame) the mac addresses with their features
    """
    malls = {
        'Mall of Mauritius': 'mauritius',
        'Phoenix Mall': 'phoenix',
        'Home & Leisure': 'home_and_leisure',
    }

    if version == 1:
        mac_address_df = pd.read_csv(
            data_path + '{}_features.csv'.format(malls[mall])
        )

        if signal_type == 1:
            mac_address_df = mac_address_df[mac_address_df.shopper_label == 1]
        elif signal_type == 0.5:
            mac_address_df = mac_address_df[mac_address_df.shopper_label == 0.5]
        elif signal_type == 0:
            mac_address_df = mac_address_df[mac_address_df.shopper_label == 0]
    elif version == 2:
        mac_address_df = pd.read_csv(
            data_path + '{}_featuresv2.csv'.format(malls[mall])
        )
    elif version == 3:
        mac_address_df = pd.read_csv(
            data_path + '{}_featuresv3.csv'.format(malls[mall])
        )
    elif version == 4:
        mac_address_df = pd.read_csv(
            data_path + '{}_featuresv4.csv'.format(malls[mall])
        )
    else:
        raise Exception("There is no version with this number. Please choose either 1 or 2!")

    return mac_address_df


def import_shop_directory(mall='Mall of Mauritius', version=1):
    malls = {
        'Mall of Mauritius': 'mauritius',
        'Phoenix Mall': 'phoenix',
        'Home & Leisure': 'home_and_leisure',
    }

    if version == 1:
        directory_df = pd.read_csv(
            data_path + '{}_directory.csv'.format(malls[mall]),
        )
    elif version == 2:
        directory_df = pd.read_csv(
            data_path + '{}_directoryv2.csv'.format(malls[mall]),
        )
    else:
        raise Exception("There is no version with this number. Please choose either 1 or 2!")

    return directory_df


def df_to_csv(df, name, sort=False):
    if sort:
        time_sort = df.sort_values('date_time')
        mac_group = time_sort.groupby('mac_address')
        mac_group.to_csv(path_or_buf=data_path + 'clean_data_' + name + '.csv', columns=COLUMNS_TO_IMPORT, index=False)
    else:
        df.to_csv(path_or_buf=data_path + 'clean_data_' + name + '.csv', columns=COLUMNS_TO_IMPORT, index=False)


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


def centroid(position_list):
    x = [i[0] for i in position_list]
    y = [i[1] for i in position_list]
    return [np.mean(x), np.mean(y)]


def add_manufacturer_to_signal(signal_df):
    """
    Returns a list containing the manufacturer for each signal in the signal_df

    :param signal_df: (pd.DataFrame) Contains the signals
    :return: (list) containing the manufacturer
    """
    mac_cross_reference_df = pd.read_csv(data_path + 'mac_address_cross_reference.csv')
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
