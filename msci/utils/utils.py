import os
import numpy as np
import pandas as pd

COLUMNS_TO_IMPORT = ['mac_address', 'date_time', 'location', 'store_id', 'x', 'y']

dir_path = os.path.dirname(os.path.realpath(__file__))


def import_data(mall='Mall of Mauritius'):
    shopper_df = pd.read_csv(dir_path + '/../data/bag_mus_12-22-2016.csv', usecols=COLUMNS_TO_IMPORT)
    shopper_df.date_time = shopper_df.date_time.astype('datetime64[ns]')
    signal_df = shopper_df[shopper_df['location'] == mall]
    return signal_df


def df_to_csv(df, name, sort=False):
    if sort:
        time_sort = df.sort_values('date_time')
        mac_group = time_sort.groupby('mac_address')
        mac_group.to_csv(path_or_buf=dir_path + '/../data/clean_data_' + name + '.csv', columns=COLUMNS_TO_IMPORT, index=False)
    else:
        df.to_csv(path_or_buf=dir_path + '/../data/clean_data_' + name + '.csv', columns=COLUMNS_TO_IMPORT, index=False)


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


def add_device_type_signal(signal_df):
    mac_cross_reference_df = pd.read_csv(dir_path + '/../data/mac_address_cross_reference.csv')
    signal_df2 = signal_df.copy()
    signal_df2['mac_address_short'] = signal_df2.mac_address.str.replace(':', '').str.upper().str[:6]
    signal_df2 = signal_df2.merge(mac_cross_reference_df, how='left', left_on='mac_address_short', right_on='Assignment')
    return signal_df2['Organization Name'].tolist()
