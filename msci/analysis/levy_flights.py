import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import log_bin as bin

columns = ['mac_address', 'date_time', 'location', 'store_id', 'x', 'y']

p_df = pd.read_csv('../data/clean_data_phoenix.csv', usecols=columns)
p_df.date_time = p_df.date_time.astype('datetime64[ns]')
p_df = p_df.sort_values('date_time')
p_mac_group_df = p_df.groupby('mac_address')
p_macs = p_df.mac_address.drop_duplicates().tolist()

hl_df = pd.read_csv('../data/clean_data_home.csv', usecols=columns).sort_values('date_time')
hl_df.date_time = hl_df.date_time.astype('datetime64[ns]')
hl_mac_group_df = hl_df.groupby('mac_address')
hl_macs = hl_df.mac_address.drop_duplicates().tolist()

mm_df = pd.read_csv('../data/clean_data_mauritius.csv', usecols=columns).sort_values('date_time')
mm_df.date_time = mm_df.date_time.astype('datetime64[ns]')
mm_mac_group_df = mm_df.groupby('mac_address')
mm_macs = mm_df.mac_address.drop_duplicates().tolist()

df_dictionary = {
    'p': {'df': p_df, 'grouped': p_mac_group_df, 'macs': p_macs},
    'mm': {'df': mm_df, 'grouped': mm_mac_group_df, 'macs': mm_macs},
    'hl': {'df': hl_df, 'grouped': hl_mac_group_df, 'macs': hl_macs}
}


def path_length_analysis(dictionary_key):
    dictionary = df_dictionary[dictionary_key]
    p_lengths = [_path_length_of_group(dictionary['grouped'].get_group(mac)) for mac in dictionary['macs']]
    lb = _logbin([i for i in p_lengths if i > 10**2], 1, 1, 1.2)
    return p_lengths, lb


def _histogram(data):
    hist, bins = np.histogram(data, bins=1000, normed=True)
    x = (bins[:-1] + bins[1:])/2
    return x, hist


def _path_length_of_group(mac_dp):
    x = mac_dp['x'].tolist()
    y = mac_dp['y'].tolist()
    pos = list(zip(x, y))
    euclideans = np.array([_euclidean_distance(pos[i], pos[i + 1]) for i in range(len(pos) - 1)])
    return np.sum(euclideans)


def _euclidean_distance(xy1, xy2):
    """
    Returns euclidean distance between points xy1 and xy2

    :param xy1: (tuple) 1st position in (x,y)
    :param xy2: (tuple) 2nd position in (x,y)
    :return: (float) euclidean distance
    """
    return np.sqrt((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)


def duration_analysis(dictionary_key):
    dictionary = df_dictionary[dictionary_key]
    durations = [_duration(dictionary['grouped'].get_group(mac)) for mac in dictionary['macs']]
    lb = _logbin(durations, 1, 1, 1.2)
    return durations, lb


def _duration(mac_dp):
    times = mac_dp['date_time'].tolist()
    duration = _time_difference(times[0], times[-1])
    return duration


def _time_difference(t0, t1):
    """
    time difference between two timedelta objects

    :param t0: (timedelta object) first timestamp
    :param t1: (timedelta object) second timestamp
    :return: (float) number of seconds elapsed between t0, t1
    """
    tdelta = t1 - t0
    return tdelta.seconds


def _logbin(data, bin_start, first_bin_width, a):
    return bin.log_bin(data, bin_start, first_bin_width, a, drop_zeros=False)


def _speed_of_group(mac_dp):
    """
    computes speeds of mac_ids

    :param mac_dp: reduced dataframe for specific mac_id
    :return: (list) speeds at different times
    """
    x = mac_dp['x'].tolist()
    y = mac_dp['y'].tolist()
    pos = list(zip(x, y))
    times = mac_dp['date_time'].tolist()
    euclideans = np.array([_euclidean_distance(pos[i], pos[i + 1]) for i in range(len(pos) - 1)])
    dt = np.array([_time_difference(times[i], times[i + 1]) for i in range(len(times) - 1)])
    speeds = euclideans / dt
    return speeds


def _basic_plot(data, scale='log', scatter=True):
    x = data[0]
    y = data[1]
    fig = plt.figure()
    if scale == 'log':
        plt.xscale('log')
        plt.yscale('log')
    if scatter:
        plt.scatter(x, y)
    else:
        plt.plot(x, y)
    plt.plot([1,1000],[np.exp(-1.12*np.log(i)+1.15) for i in [1,1000]])
    fig.show()
