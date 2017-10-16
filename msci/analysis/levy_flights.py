import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import log_bin as bin

columns = ['mac_address', 'date_time', 'location', 'floor', 'x', 'y']

p_df = pd.read_csv('../data/clean_data_phoenix.csv', usecols=columns)
p_df.date_time = p_df.date_time.astype('datetime64[ns]')


class DataSet:

    def __init__(self, dataframe):
        self.df = dataframe
        self.macs = self.df.mac_address.drop_duplicates().tolist()
        self.time_sorted = self.df.sort_values('date_time')
        self.mac_group = self.time_sorted.groupby('mac_address')

    def path_length_analysis(self):
        plengths = [_path_length_of_group(self.mac_group.get_group(mac)) for mac in self.macs]
        lb = _logbin(plengths,1,1,1.2)
        return plengths, lb

    def duration_analysis(self):
        durations = [_duration(self.mac_group.get_group(mac)) for mac in self.macs]
        lb = _logbin(durations, 1, 1, 1.2)
        return durations, lb


def _path_length_of_group(mac_dp):
    x = mac_dp['x'].tolist()
    y = mac_dp['y'].tolist()
    pos = list(zip(x, y))
    euclideans = np.array([_euclidean_distance(pos[i], pos[i + 1]) for i in range(len(pos) - 1)])
    return np.sum(euclideans)


def _duration(mac_dp):
    times = mac_dp['date_time'].tolist()
    duration = _time_difference(times[0], times[-1])
    #print(times[0], times[-1], duration)
    return duration


def _euclidean_distance(xy1, xy2):
    """
    Returns euclidean distance between points xy1 and xy2

    :param xy1: (tuple) 1st position in (x,y)
    :param xy2: (tuple) 2nd position in (x,y)
    :return: (float) euclidean distance
    """
    return np.sqrt((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)


def _time_difference(t0, t1):
    """
    time difference between two timedelta objects

    :param t0: (timedelta object) first timestamp
    :param t1: (timedelta object) second timestamp
    :return: (float) number of seconds elapsed between t0, t1
    """
    tdelta = t1 - t0
    return tdelta.seconds


def _logbin(data,binstart,firstbinwidth,a,zeros = True):
    if zeros:
        lb = bin.log_bin(data,binstart,firstbinwidth,a)
    else:
        lb = bin.log_bin(data,binstart,firstbinwidth,a,drop_zeros = True)
    return lb


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