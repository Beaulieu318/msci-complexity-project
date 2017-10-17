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
        self.macs = self.df.mac_address.drop_duplicates().tolist()[:5]
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
    print(times[-1],times[0])
    duration = _time_difference(times[-1],times[0])
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

def clean_all_data(minimum, speed, open_time, close_time):
    p_ds = DataSet(p_df, open_time, close_time)
    m_ds = DataSet(mm_df, open_time, close_time)
    h_ds = DataSet(hl_df, open_time, close_time)
    p_ds.clean(minimum, speed)
    m_ds.clean(minimum, speed)
    h_ds.clean(minimum, speed)
    p_ds.export_csv('phoenix')
    m_ds.export_csv('mauritius')
    h_ds.export_csv('homeleisure')


class DataSetClean:

    def __init__(self, df, open_time, close_time):
        """
        :param df: the data frame to be cleaned
        :param open: ('hh:mm:ss') opening time of mall
        :param close: ('hh:mm:ss') closing time of mall
        """
        self.df = df
        self.open = open_time
        self.close = close_time

    def remove_duplicates(self):
        """
        removes identical signals that are clearly errant duplicates i.e. same time and place for a given mac_id
        """
        self.df = self.df.drop_duplicates()

    def remove_outside_hours(self):
        """
        removes mac addresses that are received outside opening hours

        :return: legitimate mac ids i.e. those that are received during opening hours
        """
        signal_out_of_hours = self.df[self.df.date_time.dt.strftime('%H:%M:%S').between(self.open, self.close)]
        mac_address_out_of_hours = signal_out_of_hours.mac_address.drop_duplicates().tolist()
        in_time_df = self.df[~self.df.mac_address.isin(mac_address_out_of_hours)]
        self.df = in_time_df
        print('Number of data points left:', len(self.df))
        mac_address_in_hours = self.df.mac_address.drop_duplicates().tolist()
        return mac_address_in_hours

    def remove_sparse_data(self, minimum):
        """
        removes mac_ids that have too few data points to be of use

        :param minimum: the threshold for number of data points for a data set to be kept
        """
        mac_group = self.df.groupby('mac_address')
        sizes = mac_group.size()
        thresh = [sizes < minimum]
        sizes_index = sizes.index.tolist()
        mac_address_sparse = [sizes_index[i] for i in range(len(sizes)) if thresh[0][i]]
        self.df = self.df[~self.df.mac_address.isin(mac_address_sparse)]
        print('Number of data points left:', len(self.df))

    def remove_unrealistic_speeds(self, speed):
        """
        removes mac ids that are moving too fast to be pedestrian movement

        :param speed: max speed allowed for pedestrian
        """
        macs = self.remove_outside_hours()
        time_sorted = self.df.sort_values('date_time')
        mac_group = time_sorted.groupby('mac_address')
        mac_too_fast = []
        mac_speeds = []
        for mac in macs:
            mac_dp = mac_group.get_group(mac)
            speeds = _speed_of_group(mac_dp)
            speeds = speeds[speeds < 100000]
            if np.mean(speeds)>speed:
                mac_too_fast.append(mac)
            mac_speeds.append(speeds)
        self.df = self.df[~self.df.mac_address.isin(mac_too_fast)]

    def clean(self, minimum, speed):
        self.remove_duplicates()
        self.remove_sparse_data(minimum)
        self.remove_unrealistic_speeds(speed)

    def export_csv(self, mall_name):
        self.df.to_csv(path_or_buf='../data/clean_data_' + mall_name + '.csv', columns=columns)


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