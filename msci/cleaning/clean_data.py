import pandas as pd
import numpy as np

COLUMNS_TO_IMPORT = ['mac_address', 'date_time', 'location', 'floor', 'x', 'y']


def remove_duplicates(shopper_df):
    """
    removes identical signals that are clearly errant duplicates i.e. same time and place for a given mac_id

    :param shopper_df: (pd.DataFrame) the signals of the shoppers
    :return: (pd.DataFrame) the cleaned signals of the shoppers
    """
    shopper_unique_df = shopper_df.drop_duplicates()
    return shopper_unique_df


def remove_outside_hours(shopper_df, open_time, close_time):
    """
    removes mac addresses that are received outside opening hours

    :param shopper_df: (pd.DataFrame) the signals of the shoppers
    :param open_time: ('hh:mm:ss') opening time of mall
    :param close_time: ('hh:mm:ss') closing time of mall
    :return: (pd.DataFrame) the cleaned signals of the shoppers
    """
    date_index_df = shopper_df.copy()
    date_index_df.index = date_index_df.date_time.astype('datetime64[ns]')
    signal_out_of_hours = date_index_df.between_time(close_time, open_time)
    mac_address_out_of_hours = signal_out_of_hours.mac_address.drop_duplicates().tolist()
    shopper_inside_hours_df = shopper_df[~shopper_df.mac_address.isin(mac_address_out_of_hours)]
    return shopper_inside_hours_df


def remove_sparse_data(shopper_df, minimum):
    """
    removes mac_ids that have too few data points to be of use

    :param shopper_df: (pd.DataFrame) the signals of the shoppers
    :param minimum: the threshold for number of data points for a data set to be kept
    :return: (pd.DataFrame) the cleaned signals of the shoppers
    """
    mac_group = shopper_df.groupby('mac_address')
    mac_address_sparse = mac_group.size()[mac_group.size() > minimum].index.tolist()
    shopper_large_data_df = shopper_df[~shopper_df.mac_address.isin(mac_address_sparse)]
    return shopper_large_data_df


def remove_unrealistic_speeds(shopper_df, speed):
    """
    removes mac ids that are moving too fast to be pedestrian movement

    :param shopper_df: (pd.DataFrame) the signals of the shoppers
    :param speed: max speed allowed for pedestrian
    :return: (pd.DataFrame) the cleaned signals of the shoppers
    """
    shopper_df.date_time = shopper_df.date_time.astype('datetime64[ns]')
    time_sorted = shopper_df.sort_values('date_time')
    mac_group = time_sorted.groupby('mac_address')

    # Remove a single mac address (can't calculate speed)
    macs = mac_group.size()[mac_group.size() > 1].index.tolist()

    mac_too_fast = []
    mac_speeds = []

    for mac in macs:
        mac_dp = mac_group.get_group(mac)
        speeds = _speed_of_group(mac_dp)
        speeds = speeds[speeds < 100000]
        if np.mean(speeds) > speed:
            mac_too_fast.append(mac)
        mac_speeds.append(speeds)
    shopper_good_speeds_df = shopper_df[~shopper_df.mac_address.isin(mac_too_fast)]
    return shopper_good_speeds_df


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


def clean(shopper_df):
    """
    cleans the dataframe containing the signals of the shoppers by:
     - removing duplicates
     - removing mac addresses with signals outside of shopping hours
     - removing mac addresses with few signals
     - removing shoppers with unrealistic speeds

    :param shopper_df: (pd.DataFrame) the signals of the shoppers
    :return: (pd.DataFrame) the cleaned signals of the shoppers
    """
    shopper_df = remove_duplicates(shopper_df)
    shopper_df = remove_outside_hours(shopper_df, '08:00:00', '22:00:00')
    shopper_df = remove_sparse_data(shopper_df, 10)
    shopper_df = remove_unrealistic_speeds(shopper_df, 10)
    return shopper_df


def main():
    shopper_df = pd.read_csv('../data/bag_mus_12-22-2016.csv', usecols=COLUMNS_TO_IMPORT)
    shopper_df.date_time = shopper_df.date_time.astype('datetime64[ns]')

    hl_df = shopper_df[shopper_df['location'] == 'Home & Leisure']
    mm_df = shopper_df[shopper_df['location'] == 'Mall of Mauritius']
    p_df = shopper_df[shopper_df['location'] == 'Phoenix Mall']

    locations = {'home': hl_df, 'mauritius': mm_df, 'phoenix': p_df}

    for location, shopper_dirty_df in locations.items():
        shopper_cleaned_df = clean(shopper_dirty_df)
        shopper_cleaned_df.to_csv(path_or_buf='../data/clean_data_' + location + '.csv', columns=COLUMNS_TO_IMPORT)


if __name__ == '__main__':
    main()
