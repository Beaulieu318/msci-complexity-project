import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from msci.cleaning.utils import *

COLUMNS_TO_IMPORT = ['mac_address', 'date_time', 'location', 'store_id', 'x', 'y']

shopper_df = pd.read_csv('../data/bag_mus_12-22-2016.csv', usecols=COLUMNS_TO_IMPORT)
shopper_df.date_time = shopper_df.date_time.astype('datetime64[ns]')
p_df = shopper_df[shopper_df['location'] == 'Phoenix Mall']

p = {'name': 'home', 'df': p_df, 'open_time': '09:30:00', 'close_time': '20:00:00'}


def remove_duplicates(shopper_df):
    """
    removes identical signals that are clearly errant duplicates i.e. same time and place for a given mac_id

    :param shopper_df: (pd.DataFrame) the signals of the shoppers
    :return: (pd.DataFrame) the cleaned signals of the shoppers
    """
    shopper_unique_df = shopper_df.drop_duplicates()
    return shopper_unique_df


def remove_outside_hours(shopper_df, open_time, close_time, analysis=False):
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
    shopper_outside_hours_df = shopper_df[shopper_df.mac_address.isin(mac_address_out_of_hours)]
    if analysis:
        return shopper_inside_hours_df, shopper_outside_hours_df
    else:
        return shopper_inside_hours_df


def remove_sparse_data(shopper_df, minimum, analysis=False):
    """
    removes mac_ids that have too few data points to be of use

    :param shopper_df: (pd.DataFrame) the signals of the shoppers
    :param minimum: the threshold for number of data points for a data set to be kept
    :return: (pd.DataFrame) the cleaned signals of the shoppers
    """
    mac_group = shopper_df.groupby('mac_address')
    mac_address_sparse = mac_group.size()[mac_group.size() >= minimum].index.tolist()
    shopper_large_data_df = shopper_df[shopper_df.mac_address.isin(mac_address_sparse)]
    shopper_sparse_data_df = shopper_df[~shopper_df.mac_address.isin(mac_address_sparse)]
    if analysis:
        return shopper_large_data_df, shopper_sparse_data_df
    else:
        return shopper_large_data_df


def remove_unrealistic_speeds(shopper_df, speed, notebook=False, analysis=False):
    """
    removes mac ids that are moving too fast to be pedestrian movement

    :param shopper_df: (pd.DataFrame) the signals of the shoppers
    :param speed: max speed allowed for pedestrian
    :param notebook: allows return of speeds for plotting purposes in notebook
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
        mac_speeds.append(np.mean(speeds))
    if notebook:
        return mac_speeds
    else:
        shopper_good_speeds_df = shopper_df[~shopper_df.mac_address.isin(mac_too_fast)]
        shopper_wrong_speeds_df = shopper_df[shopper_df.mac_address.isin(mac_too_fast)]
        if analysis:
            return shopper_wrong_speeds_df, shopper_good_speeds_df
        else:
            return shopper_good_speeds_df


def remove_long_gap(shopper_df, max_gap, analysis=False):
    """
    removes data points where a time gap between signals exceeds max_gap

    :param shopper_df: dirty data frames
    :param max_gap: largest tolerable gap between successive signals
    :param analysis: return negative df
    :return: cleaned data frame
    """
    macs = shopper_df.mac_address.drop_duplicates().tolist()
    deltas = time_delta(macs, shopper_df, plot=False, flat=False)
    print('deltas')
    exceed = [np.amax(i) > max_gap for i in deltas]
    mac_address_gap = [macs[mac] for mac in range(len(macs)) if exceed[mac]]
    shopper_no_gap_df = shopper_df[~shopper_df.mac_address.isin(mac_address_gap)]
    if analysis:
        shopper_gap_df = shopper_df[shopper_df.mac_address.isin(mac_address_gap)]
        return shopper_no_gap_df, shopper_gap_df
    else:
        return shopper_no_gap_df


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
    euclideans = np.array([euclidean_distance(pos[i], pos[i + 1]) for i in range(len(pos) - 1)])
    dt = np.array([time_difference(times[i], times[i + 1]) for i in range(len(times) - 1)])
    speeds = euclideans / dt
    return speeds


def time_delta(macs, df, plot=True, flat=True):
    """
    computes time differences for each reading

    :param macs: mac addresses
    :param df: data frame
    :param plot: whether to plot
    :param flat: whether to consider each mac address separately
    :return: time deltas
    """
    df = df.sort_values('date_time')
    mac_group = df.groupby('mac_address')
    td = []
    delta_r = []
    for mac in macs:
        times = mac_group.get_group(mac).date_time.tolist()
        x = mac_group.get_group(mac).x.tolist()
        y = mac_group.get_group(mac).y.tolist()
        pos = list(zip(x,y))
        time_deltas = [time_difference(times[i],times[i+1]) for i in range(len(times)-1)]
        r_deltas = [euclidean_distance(pos[i],pos[i+1]) for i in range(len(times)-1)]
        delta_r.append(r_deltas)
        td.append(time_deltas)
    if plot:
        fig = plt.figure()
        plt.xlabel('Difference in Time Between Readings')
        plt.ylabel('Probability')
        if flat:
            flat_td = [i for sub in td for i in sub if i is not 0]
            plt.hist(flat_td, bins=50, normed=True)
            fig.show()
            return flat_td
        else:
            for mac in range(len(td)):
                plt.hist(td[mac], bins=200, normed=True)
        fig.show()
    return td, delta_r


def delta_t_r_profile(t, r, delta_t):
    t_filter = [i for i in t if i == delta_t]
    r_filter = [r[i] for i in range(len(t)) if t[i] == delta_t]
    return r_filter


def _filter_deviation(means, std, f):
    """
    filters data whose standard deviation exceeds the mean by a factor f
    :param means: mean of time deltas
    :param std: standard deviation of time deltas
    :param f: tolerance factor
    :return: filtered data
    """
    data = list(zip(means,std))
    data = [i for i in data if i[1] < f*i[0]]
    return data


def radius_gyration(df):
    sorted = df.sort_values('date_time')
    grouped = sorted.groupby('mac_address')
    macs = df.mac_address.drop_duplicates().tolist()[0:20]
    centroids = [np.array((np.mean(grouped.get_group(i).x.tolist()), np.mean(grouped.get_group(i).y.tolist()))) for i in macs]
    gyrations = []
    for mac in range(len(macs)):
        r_cm = centroids[mac]
        x = grouped.get_group(macs[mac]).x.tolist()
        y = grouped.get_group(macs[mac]).y.tolist()
        r = [np.array(i) for i in list(zip(x, y))]
        displacements = [i - r_cm for i in r]
        displacement_sum = [i[0]**2 + i[1]**2 for i in displacements]
        gyrations.append(np.sqrt(np.mean(displacement_sum)))
    return centroids, gyrations


def length_of_stay(df, plot=True):
    sorted = df.sort_values('date_time')
    grouped = sorted.groupby('mac_address')
    macs = df.mac_address.drop_duplicates().tolist()
    groups = [grouped.get_group(i).date_time.tolist() for i in macs]
    counts = [len(i) for i in groups]
    time_deltas = [time_difference(i[0], i[-1]) for i in groups]
    if plot:
        td0 = [i for i in time_deltas if i is not 0] #gets rid of mac addresses for which there is only one reading
        fig = plt.figure()
        plt.scatter(time_deltas, counts)
        plt.xlabel('Length of Stay')
        plt.ylabel('Number of Counts')
        fig.show()
        fig = plt.figure()
        hist = plt.hist(td0, bins=50, normed=False)
        plt.xlabel('Length of Stay')
        plt.ylabel('Probability (Un-normalised)')
        fig.show()
        return time_deltas, counts, hist
    else:
        return time_deltas, counts


def count_density_variance(df, minute_resolution):
    """
    Bins times into intervals of length 'minute resolution', and finds variance of counts
    :param df: data frame
    :param minute_resolution: (str) number of minutes into which to group data
    :return: list of count variances for each mac address
    """
    sorted = df.sort_values('date_time')
    grouped = sorted.groupby('mac_address')
    macs = df.mac_address.drop_duplicates().tolist()
    cdv = []
    for mac in macs:
        time = grouped.get_group(mac).date_time.dt.round(minute_resolution + 'min').value_counts()
        count_variance = time.std()
        cdv.append(count_variance)
    return cdv


def delta_r(df):
    sorted = df.sort_values('date_time')
    grouped = sorted.groupby('mac_address')
    macs = df.mac_address.drop_duplicates().tolist()


def clean(shopper_df, minimum, speed, open_time, close_time):
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
    shopper_df = remove_outside_hours(shopper_df, open_time, close_time)
    shopper_df = remove_sparse_data(shopper_df, minimum)
    shopper_df = remove_unrealistic_speeds(shopper_df, speed)
    return shopper_df


def time_volume_analysis(df):
    fig = plt.figure()
    df.date_time.hist(bins=200)
    plt.xlabel('Time')
    plt.ylabel('Counts')
    fig.show()


def main():
    shopper_df = pd.read_csv('../data/bag_mus_12-22-2016.csv', usecols=COLUMNS_TO_IMPORT)
    shopper_df.date_time = shopper_df.date_time.astype('datetime64[ns]')

    hl_df = shopper_df[shopper_df['location'] == 'Home & Leisure']
    mm_df = shopper_df[shopper_df['location'] == 'Mall of Mauritius']
    p_df = shopper_df[shopper_df['location'] == 'Phoenix Mall']

    minimum = 10
    speed = 3

    locations = [
        {'name': 'home', 'df': hl_df, 'open_time': '09:30:00', 'close_time': '20:00:00'},
        {'name': 'mauritius', 'df': mm_df, 'open_time': '09:30:00', 'close_time': '21:00:00'},
        {'name': 'phoenix', 'df': p_df, 'open_time': '09:30:00', 'close_time': '18:00:00'}
    ]

    for location in locations:
        shopper_cleaned_df = clean(location['df'], minimum, speed, location['open_time'], location['close_time'])
        shopper_cleaned_df.to_csv(
            path_or_buf='../data/clean_data_' + location['name'] + '.csv',
            columns=COLUMNS_TO_IMPORT,
            index=False
        )


#if __name__ == '__main__':
    #main()
