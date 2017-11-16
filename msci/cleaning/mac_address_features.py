import os
import pandas as pd
import numpy as np

from msci.utils import utils

dir_path = os.path.dirname(os.path.realpath(__file__))


def create_mac_address_df(signal_df):
    mac_addresses = signal_df.mac_address.value_counts()
    mac_address_df = pd.DataFrame(mac_addresses)
    mac_address_df.rename(columns={'mac_address': 'frequency'}, inplace=True)
    mac_address_df['mac_address'] = mac_address_df.index
    mac_address_df.reset_index(inplace=True, drop=True)
    return mac_address_df


def calculate_radius_gyration(signal_df, mac_address_df):
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')
    macs = mac_address_df.mac_address.tolist()
    centroids = [np.array((np.mean(signal_mac_group.get_group(i).x.tolist()), np.mean(signal_mac_group.get_group(i).y.tolist()))) for i in macs]
    gyrations = []
    for mac in range(len(macs)):
        r_cm = centroids[mac]
        x = signal_mac_group.get_group(macs[mac]).x.tolist()
        y = signal_mac_group.get_group(macs[mac]).y.tolist()
        r = [np.array(i) for i in list(zip(x, y))]
        displacements = [i - r_cm for i in r]
        displacement_sum = [i[0]**2 + i[1]**2 for i in displacements]
        gyrations.append(np.sqrt(np.mean(displacement_sum)))
    return centroids, gyrations


def find_device_type(mac_address_df):
    mac_cross_reference_df = pd.read_csv(dir_path + '/../data/mac_address_cross_reference.csv')
    mac_address_df2 = mac_address_df.copy()
    mac_address_df2['mac_address_short'] = mac_address_df2.mac_address.str.replace(':', '').str.upper().str[:6]
    mac_address_df2 = mac_address_df2.merge(mac_cross_reference_df, how='left', left_on='mac_address_short', right_on='Assignment')
    return mac_address_df2['Organization Name']


def calculate_count_density_variance(signal_df, mac_address_df, minute_resolution=5):
    """
    Bins times into intervals of length 'minute resolution', and finds variance of counts
    :param signal_df: data frame
    :param mac_address_df: (pd.DataFrame)
    :param minute_resolution: (str) number of minutes into which to group data
    :return: list of count variances for each mac address
    """
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')
    macs = mac_address_df.mac_address.tolist()
    cdv = []
    for mac in macs:
        time = signal_mac_group.get_group(mac).date_time.dt.round(str(minute_resolution) + 'min').value_counts()
        count_variance = time.std()
        cdv.append(count_variance)
    return cdv


def calculate_length_of_stay(signal_df, mac_address_df):
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')
    macs = mac_address_df.mac_address.tolist()
    groups = [signal_mac_group.get_group(i).date_time.tolist() for i in macs]
    time_deltas = [utils.time_difference(i[0], i[-1]) for i in groups]
    return time_deltas


def is_out_of_hours(signal_df, mac_address_df):
    macs = mac_address_df.mac_address.tolist()
    date_index_df = signal_df.copy()
    date_index_df.index = date_index_df.date_time.astype('datetime64[ns]')
    signal_out_of_hours = date_index_df.between_time('03:00:00', '05:00:00')
    mac_address_out_of_hours = signal_out_of_hours.mac_address.drop_duplicates().tolist()
    macs_is_out_of_hours = [1 if mac in mac_address_out_of_hours else 0 for mac in macs]
    return macs_is_out_of_hours


def calculate_average_speed(signal_df, mac_address_df):
    macs = mac_address_df.mac_address.tolist()
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')
    av_speeds = []
    for mac in macs:
        mac_signals_df = signal_mac_group.get_group(mac)
        av_speed = np.nan
        if len(mac_signals_df) > 1:
            columns_to_diff = ['date_time', 'x', 'y']
            mac_diffs_df = (mac_signals_df[columns_to_diff].shift(-1) - mac_signals_df[columns_to_diff]).iloc[:-1]
            mac_diffs_df = mac_diffs_df.assign(secs=mac_diffs_df.date_time.dt.seconds)
            mac_diffs_df = mac_diffs_df[mac_diffs_df.secs > 0]
            mac_speeds_df = (mac_diffs_df.x ** 2 + mac_diffs_df.y ** 2) ** 0.5 / mac_diffs_df.secs
            av_speed = mac_speeds_df.mean()
        av_speeds.append(av_speed)
    return av_speeds


def calculate_average_turning_angle(signal_df, mac_address_df):
    macs = mac_address_df.mac_address.tolist()
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')

    dot = lambda vec1, vec2: vec1[:, 0] * vec2[:, 0] + vec1[:, 1] * vec2[:, 1]
    mod = lambda vec: np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)

    av_turning_angle = []
    av_turning_angle_velocity = []
    for mac in macs:
        mac_signals_df = signal_mac_group.get_group(mac)

        turning_angle = np.nan
        turning_angle_velocity = np.nan

        if len(mac_signals_df) > 2:
            columns_to_diff = ['x', 'y']
            coords = mac_signals_df[columns_to_diff].as_matrix()
            vectors = coords[1:] - coords[:-1]
            u = vectors[1:]
            v = vectors[:-1]
            cos_angle = dot(u, v) / (mod(u) * mod(v))
            mac_turning_angles = np.arccos(cos_angle)
            turning_angle = np.nanmean(mac_turning_angles)

            if len(mac_turning_angles) > 1:
                mac_turning_angle_velocities = np.array(mac_turning_angles)[1:] - np.array(mac_turning_angles[:-1])
                turning_angle_velocity = np.nanmean(mac_turning_angle_velocities)

        av_turning_angle.append(turning_angle)
        av_turning_angle_velocity.append(turning_angle_velocity)

    return av_turning_angle, av_turning_angle_velocity


def calculate_path_length(signal_df, mac_address_df):
    macs = mac_address_df.mac_address.tolist()
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')

    mod = lambda vec: np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)

    av_path_length = []
    for mac in macs:
        mac_signals_df = signal_mac_group.get_group(mac)

        path_length = np.nan

        if len(mac_signals_df) > 1:
            columns_to_diff = ['x', 'y']
            coords = mac_signals_df[columns_to_diff].as_matrix()
            vectors = coords[1:] - coords[:-1]
            mac_path_lengths = mod(vectors)
            path_length = np.nanmean(mac_path_lengths)

        av_path_length.append(path_length)

    return av_path_length


def add_probability_columns(mac_address_df):
    mac_address_df['shopper'] = 0.25
    mac_address_df['mall_worker'] = 0.25
    mac_address_df['stationary_device'] = 0.25
    mac_address_df['other'] = 0.25
    mac_address_df['resolved'] = False


def create_mac_address_features(mall, export_location):
    signal_df = utils.import_signals(mall)
    mac_address_df = create_mac_address_df(signal_df)
    mac_address_df['radius_of_gyration'] = calculate_radius_gyration(signal_df, mac_address_df)
    mac_address_df['manufacturer'] = find_device_type(mac_address_df)
    mac_address_df['count_density_variance'] = calculate_count_density_variance(signal_df, mac_address_df)
    mac_address_df['length_of_stay'] = calculate_length_of_stay(signal_df, mac_address_df)
    mac_address_df['is_out_of_hours'] = is_out_of_hours(signal_df, mac_address_df)
    mac_address_df['av_speed'] = calculate_average_speed(signal_df, mac_address_df)
    mac_address_df['av_turning_angle'], mac_address_df['av_turning_angle_velocity'] = \
        calculate_average_turning_angle(signal_df, mac_address_df)
    add_probability_columns(mac_address_df)
    mac_address_df['av_path_length'] = calculate_path_length(signal_df, mac_address_df)
    mac_address_df.to_csv(export_location, index=False)
