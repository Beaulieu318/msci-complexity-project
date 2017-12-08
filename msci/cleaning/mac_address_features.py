import os
import pandas as pd
import numpy as np

from msci.utils import utils

dir_path = os.path.dirname(os.path.realpath(__file__))


def create_mac_address_df(signal_df):
    """
    Creates the mac_address_df (or feature_df) dataframe which has each unique mac address (device)

    :param signal_df: (pd.DataFrame) Contains the signals dataframe
    :return: (pd.DataFrame) The unique mac addresses with initial features
    """
    mac_addresses = signal_df.mac_address.value_counts()
    mac_address_df = pd.DataFrame(mac_addresses)
    mac_address_df.rename(columns={'mac_address': 'frequency'}, inplace=True)
    mac_address_df['mac_address'] = mac_address_df.index
    mac_address_df.reset_index(inplace=True, drop=True)
    return mac_address_df


def calculate_radius_gyration(signal_df, mac_address_df):
    """
    Calculates the radius of gyration which is a measure of how far the mac address moves from their central position

    :param signal_df: (pd.DataFrame) The signals
    :param mac_address_df: (pd.DataFrame) The mac addresses
    :return: (list) The radius of gyration for each of the mac addresses
    """
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
    """
    Finds the manufacturer for each of the mac addresses

    :param mac_address_df: (pd.DataFrame) The mac addresses
    :return: (list) the manufacturer for the mac addresses
    """
    mac_cross_reference_df = pd.read_csv(dir_path + '/../data/mac_address_cross_reference.csv')
    mac_address_df2 = mac_address_df.copy()
    mac_address_df2['mac_address_short'] = mac_address_df2.mac_address.str.replace(':', '').str.upper().str[:6]
    mac_address_df2 = mac_address_df2.merge(
        mac_cross_reference_df,
        how='left', left_on='mac_address_short',
        right_on='Assignment'
    )
    return mac_address_df2['Organization Name']


def calculate_count_density_variance(signal_df, mac_address_df, minute_resolution=5):
    """
    Bins times into intervals of length 'minute resolution', and finds variance of counts

    :param signal_df: (pd.DataFrame) The signals
    :param mac_address_df: (pd.DataFrame) The mac addresses
    :param minute_resolution: (str) number of minutes into which to group data
    :return: (list) count variances for each mac address
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
    """
    Calculates the total length of stay in seconds for each of the mac addresses

    :param signal_df: (pd.DataFrame) The signals
    :param mac_address_df: (pd.DataFrame) The mac addresses
    :return: (list) the length of stay for each of the mac addresses
    """
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')
    macs = mac_address_df.mac_address.tolist()
    groups = [signal_mac_group.get_group(i).date_time.tolist() for i in macs]
    time_deltas = [utils.time_difference(i[0], i[-1]) for i in groups]
    return time_deltas


def is_out_of_hours(signal_df, mac_address_df):
    """
    Returns a boolean if the mac address is definitely out of hours (present from 3am to 5am)

    :param signal_df: (pd.DataFrame) The signals
    :param mac_address_df: (pd.DataFrame) The mac addresses
    :return: (list) A boolean if each mac address is out of hours
    """
    macs = mac_address_df.mac_address.tolist()
    date_index_df = signal_df.copy()
    date_index_df.index = date_index_df.date_time.astype('datetime64[ns]')
    signal_out_of_hours = date_index_df.between_time('03:00:00', '05:00:00')
    mac_address_out_of_hours = signal_out_of_hours.mac_address.drop_duplicates().tolist()
    macs_is_out_of_hours = [1 if mac in mac_address_out_of_hours else 0 for mac in macs]
    return macs_is_out_of_hours


def calculate_average_speed(signal_df, mac_address_df):
    """
    Calculates the average speed for each mac address

    :param signal_df: (pd.DataFrame) The signals
    :param mac_address_df: (pd.DataFrame) The mac addresses
    :return: (list) The average speed of each mac address
    """
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


def calculate_turning_angle(signal_df, mac_address_df):
    """
    Calculate the average and total turning angle as well as the change in turning angle (turning angle velocity)

    :param signal_df: (pd.DataFrame) The signals
    :param mac_address_df: (pd.DataFrame) The mac addresses
    :return: (list list list) Three list containing information about the turning angle of each mac address
    """
    macs = mac_address_df.mac_address.tolist()
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')

    dot = lambda vec1, vec2: vec1[:, 0] * vec2[:, 0] + vec1[:, 1] * vec2[:, 1]
    mod = lambda vec: np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)

    av_turning_angles = []
    total_turning_angles = []
    av_turning_angle_velocities = []
    for mac in macs:
        mac_signals_df = signal_mac_group.get_group(mac)

        av_turning_angle = np.nan
        av_turning_angle_velocity = np.nan
        total_turning_angle = np.nan

        if len(mac_signals_df) > 2:
            columns_to_diff = ['x', 'y']
            coords = mac_signals_df[columns_to_diff].as_matrix()
            vectors = coords[1:] - coords[:-1]
            u = vectors[1:]
            v = vectors[:-1]
            cos_angle = dot(u, v) / (mod(u) * mod(v))
            mac_turning_angles = np.arccos(cos_angle)
            av_turning_angle = np.nanmean(mac_turning_angles)
            total_turning_angle = np.nansum(np.abs(mac_turning_angles))

            if len(mac_turning_angles) > 1:
                mac_turning_angle_velocities = np.array(mac_turning_angles)[1:] - np.array(mac_turning_angles[:-1])
                av_turning_angle_velocity = np.nanmean(mac_turning_angle_velocities)

        av_turning_angles.append(av_turning_angle)
        total_turning_angles.append(total_turning_angle)
        av_turning_angle_velocities.append(av_turning_angle_velocity)

    return av_turning_angles, total_turning_angles, av_turning_angle_velocities


def calculate_path_length(signal_df, mac_address_df):
    """
    Calculates the average and total path length for each mac address

    :param signal_df: (pd.DataFrame) The signals
    :param mac_address_df: (pd.DataFrame) The mac addresses
    :return: (list list) Two lists containing the average and total path length
    """
    macs = mac_address_df.mac_address.tolist()
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')

    mod = lambda vec: np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)

    av_path_lengths = []
    total_path_lengths = []
    for mac in macs:
        mac_signals_df = signal_mac_group.get_group(mac)

        av_path_length = np.nan
        total_path_length = np.nan

        if len(mac_signals_df) > 1:
            columns_to_diff = ['x', 'y']
            coords = mac_signals_df[columns_to_diff].as_matrix()
            vectors = coords[1:] - coords[:-1]
            mac_path_lengths = mod(vectors)
            av_path_length = np.nanmean(mac_path_lengths)
            total_path_length = np.nansum(mac_path_lengths)

        av_path_lengths.append(av_path_length)
        total_path_lengths.append(total_path_length)

    return av_path_lengths, total_path_lengths


def calculate_straightness(signal_df, mac_address_df):
    """
    Calculate the straightness (add two paths together and divide by the total displacement) of each mac address

    :param signal_df: (pd.DataFrame) The signals
    :param mac_address_df: (pd.DataFrame) The mac addresses
    :return: (list) The straightness of each mac address
    """
    macs = mac_address_df.mac_address.tolist()
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')

    mod = lambda vec: np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)

    av_straightnesses = []
    for mac in macs:
        mac_signals_df = signal_mac_group.get_group(mac)

        av_straightness = np.nan

        if len(mac_signals_df) > 2:
            columns_to_diff = ['x', 'y']
            coords = mac_signals_df[columns_to_diff].as_matrix()
            vectors = coords[1:] - coords[:-1]
            u = vectors[1:]
            v = vectors[:-1]
            t = coords[2:] - coords[:-2]
            mac_path_lengths = (mod(u) + mod(v)) / mod(t)
            mac_path_lengths = mac_path_lengths[np.isfinite(mac_path_lengths)]
            av_straightness = np.nanmean(mac_path_lengths)

        av_straightnesses.append(av_straightness)

    return av_straightnesses


def add_wifi_type(signal_df, mac_address_df):
    """
    Add the wifi type (a feature in the signal_df dataframe) which shows where a user is thought to be a shopper
    N.B. This changes the mac_address_df entered as a parameter

    :param signal_df: (pd.DataFrame) The signals
    :param mac_address_df: (pd.DataFrame) The mac addresses
    """
    mac_address_df['wifi_type'] = np.nan

    wifi_types = ['wifiuser', 'lawifiuser', 'Discovered-AP', 'fatti Bot ', 'unknown']
    for wifi_type in wifi_types:
        signal_wifi_type = signal_df[signal_df.wifi_type == wifi_type].mac_address.tolist()
        mac_address_df.loc[
            mac_address_df.wifi_type.isnull() &
            (mac_address_df.mac_address.isin(signal_wifi_type)), 'wifi_type'] = wifi_type


def find_start_end_coordinate(signal_df, mac_address_df):
    """
    Finds the first and last coordinates of the mac address path

    :param signal_df: (pd.DataFrame) The signals
    :param mac_address_df: (pd.DataFrame) The mac addresses
    :return: (list list) Two list containing a list of x, y coordinates of the start and end positions
    """
    macs = mac_address_df.mac_address.tolist()
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')
    start_coordinates = []
    end_coordinates = []
    for mac in macs:
        mac_signals_df = signal_mac_group.get_group(mac)
        start_coordinates.append([mac_signals_df.x.iloc[0], mac_signals_df.y.iloc[0]])
        end_coordinates.append([mac_signals_df.x.iloc[-1], mac_signals_df.y.iloc[-1]])
    return start_coordinates, end_coordinates


def create_mac_address_features(mall='Mall of Mauritius', export_location=None):
    """
    Creates the mac address features dataframe which is used to determine whether the device is a shopper

    :param mall: (string) The name of the mall
    :param export_location: (string or None) The location of where this is exported
    :return: (pd.DataFrame) If export location is None, returns the dataframe created
    """
    signal_df = utils.import_signals(mall)
    mac_address_df = create_mac_address_df(signal_df)
    mac_address_df['centroid'], \
        mac_address_df['radius_of_gyration'] = \
        calculate_radius_gyration(signal_df, mac_address_df)
    mac_address_df['manufacturer'] = find_device_type(mac_address_df)
    mac_address_df['count_density_variance'] = calculate_count_density_variance(signal_df, mac_address_df)
    mac_address_df['length_of_stay'] = calculate_length_of_stay(signal_df, mac_address_df)
    mac_address_df['is_out_of_hours'] = is_out_of_hours(signal_df, mac_address_df)
    mac_address_df['av_speed'] = calculate_average_speed(signal_df, mac_address_df)
    mac_address_df['av_turning_angle'], \
        mac_address_df['total_turning_angle'], \
        mac_address_df['av_turning_angle_velocity'] = \
        calculate_turning_angle(signal_df, mac_address_df)
    mac_address_df['av_path_length'], \
        mac_address_df['total_path_length'] = \
        calculate_path_length(signal_df, mac_address_df)
    mac_address_df['av_straightness'] = calculate_straightness(signal_df, mac_address_df)
    mac_address_df['av_speed_from_total'] = mac_address_df['total_path_length'] / mac_address_df['length_of_stay']
    mac_address_df['turning_angle_density'] = \
        mac_address_df['total_turning_angle'] / mac_address_df['total_path_length']
    mac_address_df['start_coordinate'], \
        mac_address_df['end_coordinate'] = \
        find_start_end_coordinate(signal_df, mac_address_df)
    add_wifi_type(signal_df, mac_address_df)
    if export_location:
        mac_address_df.to_csv(export_location, index=False)
    else:
        return mac_address_df
