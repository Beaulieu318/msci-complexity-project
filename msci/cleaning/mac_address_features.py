import pandas as pd
import numpy as np

from msci.cleaning.utils import *

COLUMNS_TO_IMPORT = ['mac_address', 'date_time', 'location', 'store_id', 'x', 'y']


def import_data(mall='Phoenix Mall'):
    shopper_df = pd.read_csv('../data/bag_mus_12-22-2016.csv', usecols=COLUMNS_TO_IMPORT)
    shopper_df.date_time = shopper_df.date_time.astype('datetime64[ns]')
    signal_df = shopper_df[shopper_df['location'] == mall]
    return signal_df


def create_mac_address_df(signal_df):
    mac_addresses = signal_df.mac_address.value_counts()
    mac_address_df = pd.DataFrame(mac_addresses)
    mac_address_df.rename(columns={'mac_address': 'count'}, inplace=True)
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
    mac_cross_reference_df = pd.read_csv('../data/mac_address_cross_reference.csv')
    mac_address_df2 = mac_address_df.copy()
    mac_address_df2['mac_address_short'] = mac_address_df2.mac_address.str.replace(':', '').str.upper().str[:6]
    mac_address_df2 = mac_address_df2.merge(mac_cross_reference_df, how='left', left_on='mac_address_short', right_on='Assignment')
    return mac_address_df2['Organization Name']


def count_density_variance(signal_df, mac_address_df, minute_resolution):
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
        time = signal_mac_group.get_group(mac).date_time.dt.round(minute_resolution + 'min').value_counts()
        count_variance = time.std()
        cdv.append(count_variance)
    return cdv


def total_delta_t(df):
    sorted = df.sort_values('date_time')
    grouped = sorted.groupby('mac_address')
    macs = df.mac_address.drop_duplicates().tolist()
    groups = [grouped.get_group(i).date_time.tolist() for i in macs]
    counts = [len(i) for i in groups]
    time_deltas = [time_difference(i[0], i[-1]) for i in groups]
    return time_deltas, counts
