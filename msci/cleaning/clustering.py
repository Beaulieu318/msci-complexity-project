import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import scale
from msci.utils import utils
import msci.utils.plot as pfun
import time

mac_address_df = pd.read_csv('../data/mac_address_features.csv')

mac_address_clean_df = mac_address_df.dropna()

samples = mac_address_clean_df.as_matrix(columns=['gyration', 'cdv', 'length_of_stay'])
samples_scaled = scale(samples)

manufacturers_df = {
    'HUAWEI TECHNOLOGIES CO.,LTD': False,
     'Xiaomi Communications Co Ltd': False,
     'Samsung Electronics Co.,Ltd': False,
     'Ruckus Wireless': True,
     'SAMSUNG ELECTRO-MECHANICS(THAILAND)': False,
     'HTC Corporation': False,
     'LG Electronics (Mobile Communications)': False,
     'Intel Corporate': True,
     'Hon Hai Precision Ind. Co.,Ltd.': True,
     'Murata Manufacturing Co., Ltd.': True,
     'TCT mobile ltd': False,
     'Apple, Inc.': False,
     'Motorola (Wuhan) Mobility Technologies Communication Co., Ltd.': False,
     'Ubiquiti Networks Inc.': True,
     'Sony Mobile Communications AB': False,
     'InPro Comm': True,
     'SAMSUNG ELECTRO MECHANICS CO., LTD.': False,
     'Microsoft Corporation': False,
     'Nokia Corporation': False,
     'Lenovo Mobile Communication Technology Ltd.': False,
     'BlackBerry RTS': False,
     'WISOL': False,
     'Motorola Mobility LLC, a Lenovo Company': False,
     'Microsoft Mobile Oy': False
    }


def df_to_matrix(df, columns=['gyration', 'cdv', 'length_of_stay', 'cluster']):
    return df.as_matrix(columns=columns)


def k_means(sample_data):
    model = KMeans(n_clusters=3)
    model.fit(sample_data)
    labels = model.predict(sample_data)
    mac_address_clean_df['cluster'] = labels
    return mac_address_clean_df


def separate_clusters(df):
    number_of_clusters = len(df.cluster.drop_duplicates().tolist())
    clusters = []
    for cluster in range(number_of_clusters):
        clusters.append(df[df['cluster'] == cluster])
    return clusters


def feature_statistics(df, feature):
    """
    calculates mean, standard deviation of data features
    :param df: clustering data frame
    :param feature: (str) name of feature you want to analyse e.g. gyration, cdv, length_of_stay
    :return: means, standard deviations of all feature data in each cluster
    """
    number_of_clusters = len(df.cluster.drop_duplicates().tolist())
    means = [df[df['cluster'] == i][feature].mean() for i in range(number_of_clusters)]
    std = [df[df['cluster'] == i][feature].std() for i in range(number_of_clusters)]
    return means, std


def statistics_by_manufacturer(df, feature, plot=True):
    """
    calculates statistics for different manufacturers
    :param df: data frame
    :param feature: (str) name of feature you want to analyse e.g. gyration, cdv, length_of_stay
    :return: lists of manufacturer, mean statistic, and standard deviation statistic
    """
    manufacturers = df.manufacturer.drop_duplicates().tolist()
    grouped = df.groupby('manufacturer')
    man_lengths = [len(grouped.get_group(i)) for i in manufacturers]
    manufacturers = [manufacturers[i] for i in range(len(manufacturers)) if man_lengths[i] >= 70]
    means = [grouped.get_group(i)[feature].mean() for i in manufacturers]
    std = [grouped.get_group(i)[feature].std() for i in manufacturers]
    if plot:
        fig = plt.figure()
        hist, bins = np.histogram(means, bins=len(means))
        center = (bins[:-1] + bins[1:]) / 2
        mask1 = center < 30
        mask2 = center >= 30
        plt.bar(center[mask1], hist[mask1], color='red', label='Stationary')
        plt.bar(center[mask2], hist[mask2], color='blue', label='Moving')
        plt.xlabel(feature)
        plt.ylabel('Un-normalised Probability')
        plt.legend(loc=2)
        fig.show()
    return manufacturers, means, std


def plot_3d(sep_clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r']
    for dim in range(len(sep_clusters)):
        matrix = df_to_matrix(sep_clusters[dim])
        x = np.array([i[0] for i in matrix])
        y = np.array([i[1] for i in matrix])
        z = np.array([i[2] for i in matrix])
        print(colors[dim])
        ax.scatter(x, y, z, zdir='z', s=20, color=colors[dim])
    ax.set_xlabel('Radius of Gyration')
    ax.set_ylabel('Count Density Variance')
    ax.set_zlabel('Length of Stay')
    fig.show()


def identify_duplicate_data(df):
    df['manufacturer'] = utils.add_device_type_signal(df)
    macs = df.mac_address.drop_duplicates().tolist()
    df = df.sort_values('date_time')
    grouped = df.groupby('mac_address')
    time_duplicate_boolean = []
    duplicates = []
    for mac in macs:
        print(mac)
        group = grouped.get_group(mac)
        times = group.date_time.tolist()
        time_dup = [times[i] == times[i+1] for i in range(len(times) - 1)]
        indices = [i for i, x in enumerate(time_dup) if x]
        dup = time_dup.count(True) >= 1
        if dup:
            x = group.x.tolist()
            y = group.y.tolist()
            pos_diff = [i for i in indices if (x[i], y[i]) != (x[i+1], y[i+1])]
            duplicates.append([mac, pos_diff])
        time_duplicate_boolean.append(dup)
    duplicate_macs = [macs[i] for i in range(len(macs)) if time_duplicate_boolean[i]]
    duplicates = [i for i in duplicates if len(i[1]) > 0]
    return df, duplicate_macs, duplicates


def analyse_duplicates(df, duplicate_mac, duplicate_indices, plot=True):
    grouped = df.groupby('mac_address')
    group = grouped.get_group(duplicate_mac)
    times = group.date_time.tolist()
    pos = list(zip(group.x.tolist(), group.y.tolist()))
    info = list(zip(times, pos))
    dup = [(info[i], info[i+1]) for i in duplicate_indices]
    x = [[i[1][0] for i in j] for j in dup]
    y = [[i[1][1] for i in j] for j in dup]
    if plot:
        pfun.plot_points_on_map(x, y)
    return dup, x, y


def position_difference_analysis(df, duplicates):
    t0 = time.time()
    dups = [analyse_duplicates(df, i[0], i[1], plot=False)[0] for i in duplicates[:20]]
    print(time.time() - t0)
    distance_dist = []
    i = 0
    for dup in dups:
        print(i)
        distances = [utils.euclidean_distance(i[0][1], i[1][1]) for i in dup]
        distance_dist.append(distances)
        i += 1
    return distance_dist


def identify_data_gaps(df, manufacturer, plot=True):
    df = df.sort_values('date_time')
    manufacturer_group = df.groupby('manufacturer').get_group(manufacturer)
    macs = manufacturer_group.mac_address.drop_duplicates().tolist()
    grouped = df.groupby(['manufacturer', 'mac_address'])
    time_interval_data = []
    for mac in macs:
        group = grouped.get_group((manufacturer, mac))
        times = group.date_time.tolist()
        t_deltas = [utils.time_difference(times[i], times[i+1]) for i in range(len(times) - 1)]
        time_interval_data.append(t_deltas)
    if plot:
        fig = plt.figure()
        plt.plot([0, 50], [1800, 1800], linestyle='dashed')
        for mac in time_interval_data[:20]:
            plt.plot(np.arange(len(mac)), mac)
        fig.show()
    return time_interval_data


def get_manufacturers_group(df, manufacturer):
    grouped = df.groupby('manufacturer')
    return grouped.get_group(manufacturer)


def gyration_manufacturer(df, data_type=None, specific_manufacturer='Samsung Electronics Co.,Ltd'):
    manufacturers = df.manufacturer.drop_duplicates().tolist()
    grouped = df.groupby('manufacturer')
    man_lengths = [len(grouped.get_group(i)) for i in manufacturers]
    manufacturers = [manufacturers[i] for i in range(len(manufacturers)) if man_lengths[i] >= 70]
    stationary_manufacturers = [i for i in manufacturers if manufacturers_df[i]]
    moving_manufacturers = [i for i in manufacturers if manufacturers_df[i] is False]

    if data_type is None:
        manufacturers = stationary_manufacturers
        fig = plt.figure()

    if data_type is 'stationary':
        manufacturers = stationary_manufacturers
        fig = plt.figure()

    if data_type is 'moving':
        manufacturers = moving_manufacturers
        fig = plt.figure()

    if data_type is 'specific':
        manufacturers = [specific_manufacturer]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

    colors = cm.rainbow(np.linspace(0, 1, len(manufacturers)))

    for i in range(len(manufacturers)):
        group = grouped.get_group(manufacturers[i])
        gyrations = group.gyration.tolist()
        if data_type is 'specific':
            plt.suptitle('Analysis of Radius of Gyration for ' + specific_manufacturer)
            axes[0].hist(gyrations, bins=len(gyrations), normed=True, color=colors[i], alpha=0.5, label=manufacturers[i], histtype='step')
            axes[0].set_xlabel('Radius of Gyration')
            axes[0].set_ylabel('Probability')
            lengths = group['count'].tolist()
            axes[1].scatter(gyrations, lengths)
            axes[1].set_xlabel('Radius of Gyration')
            axes[1].set_ylabel('Counts')
        else:
            plt.hist(gyrations, bins=50, normed=True, color=colors[i], alpha=0.5, label=manufacturers[i], histtype='step')
            plt.xlabel('Radius of Gyration')
            plt.ylabel('Probability')
    plt.legend(loc=1)
    fig.show()


def bar_chart(manufacturers, data, feature):
    n = len(manufacturers)

    ind = np.arange(n)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(ind, data, width, color='r')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Manufacturer')
    ax.set_title(feature + 'Statistics for Different Manufacturers')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(manufacturers, rotation='vertical')

    fig.show()


