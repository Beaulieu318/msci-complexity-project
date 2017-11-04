import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import scale

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


def statistics_by_manufacturer(df, feature):
    manufacturers = df.manufacturer.drop_duplicates().tolist()
    grouped = df.groupby('manufacturer')
    man_lengths = [len(grouped.get_group(i)) for i in manufacturers]
    manufacturers = [manufacturers[i] for i in range(len(manufacturers)) if man_lengths[i] >= 70]
    means = [grouped.get_group(i)[feature].mean() for i in manufacturers]
    #std = [grouped.get_group(i)[feature].std() for i in manufacturers]
    return list(zip(means, manufacturers))
    #return means, std, manufacturers


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


def gyration_manufacturer(df, data_type=None, specific_manufacturer='Samsung Electronics Co.,Ltd'):
    manufacturers = df.manufacturer.drop_duplicates().tolist()
    grouped = df.groupby('manufacturer')
    man_lengths = [len(grouped.get_group(i)) for i in manufacturers]
    manufacturers = [manufacturers[i] for i in range(len(manufacturers)) if man_lengths[i] >= 70]
    stationary_manufacturers = [i for i in manufacturers if manufacturers_df[i]]
    moving_manufacturers = [i for i in manufacturers if manufacturers_df[i] is False]

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
