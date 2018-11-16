import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import msci.utils.plot as pfun
from msci.utils import utils


def store_id_unrealistic_v(df, ht):
    grouped = df.groupby('mac_address')
    groups = [grouped.get_group(i[0]) for i in ht]
    sid = []
    count = []
    for g in range(len(groups)):
        count.append(len(groups[g]))
        stores = groups[g].store_id.tolist()
        stores_v = []
        for i in range(len(stores)):
            if i in ht[g][1]:
                stores_v.append(stores[i-1])
                stores_v.append(stores[i])
                stores_v.append(stores[i+1])
        sid.append([ht[g][0], stores_v])
    nancount = [i[1].count(np.nan)/len(i[1]) for i in sid]
    return sid, nancount, count


def plot_nan(df):
    grouped = df.groupby('mac_address')
    macs = df.mac_address.drop_duplicates().tolist()
    groups = [grouped.get_group(i) for i in macs]
    xs = []
    ys = []
    for g in groups:
        x = g.x.tolist()
        y = g.y.tolist()
        stores = g.store_id.tolist()
        x = [x[i] for i in range(len(x)) if stores[i] is np.nan]
        y = [y[i] for i in range(len(y)) if stores[i] is np.nan]
        xs.append(x)
        ys.append(y)
    return xs, ys


def duplicate_by_manufacturer(df, duplicate_macs):
    df = df[df.mac_address.isin(duplicate_macs)]
    grouped = df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in duplicate_macs]
    print('d')
    manufacturers = [i.manufacturer.drop_duplicates().tolist() for i in groups]
    return groups, manufacturers


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


def length_of_stay(length, dev_type, plot=False):
    """
    likelihood function for the length of stay of a device

    :param length: (int) length of stay in seconds
    :param dev_type: (string) device type to be tested e.g. stationary
    :param plot: (boolean) plot or no plot
    :return: (float) likelihood
    """
    type_data = {'stationary': [23*60*60, 1*60*60], 'shopper': [2*60*60, 8*60*60], 'worker': []}
    mu = type_data[dev_type][0]
    sigma = type_data[dev_type][1]
    if plot:
        plot_dist(mu, sigma, 'Length of Stay (s)')
    return np.exp(-(length - mu)**2/(2*sigma**2))/np.sqrt(2*math.pi*sigma)


def radius_likelihood(r_g, dev_type, plot=False):
    """
    likelihood function for the length of stay of a device

    :param r_g: (float) radius of gyration for path
    :param dev_type: (string) device type to be tested e.g. stationary
    :param plot: (boolean) plot or no plot
    :return: (float) likelihood
    """
    type_data = {'stationary': [4, 8], 'shopper': [40, 20], 'worker': []}
    mu = type_data[dev_type][0]
    sigma = type_data[dev_type][1]
    if plot:
        plot_dist(mu, sigma, 'Radius of Gyration')
    return np.exp(-(r_g - mu) ** 2 /(2*sigma**2))/np.sqrt(2*math.pi*sigma)


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