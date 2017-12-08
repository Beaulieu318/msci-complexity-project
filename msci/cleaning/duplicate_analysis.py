import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import pandas as pd
import msci.utils.plot as pfun
from msci.utils import utils
import time
import os
from collections import defaultdict

dir_path = os.path.dirname(os.path.realpath(__file__))

mac_address_df = pd.read_csv(dir_path + '/../data/mac_address_features.csv')
mac_address_clean_df = mac_address_df.dropna()

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


def period_analysis(signal_df, macs):
    grouped = signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in macs]
    all_periods = []
    all_times = []
    for group in groups:
        times = group.date_time.tolist()
        periods = [utils.time_difference(times[j], times[j+1]) for j in range(len(group) - 1)]
        all_times.append(times)
        all_periods.append(periods)
    return all_periods, all_times


def out_of_hour_periods(p_analysis):
    period_splits = []
    for i in range(len(p_analysis[1])):
        before_hours = [
            p_analysis[1][i].index(t) for t in p_analysis[1][i] if t < pd.tslib.Timestamp('2016-12-22 08:00:00')
        ]
        after_hours = [
            p_analysis[1][i].index(t) for t in p_analysis[1][i] if t > pd.tslib.Timestamp('2016-12-22 22:00:00')
        ]
        out_indices = before_hours + after_hours
        out_periods = [p_analysis[0][i][j] for j in range(len(p_analysis[0][i])) if j in out_indices]
        in_periods = [p_analysis[0][i][j] for j in range(len(p_analysis[0][i])) if j not in out_indices]
        period_splits.append([out_periods, in_periods])
    return period_splits


def period_plot(p_analysis, feature_df):
    all_periods = p_analysis[0]
    all_times = p_analysis[1]
    feature_df = feature_df[feature_df.frequency > 10]
    macs = feature_df.mac_address.tolist()
    fig = plt.figure()
    for i in range(len(macs[:3])):
        if feature_df.iloc[i].is_out_of_hours == 1:
            plt.plot(all_times[i][1:], all_periods[i], color='r')
        else:
            plt.plot(all_times[i][1:], all_periods[i], color='b')
    plt.xlabel('Time')
    plt.ylabel('Pinging Period')
    fig.tight_layout()
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


def analyse_duplicates(grouped_df, duplicate_mac, duplicate_indices, plot=True):
    group = grouped_df.get_group(duplicate_mac)
    times = group.date_time.tolist()
    pos = list(zip(group.x.tolist(), group.y.tolist()))
    info = list(zip(times, pos))
    dup = [(info[i], info[i+1]) for i in duplicate_indices]
    x = [[i[1][0] for i in j] for j in dup]
    y = [[i[1][1] for i in j] for j in dup]
    if plot:
        pfun.plot_points_on_map(x, y)
    return dup, x, y


def duplicate_point_continuity(df, duplicate_mac, duplicate_indices):
    ind = duplicate_indices
    grouped = df.groupby('mac_address')
    group = grouped.get_group(duplicate_mac)
    times = group.date_time.tolist()
    time_cont = [times[i-1:i+3] for i in ind]
    pos = list(zip(group.x.tolist(), group.y.tolist()))
    pos_cont = [pos[i-1:i+3] for i in ind]
    vel_cont = []
    for i in range(len(ind)):
        euc = np.array([utils.euclidean_distance(pos_cont[i][j], pos_cont[i][j+1]) for j in range(len(pos_cont[i]) - 1)])
        time_diff = np.array([utils.time_difference(time_cont[i][j], time_cont[i][j+1]) for j in range(len(time_cont[i]) - 1)])
        vel_cont.append(euc/time_diff)
    group_euc = np.array([utils.euclidean_distance(pos[i], pos[i+1]) for i in range(len(pos) - 1)])
    group_td = np.array([utils.time_difference(times[i], times[i+1]) for i in range(len(times) - 1)])
    group_velocity = group_euc/group_td
    group_velocity = [group_velocity[i] for i in range(len(group_velocity)) if i not in ind]
    group_velocity = np.trim_zeros(np.array([i for i in group_velocity if np.isnan(i) == False]))
    return time_cont, pos_cont, vel_cont, group_velocity


def position_difference_analysis(df, duplicates, distances=False):
    grouped = df.grouby('mac_address')
    t0 = time.time()
    dups = [analyse_duplicates(grouped, i[0], i[1], plot=False)[0] for i in duplicates[:200]]
    print(time.time() - t0)
    distance_dist = []
    if distances:
        for dup in dups:
            distances = [utils.euclidean_distance(i[0][1], i[1][1]) for i in dup]
            distance_dist.append(distances)
        return distance_dist
    else:
        return dups


def duplicate_fill(df, largest_separation):
    """
    function to account for data points that have identical times but different positions
    if discrepancy in position is greater than 'largest_separation', the position that minimises the deviation from
    path is kept. if discrepancy is less the 'largest_separation', the average of the positions is used.
    :param df: (pd.DataFrame) dirty data frame
    :param largest_separation: (int) largest distance threshold for analysis
    :return: (pd.DataFrame) the original DataFrame without two location at the same time
    """
    all_macs = df.mac_address.drop_duplicates().tolist()
    print(len(all_macs))
    grouped = df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in all_macs]
    new_data = []
    for g in range(len(groups)):
        times = groups[g].date_time.tolist()
        d = defaultdict(list)
        for i, item in enumerate(times):
            d[item].append(i)
        d = {k: v for k, v in d.items() if len(v) > 1}
        if len(d) > 0:
            x = groups[g].x.tolist()
            y = groups[g].y.tolist()
            pos = list(zip(x, y))
            for entry in d:
                pos_dups = [pos[i] for i in d[entry]]
                eucs = [utils.euclidean_distance(pos_dups[i], pos_dups[i+1]) for i in range(len(pos_dups) - 1)]
                euc_greater = [i for i in eucs if i > largest_separation]
                if len(euc_greater) > 0:
                    if len(pos) > d[entry][-1] + 2:
                        dists = [np.mean([utils.euclidean_distance(pos[d[entry][0]-1], j),
                                          utils.euclidean_distance(pos[d[entry][-1]+2], j)]) for j in pos_dups]
                        new_pos = pos_dups[np.argmin(dists)]
                    else:
                        dists = [utils.euclidean_distance(pos[d[entry][0] - 1], j) for j in pos_dups]
                        new_pos = pos_dups[np.argmin(dists)]
                else:
                    pos_dups_zip = list(zip(*pos_dups))
                    new_pos = (np.mean(pos_dups_zip[0]), np.mean(pos_dups_zip[1]))
                d[entry] = new_pos
        mac = all_macs[g]
        for i in d:
            new_data.append([mac, i, d[i][0], d[i][1]])

    new_df = pd.DataFrame(new_data, columns=['mac_address', 'date_time', 'x_new', 'y_new'])
    merged_df = df.merge(new_df, how='left', on=['mac_address', 'date_time'])

    new_coordinate_mask = merged_df.x_new.notnull()
    merged_df.loc[new_coordinate_mask, 'x'] = merged_df.loc[new_coordinate_mask, 'x_new']
    merged_df.loc[new_coordinate_mask, 'y'] = merged_df.loc[new_coordinate_mask, 'y_new']
    unique_columns = ['mac_address', 'date_time', 'location', 'x', 'y']
    merged_df.drop_duplicates(subset=unique_columns, inplace=True)
    merged_df.reset_index(inplace=True)
    del merged_df['x_new']
    del merged_df['y_new']
    return merged_df


def higher_order_duplicate_fill(df, velocity_limit, triangulation_error):
    all_macs = df.mac_address.drop_duplicates().tolist()
    print(len(all_macs))
    grouped = df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in all_macs]
    macs = [all_macs[i] for i in range(len(all_macs)) if len(groups[i]) > 1]
    groups = [g for g in groups if len(g) > 1]
    print(len(groups))
    time_culprits = []
    time_culprits_error = []
    for g in range(len(groups)):
        ts = []
        ts_error = []
        x = groups[g].x.tolist()
        y = groups[g].y.tolist()
        pos = list(zip(x, y))
        times = groups[g].date_time.tolist()
        delta_t = np.array([utils.time_difference(times[i], times[i+1]) for i in range(len(times)-1)])
        euclideans = np.array([utils.euclidean_distance(pos[i], pos[i + 1]) for i in range(len(pos) - 1)])
        euclideans_error = euclideans - 2*triangulation_error*np.ones(len(euclideans))
        velocities = euclideans/delta_t
        velocities_error = euclideans_error/delta_t
        for v in range(len(velocities)):
            if velocities[v] > velocity_limit:
                ts.append((delta_t[v], times[v], times[v+1], pos[v], pos[v+1]))
        time_culprits.append(ts)
        for v in range(len(velocities)):
            if velocities_error[v] > velocity_limit:
                ts_error.append(v)
        if len(ts_error) > 1:
            time_culprits_error.append([macs[g], ts_error, len(groups[g]), groups[g].mac_address.unique()])
    return time_culprits_error


def plot_unrealistic_speeds(df, ht):
    grouped = df.groupby('mac_address')
    macs = [i[0] for i in ht]
    groups = [grouped.get_group(i) for i in macs]
    groups = [g for g in groups if len(g) > 1]
    xs = []
    ys = []
    for g in range(len(groups)):
        print(g)
        x = groups[g].x.tolist()
        y = groups[g].y.tolist()
        x = [x[i] for i in range(len(x)) if i in ht[0][g][1]]
        y = [y[i] for i in range(len(y)) if i in ht[0][g][1]]
        xs.append(x)
        ys.append(y)
    pfun.plot_points_on_map(xs, ys)


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