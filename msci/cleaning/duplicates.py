import time
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict
from tqdm import tqdm_notebook as tqdm

from msci.utils import utils
import msci.utils.plot as pfun

dir_path = os.path.dirname(os.path.realpath(__file__))


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
    macs = df.mac_address.drop_duplicates().tolist()
    df = df.sort_values('date_time')
    grouped = df.groupby('mac_address')
    time_duplicate_boolean = []
    duplicates = []
    no_dup = 0
    for mac in macs:
        group = grouped.get_group(mac)
        times = group.date_time.tolist()
        time_dup = [times[i] == times[i+1] for i in range(len(times) - 1)]
        no_dup += time_dup.count(True)
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
    return df, duplicate_macs, duplicates, no_dup


def analyse_duplicates(grouped_df, duplicate_mac, duplicate_indices, plot=True):
    group = grouped_df.get_group(duplicate_mac)
    times = group.date_time.tolist()
    pos = list(zip(group.x.tolist(), group.y.tolist()))
    info = list(zip(times, pos))
    dup = [(info[i], info[i+1]) for i in duplicate_indices]
    x = [[i[1][0] for i in j] for j in dup]
    y = [[i[1][1] for i in j] for j in dup]
    if plot:
        pfun.plot_points_on_map(x, y, label=True)
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
    grouped = df.groupby('mac_address')
    t0 = time.time()
    dups = [analyse_duplicates(grouped, i[0], i[1], plot=False)[0] for i in duplicates[:5000]]
    print(time.time() - t0)
    distance_dist = []
    if distances:
        for dup in dups:
            distances = [utils.euclidean_distance(i[0][1], i[1][1]) for i in dup]
            distance_dist.append(distances)
        return distance_dist
    else:
        return dups


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


"""
Main functions to clean the signals from miscellaneous points
"""


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
    grouped = df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in all_macs]
    new_data = []
    for g in tqdm(range(len(groups)), desc='Duplicate Fill'):
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
    merged_df.reset_index(inplace=True, drop=True)
    del merged_df['x_new']
    del merged_df['y_new']
    return merged_df
