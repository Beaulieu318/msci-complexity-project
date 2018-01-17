import msci.utils.plot as pfun
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from msci.utils import utils
from msci.utils import log_bin


def path_length(signal_df):
    macs = signal_df.mac_address.drop_duplicates().tolist()
    grouped = signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in macs]
    path_length_distribution = []
    time_diff = []
    for group in groups:
        times = group.date_time.tolist()
        x = group.x.tolist()
        y = group.y.tolist()
        pos = list(zip(x, y))
        time_deltas = [utils.time_difference(times[i], times[i+1]) for i in range(len(times) - 1)]
        path_lengths = [utils.euclidean_distance(pos[i], pos[i+1]) for i in range(len(pos) - 1)]
        norm_path_lengths = [10*(i/j) for (i, j) in list(zip(path_lengths, time_deltas))]
        path_length_distribution.append(norm_path_lengths)
        time_diff.append(time_deltas)
    flat_pl = [i for j in path_length_distribution for i in j]
    return flat_pl


def diffusive(signal_df):
    macs = signal_df.mac_address.drop_duplicates().tolist()[300:320]
    grouped = signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in macs]
    all_times = []
    all_rms = []
    for group in groups:
        x = group.x.tolist()
        y = group.y.tolist()
        pos = list(zip(x, y))
        times = group.date_time.tolist()
        time_deltas = [utils.time_difference(times[0], times[i]) for i in range(len(times))]
        origin = pos[0]
        r2 = [(utils.euclidean_distance(origin, i))**2 for i in pos]
        mu_r2 = [np.mean(r2[0:i]) for i in range(len(r2))]
        print(len(time_deltas), len(mu_r2))
        all_times.append(time_deltas)
        all_rms.append(mu_r2)
    flat_times = [i for j in all_times for i in j]
    flat_rms = [i for j in all_rms for i in j]
    return flat_times, flat_rms


def duration_distribution(signal_df):
    macs = signal_df.mac_address.drop_duplicates().tolist()
    grouped = signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in macs]
    durations = []
    for group in groups:
        t0 = group.date_time.tolist()[0]
        t1 = group.date_time.tolist()[-1]
        duration = utils.time_difference(t0, t1)
        durations.append(duration)
    return durations


def whole_path_dist(signal_df):
    macs = signal_df.mac_address.drop_duplicates().tolist()
    grouped = signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in macs]
    total_path_lengths = []
    for group in groups:
        x = group.x.tolist()
        y = group.y.tolist()
        pos = list(zip(x, y))
        path_lengths = [utils.euclidean_distance(pos[i], pos[i + 1]) for i in range(len(pos) - 1)]
        total_length = np.sum(path_lengths)
        total_path_lengths.append(total_length)
    return total_path_lengths


def plot_dist(distribution, metric='Distance'):
    malls = list(distribution.keys())
    fig = plt.figure()
    for mall in malls:
        lb_centers, lb_counts = bin(distribution[mall])
        hist, bins = np.histogram(distribution[mall], bins=2000, normed=True)
        center = (bins[:-1] + bins[1:]) / 2
        log_center = np.log(center)
        log_hist = np.log(hist)
        log_lbx = np.log(lb_centers)
        log_lby = np.log(lb_counts)
        plt.scatter(log_center, log_hist, s=0.5, label=mall + ' raw')
        plt.plot(log_lbx, log_lby, label=mall + ' log bin')
    plt.xlabel(metric + ' (log)')
    plt.ylabel('PDF (log)')
    plt.legend()
    fig.show()


def all_mall_dists(func):
    dists = {}
    for mall in ['Mall of Mauritius', 'Phoenix Mall', 'Home & Leisure']:
        signal_df = utils.import_signals(mall=mall, signal_type=1)
        dist = func(signal_df)
        print(len(dist))
        dists[mall] = dist
    return dists


def bin(data, bin_start=1., first_bin_width=1.4, a=1.6, drop_zeros=True):
    return log_bin.log_bin(data, bin_start, first_bin_width, a, drop_zeros=drop_zeros)
