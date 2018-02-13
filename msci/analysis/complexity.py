import msci.utils.plot as pfun
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from msci.utils import utils
from msci.utils import log_bin
import time
from scipy.spatial.distance import directed_hausdorff
from itertools import product
from scipy import stats

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
    log_points = []
    for mall in malls:
        #lb_centers, lb_counts = bin(distribution[mall])
        hist, bins = np.histogram(distribution[mall], bins=2000, normed=True)
        center = (bins[:-1] + bins[1:]) / 2
        log_center = np.log10(center)
        log_hist = np.log10(hist)
        #log_lbx = np.log10(lb_centers)
        #log_lby = np.log10(lb_counts)
        plt.scatter(log_center, log_hist, s=0.5, label=mall + ' raw')
        #plt.plot(log_lbx, log_lby, label=mall + ' log bin')
        #log_points.append([log_lbx, log_lby])
    plt.xlabel(metric + ' (log)')
    plt.ylabel('PDF (log)')
    plt.legend()
    fig.show()
    return log_points


def all_mall_dists(func):
    dists = {}
    for mall in ['Mall of Mauritius', 'Phoenix Mall', 'Home & Leisure']:
        signal_df = utils.import_signals(mall=mall, signal_type=1)
        dist = func(signal_df)
        print(len(dist))
        dists[mall] = dist
    return dists


def discrete_difference(x, y):
    y_diff = np.array([y[i+1] - y[i] for i in range(len(y) -1)])
    x_diff = np.array([x[i+1] - x[i] for i in range(len(x) -1)])
    diff = y_diff/x_diff
    return diff


def bin(data, bin_start=1., first_bin_width=1.4, a=1.6, drop_zeros=True):
    return log_bin.log_bin(data, bin_start, first_bin_width, a, drop_zeros=drop_zeros)


def shop_areas(signal_df, store_only=True):
    signal_df = signal_df.dropna()
    store_ids = signal_df.store_id.drop_duplicates().tolist()
    if store_only:
        store_ids = [i for i in store_ids if i[0] == 'B']
    areas = {}
    dimensions = {}
    store_id_df = signal_df.groupby('store_id')
    store_data = [store_id_df.get_group(i) for i in store_ids]
    for store in range(len(store_data)):
        print(store)
        x = store_data[store].x.tolist()
        y = store_data[store].y.tolist()
        pos = list(zip(x, y))
        key = list(set(x))
        xy_dict = {k: [] for k in key}
        for (i, j) in pos:
            xy_dict[i].append(j)
        area = []
        dims = {}
        for k in key:
            y_max = np.amax(xy_dict[k])
            y_min = np.amin(xy_dict[k])
            area.append(y_max - y_min)
            dims[k] = [y_min, y_max]
        dimensions[store_ids[store]] = dims
        areas[store_ids[store]] = np.sum(area)
    return store_data, areas, dimensions


def efficient_area(signal_df, store_only=True):
    stores = signal_df.store_id.drop_duplicates().tolist()
    if store_only:
        stores = [i for i in stores if i[0] == 'B']
    grouped = signal_df.groupby('store_id')
    groups = [grouped.get_group(i) for i in stores]
    areas = {}
    plot_points = {}
    for g in range(len(groups)):
        area = 0
        ym = {}
        x = np.round(groups[g].x.tolist()).astype('int')
        y = np.round(groups[g].y.tolist())
        bins = np.nonzero(np.bincount(x))[0]
        for i in bins:
            ysub = np.array(y)[np.where(x == i)[0]]
            ymax = np.amax(ysub)
            ymin = np.amin(ysub)
            ym[i] = [ymin, ymax]
            l = ymax-ymin
            if l != 0:
                area += ymax-ymin
            else:
                area += 1
        areas[stores[g]] = area
        plot_points[stores[g]] = ym
    return groups, areas, plot_points


def plot_shop_area(dimensions, store_id):
    dim_dict = dimensions[store_id]
    x = list(dim_dict.keys())
    y_max = [dim_dict[i][1] for i in x]
    y_min = [dim_dict[i][0] for i in x]
    fig = plt.figure()
    plt.scatter(x, y_min, color='r')
    plt.scatter(x, y_max, color='b')
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y_min[i], y_max[i]], color='g')
    plt.style.use('seaborn')
    plt.xlabel('x')
    plt.ylabel('y')
    fig.show()
    return x + x, y_max + y_min


def visitors_area(signal_df, store_only=True):
    if store_only:
        store_id_df, areas = efficient_area(signal_df)[:2]
    else:
        store_id_df, areas = efficient_area(signal_df, store_only=False)[:2]
    areas = [areas[a] for a in list(areas)]
    visitor_numbers = [len(i.mac_address.tolist()) for i in store_id_df]
    log_vis = np.log10(visitor_numbers)
    log_ar = np.log10(areas)
    fig = plt.figure()
    plt.scatter(log_ar, log_vis, color='C1')
    plt.xlabel(r'Shop Area, $m^2$ (log)', fontsize=35)
    plt.ylabel('Number of Visitors (log)', fontsize=35)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ar, log_vis)
    x = np.linspace(0, 4, 5)
    y = [slope*i + intercept for i in x]
    plt.plot(x, y, color='C0', linestyle='dashed', label=r'$\tau=$' + str(round(slope, 3)))
    print(slope, intercept, r_value**2)
    #plt.legend()
    plt.xlim((0, 4))
    plt.style.use('ggplot')
    fig.show()
    return areas, visitor_numbers


def fraction_visited(signal_df):
    macs = signal_df.mac_address.drop_duplicates().tolist()
    stores = signal_df.store_id.drop_duplicates().tolist()
    grouped_df = signal_df.groupby('mac_address')
    groups = [grouped_df.get_group(i) for i in macs]
    stores_visited = []
    for group in groups:
        stores_v = group.store_id.tolist()
        stores_visited.append(len(list(set(stores_v)))/len(stores))
    return stores_visited


def haussdorf_distance_manual(path_a, path_b):
    """
    computes Haussdorf distance between two paths a and b.
    Note: h(a,b) not necessarily equal to h(b, a)
    :param path_a: (list of tuples) x,y coordinates of path a
    :param path_b: (list of tuples) x,y coordinates of path b
    :param manual: (Boolean) standard python library or own code
    :return: Haussdorf distance H(a,b)
    """
    if manual:
        h = 0
        for a in path_a:
            ab_distances = [utils.euclidean_distance(a, b) for b in path_b]
            min_ab = np.amin(ab_distances)
            if min_ab > h:
                h = min_ab
    else:
        h = directed_hausdorff(np.array(path_a), np.array(path_b))[0]
    return h


def haussdorf_distance(path_a, paths):
    """
    computes Haussdorf distance between path a and other paths, b.
    Note: h(a,b) not necessarily equal to h(b, a)
    :param path_a: (list of tuples) x,y coordinates of path a
    :param paths: (list of list of tuples) x,y coordinates of b paths
    :return: Haussdorf distance H(a,b)
    """
    hab = [directed_hausdorff(np.array(path_a), np.array(p))[0] for p in paths]
    hba = [directed_hausdorff(np.array(p), np.array(path_a))[0] for p in paths]
    greater = int(hab > hba)
    less = int(hba > hab)
    h = hab*greater + hba*less
    return h


def pairwise_haussdorf_fast(pos, plot=False, normal=True):
    t0 = time.time()
    H = []
    for i in range(len(pos)):
        H.append(haussdorf_distance(pos[i], pos))
    print(time.time() - t0)
    H = np.array(H).reshape(len(pos), len(pos))
    if normal:
        H = H/np.amax(H)
    if plot:
        plot_heat(H)
    return H


def plot_heat(H):
    fig = plt.figure()
    plt.imshow(H)
    plt.colorbar()
    fig.show()


def matrix_correlation(matrix_a, matrix_b):
    a_flat = matrix_a.flatten()
    b_flat = matrix_b.flatten()
    corr = np.corrcoef(a_flat, b_flat)
    return corr


def undirected_haussdorf(path_a, path_b, manual=False):
    hab = haussdorf_distance(path_a, path_b, manual=manual)
    hba = haussdorf_distance(path_b, path_a, manual=manual)
    return np.amax([hab, hba])


def position_dictionary(signal_df, list_type=True):
    signal_df = signal_df.sort_values('date_time')
    macs = signal_df.sort_values('mac_address').mac_address.drop_duplicates().tolist()
    grouped_df = signal_df.groupby('mac_address')
    groups = [grouped_df.get_group(i) for i in macs]
    if list_type:
        positions = [list(zip(groups[i].x.tolist(), groups[i].y.tolist())) for i in range(len(groups))]
        return positions
    else:
        positions = {macs[i]: list(zip(groups[i].x.tolist(), groups[i].y.tolist())) for i in range(len(groups))}
        return positions


# def pairwise_haussdorf(positions, macs):
#     ph = {}
#     i = 0
#     for mac1 in macs:
#         i += 1
#         j = 0
#         print('mac', i)
#         for mac2 in macs[i+1:]:
#             j += 1
#             print(j)
#             if mac1 != mac2:
#                 ph[(mac1, mac2)] = undirected_haussdorf(positions[mac1], positions[mac2])
#                 ph[(mac1, mac2)] = undirected_haussdorf(positions[mac1], positions[mac2])
#     return ph


def pairwise_haussdorf(positions, macs):
    ph = np.zeros((len(macs), len(macs)))
    pairwise_combinations = list(product(range(len(macs)), range(len(macs))))
    for (i, j) in pairwise_combinations:
        t0 = time.time()
        if j > i:
            #print(j)
            hd = undirected_haussdorf(positions[macs[i]], positions[macs[j]])
            ph[i][j] = hd
        print(time.time() - t0)
    return ph
