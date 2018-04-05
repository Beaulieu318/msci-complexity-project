import pandas as pd
import numpy as np

from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

import seaborn as sns

from msci.utils import utils
import msci.utils.plot as pfun
from msci.analysis.networks import *
from msci.utils.plot import create_count_of_shoppers_gif, create_count_of_shoppers_image
from msci.utils.plot import plot_path_jn, plot_histogram_jn, plot_points_on_map_jn
from msci.modelling.network import simulation, environment, shopper

from sklearn import preprocessing
from scipy.stats import linregress

matplotlib.style.use('ggplot')

data_import = False
simulate = True
return_p = False

if data_import:
    mac_address_df = utils.import_mac_addresses(version=4)
    signal_df = utils.import_signals(version=4)

    shopper_df = mac_address_df.query("bayesian_label == 'Shopper'")
    shopper_df = shopper_df.query("wifi_type != 'lawifiuser'")

    shop_df = utils.import_shop_directory(mall='Mall of Mauritius', version=2)

    r_signal_df = signal_df[
        signal_df.store_id.notnull() &
        (signal_df.store_id.str[0] == 'B') &
        signal_df.mac_address.isin(shopper_df.mac_address)
        ]

    shopper_df = shopper_df[shopper_df.mac_address.isin(
        r_signal_df.mac_address.drop_duplicates().tolist()
        )]


# def markov_chain_pre(r_signal_df):
#     le = preprocessing.LabelEncoder()
#     le.fit(r_signal_df.sort_values('store_id').store_id.unique())
#     K = len(le.classes_)
#     le_Y = le.transform(r_signal_df.sort_values('mac_address').store_id)

#     onehot = preprocessing.OneHotEncoder()
#     onehot.fit(le_Y[:, np.newaxis])
#     onehot_Y = onehot.transform(le_Y[:, np.newaxis])

#     return onehot_Y, K, le


# def matrix_evaluation(onehot_Y, K, mac_address_len, le):
#     N = np.zeros((K, K))
#     N1 = np.zeros(K)

#     L = len(mac_address_len)

#     for l in range(L):
#         seq_start = sum(mac_address_len[:l])
#         seq_end = sum(mac_address_len[:l + 1])

#         seq = onehot_Y[seq_start: seq_end].toarray()

#         T = len(seq)

#         N1 += seq[0]

#         if T > 1:
#             for t in range(1, T):
#                 N += np.outer(seq[t - 1], seq[t])

#     pi = N1 / N1.sum()
#     A = (N.T / N.sum(axis=1)).T

#     return le.inverse_transform(range(len(pi))), pi, A


# def analyse(names, pi, A, shop_df, save=False):
#     val = np.dot(pi, A.dot(A).dot(A).dot(A).dot(A).dot(A).dot(A).dot(A).dot(A).dot(A).dot(A))

#     pi_t = np.copy(pi)
#     val2 = pi_t

#     for i in range(19):
#         pi_t = pi_t.dot(A)
#     pi_t /= sum(pi_t)
#     val2 += pi_t
#     val2 /= sum(val2)

#     store_pi_df = pd.DataFrame(
#         np.array([
#             names,
#             pi.astype(float),
#             val / sum(val),
#             val2 / sum(val2),
#         ]).T,
#         columns=('store_id', 'pi', 'piA', 'avpiA')
#     )

#     if save:
#         np.savez(
#             'shop_markov_data',
#             shop_names=le.inverse_transform(range(len(pi))),
#             transition_matrix=A,
#             initial_probabilities=pi,
#         )

#     shop_probs_df = pd.merge(shop_df, store_pi_df, on='store_id', how='left')
#     shop_probs_df.sort_values('piA', ascending=False)

#     return store_pi_df, shop_probs_df, val


def manual_matrix(r_signal_df, return_permitted=True, jn=False):
    macs = r_signal_df.mac_address.drop_duplicates().tolist()
    store_names = r_signal_df.store_id.drop_duplicates().tolist()
    grouped = r_signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in macs]
    stores = [i.store_id.tolist() for i in groups]
    store_transitions = {i: [] for i in store_names}
    store_initial = {i: 0 for i in store_names}
    for store in stores:
        store_initial[store[0]] += 1
        for i in range(len(store) - 1):
            if return_permitted:
                store_transitions[store[i]].append(store[i + 1])
            else:
                if store[i + 1] != store[i]:
                    store_transitions[store[i]].append(store[i + 1])

    transition_matrix = {}

    for i in store_names:
        row_transition = np.array([store_transitions[i].count(j) for j in store_names])
        transition_matrix[i] = row_transition / np.sum(row_transition)

    sum_initial = sum(store_initial.values())

    # return store_initial

    normalised_initial = {i: j / sum_initial for (i, j) in list(zip(store_names, list(store_initial.values())))}

    largest_pi = max(normalised_initial.keys(), key=(lambda key: normalised_initial[key]))
    print(largest_pi, normalised_initial[largest_pi])

    array_transition = np.array([transition_matrix[i] for i in store_names]).T
    array_initial = np.array([normalised_initial[i] for i in store_names])

    if jn:
        return store_names, array_transition, array_initial
    else:
        return store_names, store_transitions, transition_matrix, normalised_initial, array_transition, largest_pi


def initial_store(r_signal_df):
    macs = r_signal_df.mac_address.drop_duplicates().tolist()
    store_names = r_signal_df.store_id.drop_duplicates().tolist()
    grouped = r_signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in macs]
    store_lists = [i.store_id.tolist() for i in groups]
    first_store = [i[0] for i in store_lists]
    initial_dict = {i: first_store.count(i) for i in store_names}
    sum_initial = sum(initial_dict.values())
    assert (sum_initial == len(macs))
    normalised_initial = {i: initial_dict[i] / sum_initial for i in initial_dict}
    return initial_dict, normalised_initial


def compare(r_signal_df, mac_address_len):
    pre = markov_chain_pre(r_signal_df)
    matrix = matrix_evaluation(pre[0], pre[1], mac_address_len, pre[2])
    manual = manual_matrix(r_signal_df)
    bb_initial = matrix[0][np.argmax(matrix[-2])]
    manual_initial = manual[-1]
    return bb_initial, manual_initial


def ssv_eigen(A):
    eig = np.linalg.eig(A)
    mod_eig = [abs(e) for e in eig[0]]
    eig1 = np.array([1 - me for me in mod_eig])
    print(np.argmin(eig1 ** 2))
    eigenvec = eig[1][np.argmin(eig1 ** 2)]
    real_eigenvec = abs(eigenvec)
    return real_eigenvec


def steady_state_vector(A, pi, repeats, analytic=False):
    pis = [pi]
    residues = []
    if analytic:
        I = np.eye(len(A), dtype=int)
        P = I - A
        q = np.linalg.solve(P, np.zeros(len(A)).reshape(len(A), 1))
        return q, P
    else:
        for i in range(repeats):
            pi = A.dot(pi)
            residues.append((pi - pis[-1]) ** 2)
            pis.append(copy.deepcopy(pi))
        return pi, [np.sum(r) for r in residues], pis


def process(names, A, pi, shop_df, rep, save=False):
    ssv = steady_state_vector(A, pi, repeats=rep, analytic=False)
    steady_state = ssv[0]
    data = [names, pi.tolist(), A.dot(pi).tolist(), steady_state.tolist()]
    data_T = list(map(list, zip(*data)))
    store_pi_df = pd.DataFrame(data_T, columns=('store_id', 'pi', 'piA', 'avpiA'))
    shop_probs_df = pd.merge(shop_df, store_pi_df, on='store_id', how='left')
    shop_probs_df.sort_values('piA', ascending=False)
    if save:
        np.savez(
            'shop_markov_data_nrp',
            shop_names=names,
            transition_matrix=A,
            initial_probabilities=pi,
        )
    return shop_probs_df


def ingoing_store_probabilities(transition_matrix, store_names, dict=True):
    if dict:
        ingoing = {}
        for i in range(len(store_names)):
            ing = [transition_matrix[j][i] for j in store_names]
            ingoing[store_names[i]] = np.sum(ing)
        return ingoing
    else:
        ingoing = np.sum(transition_matrix.T[i] for i in range(len(store_names)))
        return ingoing


def different_shop_number_pi(A, pi, shopper_df):
    number_of_shops_visited = shopper_df.number_of_shops.as_matrix()
    return number_of_shops_visited

"""
Simulation
"""

from importlib import reload
from matplotlib import animation, rc
from IPython.display import HTML

import copy
import datetime

from scipy.stats import gaussian_kde

if simulate:

    mac_address_df = utils.import_mac_addresses(version=4)
    signal_df = utils.import_signals(version=4)
    shop_df = utils.import_shop_directory(mall='Mall of Mauritius', version=2)
    shopper_df = mac_address_df[(mac_address_df.dbscan_label == 'Shopper') & (mac_address_df.wifi_type != 'lawifiuser')]
    
    r_signal_df = signal_df[
    signal_df.store_id.notnull() &
    (signal_df.store_id.str[0] == 'B') &
    signal_df.mac_address.isin(shopper_df.mac_address)
    ]

    shopper_df = shopper_df[shopper_df.mac_address.isin(
    r_signal_df.mac_address.drop_duplicates().tolist()
    )]

    MAX_SHOPPERS = len(shopper_df)

    shop_df.centroid = shop_df.centroid.apply(lambda x: x[1:-1].split(','))

    environment_ = environment.Environment(shop_df)

    shopper_df.start_time = pd.to_datetime(shopper_df.start_time)

    start_date_time = datetime.datetime(year=2016, month=12, day=22, hour=9)
    end_date_time = datetime.datetime(year=2016, month=12, day=23, hour=0)

    environment = copy.deepcopy(environment_)

    sim = simulation.NetworkSimulation(environment, max_shoppers=MAX_SHOPPERS, start_date_time=start_date_time, end_date_time=end_date_time)

    if return_p:
        shop_markov_data = np.load('transition_return.npz')
    else:
        shop_markov_data = np.load('transition_no_return.npz')
    
    sim.A, sim.pi = environment.realign_transition_matrix(
    shop_markov_data['shop_names'], 
    shop_markov_data['transition_matrix'].T,
    shop_markov_data['initial_probabilities'],
    )

    length_of_stay_distribution = gaussian_kde((shopper_df.length_of_stay / 60).as_matrix())

    arrival_distribution = gaussian_kde(
    (pd.to_datetime(shopper_df.start_time) - start_date_time).dt.round('15min').dt.total_seconds().as_matrix()
    )

    sim.length_of_stay_distribution = length_of_stay_distribution
    sim.arrival_distribution = arrival_distribution

    sim.iterate(max_iterations=1000)


def arrival_hist(shopper_df):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))

    binwidth = 100

    data = shopper_df[
        (pd.datetime(2016, 12, 22, 9) < shopper_df.start_time) &
        (pd.datetime(2016, 12, 22, 12) > shopper_df.start_time)
    ].length_of_stay
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), histtype='step', label='9-12');

    data = shopper_df[
        (pd.datetime(2016, 12, 22, 12) < shopper_df.start_time) &
        (pd.datetime(2016, 12, 22, 15) > shopper_df.start_time)
    ].length_of_stay
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), histtype='step', label='12-15');

    data = shopper_df[
        (pd.datetime(2016, 12, 22, 15) < shopper_df.start_time) &
        (pd.datetime(2016, 12, 22, 18) > shopper_df.start_time)
    ].length_of_stay
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), histtype='step', label='15-18');

    data = shopper_df[
        (pd.datetime(2016, 12, 22, 18) < shopper_df.start_time) &
        (pd.datetime(2016, 12, 22, 21) > shopper_df.start_time)
    ].length_of_stay
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), histtype='step', label='18-21');

    data = shopper_df[
        (pd.datetime(2016, 12, 22, 21) < shopper_df.start_time) &
        (pd.datetime(2016, 12, 23, 0) > shopper_df.start_time)
    ].length_of_stay
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), histtype='step', label='21-24');

    ax.legend();


def actual_sim_hist(signal_df, shopper_df, sim):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))

    plot_histogram_jn(
        signal_df[signal_df.mac_address.isin(shopper_df.mac_address)], 
        axes=ax,
        label='Actual Shoppers',
    )

    plot_histogram_jn(
        sim.signal_df, 
        axes=ax,
        label='Simulated Shoppers',
    );


def length_os(shopper_df, sim):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))

    binwidth = 0.05

    data = (shopper_df.length_of_stay / 60 / 60).as_matrix()
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), histtype='step', label='Actual Shoppers');

    data = (sim.mac_address_df.length_of_stay / 60 / 60).as_matrix()
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), histtype='step', label='Simulated Shoppers');

    ax.set_ylabel('No. of shoppers')
    ax.set_xlabel('Length of stay (hours)')

    ax.legend();

    plt.show()


def start_time(shopper_df, sim):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))

    binwidth = 500

    start_time = min(pd.to_datetime(sim.mac_address_df.start_time))

    data = (pd.to_datetime(shopper_df.start_time) - start_time).dt.round('15min').dt.total_seconds()
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), histtype='step', label='Actual Shoppers')


    data = (pd.to_datetime(sim.mac_address_df.start_time) - start_time).dt.round('15min').dt.total_seconds()
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), histtype='step', label='Simulated Shoppers');

    ax.set_ylabel('No. of shoppers')
    ax.set_xlabel('Start time')

    ax.legend();

    plt.show()


def number_of_shops(shopper_df, sim):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))

    binwidth = 1

    data = shopper_df.number_of_shops.as_matrix()
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), histtype='step', label='Actual Shoppers')

    data = sim.mac_address_df.number_of_shops.as_matrix()
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), histtype='step', label='Simulated Shoppers');

    ax.set_ylabel('No. of shoppers')
    ax.set_xlabel('Number of shops visited')

    ax.legend();

    plt.show()


def store_area_plot(shop_df, sim, r_signal_df):

    shop_df['sim_count'] = add_count_of_shoppers(sim.signal_df, shop_df)
    shop_df['act_count'] = add_count_of_shoppers(r_signal_df, shop_df)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 10))

    area_shop_df = shop_df[
        (shop_df.area > 0) & 
        (shop_df.act_count > 0) & 
        (shop_df.sim_count > 0) &
        shop_df.sim_count.notnull()
    ]

    ax.scatter(area_shop_df.area, area_shop_df.act_count, label='Actual Shoppers')
    slope, intercept, x_value, p_value, std_err = linregress(
        np.log10(area_shop_df.area), np.log10(area_shop_df.act_count)
    )
    area_fit = np.linspace(1, 10**4, 10)
    count_of_shoppers_fit = [10**intercept*x**slope for x in area_fit]
    ax.plot(
        area_fit, count_of_shoppers_fit, 'r--', 
        label='Power Law Fit ($Ax^{\gamma}$) \n $\gamma=%.2f \pm %.2f$' % (slope, std_err)
    )

    ax.scatter(area_shop_df.area, area_shop_df.sim_count, label='Simulated Shoppers')
    slope, intercept, x_value, p_value, std_err = linregress(
        np.log10(area_shop_df.area), np.log10(area_shop_df.sim_count)
    )
    area_fit = np.linspace(1, 10**4, 10)
    count_of_shoppers_fit = [10**intercept*x**slope for x in area_fit]
    ax.plot(
        area_fit, count_of_shoppers_fit, 'b--', 
        label='Power Law Fit ($Ax^{\gamma}$) \n $\gamma=%.2f \pm %.2f$' % (slope, std_err)
    )

    ax.set_xlabel('Area of store ($m^2$)', fontsize=15)
    ax.set_ylabel('Number of shoppers', fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend(fontsize=15);

    plt.show()

    return area_shop_df.area, area_shop_df.sim_count, area_fit, count_of_shoppers_fit


def add_count_of_shoppers(signal_df, shop_df):
    count_of_shoppers = []
    signal_group = signal_df.groupby('store_id')
    for shop in tqdm(shop_df.store_id.tolist(), desc='Count of shoppers'):
        try:
            group = signal_group.get_group(shop)
            count_of_shoppers.append(len(group.mac_address.unique()))
        except:
            count_of_shoppers.append(np.nan)
    return count_of_shoppers


