from msci.analysis.networks import *
from msci.analysis.complexity import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import *
from sklearn.cluster import *
import networkx as nx
import community as comm
import itertools
import scipy as sp
from msci.cleaning.store_ids import *
import os
import pandas as pd
from matplotlib_venn import venn3, venn3_circles
import scipy as sp
from msci.utils.utils import data_path


def generate_girv_newman_matrix(clean_df, i1, i2):
    shopper_shop_pivot = create_shopper_shop_pivot(clean_df)
    analysis_mac_addresses = shopper_shop_pivot.index.drop_duplicates().tolist()[i1:i2]
    shopper_shop_pivot = shopper_shop_pivot[shopper_shop_pivot.index.isin(analysis_mac_addresses)]
    shopper_similarity_df = create_similarity_matrix(shopper_shop_pivot, num_entities=i2 - i1 + 1, skip_entities=0)
    G = shopper_similarity_df.as_matrix()
    return G


def read_lm(csv_file=data_path + 'lm.csv'):
    lm_df = pd.read_csv(csv_file)
    lm_grouped = lm_df.groupby('community')
    communities = lm_df.community.drop_duplicates().tolist()
    lm_groups = [lm_grouped.get_group(i).index.tolist() for i in communities]
    return lm_df, lm_groups


def shops_visited_by_community(signal_df, community_grouped):
    macs = signal_df.mac_address.drop_duplicates().tolist()
    mac_groups = [macs[i] for i in community_grouped]
    return mac_groups


def create_shopper_shop_pivot(signal_df, matrix=True, sparse=False, sub=False):
    clean_signal_df = signal_df[
        signal_df.store_id.notnull() &
        signal_df.store_id.str.contains('B')
        ]
    if sub:
        macs = clean_signal_df.mac_address.drop_duplicates().tolist()
        mac_sub = np.array(macs)[np.random.choice(len(macs), 1000, replace=False)]
        clean_signal_df = clean_signal_df[clean_signal_df.mac_address.isin(mac_sub)]

    shopper_shop_pivot = pd.pivot_table(clean_signal_df, values='x', index='mac_address', columns='store_id',
                                        aggfunc=len)
    shopper_shop_pivot = shopper_shop_pivot.fillna(0)
    if matrix:
        shopper_shop_pivot = shopper_shop_pivot.as_matrix()
        if sparse:
            shopper_shop_pivot = sp.sparse.csr_matrix(shopper_shop_pivot)
            # shopper_shop_pivot = sp.sparse.csr_matrix(shopper_shop_pivot.values)
    if sub:
        return mac_sub, shopper_shop_pivot
    else:
        return shopper_shop_pivot


def pivot_to_common(pivot_matrix, zero_fill=True, normalise=True):
    reduced = pivot_matrix.copy()
    reduced[reduced > 1] = 1
    common = reduced.dot(reduced.transpose())
    if normalise:
        d = np.diagonal(common)[:, np.newaxis]
        common = common / d
        assert np.sum((np.diagonal(common) == np.ones(np.shape(common)[0])).astype(int)) == np.shape(common)[0]
    if zero_fill:
        np.fill_diagonal(common, 0)
    return common


def generate_haussdorf_matrix(clean_df, i1, i2, zero=False):
    shopper_shop_pivot = create_shopper_shop_pivot(clean_df)
    analysis_mac_addresses = shopper_shop_pivot.index.drop_duplicates().tolist()[i1:i2]
    pos_dict = position_dictionary(clean_df[clean_df.mac_address.isin(analysis_mac_addresses)], list_type=True)
    if zero:
        G = generate_girv_newman_matrix(clean_df, i1, i2)
        not_zero = np.nonzero(G.flatten())[0]
        mask = np.in1d(range(len(G.flatten())), not_zero).astype('int')
        ph = pairwise_haussdorf_fast(sorted(pos_dict), plot=False, normal=False)
        ph_zeroed = (ph.flatten() * mask).reshape(np.shape(ph))
        return ph_zeroed / np.amax(ph_zeroed)
    else:
        ph = pairwise_haussdorf_fast(sorted(pos_dict), plot=False)
        return ph


def haussdorf_community(clean_df, i1, i2, zero=False, correlation=True):
    """
    finds correlation of girv newman matrix and haussdorf matrix
    :param clean_df:
    :param i1:
    :param i2:
    :param correlation:
    :return:
    """
    G = generate_girv_newman_matrix(clean_df, i1, i2)
    if zero:
        ph = generate_haussdorf_matrix(clean_df, i1, i2, zero=True)
    else:
        ph = generate_haussdorf_matrix(clean_df, i1, i2, zero=False)
    assert np.shape(G) == np.shape(ph)
    if correlation:
        corr = matrix_correlation(G, ph)
        return corr


def haussdorf_clustering(H, method='dbscan'):
    if method == 'dbscan':
        cluster = dbscan(H)
        return cluster


def store_id_mapping(signal_df):
    stores = signal_df.store_id.drop_duplicates().tolist()
    store_dict = {stores[i]: i for i in range(len(stores))}
    return store_dict


def relabel_store_df(signal_df):
    """
    maps store ids onto unique integer identify - for ease of use in numpy array manipulation
    :param signal_df: (pd df) data frame to relabel
    :return: (pd df) relabelled df
    """
    store_dict = store_id_mapping(signal_df)
    signal_df['store_id'] = signal_df['store_id'].replace(store_dict)
    return signal_df


def common_stores(signal_df, graph=False, zero_diagonal=True, normalise=True):
    macs = signal_df.mac_address.drop_duplicates().tolist()[:200]
    grouped = signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in macs]
    stores_visited = [np.array(g.store_id.drop_duplicates().tolist()) for g in groups]
    return stores_visited
    mutual_stores = np.zeros((len(macs), len(macs)))
    mutual_dictionary = {}
    t0 = time.time()
    for i in range(len(macs)):
        for j in range(len(macs)):
            common = np.in1d(stores_visited[i], stores_visited[j]).astype('int')
            stores = stores_visited[i] * common
            stores = stores[np.nonzero(stores)[0]]
            mutual_stores[i][j] = np.sum(common)
            mutual_dictionary[(i, j)] = stores
    print(time.time() - t0)
    if zero_diagonal:
        np.fill_diagonal(mutual_stores, 0)
    if graph:
        G = nx.from_numpy_matrix(mutual_stores, create_using=nx.MultiGraph())
        return G, mutual_stores, mutual_dictionary
    else:
        return mutual_stores, mutual_dictionary


def girv(G):
    gn = nx.algorithms.community.girvan_newman(G)
    # communities = tuple(sorted(c) for c in next(gn))
    return gn


def graph_modularity(G):
    m = np.sum(G)
    modularity = 0
    for i in range(len(G)):
        for j in range(len(G)):
            modularity += G[i][j] - np.sum(G[i]) * np.sum(G[j]) / (2 * m)


def adjacency(so=True, excel=False, graph=False, sparse=True):
    mm_df = utils.import_signals(signal_type=1)
    mm_df = mm_df[mm_df.store_id.notnull()]
    clean_df = clean_store_id(mm_df, store_only=so)
    macs = clean_df.mac_address.drop_duplicates().tolist()
    pivot = create_shopper_shop_pivot(clean_df)
    common = pivot_to_common(pivot)

    if excel:
        # np.savetxt('adjacency.csv', common, delimiter=",")
        df = pd.DataFrame(common)
        df.to_csv('adjacency.csv', index=False, header=False)
    if graph:
        if sparse:
            G = nx.from_scipy_sparse_matrix(common, create_using=nx.MultiDiGraph())
            G = nx.relabel_nodes(G, dict(enumerate(macs)))
        else:
            G = nx.from_numpy_matrix(common, create_using=nx.MultiDiGraph)
            G = nx.relabel_nodes(G, dict(enumerate(macs)))
        return G, macs
    return common, macs


def louvain_mod(G, macs):
    lm = comm.best_partition(G)
    return macs, lm


def louvain_graph(network, lm_file='lm_weighted.csv'):
    df = pd.read_csv(data_path + '' + lm_file)
    macs = df['mac address'].tolist()
    comms = df['community'].tolist()

    for i in range(len(macs)):
        network.node[i]['community'] = comms[i]

    return network


def similar_shops(signal_df, lm_file='lm_weighted.csv'):
    df = pd.read_csv(data_path + '' + lm_file)
    macs = df['mac address'].tolist()
    comms = df['community'].tolist()

    grouped_df = df.groupby('community')
    mac_groups = [grouped_df.get_group(i)['mac address'].tolist() for i in range(3)]

    signal_df0 = signal_df[signal_df.mac_address.isin(mac_groups[0])]
    signal_df1 = signal_df[signal_df.mac_address.isin(mac_groups[1])]
    signal_df2 = signal_df[signal_df.mac_address.isin(mac_groups[2])]

    stores0 = signal_df0.store_id.drop_duplicates().tolist()
    stores1 = signal_df1.store_id.drop_duplicates().tolist()
    stores2 = signal_df2.store_id.drop_duplicates().tolist()

    return stores0, stores1, stores2


def venn_diagram(sets, set_labels=('community 1', 'community 2', 'community 3'),
                 title='Stores Visited by Different Communities'):
    fig = plt.figure()
    v = venn3(sets, set_labels)
    plt.title(title)
    fig.show()


def visits_from_each_community(signal_df, lm_file='lm_subset_weighted.csv', full=False):
    df = pd.read_csv(data_path + '' + lm_file)
    grouped_df = df.groupby('community')
    groups = [grouped_df.get_group(i) for i in range(3)]

    macs = df.mac_address.tolist()
    comms = df.community.tolist()

    signal_df = signal_df[signal_df.mac_address.isin(macs)]
    full_df = pd.merge(signal_df, df, on='mac_address', how='left')
    if full:
        return full_df

    comm_grouped = full_df.groupby('community')
    comm_groups = [comm_grouped.get_group(i) for i in range(len(full_df.community.drop_duplicates().stolist()))]

    store_list = full_df.store_id.drop_duplicates().tolist()
    store_grouped = full_df.groupby('store_id')
    store_groups = [store_grouped.get_group(i) for i in store_list]

    return comm_grouped, comm_groups, store_grouped, store_groups, store_list


def store_pie_chart(store_grouped, store_name):
    sg = store_grouped.get_group(store_name)
    communities = np.bincount(sg.community.tolist())
    labels = ['community ' + str(i) for i in range(len(communities))]
    fig = plt.figure()
    plt.pie(communities, labels=labels, autopct='%1.1f%%')
    fig.show()


def community_pie_chart(signal_df, lm_file='lm_subset_unweighted.csv'):
    full_df = visits_from_each_community(signal_df, lm_file, full=True)
    comm_grouped = full_df.groupby('community')
    # return comm_grouped
    store_visits = {
        comm.community.tolist()[0]: {
            store: comm.store_id.tolist().count(store) for store in comm.store_id.unique().tolist()
        }
        for comm in [comm_grouped.get_group(i) for i in range(4)]
    }
    return store_visits


def community_pie_charts(store_visit_dictionary, community):
    store_dict = store_visit_dictionary[community]
    fig = plt.figure()
    plt.pie(store_dict.values(), labels=list(store_dict.keys()), autopct='%1.1f%%')
    fig.show()


def modularity(community_dictionary, common_stores_matrix, directed=True):
    """
    Function to compute modularity of a weighted network

    https://arxiv.org/pdf/0803.0476.pdf (undirected case)
    https://arxiv.org/pdf/0801.1647.pdf (directed case)

    :param community_dictionary: (dict) {i: c} i \in shopper_population, c = community number
    :param common_stores_matrix: (numpy matrix) weighted matrix for mutual stores visited 
    :return: (float) Modularity, Q
    """
    m = 0.5 * np.sum(common_stores_matrix)
    Q = 0
    for i in range(len(community_dictionary)):
        for j in range(len(community_dictionary)):
            if community_dictionary[i] == community_dictionary[j]:
                Aij = community[i][j]
                if directed:
                    kikj = np.sum(common_stores_matrix[i] * np.sum(common_stores_matrix.T[j]))
                    Q += Aij / m - kikj / (m ** 2)
                else:
                    kikj = np.sum(common_stores_matrix[i]) * np.sum(common_stores_matrix[j])
                    Q += (Aij - kikj / (2 * m)) / (2 * m)
    return Q

# def louvain_modularity(G):
#     m = np.sum(G)
#     number_of_nodes = np.shape(G)[0]
#     communities = {i: [i] for i in range(number_of_nodes)}
#     inverse_communities = {i: i for i in range(number_of_nodes)}
#     local_maximum = True
#     while local_maximum:
#         change = 0
#         for i in range(number_of_nodes):
#             #print(i)
#             k_i = np.sum(G[i])
#             delta_q = []
#             neighbours = np.nonzero(G[i])[0]
#             for j in neighbours:
#                 j_community_members = communities[inverse_communities[j]]
#                 if len(j_community_members) > 1:
#                     community_neighbour_pairs = list(itertools.product(j_community_members, j_community_members))
#                     sigma_in = np.sum([G[i][j] for (i, j) in community_neighbour_pairs])
#                 else:
#                     sigma_in = 0
#                 sigma_t = np.sum([np.sum(G[cm]) for cm in j_community_members])
#                 k_i_in = np.sum(G[i][np.array(j_community_members)])
#                 dq = (sigma_in + k_i_in)/(2*m) - ((sigma_t + k_i)/(2*m))**2 - sigma_in/(2*m) + (sigma_t/(2*m))**2 + (k_i/(2*m))**2
#                 delta_q.append(dq)
#             if np.amax(delta_q) <= 0:
#                 print(np.amax(delta_q))
#             new_community = inverse_communities[neighbours[np.argmax(delta_q)]]
#             if new_community != inverse_communities[i]:
#                 print(i, new_community)
#                 change += 1
#                 communities[new_community].append(communities[inverse_communities[i]].pop(communities[inverse_communities[i]].index(i)))
#             if not communities[inverse_communities[i]]:
#                 del communities[inverse_communities[i]]
#             inverse_communities[i] = new_community
#         print('COMMUNITY CHANGES', change)
#         if change == 0:
#             local_maximum = False
#         lm_iterations.append([communities, inverse_communities])
#     return communities, inverse_communities


# def louvain_modularity_(G):
#     G_t = G.transpose()
#     m = np.sum(G)
#     number_of_nodes = np.shape(G)[0]
#     communities = {i: [i] for i in range(number_of_nodes)}
#     inverse_communities = {i: i for i in range(number_of_nodes)}
#     local_maximum = True
#     changes = [0, 1, 0, 1, 0]
#     while local_maximum:
#         t0 = time.time()
#         change = 0
#         for i in range(number_of_nodes):
#             delta_q = []
#             neighbours = np.nonzero(G[i])[0]

#             i_community_members = communities[inverse_communities[i]]
#             nu_to_c1 = np.sum([G[i, i_community_members]])
#             c1_to_nu = np.sum([G[i_community_members, i]])

#             out_i = np.sum(G[i])
#             in_i = np.sum(G_t[i])

#             c_veci = np.sum([G[i_community_members]])
#             c_inv_veci = np.sum([G_t[i_community_members]])

#             # print('neighbours', neighbours)
#             if len(neighbours) > 0:
#                 for j in neighbours:

#                     # print(j)

#                     j_community_members = communities[inverse_communities[j]]

#                     # print(i_community_members)
#                     # print(j_community_members)

#                     nu_to_c2 = np.sum([G[i, j_community_members]])
#                     c2_to_nu = np.sum([G[j_community_members, i]])


#                     # print('nu to c1', nu_to_c1)
#                     # print('nu to c2', nu_to_c2)
#                     # print('c1 to nu', c1_to_nu)
#                     # print('c2 to nu', c2_to_nu)

#                     # print('out', out_i)
#                     # print('in', in_i)

#                     c_vecj = np.sum([G[j_community_members]])
#                     c_inv_vecj = np.sum([G_t[j_community_members]])

#                     # print('invcveci', c_inv_veci)
#                     # print('invcvecj', c_inv_vecj)

#                     dq = (nu_to_c2 - nu_to_c1 + c2_to_nu - c1_to_nu + (out_i*(c_veci - c_vecj) + in_i*(c_inv_veci - c_inv_vecj))/m)/m
#                     #print('dq', dq)
#                     delta_q.append(dq)

#                 dq_max = np.amax(delta_q)
#                 if dq_max > 0:
#                     max_positions = [i for i, j in enumerate(delta_q) if j == dq_max]
#                     new_community = inverse_communities[neighbours[max_positions[-1]]]
#                     if new_community != inverse_communities[i]:
#                         change += 1
#                         communities[new_community].append(communities[inverse_communities[i]].pop(communities[inverse_communities[i]].index(i)))
#                     if not communities[inverse_communities[i]]:
#                         del communities[inverse_communities[i]]
#                     inverse_communities[i] = new_community
#         print('COMMUNITY CHANGES', change)
#         print('Cycle Time', time.time() - t0)
#         lm_iterations.append(communities)
#         changes.append(change)
#         print('change test', change, set(changes[-5:]), changes[-1])
#         if set(changes[-3:]) == set(changes[-6:-3]):
#             local_maximum = False
#     return communities, inverse_communities, changes[-1]


# def louvain_modularity_inv(G):
#     G_t = G.transpose()
#     m = np.sum(G)
#     print(m)
#     number_of_nodes = np.shape(G)[0]
#     inverse_communities = np.arange(number_of_nodes)
#     local_maximum = True
#     changes = [0, 1, 0, 1, 0]
#     while local_maximum:
#         t0 = time.time()
#         change = 0
#         for i in range(number_of_nodes):
#             delta_q = []
#             neighbours = np.nonzero(G[i])[0]

#             current_community = inverse_communities[i]
#             i_community_members = np.where(inverse_communities == current_community)
#             nu_to_c1 = np.sum([G[i, i_community_members]])
#             c1_to_nu = np.sum([G[i_community_members, i]])

#             out_i = np.sum(G[i])
#             in_i = np.sum(G_t[i])

#             c_veci = np.sum([G[i_community_members]])
#             c_inv_veci = np.sum([G_t[i_community_members]])

#             # print('neighbours', neighbours)
#             if len(neighbours) > 0:
#                 for j in neighbours:

#                     # print(j)

#                     j_current_community = inverse_communities[j]
#                     j_community_members = np.where(inverse_communities == j_current_community)

#                     # print(i_community_members)
#                     # print(j_community_members)

#                     nu_to_c2 = np.sum([G[i, j_community_members]])
#                     c2_to_nu = np.sum([G[j_community_members, i]])


#                     # print('nu to c1', nu_to_c1)
#                     # print('nu to c2', nu_to_c2)
#                     # print('c1 to nu', c1_to_nu)
#                     # print('c2 to nu', c2_to_nu)

#                     # print('out', out_i)
#                     # print('in', in_i)

#                     c_vecj = np.sum([G[j_community_members]])
#                     c_inv_vecj = np.sum([G_t[j_community_members]])

#                     # print('invcveci', c_inv_veci)
#                     # print('invcvecj', c_inv_vecj)

#                     dq = (nu_to_c2 - nu_to_c1 + c2_to_nu - c1_to_nu + (out_i*(c_veci - c_vecj) + in_i*(c_inv_veci - c_inv_vecj))/m)/m
#                     #print('dq', dq)
#                     delta_q.append(dq)

#                 return delta_q
#                 dq_max = np.amax(delta_q)
#                 if dq_max > 0:
#                     max_positions = [i for i, j in enumerate(delta_q) if j == dq_max]
#                     new_community = inverse_communities[neighbours[max_positions[-1]]]
#                     if new_community != inverse_communities[i]:
#                         change += 1
#                         #communities[new_community].append(communities[inverse_communities[i]].pop(communities[inverse_communities[i]].index(i)))
#                     #if not communities[inverse_communities[i]]:
#                         #del communities[inverse_communities[i]]
#                         inverse_communities[i] = new_community
#         return inverse_communities
#         print('COMMUNITY CHANGES', change)
#         print('Cycle Time', time.time() - t0)
#         changes.append(change)
#         print('change test', change, set(changes[-5:]), changes[-1])
#         if set(changes[-3:]) == set(changes[-6:-3]):
#             local_maximum = False
#     return inverse_communities, changes[-1]
