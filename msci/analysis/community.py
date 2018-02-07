from msci.analysis.networks import *
from msci.analysis.complexity import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import *
from sklearn.cluster import *
import networkx as nx
import community
import itertools
import scipy as sp


def generate_girv_newman_matrix(clean_df, i1, i2):
    shopper_shop_pivot = create_shopper_shop_pivot(clean_df)
    analysis_mac_addresses = shopper_shop_pivot.index.drop_duplicates().tolist()[i1:i2]
    shopper_shop_pivot = shopper_shop_pivot[shopper_shop_pivot.index.isin(analysis_mac_addresses)]
    shopper_similarity_df = create_similarity_matrix(shopper_shop_pivot, num_entities=i2 - i1 + 1, skip_entities=0)
    G = shopper_similarity_df.as_matrix()
    return G


def create_shopper_shop_pivot(signal_df, matrix=True):
    clean_signal_df = signal_df[
        signal_df.store_id.notnull() &
        signal_df.store_id.str.contains('B')
        ]
    shopper_shop_pivot = pd.pivot_table(clean_signal_df, values='x', index='mac_address', columns='store_id',
                                        aggfunc=len)
    shopper_shop_pivot = shopper_shop_pivot.fillna(0)
    if matrix:
        shopper_shop_pivot = shopper_shop_pivot.as_matrix()
        #shopper_shop_pivot = sp.sparse.csr_matrix(shopper_shop_pivot.values)
    return shopper_shop_pivot


def pivot_to_common(pivot_matrix, zero_fill = True):
    reduced = pivot_matrix.copy()
    reduced[reduced > 1] = 1
    common = reduced.dot(reduced.transpose())
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
        ph_zeroed = (ph.flatten()*mask).reshape(np.shape(ph))
        return ph_zeroed/np.amax(ph_zeroed)
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


def common_stores(signal_df, graph=False, zero_diagonal=True):
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
            stores = stores_visited[i]*common
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
    #communities = tuple(sorted(c) for c in next(gn))
    return gn


def graph_modularity(G):
    m = np.sum(G)
    modularity = 0
    for i in range(len(G)):
        for j in range(len(G)):
            modularity += G[i][j] - np.sum(G[i])*np.sum(G[j])/(2*m)


lm_iterations = []


def louvain_modularity(G):
    m = np.sum(G)
    number_of_nodes = np.shape(G)[0]
    communities = {i: [i] for i in range(number_of_nodes)}
    inverse_communities = {i: i for i in range(number_of_nodes)}
    local_maximum = True
    while local_maximum:
        change = 0
        for i in range(number_of_nodes):
            #print(i)
            k_i = np.sum(G[i])
            delta_q = []
            neighbours = np.nonzero(G[i])[0]
            for j in neighbours:
                j_community_members = communities[inverse_communities[j]]
                if len(j_community_members) > 1:
                    community_neighbour_pairs = list(itertools.product(j_community_members, j_community_members))
                    sigma_in = np.sum([G[i][j] for (i, j) in community_neighbour_pairs])
                else:
                    sigma_in = 0
                sigma_t = np.sum([np.sum(G[cm]) for cm in j_community_members])
                k_i_in = np.sum(G[i][np.array(j_community_members)])
                dq = (sigma_in + k_i_in)/(2*m) - ((sigma_t + k_i)/(2*m))**2 - sigma_in/(2*m) + (sigma_t/(2*m))**2 + (k_i/(2*m))**2
                delta_q.append(dq)
            if np.amax(delta_q) <= 0:
                print(np.amax(delta_q))
            new_community = inverse_communities[neighbours[np.argmax(delta_q)]]
            if new_community != inverse_communities[i]:
                print(i, new_community)
                change += 1
                communities[new_community].append(communities[inverse_communities[i]].pop(communities[inverse_communities[i]].index(i)))
            if not communities[inverse_communities[i]]:
                del communities[inverse_communities[i]]
            inverse_communities[i] = new_community
        print('COMMUNITY CHANGES', change)
        if change == 0:
            local_maximum = False
        lm_iterations.append([communities, inverse_communities])
    return communities, inverse_communities


def louvain_modularity_(G):
    G_t = G.transpose()
    m = np.sum(G)
    number_of_nodes = np.shape(G)[0]
    communities = {i: [i] for i in range(number_of_nodes)}
    inverse_communities = {i: i for i in range(number_of_nodes)}
    local_maximum = True
    changes = [0, 1, 0, 1, 0]
    while local_maximum:
        t0 = time.time()
        change = 0
        for i in range(number_of_nodes):
            delta_q = []
            neighbours = np.nonzero(G[i])[0]
            # print('neighbours', neighbours)
            if len(neighbours) > 0:
                for j in neighbours:

                    # print(j)

                    j_community_members = communities[inverse_communities[j]]
                    i_community_members = communities[inverse_communities[i]]

                    # print(i_community_members)
                    # print(j_community_members)

                    nu_to_c2 = np.sum([G[i][n] for n in j_community_members])
                    nu_to_c1 = np.sum(G[i][n] for n in i_community_members)
                    c2_to_nu = np.sum([G[n][i] for n in j_community_members])
                    c1_to_nu = np.sum([G[n][i] for n in i_community_members])

                    # print('nu to c1', nu_to_c1)
                    # print('nu to c2', nu_to_c2)
                    # print('c1 to nu', c1_to_nu)
                    # print('c2 to nu', c2_to_nu)

                    out_i = np.sum(G[i])
                    in_i = np.sum(G_t[i])

                    # print('out', out_i)
                    # print('in', in_i)

                    c_veci = np.sum([G[i_c] for i_c in i_community_members])
                    c_vecj = np.sum([G[j_c] for j_c in j_community_members])

                    # print('cveci', c_veci)
                    # print('cvecj', c_vecj)

                    c_inv_veci = np.sum([G_t[i_c] for i_c in i_community_members])
                    c_inv_vecj = np.sum([G_t[j_c] for j_c in j_community_members])

                    # print('invcveci', c_inv_veci)
                    # print('invcvecj', c_inv_vecj)

                    dq = (nu_to_c2 - nu_to_c1 + c2_to_nu - c1_to_nu + (out_i*(c_veci - c_vecj) + in_i*(c_inv_veci - c_inv_vecj))/m)/m
                    #print('dq', dq)
                    delta_q.append(dq)

                dq_max = np.amax(delta_q)
                if dq_max > 0:
                    max_positions = [i for i, j in enumerate(delta_q) if j == dq_max]
                    new_community = inverse_communities[neighbours[max_positions[-1]]]
                    if new_community != inverse_communities[i]:
                        change += 1
                        communities[new_community].append(communities[inverse_communities[i]].pop(communities[inverse_communities[i]].index(i)))
                    if not communities[inverse_communities[i]]:
                        del communities[inverse_communities[i]]
                    inverse_communities[i] = new_community
        print('COMMUNITY CHANGES', change)
        print('Cycle Time', time.time() - t0)
        lm_iterations.append(communities)
        changes.append(change)
        print('change test', change, set(changes[-5:]), changes[-1])
        if set(changes[-3:]) == set(changes[-6:-3]):
            local_maximum = False
    return communities, inverse_communities, changes[-1]
