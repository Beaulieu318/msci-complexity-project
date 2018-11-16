import copy
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm


def create_adjacency_matrices(signal_df, sliding_interval=30, window_size=60):
    """
    Creates an array of adjacency matrices which find the number of shopper which move from shop i to shop j

    :param signal_df: (pd.DataFrame) contains all the signals from a given mall
    :param sliding_interval: (int) the number of minutes between each sampled window
    :param window_size: (int) the size of the window in minutes
    :return: (np.array) an array of adjacency matrices which increase in time with intervals of `sliding_interval`
    """
    signal_df = signal_df[
        (signal_df.date_time.dt.time > pd.Timestamp('7:00:00').time())
    ]

    store_ids = np.sort(signal_df[signal_df.store_id.notnull()].store_id.unique())
    store_dict = {key: {'index': value} for (key, value) in zip(store_ids, range(len(store_ids)))}
    store_dict = calculate_average_store_position(signal_df, store_dict)

    adjacency_matrices = []
    count_of_shoppers = []
    frame_times = {}

    start_time = min(signal_df.date_time)
    end_time = max(signal_df.date_time)

    number_of_frames = int(
        (end_time - start_time - pd.Timedelta(minutes=window_size))
                / pd.Timedelta(minutes=sliding_interval)
    )

    for frame in tqdm(range(number_of_frames)):
        window_start_time = start_time + pd.Timedelta(minutes=sliding_interval) * frame
        window_end_time = window_start_time + pd.Timedelta(minutes=window_size)

        signal_window_df = signal_df[
            (signal_df.date_time > window_start_time) &
            (signal_df.date_time < window_end_time)
            ]

        signal_matrix = signal_window_df.as_matrix(['mac_address', 'store_id'])
        adjacency_matrices.append(create_adjacency_matrix(signal_matrix, store_dict))
        count_of_shoppers.append(count_shoppers_in_store(signal_matrix, store_dict))
        frame_times[frame] = window_start_time

    return np.array(adjacency_matrices), np.array(count_of_shoppers), store_dict, frame_times


def create_adjacency_matrix(signal_matrix, store_dict):
    """
    :param signal_matrix: (np.array) the mac_address, store_id for a given window
    :param store_dict: (dict) the store ids as keys and index in the adjacency matrix as values
    :return: (np.array) adjacency matrix with the number of shoppers going from store i to store j in the signal_matrix
    """
    num_stores = len(store_dict)
    adjacency_matrix = np.zeros((num_stores, num_stores))
    mac_addresses = np.unique(signal_matrix[:, 0])

    for mac_address in mac_addresses:
        mac_address_indices = np.where(signal_matrix[:, 0] == mac_address)

        # remove nans from stores
        stores = [store for store in signal_matrix[mac_address_indices[0]][:, 1] if store is not np.nan]

        store_from = np.nan
        for store_to in stores:

            if (store_from is not store_to) and (store_from is not np.nan) and (store_to is not np.nan):
                store_from_index = store_dict[store_from]['index']
                store_to_index = store_dict[store_to]['index']

                adjacency_matrix[store_from_index][store_to_index] += 1

            store_from = store_to

    return adjacency_matrix


def count_shoppers_in_store(signal_matrix, store_dict):
    """
    :param signal_matrix: (np.array) the mac_address, store_id for a given window
    :param store_dict: (dict) the store ids as keys and index in the adjacency matrix as values
    :return: (np.array) the store_dict dict with the number of shoppers during that period in the store
    """
    store_dict_count = copy.deepcopy(store_dict)
    for store in store_dict_count:
        store_dict_count[store]['frequency'] = len(
            np.unique(signal_matrix[np.where(signal_matrix[:, 1] == store)])
        )

    return store_dict_count


def calculate_average_store_position(signal_df, store_dict):
    """
    :param signal_df: (pd.DataFrame) contains all the signals from a given mall
    :param store_dict: 
    :return: 
    """
    store_dict_position = copy.deepcopy(store_dict)
    for store in store_dict_position:
        store_dict_position[store]['x'] = signal_df[signal_df.store_id == store]['x'].mean()
        store_dict_position[store]['y'] = signal_df[signal_df.store_id == store]['y'].mean()

    return store_dict_position


def calculate_total_count_of_shoppers(signal_df):
    """
    :param signal_df: (pd.DataFrame) contains all the signals from a given mall
    :return:
    """
    signal_df = signal_df[
        (signal_df.date_time.dt.time > pd.Timestamp('7:00:00').time())
    ]

    store_ids = np.sort(signal_df[signal_df.store_id.notnull()].store_id.unique())
    store_dict = {key: {'index': value} for (key, value) in zip(store_ids, range(len(store_ids)))}
    store_dict = calculate_average_store_position(signal_df, store_dict)
    signal_matrix = signal_df.as_matrix(['mac_address', 'store_id'])

    return count_shoppers_in_store(signal_matrix, store_dict)


def calculate_in_degree_rank(total_shopper_count_df):
    """
    Calculates the rank in the frequency table for the degrees (number of people going to a shop)

    :param total_shopper_count_df: (pd.Dataframe) contains the stores with the frequency (count of vistors for each shop)
    :return: (list, list) the degrees and the rank which can be plotted on the x and y axis respecitively
    """
    total_shopper_count_df = total_shopper_count_df.sort_values('frequency')
    rank = range(len(total_shopper_count_df))[::-1]
    degrees = total_shopper_count_df['frequency']

    return degrees, rank


def calculate_in_degree_rank_probability(total_shopper_count_df):
    """
    Calculates the rank in the frequency table for the degrees (number of people going to a shop)

    :param total_shopper_count_df: (pd.Dataframe) contains the stores with the frequency (count of vistors for each shop)
    :return: (list, list) the degrees and the rank which can be plotted on the x and y axis respecitively
    """
    total_shopper_count_df = total_shopper_count_df.sort_values('frequency')
    total_shopper_count_group = total_shopper_count_df.groupby('frequency')
    N = len(total_shopper_count_group)
    P = lambda k: total_shopper_count_group.count()['index'][k] / total_shopper_count_group.count()['index'].sum()
    degrees = total_shopper_count_group.count().index

    rank = []
    for i in range(len(degrees)):
        r_i = 0
        for k in degrees[i:]:
            r_i += P(k)
        r_i *= N
        rank.append(r_i)

    return degrees, rank


def create_shopper_shop_pivot(signal_df):
    clean_signal_df = signal_df[
        signal_df.store_id.notnull() &
        signal_df.store_id.str.contains('B')
    ]
    shopper_shop_pivot = pd.pivot_table(clean_signal_df, values='x', index='mac_address', columns='store_id', aggfunc=len)
    return shopper_shop_pivot


def create_similarity_matrix(shopper_shop_df, num_entities=100, skip_entities=0):
    """
    Creates a shop or shopper similarity matrix.
    The x-axis is entities which the similarity is based on and y-axis is overlap which is being counted.
    E.g. a shopper shopper matrix will contain the number shops both shopper_i and shopper_j both went to
    E.g. a shop shop matrix will contain the number of shoppers that both went to shop_i and shop_j

    :param shopper_shop_df: (pd.DataFrame) the shoppers on one axis with the shops on the other
    :return: (pd.DataFrame) entity entity matrix with the number of overlap in the value (entity is shop or shopper)
    """
    shopper_shop_df[shopper_shop_df > 0] = 1
    # shopper_shop_df = shopper_shop_df[shopper_shop_df.notnull().sum(axis=1) == 5]
    shopper_shop_index_df = (shopper_shop_df * np.arange(shopper_shop_df.shape[1]))[
                            skip_entities:skip_entities + num_entities]
    similarity_df = pd.DataFrame(0, shopper_shop_index_df.index, shopper_shop_index_df.index)

    for i in range(len(similarity_df.index)):
        entity_index_i = shopper_shop_index_df.iloc[i][shopper_shop_index_df.iloc[i].notnull()]
        for j in range(i + 1, (len(similarity_df.index))):
            entity_index_j = shopper_shop_index_df.iloc[j][shopper_shop_index_df.iloc[j].notnull()]
            num_overlap = len(list(set(entity_index_i) & set(entity_index_j)))
            similarity_df.iloc[i, j] = num_overlap

    norm = shopper_shop_index_df.notnull().sum(axis=1)

    return similarity_df / norm + similarity_df.T / norm