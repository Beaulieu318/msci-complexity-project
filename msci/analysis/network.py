import numpy as np
import pandas as pd


def create_adjacency_matrices(signal_df, sliding_interval=30, window_size=60):
    """
    Creates an array of adjacency matrices which find the number of shopper which move from shop i to shop j

    :param signal_df: (pd.DataFrame) the number of signals in the dataframe
    :param sliding_interval: (int) the number of minutes between each sampled window
    :param window_size: (int) the size of the window in minutes
    :return: (np.array) an array of adjacency matrices which increase in time with intervals of `sliding_interval`
    """
    signal_df = signal_df[signal_df.date_time.dt.time > pd.Timestamp('7:00:00').time()]

    store_ids = np.sort(signal_df[signal_df.store_id.notnull()].store_id.unique())
    store_ids_indices = {key: value for (key, value) in zip(store_ids, range(len(store_ids)))}

    adjacency_matrices_with_time = []

    start_time = min(signal_df.date_time)
    end_time = max(signal_df.date_time)
    window_start_time = start_time

    while window_start_time < (end_time - pd.Timedelta(minutes=window_size)):
        print(window_start_time)
        window_end_time = window_start_time + pd.Timedelta(minutes=window_size)

        signal_window_df = signal_df[
            (signal_df.date_time > window_start_time) &
            (signal_df.date_time < window_end_time)
            ]

        signal_matrix = signal_window_df.as_matrix(['mac_address', 'store_id'])
        adjacency_matrices_with_time.append(create_adjacency_matrix(store_ids_indices, signal_matrix))

        window_start_time = window_start_time + pd.Timedelta(minutes=sliding_interval)

    return np.array(adjacency_matrices_with_time)


def create_adjacency_matrix(store_ids_indices, signal_matrix):
    """
    :param store_ids_indices: (dict) the store ids as keys and index in the adjacency matrix as values
    :param signal_matrix: (np.array) the mac_address, store_id for a given window
    :return: (np.array) adjacency matrix with the number of shoppers going from store i to store j in the signal_matrix
    """
    num_stores = len(store_ids_indices)
    adjacency_matrix = np.zeros((num_stores, num_stores))
    mac_addresses = np.unique(signal_matrix[:, 0])

    for mac_address in mac_addresses:
        mac_address_indices = np.where(signal_matrix[:, 0] == mac_address)

        # remove nans from stores
        stores = [store for store in signal_matrix[mac_address_indices[0]][:, 1] if store is not np.nan]

        store_from = np.nan
        for store_to in stores:

            if (store_from is not store_to) and (store_from is not np.nan) and (store_to is not np.nan):
                store_from_index = store_ids_indices[store_from]
                store_to_index = store_ids_indices[store_to]

                adjacency_matrix[store_from_index][store_to_index] += 1

            store_from = store_to

    return adjacency_matrix
