import pandas as pd
import numpy as np

from msci.utils import utils


def remove_duplicates(signal_df):
    """
    removes identical signals that are clearly errant duplicates i.e. same time and place for a given mac_id

    :param signal_df: (pd.DataFrame) the signals of the shoppers
    :return: (pd.DataFrame) the cleaned signals of the shoppers
    """
    signal_unique_df = signal_df.drop_duplicates()
    return signal_unique_df


def remove_mac_address_with_single_signals(signal_df):
    """
    removes mac addresses that have 1 signal

    :param signal_df: (pd.DataFrame) the signals of the shoppers
    :return: (pd.DataFrame) the cleaned signals of the shoppers
    """
    mac_group = signal_df.groupby('mac_address')
    mac_address_sparse = mac_group.size()[mac_group.size() > 1].index.tolist()
    signal_large_df = signal_df[signal_df.mac_address.isin(mac_address_sparse)]
    return signal_large_df


def initial_clean_signals(mall, export_location):
    signal_df = utils.import_signals(mall)
    signal_df.to_csv(export_location, index=False)
