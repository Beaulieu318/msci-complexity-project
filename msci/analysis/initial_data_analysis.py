import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import numpy as np
import math


class DataSet:

    def __init__(self, df):
        self.df = df

    def data_anatomy(self):
        """
        Basic analysis of compositions of data set
        :return: number of signals, number of unique signals, number of unique IDs
        """
        rows = len(self.df)
        unique_rows = len(self.df.drop_duplicates())
        unique_ids = len(self.df['mac_address'].drop_duplicates())
        return rows, unique_rows, unique_ids

    def volume_analysis(self):
        """
        Analysis of signals over time
        :return: outputs histograms of customer volume over time
        """
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax = self.df.date_time.hist(bins=50)
        ax.set_title('Histogram of shoppers against time')
        ax.set_xlabel('Time (month-day hour)')
        ax.set_ylabel('Count of shoppers (no.)')
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax = self.df.loc[self.df.drop_duplicates('mac_address').index].date_time.hist(bins=50)
        ax.set_title('Histogram of shoppers against time')
        ax.set_xlabel('Time (month-day hour)')
        ax.set_ylabel('Count of shoppers (no.)')

    def group_df(self):
        """
        groups dataframe by mac address to isolate movements of individuals
        :return: grouped data frame
        """
        mac_df = self.df.groupby('mac_address')
        return mac_df

    def size_analysis(self, mac_df):
        """
        Analysis of distributions of signal numbers
        :param mac_df: mac sorted data frame
        :return: mean, std of signals sizes
        """
        sizes = mac_df.size()
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        return mean_size, std_size

    def all_mac_indices(self, mac_index):
        macs = self.df['mac_address'].tolist()
        indices = [i for i, x in enumerate(macs) if x == macs[mac_index]]
        return indices

