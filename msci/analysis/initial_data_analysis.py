import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import numpy as np
import math

#shopper_df = pd.read_csv('../data/bag_mus_12-22-2016.csv')
#shopper_df.date_time = shopper_df.date_time.astype('datetime64[ns]')

class DataSet:

    def __init__(self, df):
        self.df = df
        self.data_dictionary = {}

    def data_anatomy(self):
        rows = len(self.df)
        unique_rows = len(self.df.drop_duplicates())
        unique_ids = len(self.df['mac_address'].drop_duplicates())
        return rows, unique_rows, unique_ids

    def volume_analysis(self):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax = self.df.date_time.hist(bins=50)
        ax.set_title('Histogram of shoppers against time')
        ax.set_xlabel('Time (month-day hour)')
        ax.set_ylabel('Count of shoppers (no.)')
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax = self.df.loc[shopper_df.drop_duplicates('mac_address').index].date_time.hist(bins=50)
        ax.set_title('Histogram of shoppers against time')
        ax.set_xlabel('Time (month-day hour)')
        ax.set_ylabel('Count of shoppers (no.)')

    def signal_numbers(self):
        dictionary = self.df_to_dictionary()
        unique_id_list = self.df['mac_address'].drop_duplicates().tolist()
        signal_magnitudes = [len(dictionary[i]) for i in unique_id_list]
        return unique_id_list, signal_magnitudes

    def sort_data_frame(self):
        sorted_df = self.df.sort_values(['mac_address'])
        positions = list(zip(sorted_df['x'], sorted_df['y']))
        macs = np.array(sorted_df['mac_address'].tolist())
        changes = np.where(macs[:-1] != macs[1:])[0]
        changes = np.insert(changes, 0, 0)
        return sorted_df, positions, macs, changes

    def df_to_dictionary(self):
        sorted_data = self.sort_data_frame()
        positions = sorted_data[1]
        macs = sorted_data[2]
        changes = sorted_data[3]
        signal_magnitudes = []
        for i in range(len(changes)-1):
            self.data_dictionary[macs[changes[i]+1]] = positions[changes[i]:changes[i+1]]
            signal_magnitudes.append(len(self.data_dictionary[macs[changes[i]+1]]))
        return self.data_dictionary, signal_magnitudes

    def all_mac_indices(self, mac_index):
        macs = self.df['mac_address'].tolist()
        indices = [i for i, x in enumerate(macs) if x == macs[mac_index]]
        return indices

