import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

columns = ['mac_address', 'date_time', 'location', 'floor', 'x', 'y']

#p_df = pd.read_csv('../data/clean_data_phoenix.csv', usecols=columns)
#p_df.date_time = p_df.date_time.astype('datetime64[ns]')


class DataSet:

    def __init__(self, dataframe):
        self.df = dataframe

    def path_length(self):
        macs = self.df.mac_address.drop_duplicates().tolist()
        mac_group = self.df.groupby('mac_address')
        return macs,mac_group


def _euclidean_distance(xy1, xy2):
    return ((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)**0.5


def tdelta(t0, t1):
    tdelta = t1 - t0
    return tdelta.seconds