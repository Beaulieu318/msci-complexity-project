import pandas as pd
import numpy as np
import copy

from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

import seaborn as sns

from msci.utils import utils
from msci.analysis.networks import *
import msci.utils.plot as pfun
from msci.analysis.community_analysis import modularity, create_shopper_shop_pivot, pivot_to_common

from hmmlearn import hmm
from sklearn import preprocessing

matplotlib.style.use('ggplot')

data_import = False

if data_import:
    mac_address_df = utils.import_mac_addresses(version=3)
    signal_df = utils.import_signals(version=4)

    shopper_df = mac_address_df[mac_address_df.dbscan_label == 'Shopper']

    # shop_df = utils.import_shop_directory(mall='Mall of Mauritius', version=2)

    clean_signal_shop_df = signal_df[
        signal_df.store_id.notnull() &
        (signal_df.store_id.str[0] == 'B') &
        signal_df.mac_address.isin(shopper_df.mac_address.tolist())
        ]

    signal_mac_group = clean_signal_shop_df.groupby('mac_address')
    shopper_in_signal_df = shopper_df[shopper_df.mac_address.isin(clean_signal_shop_df.mac_address.tolist())]


def common(signal_df):
    pivot = create_shopper_shop_pivot(signal_df)
    return pivot_to_common(pivot)


class HiddenMarkovModel:
    def __init__(self, shopper_in_signal_df, signal_mac_group):
        self.shopper_df = shopper_in_signal_df
        self.mac_group = signal_mac_group
        self.macs = self.shopper_df.mac_address.tolist()
        self.data = {}
        self.parameter_encoder = {}
        self.encoded_data = {}

    def observable_sequence_lengths(self):
        lengths = []
        for i in range(len(self.shopper_df)):
            lengths.append(len(self.mac_group.get_group(self.macs[i])))
        return np.array(lengths)

    def parameter_data(self, attributes):

        for param in attributes:
            Y_discrete = np.array([])

            for i in range(len(self.shopper_df)):
                seq = self.mac_group.get_group(self.macs[i])[param].as_matrix()
                Y_discrete = np.concatenate([Y_discrete, seq])
                self.data[param] = Y_discrete

    def encode_parameter_data(self, attributes):
        """
        creates relevant data encoders and stores in dictionary in class

        :params attributes: (list) list of parameters to be used in HMM e.g. store_id
        """
        for param in attributes:
            l = preprocessing.LabelEncoder()
            l.fit(self.data[param])
            l_data = l.transform(self.data[param])
            self.parameter_encoder[param] = l
            self.encoded_data[param] = l_data

    def build_model(self, number_of_clusters, iterations):
        self.lengths = self.observable_sequence_lengths()
        self.model = hmm.MultinomialHMM(n_components=number_of_clusters, n_iter=iterations)
        self.data_to_run = np.concatenate([self.encoded_data[i][:, np.newaxis] for i in self.encoded_data], axis=1)
        self.model.fit(self.data_to_run)

    def cluster(self):
        return self.model.predict(self.data_to_run)

    def run(self, attributes, number_of_clusters, iterations):
        self.parameter_data(attributes)
        self.encode_parameter_data(attributes)
        self.build_model(number_of_clusters, iterations)


def hmm_model(shopper_in_signal_df, signal_mac_group, number_of_clusters, iterations):
    Y_dHMM = np.array([])
    lengths = []

    macs = shopper_in_signal_df.mac_address.tolist()

    for i in range(len(shopper_in_signal_df)):
        seq = signal_mac_group.get_group(macs[i]).store_id.as_matrix()
        lengths.append(len(seq))
        Y_dHMM = np.concatenate([Y_dHMM, seq])

    lengths = np.array(lengths)

    le = preprocessing.LabelEncoder()
    le.fit(Y_dHMM)
    le_Y_dHMM = le.transform(Y_dHMM)

    model = hmm.MultinomialHMM(n_components=number_of_clusters, n_iter=iterations)

    model.fit(le_Y_dHMM[:, np.newaxis], lengths=lengths)

    return model, le_Y_dHMM, lengths, model.transmat_, model.emissionprob_, model.startprob_, le


def hmm_clustering(model, le_Y_dHMM, lengths):
    macs_predictions = model.predict(le_Y_dHMM[:, np.newaxis], lengths=lengths)

    start_index = 0

    hmm_probs = []

    for l_i in range(len(lengths)):
        mac_predictions = macs_predictions[start_index:start_index + lengths[l_i]]
        mac_prediciton = max(mac_predictions)
        hmm_probs.append(mac_prediciton)
        start_index += lengths[l_i]

    return hmm_probs


def community_pie_charts(store_visit_dictionary, community):
    store_dict = store_visit_dictionary[community]
    fig = plt.figure()
    plt.pie(store_dict.values(), labels=list(store_dict.keys()), autopct='%1.1f%%')
    fig.show()


def emmision_pie(emission_matrix, clean_signal_shop_df):
    fig = plt.figure()
    plt.pie(emission_matrix, labels=clean_signal_shop_df.store_id.drop_duplicates().tolist())
    fig.show()


    # print(model.transmat_)
    # print(model.emissionprob_)
    # print(model.startprob_)
