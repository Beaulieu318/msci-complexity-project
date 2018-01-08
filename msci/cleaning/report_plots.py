import msci.utils.plot as pfun
from msci.utils import utils
import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import seaborn as sns
import pandas as pd
import matplotlib_venn as venn
import time
import matplotlib.dates as mdates
import plotly.plotly as py
import plotly.graph_objs as go
from datetime import datetime
import plotly.tools as tls
import pandas_datareader.data as web
import matplotlib as mpl


mm_df = utils.import_signals()


FEATURE_LIST = [
    'frequency',
    'length_of_stay',
    'radius_of_gyration',
    'count_density_variance',
    'av_speed',
    'av_turning_angle',
    'total_turning_angle',
    'av_turning_angle_velocity',
    'av_path_length',
    'total_path_length',
    'av_straightness',
]


def plot_time(df, minute_resolution, signal=True):
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('ggplot')
    #plt.style.use('dark_background')
    df.date_time = df.date_time.dt.round(str(minute_resolution) + 'min')
    signal_time_df = df.groupby('date_time')
    times = df.date_time.drop_duplicates().tolist()
    datetimes = [i.to_pydatetime() for i in times]
    counts = [len(signal_time_df.get_group(i)) for i in times]
    mac_counts = [len(signal_time_df.get_group(i).mac_address.unique()) for i in times]
    if signal:
        frame = pd.DataFrame(counts, index=datetimes, columns=['count'])
        frame.plot(legend=None)
        plt.xlabel('Time')
        plt.ylabel('Signal Count')
        plt.show()
    else:
        frame = pd.DataFrame(mac_counts, index=datetimes, columns=['count'])
        frame.plot(legend=None, color='b')
        plt.xlabel('Time')
        plt.ylabel('Device Count')
        plt.show()

def likelihoods(feature_df, feature):
    feature_df = feature_df[feature_df.frequency > 10]
    in_hours_df = feature_df[feature_df.is_out_of_hours == 0]
    out_hours_df = feature_df[feature_df.is_out_of_hours == 1]
    feature_in = in_hours_df[feature].tolist()
    feature_out = out_hours_df[feature].tolist()
    fig = plt.figure()
    plt.hist(feature_out, bins=50, color='C0')
    plt.xlabel(r'Length of Stay, $s$')
    plt.ylabel('PDF')
    plt.title('Length of Stay Histogram for Signals Out of Hours')
    fig.show()
    fig = plt.figure()
    plt.hist(feature_in, bins=50, color='C1')
    plt.xlabel(r'Length of Stay, $s$')
    plt.ylabel('PDF')
    plt.title('Length of Stay Histogram for Signals Not Out of Hours')
    fig.show()
    return feature_in, feature_out


def likelihood_function_generator(mac_address_df, feature, dev_type):
    """
    Calculates the likelihood function of a feature given that is in or out of hours

    :param mac_address_df: (pd.DataFrame) Contains the mac address features
    :param feature: (str) The feature which the likelihood is calculated from
    :param dev_type: (string) type of device: whether the mac address is out of hours i.e. stationary
    :param plot: (boolean) plot or not
    :return: (function) The pdf of the likelihood function
    """
    mac_address_high_count_df = mac_address_df[mac_address_df.frequency > 10]
    if dev_type == 'stationary':
        values = mac_address_high_count_df[mac_address_high_count_df.is_out_of_hours == 1][feature].values.ravel()
    elif dev_type == 'shopper':
        values = mac_address_high_count_df[mac_address_high_count_df.is_out_of_hours == 0][feature].values.ravel()
    else:
        raise Exception("The dev_type is not a valid entry. Needs to be either 'stationary' or 'shopper'")
    values = values[np.isfinite(values)]
    func = stats.kde.gaussian_kde(values)
    return func


def kde_test(feature_df, feature):
    """
    compares the kde fit function to the numerical pdf generated from data

    :param feature_df: mac address data frame
    :param feature: feature tested
    :param dev_type: stationary or shopper
    :return: None
    """
    feature_df = feature_df.dropna()
    function_stat = likelihood_function_generator(feature_df, feature, dev_type='stationary')
    data_stat = feature_df[feature_df.is_out_of_hours == 1][feature].tolist()
    function_mov = likelihood_function_generator(feature_df, feature, dev_type='shopper')
    data_mov = feature_df[feature_df.is_out_of_hours == 0][feature].tolist()
    max_value_stat = np.amax(data_stat)
    max_value_mov = np.amax(data_mov)
    print(1.2*max_value_mov, 1.2*max_value_stat)
    x_stat = np.linspace(0, 1.2*max_value_stat, num=1000)
    x_mov = np.linspace(0, 1.2*max_value_mov, num=1000)
    y_stat = [function_stat(i) for i in x_stat]
    y_mov = [function_mov(i) for i in x_mov]
    fig = plt.figure()
    #plt.hist(data_stat, bins=25, normed=True, color='C0', alpha=0.25)
    plt.plot(x_stat, y_stat, color='C0', label=r'P(' + 'Length of Stay' + r' | d = ' + 'Stationary)')
    #plt.hist(data_mov, bins=25, normed=True, color='C1', alpha=0.25)
    plt.plot(x_mov, y_mov, color='C1', label=r'P(' + 'Length of Stay' + r' | d = ' + 'Moving)')
    plt.xlabel('Average Straightness')
    #plt.ylabel('PDF')
    #plt.title('Likelihood Function Measurement')
    #plt.legend(loc=2)
    #plt.ylim((0, 1.2*np.amax(y_mov)))
    fig.show()


def venn_diagram(mac_list, set_labels):
    fig = plt.figure()
    sets = [set(i) for i in mac_list]
    if len(mac_list) == 2:
        venn.venn2(sets, set_labels)
    if len(mac_list) == 3:
        venn.venn3(sets, set_labels)
    fig.tight_layout()
    fig.show()


def normal_sig_test(mu, sig, sl):
    x = np.linspace(-5, 5, 200)
    y = [np.exp(-(i-mu)**2/(2*sig**2)) for i in x]
    fig = plt.figure()
    plt.plot(x, y, label='Distribution')
    ysl = np.exp(-(sl-mu)**2/(2*sig**2))
    plt.plot([sl, sl], [0, ysl], color='b', linestyle='dashed', linewidth=2, label='Significance Level')
    x_sl = np.linspace(sl, 5)
    y_sl = [np.exp(-(i-mu)**2/(2*sig**2)) for i in x_sl]
    plt.fill_between(x_sl, 0, y_sl, alpha=0.25, color='b')
    plt.xlabel(r'$x$')
    plt.ylabel('PDF')
    plt.legend()
    fig.show()


def sig_level(probs, sl):
    fig = plt.figure()
    plt.hist(probs, bins=75, color='C1', label='Classified Shopper with Confidence')
    plt.xlabel('Posterior Probability')
    plt.ylabel('Distribution of Posterior Probabilities')
    plt.xlim((0.95, 1))
    plt.legend(loc=2)
    fig.show()


def path_length_distribution(signal_df):
    macs = signal_df.mac_address.tolist()[:50]
    grouped = signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in macs]
    path_length_dist = []
    for group in groups:
        x = group.x.tolist()
        y = group.y.tolist()
        pos = list(zip(x, y))
        pos_diff = [utils.euclidean_distance(pos[i], pos[i + 1]) for i in range(len(pos) - 1)]
        path_length_dist.append(pos_diff)
    return path_length_dist


def hex_to_bin(mac):
    n = int(mac.replace(':', ''), 16)
    binary = bin(n)
    binary = binary[0] + '0' + binary[2:]
    return binary


def reverse_mac(mac):
    binary = hex_to_bin(mac)
    split_bin = [binary[8*i:(8*i + 8)] for i in range(6)]
    split_bin_reverse = [i[::-1] for i in split_bin]
    reverse = ''.join(split_bin_reverse)
    n = int(reverse, 2)
    mac = '%012x'%n
    mac = [mac[2*i:(2*i+2)] for i in range(6)]
    formatted_mac = mac[0]
    for i in range(1, 6):
        formatted_mac += ':'
        formatted_mac += mac[i]
    print(binary)
    print(split_bin)
    return formatted_mac


def oui_bar(feature_df):
    fig = plt.figure()
    feature_df.manufacturer.value_counts()[0:10].plot.bar()
    plt.xlabel('Manufacturer')
    plt.ylabel('Number of Devices')
    fig.show()


def oui_volume(signal_df):
    signal_df['manufacturer'] = utils.add_manufacturer_to_signal(signal_df)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
    for company in signal_df.groupby('mac_address').head(1).manufacturer.value_counts()[:30].index:
        company_signal_df = signal_df[signal_df.manufacturer == company]
        pfun.plot_histogram_jn(company_signal_df, axes=axes, label=company)
