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
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA


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


def bayes_array(data, prior, likelihood):
    """
    applies Bayes' theorem to the observed data based on a list of priors and a given likelihood function
    :param data: measurement
    :param prior: (list of arrays)
    :param likelihood: (function)
    :return:
    """
    likelihoods_stationary = likelihood[0](data)
    likelihoods_shopper = likelihood[1](data)
    posterior_stationary = likelihoods_stationary*prior[0]
    posterior_shopper = likelihoods_shopper*prior[1]
    sums = np.array([np.sum(i) for i in list(zip(posterior_stationary, posterior_shopper))])
    normal_stationary_posterior = posterior_stationary / sums
    normal_shopper_posterior = posterior_shopper / sums
    return np.array([normal_stationary_posterior, normal_shopper_posterior])


def prior_generator(p_stationary, mac_length):
    """
    generates array of priors for use as first step in sequential Bayes

    :param p_stationary: (float) between 0 and 1
    :param mac_length: (int) number of mac addresses used
    :return: array of priors
    """
    p_stat = np.array([p_stationary for i in range(mac_length)])
    p_shop = np.array([1 - p_stationary for i in range(mac_length)])
    return np.array([p_stat, p_shop])


"""
Likelihood Functions
"""


def likelihood_function_generator(mac_address_df, feature, dev_type, plot=False):
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
    if dev_type == 'shopper':
        values = mac_address_high_count_df[mac_address_high_count_df.is_out_of_hours == 0][feature].values.ravel()
    values = values[np.isfinite(values)]
    #print(len(values))
    func = stats.kde.gaussian_kde(values)
    if plot:
        plot_dist(func, feature, np.amax(values))
    integral = integrate.quad(func, 0, 100000)
    #print(feature, integral)
    return func


def likelihood_dictionary(feature_df, feature_list):
    """
    generates a dictionary of likelihood functions for each feature in feature_list

    :param feature_df: (df) data frame of observable measures
    :param feature_list: (list of strings) features measured
    :return: (dictionary) of likelihood functions
    """
    feature_likelihoods = {}
    for feature in feature_list:
        feature_likelihoods[feature] = [
            likelihood_function_generator(feature_df, feature, dev_type='stationary'),
            likelihood_function_generator(feature_df, feature, dev_type='shopper')]
    return feature_likelihoods


"""
Analysis
"""


def sequential(prior, feature_df, feature_list):
    """
    applies bayes_array in sequence for a range of observables

    :param feature_list: (list of strings) list of features to be tested
    :param prior: (float) initial P(stationary)
    :param feature_df: data frame
    :return: array of posteriors
    """
    priors = prior_generator(prior, len(feature_df))
    feature_likelihoods = likelihood_dictionary(feature_df, feature_list)
    prob_estimates = [priors]
    for feature in feature_list:
        print(feature)
        data = feature_df[feature].tolist()
        likelihood = feature_likelihoods[feature]
        posterior = bayes_array(data, prob_estimates[-1], likelihood)
        prob_estimates.append(posterior)
    return prob_estimates


def plot_probability_trace(prob_estimates, feature_list, stationary_percentage=[]):
    """
    plots sequence of posterior probabilities

    :param prob_estimates: data
    :param feature_list: (list of strings) list of features tested
    :return: None
    """
    stationary = [i[0] for i in prob_estimates]
    shopper = [i[1] for i in prob_estimates]
    print(feature_list)
    if len(stationary_percentage) > 0:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 8))
        axes[1].plot(range(len(feature_list)), stationary_percentage, linewidth=5)
        axes[1].set_xlabel('Feature Sequence', fontsize=20)
        axes[1].set_ylabel('Percentage of Stationary Devices (SL)')
        axes[1].set_ylim((0, 1.2*np.amax(stationary_percentage)))
        for mac in range(len(prob_estimates[0][0]) - 500, len(prob_estimates[0][0])):
            y = [i[mac] for i in stationary]
            axes[0].plot(range(len(feature_list)), y)
        axes[0].set_xlabel('Feature Sequence', fontsize=20)
        axes[0].set_ylabel('P(Stationary)')
        axes[0].set_ylim((0, 1))
    else:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 8))
        for mac in range(len(prob_estimates[0][0]) - 500, len(prob_estimates[0][0])):
            y = [i[mac] for i in stationary]
            axes[0].plot(range(len(feature_list)), y)
        for mac in range(len(prob_estimates[0][0]) - 500, len(prob_estimates[0][0])):
            y = [i[mac] for i in shopper]
            axes[1].plot(range(len(feature_list)), y)
        axes[0].set_xlabel('Feature Sequence', fontsize=20)
        axes[0].set_ylabel('P(Stationary)')
        axes[0].set_ylim((0, 1))
        axes[1].set_xlabel('Feature Sequence', fontsize=20)
        axes[1].set_ylabel('P(Shopper)')
        axes[1].set_ylim((0, 1))
    plt.title('Sequential Bayesian Inference for Device Classification')
    plt.xticks(range(len(feature_list)), feature_list, rotation='vertical')
    fig.tight_layout()
    fig.show()


def inference_result_analysis(posteriors, feature_df, confidence, signal_df, stage=-1, plot_path=True):
    macs = feature_df.mac_address.tolist()
    manufacturers = feature_df.manufacturer.tolist()
    final_probabilities = posteriors[stage]
    stationary_condition = final_probabilities[0] > confidence
    stationary_devices = [macs[i] for i in range(len(stationary_condition)) if stationary_condition[i]]
    stationary_manufacturer = [manufacturers[i] for i in range(len(stationary_condition)) if stationary_condition[i]]
    print('Number of Stationary Devices = ', len(stationary_devices))
    if plot_path:
        pfun.plot_path(signal_df, stationary_devices[:30])
    return stationary_manufacturer, stationary_devices


def inference_progress(posteriors, feature_df, confidence, signal_df):
    number_of_macs = len(feature_df)
    progress = [
        inference_result_analysis(posteriors, feature_df, confidence, signal_df, stage=i, plot_path=False)
        for i in range(len(posteriors))
    ]
    return [100*len(i[0])/number_of_macs for i in progress]


def plot_dist(func, feature, max_value):
    """
    plots likelihood function

    :param func: likelihood function (output from likelihood_function_generator
    :param feature: (string) feature
    :param max_value: (float) maximum x value to inform linspace
    :return: None
    """
    x = np.linspace(0, 1.2*max_value, num=1000)
    y = [func(i) for i in x]
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel(feature)
    plt.ylabel('PDF')
    fig.show()


def kolmogorov_smirnov(feature_df, feature, x_max):
    like_stationary = likelihood_function_generator(feature_df, feature, 'stationary', plot=False)
    like_shopper = likelihood_function_generator(feature_df, feature, 'shopper', plot=False)
    stationary_vals = np.array([like_stationary(i) for i in np.linspace(0, x_max, num=100)])
    shopper_vals = np.array([like_shopper(i) for i in np.linspace(0, x_max, num=100)])
    stationary_pdf = np.cumsum(stationary_vals)
    shopper_pdf = np.cumsum(shopper_vals)
    ks = stats.ks_2samp(stationary_pdf, shopper_pdf)
    return ks


def ks_results(feature_df, feature_list, statistic=True):
    x_max = [np.nanmax(feature_df[i].values) for i in feature_list]
    if statistic:
        ks = [kolmogorov_smirnov(feature_df, feature_list[i], x_max[i]).statistic for i in range(len(feature_list))]
    else:
        ks = [kolmogorov_smirnov(feature_df, feature_list[i], x_max[i]).pvalue for i in range(len(feature_list))]
    fig = plt.figure()
    plt.scatter(range(len(feature_list)), ks)
    plt.xticks(range(len(feature_list)), feature_list, rotation='vertical')
    plt.xlabel('Feature Type')
    plt.ylabel('KS-Test p-value')
    fig.tight_layout()
    fig.show()
    return ks


def kde_test(feature_df, feature, dev_type):
    """
    compares the kde fit function to the numerical pdf generated from data

    :param feature_df: mac address data frame
    :param feature: feature tested
    :param dev_type: stationary or shopper
    :return: None
    """
    if dev_type == 'stationary':
        function = likelihood_function_generator(feature_df, feature, dev_type='stationary')
        data = feature_df[feature_df.is_out_of_hours == 1][feature].tolist()
    if dev_type == 'shopper':
        function = likelihood_function_generator(feature_df, feature, dev_type='shopper')
        data = feature_df[feature_df.is_out_of_hours == 0][feature].tolist()
    max_value = np.amax(data)
    x = np.linspace(0, 1.2*max_value, num=1000)
    y = [function(i) for i in x]
    fig = plt.figure()
    plt.hist(data, bins=80, normed=True)
    plt.plot(x, y)
    fig.show()


"""
Correlation Analysis
"""


def pairplot(feature_df, feature_list):
    """
    Creates the pair plot which shows kdes on the diagonal and correlation on the off axis

    :param feature_df: (pd.DataFrame) the features
    :param feature_list: (list) A list of the features
    """
    fig = plt.figure()
    features = ['is_out_of_hours'] + feature_list
    g = sns.pairplot(
        feature_df[features].dropna(),
        vars=feature_list,
        hue="is_out_of_hours", diag_kind="kde", dropna=True
    )

    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    fig.show()


def ks_statistic(feature_df, feature_list):
    statistic = []
    for f in feature_list:
        v1 = feature_df[feature_df.is_out_of_hours == 0][f].values.ravel()
        v1 = v1[np.isfinite(v1)]
        v2 = feature_df[feature_df.is_out_of_hours == 1][f].values.ravel()
        v2 = v2[np.isfinite(v2)]
        statistic.append(stats.ks_2samp(v1, v2)[0])
    feat_stats_series = pd.Series(index=feature_list, data=statistic)
    feat_stats_series.sort_values(ascending=False).plot(kind='bar')
    plt.xticks(rotation='45')


def venn_diagram(mac_list, set_labels):
    fig = plt.figure()
    sets = [set(i) for i in mac_list]
    if len(mac_list) == 2:
        venn.venn2(sets, set_labels)
    if len(mac_list) == 3:
        venn.venn3(sets, set_labels)
    fig.tight_layout()
    fig.show()


def recursive_bayesian(feature_df, feature_list, prior, confidence, signal_df):
    """
    applies Bayesian inference recursively until convergent condition is met

    :param feature_df: feature data frame
    :param feature_list: feature list
    :param prior: prior probability of stationary device
    :param confidence: confidence interval for stationary device
    :param signal_df: signal data frame
    :return: all stationary macs from recursive regime
    """
    all_macs = feature_df.mac_address.tolist()
    stationary_threshold = 1
    all_stationary = []
    count = 2
    while stationary_threshold > 0 and count > 1:
        post = sequential(prior, feature_df, feature_list)
        prog = inference_result_analysis(post, feature_df, confidence, signal_df, stage=-1, plot_path=False)
        stationary_devices = prog[1]
        all_stationary.append(stationary_devices)
        feature_df = feature_df[~feature_df.mac_address.isin(stationary_devices)]
        count = len(feature_df[feature_df.is_out_of_hours == 1])
        print('count', count)
        stationary_threshold = len(stationary_devices)
        print('stationary threshold', stationary_threshold)
    flat_stationary = [i for j in all_stationary for i in j]
    all_shopper = [i for i in all_macs if i not in flat_stationary]
    return all_stationary, all_shopper, flat_stationary, stationary_threshold


def evaluate_recursion(rb, feature_df, plot=False):
    """
    identifies mac addresses that are nominally stationary i.e. give signals out of hours but are then reclassified
    as non-stationary by the the recursive Bayesian. Naively these could be the mall staff.

    :param rb:
    :param feature_df:
    :return:
    """
    out_of_hours = feature_df[feature_df.is_out_of_hours == 1].mac_address.tolist()
    #in_hours = feature_df[feature_df.is_out_of_hours == 0].mac_address.tolist()
    flat_rb = [i for j in rb[0] for i in j]
    moving_out = [i for i in out_of_hours if i not in flat_rb]
    if plot:
        venn_diagram([out_of_hours, flat_rb, rb[1]], ['Out of hours', 'Classified stationary', 'Classified shopper'])
    moving_out_df = feature_df[feature_df.mac_address.isin(moving_out)]
    return moving_out, moving_out_df


def stationary_manufacturer(manufacturer_list, rb, feature_df):
    """


    :param manufacturer_list:
    :param rb:
    :param feature_df:
    :return:
    """
    flat_stationary = [i for j in rb[0] for i in j]
    mac_set = []
    mac_labels = []
    for manufacturer in manufacturer_list:
        df = feature_df[feature_df.manufacturer == manufacturer]
        mac_set.append(df.mac_address.tolist())
        mac_labels.append(manufacturer)
    mac_set.append(flat_stationary)
    mac_labels.append('stationary')
    venn_diagram(mac_set, mac_labels)


def subset_bayesian(subset_size, iterations, feature_df, feature_list, prior, confidence, signal_df):
    """
    explores robustness of bayesian inference by considering random subsets of the data

    :param subset_size: (int) size of subset to be considered
    :param iterations: (int) how many random subsets to test
    :param feature_df: (df) feature data frame
    :param feature_list: (list of strings) list of features for inference chain
    :param prior: (float) prior probability of stationary
    :param confidence: (float) confidence level needed for stationary classification
    :param signal_df: signal data frame
    :return: information on stationary classification for each mac address over the subset iterations
    """
    stationary_macs_by_iteration = {}
    all_macs = feature_df.mac_address.tolist()
    stationary_macs = {}
    times = []
    end_type = []
    for mac in all_macs:
        stationary_macs[mac] = []
    for i in range(iterations):
        t0 = time.time()
        print(i)
        random_indices = np.random.choice(np.arange(len(feature_df)), size=subset_size)
        random_macs = [all_macs[mac] for mac in range(len(feature_df)) if mac in random_indices]
        random_feature_df = feature_df[feature_df.index.isin(random_indices)]
        rb = recursive_bayesian(random_feature_df, feature_list, prior, confidence, signal_df)
        end_type.append(rb[-1])
        stationary_macs_by_iteration[str(i)] = [random_macs, rb[2]]
        for j in random_macs:
            if j in rb[2]:
                stationary_macs[j].append(True)
            else:
                stationary_macs[j].append(False)
        times.append(time.time() - t0)
    return stationary_macs, stationary_macs_by_iteration, times


def subset_analysis(subset_bayesian_result, feature_df):
    sbr = subset_bayesian_result
    #macs = feature_df.mac_address.tolist()
    breakdown = {mac: [info.count(True), info.count(False)] for mac, info in sbr.items()}
    inconsistent = {mac: info for mac, info in breakdown.items() if 0 not in info}
    empty = {mac: info for mac, info in breakdown.items() if len(info) == 0}
    completely_consistent = {mac: info for mac, info in breakdown.items() if 0 in info and info[0] != info[1]}
    return breakdown, completely_consistent, empty, inconsistent


def plot_subset(completely_consistent, inconsistent):
    """
    plots pie chart to show fraction of consistently classified stationary devices
    plots distribution of fractions among inconsistently classified devices

    :param completely_consistent: (dictionary) macs corresponding to consistently classified devices
    :param inconsistent: (dictionary) macs corresponding to inconsistently classified devices
    :return: None
    """
    macs = list(inconsistent.keys())
    true_array = [inconsistent[i][0]/np.sum(inconsistent[i]) for i in macs]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 12))

    axes[0].pie([len(completely_consistent), len(inconsistent)],
            labels=['robust', 'inconsistent'], explode=[0, 0.1], shadow=False, autopct='%1.0f%%', labeldistance=0.8)
    axes[1].hist(true_array, bins=50)
    axes[1].set_xlabel('fraction of stationary classifications')
    axes[1].set_ylabel('probability_density_function')
    plt.suptitle('Bayesian Inference Subset Analysis')
    fig.show()


def principal_component_analysis(feature_df, feature_list):
    feature_data = [feature_df[feature].tolist() for feature in feature_list]
    data_pca = np.array(list(map(list, zip(*feature_data))))
    pca = PCA(n_components=len(feature_list))
    pca.fit(data_pca)
    variance = pca.explained_variance_ratio_
    singular = pca.singular_values_
    return feature_data, data_pca, pca, variance, singular
