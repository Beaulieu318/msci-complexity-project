import msci.utils.plot as pfun
from msci.utils import utils
import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate


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
    func = stats.kde.gaussian_kde(values)
    if plot:
        plot_dist(func, feature, np.amax(values))
    integral = integrate.quad(func, 0, 100000)
    print(feature, integral)
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


def plot_probability_trace(prob_estimates, feature_list):
    """
    plots sequence of posterior probabilities

    :param prob_estimates: data
    :param feature_list: (list of strings) list of features tested
    :return: None
    """
    stationary = [i[0] for i in prob_estimates]
    shopper = [i[1] for i in prob_estimates]
    print(feature_list)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))
    plt.ylim((0, 1))
    plt.title('Sequential Bayesian Inference for Device Classification')
    #plt.setp(axes, xticks=range(len(feature_list)), xticklabels=feature_list)
    for mac in range(len(prob_estimates[0][0]) - 500, len(prob_estimates[0][0])):
        y = [i[mac] for i in stationary]
        axes[0].plot(range(len(feature_list)), y)
    for mac in range(len(prob_estimates[0][0]) - 500, len(prob_estimates[0][0])):
        y = [i[mac] for i in shopper]
        axes[1].plot(range(len(feature_list)), y)
    axes[0].set_xlabel('Feature Sequence', fontsize=20)
    axes[0].set_ylabel('P(Stationary)')
    axes[1].set_xlabel('Feature Sequence', fontsize=20)
    axes[1].set_ylabel('P(Shopper)')
    #axes[0].set_xticks(range(len(feature_list)))
    #axes[0].set_xticklabels(feature_list, fontdict={'verticalalignment': 'baseline'})
    #axes[1].set_xticks(range(len(feature_list)))
    #axes[1].set_xticklabels(feature_list)
    plt.xticks(range(len(feature_list)), feature_list, rotation='vertical')
    fig.tight_layout()
    fig.show()


def inference_result_analysis(posteriors, feature_df, confidence, signal_df, plot_path=True):
    macs = feature_df.mac_address.tolist()
    manufacturers = feature_df.manufacturer.tolist()
    final_probabilities = posteriors[-1]
    stationary_condition = final_probabilities[0] > confidence
    stationary_devices = [macs[i] for i in range(len(stationary_condition)) if stationary_condition[i]]
    stationary_manufacturer = [manufacturers[i] for i in range(len(stationary_condition)) if stationary_condition[i]]
    print('Number of Stationary Devices = ', len(stationary_devices))
    if plot_path:
        pfun.plot_path(signal_df, stationary_devices[:30])
    return list(zip(stationary_manufacturer, stationary_devices))


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
