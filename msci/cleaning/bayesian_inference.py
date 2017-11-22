import msci.utils.plot as pfun
from msci.utils import utils
import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt


def bayes_array(data, prior, likelihood):
    """
    applies Bayes' theorem to the observed data based on a list of priors and a given likelihood function
    :param data: measurement
    :param prior: (list of arrays)
    :param likelihood: (function)
    :return:
    """
    likelihoods_stationary = np.array([likelihood[0](i) for i in data])
    likelihoods_shopper = np.array([likelihood[1](i) for i in data])
    posterior_stationary = [i[0]*j for (i,j) in list(zip(prior, likelihoods_stationary))]
    posterior_shopper = [i[1] * j for (i, j) in list(zip(prior, likelihoods_shopper))]
    normal_posterior = [i/np.sum(i) for i in list(zip(posterior_stationary, posterior_shopper))]
    return normal_posterior


def prior_generator(p_stationary, mac_length):
    """
    generates array of priors for use as first step in sequential Bayes

    :param p_stationary: (float) between 0 and 1
    :param mac_length: (int) number of mac addresses used
    :return: array of priors
    """
    p = [p_stationary, 1-p_stationary]
    priors = [p for i in range(mac_length)]
    return np.array(priors)


"""
Likelihood Functions
"""


def likelihood_function_generator(mac_address_df, feature, dev_type):
    """
    Calculates the likelihood function of a feature given that is in or out of hours

    :param mac_address_df: (pd.DataFrame) Contains the mac address features
    :param feature: (str) The feature which the likelihood is calculated from
    :param dev_type: (string) type of device: whether the mac address is out of hours i.e. stationary
    :return: (function) The pdf of the likelihood function
    """
    mac_address_high_count_df = mac_address_df[mac_address_df.frequency > 10]
    if dev_type == 'stationary':
        values = mac_address_high_count_df[mac_address_high_count_df.is_out_of_hours == 1][feature].values.ravel()
    if dev_type == 'shopper':
        values = mac_address_high_count_df[mac_address_high_count_df.is_out_of_hours == 0][feature].values.ravel()
    values = values[np.isfinite(values)]
    return stats.kde.gaussian_kde(values)


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
        data = feature_df[feature].tolist()
        likelihood = feature_likelihoods[feature]
        posterior = bayes_array(data[:200], prob_estimates[-1], likelihood)
        prob_estimates.append(posterior)
    return np.array(prob_estimates)


def plot_probability_trace(prob_estimates, feature_list):
    """
    plots sequence of posterior probabilities

    :param prob_estimates: data
    :param feature_list: (list of strings) list of features tested
    :return: None
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))
    plt.ylim((0,1))
    plt.title('Sequential Bayesian Inference for Device Classification')
    plt.setp(axes, xticks=range(len(feature_list)), xticklabels=feature_list)
    for mac in range(len(prob_estimates[0][:200])):
        y = []
        for i in range(len(feature_list)):
            y.append(prob_estimates[i][mac][0])
        axes[0].plot(range(len(feature_list)), y)
    for mac in range(len(prob_estimates[0][:200])):
        y = []
        for i in range(len(feature_list)):
            y.append(prob_estimates[i][mac][1])
        axes[1].plot(range(len(feature_list)), y)
    axes[0].set_xlabel('Feature Sequence', fontsize=20)
    axes[0].set_ylabel('P(Stationary)')
    axes[1].set_xlabel('Feature Sequence', fontsize=20)
    axes[1].set_ylabel('P(Shopper)')
    fig.tight_layout()
    fig.show()


def plot_dist(mu, sigma, feature):
    x = np.linspace(int(mu-3*sigma), int(mu+3*sigma))
    y = [np.exp(-(i-mu)**2/(2*sigma**2))/np.sqrt(2*math.pi*sigma) for i in x]
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel(feature)
    plt.ylabel('PDF')
    fig.show()

