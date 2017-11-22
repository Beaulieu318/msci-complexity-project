import msci.utils.plot as pfun
from msci.utils import utils
import numpy as np
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
    likelihoods_stationary = np.array([likelihood(i, 'stationary') for i in data])
    likelihoods_shopper = np.array([likelihood(i, 'shopper') for i in data])
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


def length_of_stay(length, dev_type, plot=False):
    """
    likelihood function for the length of stay of a device

    :param length: (int) length of stay in seconds
    :param dev_type: (string) device type to be tested e.g. stationary
    :param plot: (boolean) plot or no plot
    :return: (float) likelihood
    """
    type_data = {'stationary': [23*60*60, 1*60*60], 'shopper': [2*60*60, 8*60*60], 'worker': []}
    mu = type_data[dev_type][0]
    sigma = type_data[dev_type][1]
    if plot:
        plot_dist(mu, sigma, 'Length of Stay (s)')
    return np.exp(-(length - mu)**2/(2*sigma**2))/np.sqrt(2*math.pi*sigma)


def radius_likelihood(r_g, dev_type, plot=False):
    """
    likelihood function for the length of stay of a device

    :param r_g: (float) radius of gyration for path
    :param dev_type: (string) device type to be tested e.g. stationary
    :param plot: (boolean) plot or no plot
    :return: (float) likelihood
    """
    type_data = {'stationary': [4, 8], 'shopper': [40, 20], 'worker': []}
    mu = type_data[dev_type][0]
    sigma = type_data[dev_type][1]
    if plot:
        plot_dist(mu, sigma, 'Radius of Gyration')
    return np.exp(-(r_g - mu) ** 2 /(2*sigma**2))/np.sqrt(2*math.pi*sigma)


"""
Analysis
"""


def sequential(feature_list, priors, feature_df):
    """
    applies bayes_array in sequence for a range of observables

    :param feature_list: (list of strings) list of features to be tested
    :param priors: (array) list of initial priors
    :param feature_df: data frame
    :return: array of posteriors
    """
    feature_likelihoods = {'length_of_stay': [length_of_stay], 'gyration': [radius_likelihood]}
    prob_estimates = [priors]
    for feature in feature_list:
        data = feature_df[feature].tolist()
        likelihood = feature_likelihoods[feature][0]
        posterior = bayes_array(data, prob_estimates[-1], likelihood)
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
    for mac in range(len(prob_estimates[0][:100])):
        y = []
        for i in range(len(feature_list)):
            y.append(prob_estimates[i][mac][0])
        axes[0].plot(range(len(feature_list)), y)
    for mac in range(len(prob_estimates[0][:100])):
        y = []
        for i in range(len(feature_list)):
            y.append(prob_estimates[i][mac][1])
        axes[1].plot(range(len(feature_list)), y)
    axes[0].set_xlabel('Feature Sequence', fontsize=20)
    axes[0].set_ylabel('P(Stationary)')
    axes[1].set_xlabel('Feature Sequence', fontsize=20)
    axes[1].set_ylabel('P(Stationary)')
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

