def plot_probability_trace(prob_estimates, feature_list, stationary_percentage=[]):
    """
    Plots sequence of posterior probabilities

    :param prob_estimates: data
    :param feature_list: (list of strings) list of features tested
    :param stationary_percentage: (list)
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
        for mac in range(len(prob_estimates[0][0]) - 3000, len(prob_estimates[0][0])):
        #for mac in range(500):
            y = [i[mac] for i in stationary]
            axes[0].plot(range(len(feature_list)), y)
        for mac in range(len(prob_estimates[0][0]) - 3000, len(prob_estimates[0][0])):
        #for mac in range(500):
            y = [i[mac] for i in shopper]
            axes[1].plot(range(len(feature_list)), y)
        #axes[0].set_xlabel('Feature Sequence', fontsize=20)
        axes[0].set_ylabel('P(Stationary)')
        axes[0].set_ylim((0, 1))
        axes[1].set_xlabel('Feature Sequence', fontsize=20)
        axes[1].set_ylabel('P(Shopper)')
        axes[1].set_ylim((0, 1))
        axes[0].set_xlim((0, len(FEATURE_LIST)))
        axes[1].set_xlim((0, len(FEATURE_LIST)))
    #plt.suptitle('Sequential Bayesian Inference for Device Classification')
    plt.xticks(range(len(feature_list)), feature_list, rotation='vertical')
    fig.tight_layout()
    fig.show()