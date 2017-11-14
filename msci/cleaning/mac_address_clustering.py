from msci.utils import utils


def low_frequency_count(mac_address_df, min_frequency):
    """
    Sets the probability of other for low frequency counts to 1.
    The exact number for the low frequency count is an call min_freqency.
    N.B. This function changes the input DataFrame.

    :param mac_address_df: (pd.DataFrame) The DataFrame of mac addresses
    :param min_frequency: (int) The minimum frequency which is considered valid
    """
    unresolved_mac_address_df = mac_address_df[mac_address_df.resolved == False]
    selected_mac_address_df = unresolved_mac_address_df[unresolved_mac_address_df.frequency <= min_frequency]

    mac_address_result = mac_address_df.mac_address.isin(selected_mac_address_df.mac_address)

    mac_address_df.loc[mac_address_result, 'shopper'] = 0
    mac_address_df.loc[mac_address_result, 'mall_worker'] = 0
    mac_address_df.loc[mac_address_result, 'stationary_device'] = 0
    mac_address_df.loc[mac_address_result, 'other'] = 1

    _check_probability_total(mac_address_df)
    _set_resolved_mac_addresses(mac_address_df)


def out_of_hours(mac_address_df):
    shopper_likelihood = 0
    mall_worker_likelihood = 0.2
    stationary_device_likelihood = 0.9
    other_likelihood = 0.5

    unresolved_mac_address_df = mac_address_df[mac_address_df.resolved == False]
    mac_address_result = mac_address_df.mac_address.isin(unresolved_mac_address_df.mac_address)

    utils.bayes_bool(
        result=mac_address_df.loc[mac_address_result, 'is_out_of_hours'].values,
        likelihood=shopper_likelihood,
        prior=mac_address_df.loc[mac_address_result, 'shopper'].values
    )


def _check_probability_total(mac_address_df):
    """
    Checks if all the probabilities of the mac_address_df add to 1 for the different groups.
    Raises an exception if False.

    :param mac_address_df:
    """
    passed_probability_test = (
        ((mac_address_df.shopper +
         mac_address_df.mall_worker +
         mac_address_df.stationary_device +
         mac_address_df.other) == 1).all()
    )

    if not passed_probability_test:
        raise Exception("The probabilities don't sum to 1")


def _set_resolved_mac_addresses(mac_address_df):
    """
    Changes any resolved mac addresses to True.
    This will stop any further test, to determine the group, from iterating over them.
    N.B. This function changes the input DataFrame.

    :param mac_address_df:
    """
    mac_address_df.loc[
        (
            (mac_address_df.shopper == 1) |
            (mac_address_df.mall_worker == 1) |
            (mac_address_df.stationary_device == 1) |
            (mac_address_df.other == 1)
        ), 'resolved'] = True
