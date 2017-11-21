import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

from msci.utils.animation import RealShoppersAnimation

dir_path = os.path.dirname(os.path.realpath(__file__))


def plot_path(signal_df, mac_address_df, scatter=True):
    """
    plots paths of list of mac addresses through shopping mall

    :param signal_df: data frame
    :param mac_address_df: list of mac addresses
    :param scatter: boolean to allow for scatter or plot
    :return: None
    """

    fig = plt.figure()

    img = mpimg.imread(dir_path + '/../images/mall_of_mauritius_map.png')
    plt.imshow(img[::-1], origin='lower', extent=[-77, 470, -18, 255], alpha=0.1)

    if type(mac_address_df) == pd.core.frame.DataFrame:
        signal_group = signal_df[signal_df.mac_address.isin(mac_address_df.mac_address.tolist())].groupby('mac_address')
    elif (type(mac_address_df) == list) or (type(mac_address_df) == pd.core.series.Series):
        signal_group = signal_df[signal_df.mac_address.isin(mac_address_df)].groupby('mac_address')
    elif type(mac_address_df) == str:
        signal_group = signal_df[signal_df.mac_address == mac_address_df].groupby('mac_address')
    else:
        raise Exception('mac_address_df is not a Series, list or str of mac addresses')

    for title, group in signal_group:
        if scatter:
            plt.scatter(group.x, group.y, label=title, s=0.5)
        else:
            plt.plot(group.x, group.y, label=title)

    plt.title('Stores in Mall of Mauritius')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim((0, 350))
    plt.ylim((0, 200))
    plt.legend(loc='upper center', markerscale=5., ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.show()


def plot_path_jn(signal_df, mac_address_df, axes, scatter=True):
    """
    plots paths of list of mac addresses through shopping mall

    :param signal_df: data frame
    :param mac_address_df: list of mac addresses
    :param axes: the ax from the figure
    :param scatter: boolean to allow for scatter or plot
    :return: None
    """
    if type(mac_address_df) == pd.core.frame.DataFrame:
        signal_group = signal_df[signal_df.mac_address.isin(mac_address_df.mac_address.tolist())].groupby('mac_address')
    elif (type(mac_address_df) == list) or (type(mac_address_df) == pd.core.series.Series):
        signal_group = signal_df[signal_df.mac_address.isin(mac_address_df)].groupby('mac_address')
    elif type(mac_address_df) == str:
        signal_group = signal_df[signal_df.mac_address == mac_address_df].groupby('mac_address')
    else:
        raise Exception('mac_address_df is not a Series, list or str of mac addresses')

    img = mpimg.imread(dir_path + '/../images/mall_of_mauritius_map.png')

    axes.imshow(img[::-1], origin='lower', extent=[-77, 470, -18, 255], alpha=0.1)

    for title, group in signal_group:
        if scatter:
            plt.scatter(group.x, group.y, label=title, s=0.7)
        else:
            plt.plot(group.x, group.y, label=title)

    axes.set_title('Stores in Mall of Mauritius')
    axes.set_xlabel('x (m)')
    axes.set_ylabel('y (m)')
    axes.set_xlim((0, 350))
    axes.set_ylim((0, 200))
    axes.legend(loc='upper center', markerscale=5., ncol=3, bbox_to_anchor=(0.5, -0.1))


def plot_histogram_jn(signal_df, axes, minute_resolution=15, label=None):
    """
    This plots the histogram of mac address against time

    :param signal_df: (pd.DataFrame) The signals
    :param axes: The figure ax
    :param minute_resolution: (int) The resolution of the time
    :param label: (str) The label of the line
    :return: A plot
    """
    signal_df.date_time = signal_df.date_time.dt.round(str(minute_resolution) + 'min')
    signal_time_df = signal_df.groupby('date_time').mac_address.nunique().to_frame()
    if label is not None:
        signal_time_df.rename(columns={'mac_address': label}, inplace=True)

    ax = signal_time_df.plot(ax=axes)
    ax.set_title('Histogram of mac addresses against time')
    ax.set_xlabel('Time (hh:mm)')
    ax.set_ylabel('Count of mac addresses per {} mins (no.)'.format(minute_resolution))


def plot_points_on_map(x, y, label=False):
    fig = plt.figure()

    img = mpimg.imread(dir_path + '/../images/mall_of_mauritius_map.png')
    plt.imshow(img[::-1], origin='lower', extent=[-77, 470, -18, 255], alpha=0.1)

    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        if label:
            for j in range(len(x[i])):
                plt.annotate(str(i), (x[i][j], y[i][j]))

    plt.title('Stores in Mall of Mauritius')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim((0, 350))
    plt.ylim((0, 200))
    fig.show()


def reformat_data(signal_df, mac_address_df):
    if type(mac_address_df) == pd.core.frame.DataFrame:
        selected_signal_df = signal_df[signal_df.mac_address.isin(mac_address_df.mac_address.tolist())]
    elif (type(mac_address_df) == list) or (type(mac_address_df) == pd.core.series.Series):
        selected_signal_df = signal_df[signal_df.mac_address.isin(mac_address_df)]
    elif type(mac_address_df) == str:
        selected_signal_df = signal_df[signal_df.mac_address == mac_address_df]
    else:
        raise Exception('mac_address_df is not a Series, list or str of mac addresses')

    selected_signal_df.index = selected_signal_df.date_time
    clean_signal_df = selected_signal_df[['mac_address', 'x', 'y']]
    clean_signal_df['coords'] = list(zip(clean_signal_df.x, clean_signal_df.y))
    clean_signal_df = clean_signal_df[['mac_address', 'coords']]
    return clean_signal_df.groupby(clean_signal_df.index).apply(_create_group).tolist()


def _create_group(items):
    values = {}
    for mac, coord in list(zip(items.mac_address.tolist(), items.coords.tolist())):
        values[mac] = coord
    return values


def animate(signal_df, mac_address_df, jn=False):
    shoppers_history = reformat_data(signal_df, mac_address_df)
    anim = RealShoppersAnimation(shoppers_history, 100)
    if jn:
        return anim.run()
    else:
        a = anim.run()
        plt.show()
