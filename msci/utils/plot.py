import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

    try:
        signal_group = signal_df[signal_df.mac_address.isin(mac_address_df.mac_address.tolist())].groupby('mac_address')
    except AttributeError:
        signal_group = signal_df[signal_df.mac_address.isin(mac_address_df)].groupby('mac_address')

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


def reformat_data(signal_df, mac_address_df):
    try:
        selected_signal_df = signal_df[signal_df.mac_address.isin(mac_address_df.mac_address.tolist())]
    except AttributeError:
        selected_signal_df = signal_df[signal_df.mac_address.isin(mac_address_df)]

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


def animate(signal_df, mac_address_df):
    shoppers_history = reformat_data(signal_df, mac_address_df)
    anim = RealShoppersAnimation(shoppers_history, 100)
    return anim.run()
