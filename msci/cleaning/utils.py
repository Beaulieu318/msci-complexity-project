import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

COLUMNS_TO_IMPORT = ['mac_address', 'date_time', 'location', 'store_id', 'x', 'y']


def df_to_csv(df, name, sort=False):
    if sort:
        time_sort = df.sort_values('date_time')
        mac_group = time_sort.groupby('mac_address')
        mac_group.to_csv(path_or_buf='../data/clean_data_' + name + '.csv', columns=COLUMNS_TO_IMPORT, index=False)
    else:
        df.to_csv(path_or_buf='../data/clean_data_' + name + '.csv', columns=COLUMNS_TO_IMPORT, index=False)


def euclidean_distance(xy1, xy2):
    """
    Returns euclidean distance between points xy1 and xy2

    :param xy1: (tuple) 1st position in (x,y)
    :param xy2: (tuple) 2nd position in (x,y)
    :return: (float) euclidean distance
    """
    return np.sqrt((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)


def time_difference(t0, t1):
    """
    time difference between two timedelta objects
    :param t0: (timedelta object) first timestamp
    :param t1: (timedelta object) second timestamp
    :return: (float) number of seconds elapsed between t0, t1
    """
    tdelta = t1 - t0
    return tdelta.seconds


def plot_path(macs, df):
    """
    plots paths of list of mac addresses through shopping mall

    :param macs: list of mac addresses
    :param df: data frame
    :return: None
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))

    img = mpimg.imread("../images/mall_of_mauritius_map.png")
    axes.imshow(img[::-1], origin='lower', extent=[-77, 470, -18, 255], alpha=0.1)

    df_group = df[
        df.mac_address.isin(macs)
    ].groupby('mac_address')

    for title, group in df_group:
        group.plot(x='x', y='y', ax=axes, label=title)

    axes.set_title('Stores in Mall of Mauritius')
    axes.set_xlabel('x (m)')
    axes.set_ylabel('y (m)')
    axes.set_xlim([0, 350])
    axes.set_ylim([0, 200])
    axes.legend(loc='upper center', markerscale=10., ncol=3, bbox_to_anchor=(0.5, -0.1));
    fig.show()