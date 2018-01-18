import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import math
import numpy as np
from scipy.optimize import fmin
import matplotlib.cm as cm
from tempfile import mkstemp, mkdtemp
from imageio import imread, mimsave
from matplotlib.pyplot import clf

from msci.utils.animation import RealShoppersAnimation

dir_path = os.path.dirname(os.path.realpath(__file__))


"""
Path plots on the map of Mall of Mauritius
"""


def plot_path(signal_df, mac_address_df, scatter=True):
    """
    Plots paths of list of mac addresses through shopping mall

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
    Plots paths of list of mac addresses through shopping mall

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
            plt.scatter(group.x, group.y, label=title, s=2)
        else:
            plt.plot(group.x, group.y, label=title)

    axes.set_title('Stores in Mall of Mauritius')
    axes.set_xlabel('x (m)')
    axes.set_ylabel('y (m)')
    axes.set_xlim((0, 350))
    axes.set_ylim((0, 200))
    axes.legend(loc='upper center', markerscale=5., ncol=3, bbox_to_anchor=(0.5, -0.1))


def plot_points_on_map(x, y, label=False):
    """
    Plots list of x,y coordinates onto a map

    :param x: (list) A list of x coordinates
    :param y: (list) A list of y coordinates
    :param label: (Boolean) Where the coordinates should be labelled with numbers
    :return: A plot
    """
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


"""
Histogram plots against time
"""


def plot_histogram_jn(signal_df, axes, minute_resolution=15, label=None):
    """
    This plots the histogram of mac address against time.
    The y axis shows how many mac addresses (devices) where present over during the minute_resolution intervals.

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


"""
Animation
"""


def reformat_data(signal_df, mac_address_df):
    """
    Formats the data so that it can be understood by the animation

    :param signal_df: (pd.DataFrame) The signals data frame
    :param mac_address_df: (pd.Series) The series of mac addresses which need to be animated
    :return: The signals that used for the animation
    """
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
    """
    Create an animation of the mac addresses moving around the show

    :param signal_df: (pd.DataFrame) The signals data frame
    :param mac_address_df: (pd.Series) The series of mac addresses which need to be animated
    :param jn: (Bool) If this is run in a jupyter notebook
    :return: An animation
    """
    shoppers_history = reformat_data(signal_df, mac_address_df)
    anim = RealShoppersAnimation(shoppers_history, 100)
    if jn:
        return anim.run()
    else:
        a = anim.run()
        plt.show()


"""
Venn Diagram with more than 3 circles
"""


def multiple_venn(all_stationary, mac_list, mac_labels):
    """
    allows plotting of >3 circle venn diagram with no overlap between subsets
    ideal for showing stationary/moving breakdown of different manufacturers

    :param all_stationary: (list) stationary mac addresses
    :param mac_list: (list of lists) stationary mac addresses of different manufacturers
    :param mac_labels: (list) manufacturer names
    :return: venn diagram metrics
    """
    angles = [i*2*math.pi/len(mac_list) for i in range(len(mac_list))]
    circle_areas = [len(i) for i in mac_list]
    stationary_area = len(all_stationary)
    distances = []
    overlaps = []
    for i in range(len(circle_areas)):
        overlap = len([i for i in mac_list[i] if i in all_stationary])
        area = circle_areas[i]
        d = find_d(chord_dist_func, overlap, stationary_area, area)[0]
        distances.append(d)
        overlaps.append([overlap, len(mac_list[i]) - overlap])
    circles = [[0, 0, np.sqrt(stationary_area/math.pi)]]
    for j in range(len(angles)):
        if 0 <= angles[j] < 0.5*math.pi:
            distance = distances[j]
            alpha = distance/np.sqrt(1 + (np.tan(angles[j]))**2)
            beta = np.sqrt(distance**2 - alpha**2)
            radius = np.sqrt(circle_areas[j]/math.pi)
            circles.append([alpha, beta, radius])
        if 0.5*math.pi <= angles[j] < math.pi:
            distance = distances[j]
            alpha = -1*distance/np.sqrt(1 + (np.tan(angles[j]))**2)
            beta = np.sqrt(distance**2 - alpha**2)
            radius = np.sqrt(circle_areas[j]/math.pi)
            circles.append([alpha, beta, radius])
        if math.pi <= angles[j] < 1.5*math.pi:
            distance = distances[j]
            alpha = -1 * distance / np.sqrt(1 + (np.tan(angles[j])) ** 2)
            beta = -1*np.sqrt(distance ** 2 - alpha ** 2)
            radius = np.sqrt(circle_areas[j] / math.pi)
            circles.append([alpha, beta, radius])
        if 1.5*math.pi <= angles[j] < 2*math.pi:
            distance = distances[j]
            alpha = distance / np.sqrt(1 + (np.tan(angles[j])) ** 2)
            beta = -1*np.sqrt(distance ** 2 - alpha ** 2)
            radius = np.sqrt(circle_areas[j] / math.pi)
            circles.append([alpha, beta, radius])
    plot_circles(circles, mac_labels, overlaps)
    return distances, angles, circle_areas


def chord_dist_func(d, overlap, R, r):
    a0 = 0.5 * np.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R))
    a1 = r ** 2 * np.arccos((d ** 2 + r ** 2 - R ** 2) / (2 * d * r)) + R ** 2 * np.arccos(
        (d ** 2 + R ** 2 - r ** 2) / (2 * d * R))
    return np.abs(overlap - (a1 - a0))


def find_d(func, overlap_area, A, a):
    r = np.sqrt(a/math.pi)
    R = np.sqrt(A/math.pi)
    return fmin(func, x0=R, args=(overlap_area, R, r))


def plot_circles(list_of_circles, manufacturers, splits):
    colors = cm.rainbow(np.linspace(0, 1, len(list_of_circles)))
    fig, ax = plt.subplots()
    circles = []
    for cir in list_of_circles[1:]:
        ind = list_of_circles.index(cir) - 1
        c = plt.Circle((cir[0], cir[1]), cir[2], color=colors[list_of_circles.index(cir)])
        circles.append(c)
        ax.add_artist(c)
        plt.annotate(str(splits[ind][0]) + ':' + str(splits[ind][1]), (cir[0], cir[1]))
    c = plt.Circle((list_of_circles[0][0],
                    list_of_circles[0][1]), list_of_circles[0][2], label='Stationary', fill=False)
    ax.add_artist(c)
    plt.legend(tuple(circles), tuple(manufacturers), loc=2)
    plt.xlim((-70, 70))
    plt.ylim((-70, 70))
    fig.show()


"""
Network plots and images of people in shop
"""


def create_count_of_shoppers_image(count_of_shoppers, frame_times, count_index=10, output_fig=False):
    """
    Creates a image of the count of shoppers at a particular frame (time)

    :param count_of_shoppers: (dict) key: all the shops, values: count, x, y, index
    :param frame_times: (dict) key: frame number, value: date time
    :param count_index: (int) the current frame for the image
    :param output_fig: (bool) use True in create_count_of_shoppers_gif function
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    img = mpimg.imread("../../images/mall_of_mauritius_map.png")
    axes.imshow(img[::-1], origin='lower', extent=[-77, 470, -18, 255], alpha=0.1)

    colors = iter(cm.rainbow(np.linspace(0, 1, len(count_of_shoppers[count_index]))))
    for key in count_of_shoppers[count_index]:
        axes.scatter(
            count_of_shoppers[count_index][key]['x'],
            count_of_shoppers[count_index][key]['y'],
            s=count_of_shoppers[count_index][key]['frequency'] * 5,
            color=next(colors),
            label=key
        )

    axes.set_title('Mall of Mauritius: {}'.format(frame_times[count_index]), fontsize=20)
    axes.set_xlabel('x (m)', fontsize=15)
    axes.set_ylabel('y (m)', fontsize=15)
    axes.set_xlim((0, 350))
    axes.set_ylim((0, 200))
    # axes.legend(loc='upper center', markerscale=1., ncol=15, bbox_to_anchor=(0.5, -0.1))

    if output_fig:
        return fig


def create_count_of_shoppers_gif(count_of_shoppers, frame_times, gif_filename='count_of_shopper'):
    """
    Creates the GIF of the count of shoppers at different shops over the day

    :param count_of_shoppers: (dict) key: all the shops, values: count, x, y, index
    :param frame_times: (dict) key: frame number, value: date time
    :param gif_filename: (str) the filename for the GIF
    """
    tempdir = mkdtemp()

    pngs = []

    for frame in range(len(count_of_shoppers)):
        fig = create_count_of_shoppers_image(
            count_of_shoppers,
            frame_times,
            count_index=frame,
            output_fig=True
        )

        _, filename = mkstemp(dir=tempdir)
        filename += '.png'

        fig.savefig(filename)
        clf()  # clear figure
        pngs.append(filename)

    images = []
    for png in pngs:
        img = imread(png)
        images.append(img)
    mimsave('{}.gif'.format(gif_filename), images, duration=0.5)
