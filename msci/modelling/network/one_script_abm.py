import numpy as np
import pandas as pd
import random
import time
import datetime

import matplotlib
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import linregress
import cmath

from msci.utils import utils
import msci.utils.plot as pfun
from msci.analysis.networks import *
from msci.utils.plot import create_count_of_shoppers_gif, create_count_of_shoppers_image
from msci.utils.plot import plot_path_jn, plot_histogram_jn, plot_points_on_map_jn
from msci.modelling.network import simulation, environment, shopper


class Environment:
    def __init__(self, shop_df):

        centroids = shop_df.centroid.tolist()
        centroids = [(float(i[0]), float(i[1])) for i in centroids]

        self.angles = [_center_angle(i) for i in centroids]
        self.store_locations = centroids
        self.distance_matrix = np.zeros((len(centroids), len(centroids)))

        self.store_distances()  # finds distances between all stores
        self.shoppers = None
        self.timestamp = []
        self.x = []
        self.y = []
        self.store_id = []
        self.mac_address = []

    def store_distances(self):
        euc = lambda r1, r2: np.sqrt((r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2)
        for s1 in range(len(self.store_locations)):
            for s2 in range(len(self.store_locations)):
                self.distance_matrix[s1][s2] = euc(self.store_locations[s1], self.store_locations[s2])

    def move_basic(self, shopper, time, timestep, dir_bias=True):
        if shopper.status == 'pre':
            if time > shopper.start_time:
                shopper.status = 'shopping'
        if shopper.status == 'shopping':
            if time > shopper.leave_time:
                shopper.status = 'post'
            else:
                if not shopper.location:
                    shopper.location = random.choice(range(len(self.store_locations)))
                else:
                    current_location = shopper.location
                    possible_locations = [i for i in range(len(self.distance_matrix[current_location])) if
                                          self.distance_matrix[current_location][
                                              i] / timestep.seconds < shopper.max_speed]

                    if dir_bias:
                        angle_dev = [self.angles[current_location] - self.angles[next_location] for next_location in
                                     possible_locations]
                        forward_dev = [possible_locations[i] for i in range(len(possible_locations)) if
                                       angle_dev[i] > 0]
                        backward_dev = [possible_locations[i] for i in range(len(possible_locations)) if
                                        angle_dev[i] < 0]
                        if random.random() < 0.95:
                            shopper.location = random.choice(forward_dev)
                        else:
                            shopper.location = random.choice(backward_dev)
                    else:
                        shopper.location = random.choice(possible_locations)

                    self.x.append(self.store_locations[shopper.location][0])
                    self.y.append(self.store_locations[shopper.location][1])
                    self.mac_address.append(shopper.ID)
                    self.timestamp.append(time)
                    self.store_id.append(shopper.location)

                    self.shoppers[shopper].append(shopper.location)


class Shopper:
    def __init__(self, ID, start_time, start_time_distribution, length_of_stay_distribution, transition_matrix,
                 max_speed):
        """
        start_time_distirbution and length_of_stay_distribution are python function objects
        """
        self.start = start_time_distribution.resample(size=1)[0][0]
        self.length_of_stay = length_of_stay_distribution.resample(size=1)[0][0]
        self.start_time = start_time + datetime.timedelta(seconds=self.start)
        self.leave_time = start_time + datetime.timedelta(seconds=self.start + self.length_of_stay)
        # print(self.start, self.length_of_stay, self.start_time, self.leave_time)
        self.transition_matrix = transition_matrix
        self.location = None
        self.status = 'pre'
        self.max_speed = max_speed
        self.ID = ID
        if random.random() < 0.5:
            self.direction = 'clock'
        else:
            self.direction = 'anti_clock'


class Simulation:
    def __init__(self, start_time, timestep):
        self.time = start_time
        self.timestep = timestep
        self.signal_df = utils.import_signals(version=4)
        self.mac_address_df = utils.import_mac_addresses(version=4)
        self.shop_df = utils.import_shop_directory(mall='Mall of Mauritius', version=2)
        self.shop_df.centroid = self.shop_df.centroid.apply(lambda x: x[1:-1].split(','))
        self.shopper_df = self.mac_address_df.query("dbscan_label == 'Shopper'")
        self.r_signal_df = self.signal_df[
            (self.signal_df.store_id.str[0] == 'B') &
            (self.signal_df.mac_address.isin(self.shopper_df.mac_address))
            ]
        self.shopper_df = self.shopper_df[
            self.shopper_df.mac_address.isin(self.r_signal_df.mac_address.drop_duplicates().tolist())
        ]
        self.t0 = datetime.datetime(year=2016, month=12, day=22, hour=0)
        self.simulation_start_time = datetime.datetime(year=2016, month=12, day=22, hour=9)

        self.environment = Environment(self.shop_df)

    def generate_shoppers(self, number_of_shoppers, max_speed):
        start_times = pd.to_datetime(self.shopper_df.start_time).tolist()
        arrival_times = [utils.time_difference(self.t0, i) for i in start_times]
        arrival_dist = stats.kde.gaussian_kde(arrival_times)
        los_dist = _generate_distribution_function(self.shopper_df, 'length_of_stay')
        shoppers = [Shopper(s, self.t0, arrival_dist, los_dist, None, max_speed) for s in range(number_of_shoppers)]
        self.environment.shoppers = {i: [] for i in shoppers}

    def iterate(self, iter_numbers, phase=1):
        for i in range(iter_numbers):
            if i % 100 == 0:
                print(i)
            self.time += self.timestep
            for shopper in self.environment.shoppers:
                if phase == 1:
                    self.environment.move_basic(shopper, self.time, self.timestep)
                if phase == 2:
                    self.environment.move_basic(shopper, self.time, self.timestep)
        self.sim_shopper_df = self.get_mac_df()
        self.sim_signal_df = self.get_signal_df()

    def get_attributes(self, attribute):
        if attribute == 'start_time':
            return [i.start_time for i in self.environment.shoppers]
        if attribute == 'status':
            return [i.status for i in self.environment.shoppers]

    def get_signal_df(self):
        return pd.DataFrame({
            'mac_address': self.environment.mac_address,
            'date_time': self.environment.timestamp,
            'x': self.environment.x,
            'y': self.environment.y,
            'store_id': self.environment.store_id
        })

    def get_mac_df(self):
        sim_signal_df = self.get_signal_df()
        macs = sim_signal_df.mac_address.drop_duplicates().tolist()
        grouped = sim_signal_df.groupby('mac_address')
        groups = [grouped.get_group(i) for i in macs]
        length_of_stays = [utils.time_difference(i.date_time.tolist()[0], i.date_time.tolist()[-1]) for i in groups]
        arrival_times = [i.date_time.tolist()[0] for i in groups]
        number_of_stores = [len(i.store_id.unique()) for i in groups]
        frequency = [len(i) for i in groups]
        centroids, rgs = _calculate_radius_gyration(sim_signal_df)
        print(len(macs), len(length_of_stays), len(rgs), len(centroids), len(arrival_times), len(frequency))
        mac_address_df = pd.DataFrame({
            'mac_address': macs,
            'length_of_stay': length_of_stays,
            'start_time': arrival_times,
            'frequency': frequency,
            'centroid': centroids,
            'radius_of_gyration': rgs
        })
        return mac_address_df

    def arrival_dist(self):
        fig = plt.figure()
        data_arrival = pd.to_datetime(self.shopper_df.start_time)
        sim_arrival = pd.to_datetime(self.sim_shopper_df.start_time)
        plt.hist([utils.time_difference(self.t0, i) for i in data_arrival], bins=50)
        plt.hist([utils.time_difference(self.t0, i) for i in sim_arrival], bins=50)
        fig.show()

    def rg_dist(self):
        fig = plt.figure()
        data_rg = self.shopper_df.radius_of_gyration.tolist()
        sim_rg = self.sim_shopper_df.radius_of_gyration.tolist()
        plt.hist(data_rg, bins=50)
        plt.hist(sim_rg, bins=50)
        fig.show()

    def store_area_plot(self):
        fig = plt.figure()

        shop_list = self.shop_df.store_id.tolist()
        grouped_data = self.signal_df.groupby('store_id')
        grouped_sim = self.sim_signal_df.groupby('store_id')
        store_areas = self.shop_df.area.tolist()
        store_visitors_data = [len(grouped_data.get_group(i).mac_address.unique()) for i in shop_list]
        store_visitors_sim = [len(grouped_sim.get_group(i)) for i in range(len(shop_list))]

        log_store_area = np.log10(store_areas)
        log_store_visitors_data = np.log10(store_visitors_data)
        log_store_visitors_sim = np.log10(store_visitors_sim)

        plt.scatter(log_store_area, log_store_visitors_data, color='r')
        plt.scatter(log_store_area, log_store_visitors_sim, color='b')
        plt.xlabel('Store Area')
        plt.ylabel('Store Visitors')

        slope_data, intercept_data, x_value, p_value, std_err = linregress(
            log_store_area[np.where(np.array(store_areas) != 0)],
            log_store_visitors_data[np.where(np.array(store_areas) != 0)])
        slope_sim, intercept_sim, x_value, p_value, std_err = linregress(
            log_store_area[np.where(np.array(store_areas) != 0)],
            log_store_visitors_sim[np.where(np.array(store_areas) != 0)])

        area_fit = np.linspace(1, 10 ** 4, 10)
        plt.plot(np.log10(area_fit), np.log10([10 ** intercept_data * x ** slope_data for x in area_fit]), color='r',
                 linestyle='dashed', label=slope_data)
        plt.plot(np.log10(area_fit), np.log10([10 ** intercept_sim * x ** slope_sim for x in area_fit]), color='b',
                 linestyle='dashed', label=slope_sim)

        plt.legend()

        fig.show()


def _generate_distribution_function(dataframe, attribute):
    values = dataframe[attribute].tolist()
    func = stats.kde.gaussian_kde(values)
    return func


def _calculate_radius_gyration(signal_df, remove_consecutive=False):
    """
    Calculates the radius of gyration which is a measure of how far the mac address moves from their central position

    :param signal_df: (pd.DataFrame) The signals
    :param mac_address_df: (pd.DataFrame) The mac addresses
    :return: (list) The radius of gyration for each of the mac addresses
    """
    signal_sorted_df = signal_df.sort_values('date_time')
    signal_mac_group = signal_sorted_df.groupby('mac_address')
    macs = signal_df.mac_address.drop_duplicates().tolist()
    centroids = []
    gyrations = []
    for mac in macs:
        mac_signals_df = signal_mac_group.get_group(mac)

        if remove_consecutive:
            mac_signals_df = mac_signals_df.loc[mac_signals_df.store_id.shift(-1) != mac_signals_df.store_id]

        r = mac_signals_df[['x', 'y']].as_matrix()
        r_cm = np.mean(r, axis=0)
        displacement = r - r_cm
        rms = np.sqrt(np.mean(displacement[:, 0] ** 2 + displacement[:, 1] ** 2))
        centroids.append([r_cm[0], r_cm[1]])
        gyrations.append(rms)
    return centroids, gyrations


def stay_probability(signal_df, mac_df):
    shopper_df = mac_df.query("dbscan_label == 'Shopper'")
    macs = shopper_df.mac_address.tolist()
    signal_df = signal_df[
        (signal_df.mac_address.isin(macs)) &
        (signal_df.store_id.str[0] == 'B')
        ]
    shopper_df = shopper_df[shopper_df.mac_address.isin(signal_df.mac_address.drop_duplicates().tolist())]
    grouped = signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in shopper_df.mac_address.tolist()]
    store_sequences = [i.store_id.tolist() for i in groups]
    return groups


def wifi_type(signal_df):
    macs = signal_df.mac_address.drop_duplicates().tolist()
    grouped = signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in macs]
    wifi_types = [i.wifi_type.unique() for i in groups]
    return wifi_types


def _center_angle(p1):
    center = (175, 75)
    x = p1[0] - center[0]
    y = p1[1] - center[1]
    theta = cmath.phase(complex(x, y))
    if theta > 0:
        return theta
    else:
        return theta + 2 * np.pi


def angle_direction(a1, a2):
    if a1 == a2:
        return None
    else:
        a2 = a2 - a1
        if a2 < 0:
            a2 = 2 * np.pi + a2
        print(a2)
        if a2 < 2 * np.pi / 3:
            return -1
        elif a2 > 4 * np.pi / 3:
            return 1
        else:
            return 0


def data_angles(shop_df, signal_df, mac_address_df):
    """
    Function to find the directional bias in the dataset
    """
    shopper_df = mac_address_df.query("dbscan_label == 'Shopper'")
    signal_df = signal_df[
        (signal_df.store_id.str[0] == 'B') &
        (signal_df.mac_address.isin(shopper_df.mac_address))
        ]
    shop_df.centroid = shop_df.centroid.apply(lambda x: x[1:-1].split(','))
    store_ids = shop_df.store_id.tolist()
    centroids = shop_df.centroid.tolist()
    centroids = [(float(i[0]), float(i[1])) for i in centroids]
    angles = {i: _center_angle(j) for (i, j) in list(zip(store_ids, centroids))}
    grouped = signal_df.groupby('mac_address')
    groups = [grouped.get_group(i) for i in signal_df.mac_address.drop_duplicates().tolist()]
    directions = []
    clock_total = 0
    anti_clock_total = 0
    for group in groups:
        ids = group.store_id.tolist()
        dirs = [angle_direction(angles[ids[i]], angles[ids[i + 1]]) for i in range(len(ids) - 1)]
        clock = dirs.count(1)
        anti_clock = dirs.count(-1)
        directions.append([anti_clock, dirs.count(0), clock, len([i for i in dirs if not i])])
        clock_total += clock
        anti_clock_total += anti_clock
    clock_people = [i for i in directions if i[2] > i[0]]
    anti_clock_people = [i for i in directions if i[0] > i[2]]
    print('Number of people with clockwise bias: ', len(clock_people))
    print('Number of people with anti-clockwise bias: ', len(anti_clock_people))
    print('Average clockwise percentage: ', np.mean([i[2] / (i[2] + i[0]) for i in clock_people]))
    print('Average anti-clockwise percentage: ', np.mean([i[0] / (i[2] + i[0]) for i in anti_clock_people]))
    return directions, clock_total, anti_clock_total, clock_people, anti_clock_people


def run_simulation(tdelta, number_of_shoppers, max_speed):
    shop_df = utils.import_shop_directory(mall='Mall of Mauritius', version=2)
    shop_df.centroid = shop_df.centroid.apply(lambda x: x[1:-1].split(','))
    croids = shop_df.centroid.tolist()
    croids = [(float(i[0]), float(i[1])) for i in croids]
    start_time = datetime.datetime(year=2016, month=12, day=22, hour=8)
    end_time = datetime.datetime(year=2016, month=12, day=23, hour=0)
    iterations = int((utils.time_difference(start_time, end_time)) / tdelta)
    print('iterations', iterations)
    NS = Simulation(
        start_time,
        datetime.timedelta(seconds=tdelta)
    )
    NS.generate_shoppers(number_of_shoppers, max_speed)
    NS.iterate(iterations, phase=1)
    return NS



