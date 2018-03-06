import random
import datetime
import pandas as pd
import numpy as np

from scipy.stats import norm

from tqdm import tqdm_notebook as tqdm

from msci.modelling.network.shopper import Shopper


class NetworkSimulation:
    def __init__(self, environment, max_shoppers, start_date_time, end_date_time):
        self.environment = environment
        self.max_shoppers = max_shoppers

        self.shoppers = []
        self.shoppers_history = []
        self.signal_history = []

        self.start_date_time = start_date_time
        self.end_date_time = end_date_time
        self.total_minutes = (self.end_date_time - self.start_date_time).seconds / 60
        self.minutes_per_iteration = 0

        self.signal_df = None
        self.mac_address_df = None

        self.length_of_stay_distribution = None
        self.arrival_distribution = None

    def iterate(self, max_iterations):

        self.minutes_per_iteration = self.total_minutes / max_iterations

        self._generate_shoppers()

        for iteration in tqdm(range(max_iterations)):
            date_time = self.start_date_time + datetime.timedelta(minutes=self.minutes_per_iteration) * iteration

            self._add_new_shopper(date_time)
            self._action_shopper()
            self._save_shoppers_history()
            self._save_signal_history(date_time)

        self.create_signal_df()
        self.create_mac_address_df()

    def _generate_shoppers(self):
        self.all_shoppers = []
        self.shoppers_arrival_times = np.empty(self.max_shoppers, dtype=datetime.datetime)

        for i in range(self.max_shoppers):

            start_time = self.start_date_time + datetime.timedelta(minutes=-1)
            length_of_stay = -1

            while (start_time < self.start_date_time) and (length_of_stay < 0):
                start_time = self.start_date_time + datetime.timedelta(seconds=self.arrival_distribution.resample(size=1)[0][0])
                length_of_stay = self.length_of_stay_distribution.resample(size=1)[0][0]

            self.all_shoppers.append(
                Shopper(
                    name=len(self.all_shoppers) + 1,
                    start_time=start_time,
                    maximum_length_of_stay=length_of_stay
                )
            )
            self.shoppers_arrival_times[i] = start_time

    def _add_new_shopper(self, date_time):
        # This is wrong - there can actually be 20 shoppers made - need to change
        min_date_time = date_time - datetime.timedelta(minutes=self.minutes_per_iteration / 2)
        max_date_time = date_time + datetime.timedelta(minutes=self.minutes_per_iteration / 2)

        shopper_index = np.where(
            (self.shoppers_arrival_times > min_date_time) &
            (self.shoppers_arrival_times <= max_date_time)
        )[0]

        for i in shopper_index:
            self.shoppers.append(
                self.all_shoppers[i]
            )

    def _action_shopper(self):
        for shopper in self.shoppers:
            if shopper.shopping:
                shopper.shop(self.environment, self.minutes_per_iteration)

    def _save_shoppers_history(self):
        self.shoppers_history.append(
            dict(list([str(shopper), str(shopper.location)] for shopper in self.shoppers))
        )

    def _save_signal_history(self, date_time):

        for shopper in self.shoppers:
            if shopper.shopping:
                self.signal_history.append([
                    str(shopper),
                    date_time,
                    str(shopper.location),
                ])

    def create_signal_df(self):
        self.signal_df = pd.DataFrame(self.signal_history, columns=['mac_address', 'date_time', 'store_id'])
        self.signal_df = pd.merge(self.signal_df, self.environment.locations_df[['store_id', 'centroid']], on='store_id', how='left')
        self.signal_df['x'] = self.signal_df.centroid.str[0].astype(float)  # + norm(loc=0, scale=5).rvs(size=len(self.signal_df))
        self.signal_df['y'] = self.signal_df.centroid.str[1].astype(float)  # + norm(loc=0, scale=5).rvs(size=len(self.signal_df))
        del self.signal_df['centroid']

    def create_mac_address_df(self):
        mac_address = []
        for shopper in self.shoppers:
            mac_address.append([str(shopper), len(list(set(shopper.locations_visited))), shopper.length_of_stay*60, shopper.start_time])

        self.mac_address_df = pd.DataFrame(mac_address, columns=['mac_address', 'number_of_shops', 'length_of_stay', 'start_time'])
        self.mac_address_df = self.mac_address_df[self.mac_address_df.mac_address.isin(self.signal_df.mac_address)]
