import random
import datetime
import pandas as pd

from tqdm import tqdm_notebook as tqdm

from msci.modelling.network.shopper import Shopper


class NetworkSimulation:
    def __init__(self, environment, max_shoppers):
        self.environment = environment
        self.max_shoppers = max_shoppers

        self.shoppers = []
        self.shoppers_history = []
        self.signal_history = []

        self.start_date_time = datetime.datetime(year=2016, month=12, day=22, hour=7)
        self.end_date_time = datetime.datetime(year=2016, month=12, day=22, hour=23)
        self.total_minutes = (self.end_date_time - self.start_date_time).seconds / 60
        self.minutes_per_iteration = 0

        self.signal_df = None
        self.mac_address_df = None

        self.shopper_leave_distribution = None

    def iterate(self, max_iterations):

        self.minutes_per_iteration = self.total_minutes / max_iterations

        for iteration in tqdm(range(max_iterations)):
            date_time = self.start_date_time + datetime.timedelta(minutes=self.minutes_per_iteration) * iteration

            self._add_new_shopper(date_time)
            self._action_shopper()
            self._save_shoppers_history()
            self._save_signal_history(date_time)

    def _add_new_shopper(self, date_time):
        if len(self.shoppers) < self.max_shoppers:
            number_of_shoppers_added = random.randint(0, 20)
            for i in range(number_of_shoppers_added):
                self.shoppers.append(
                    Shopper(
                        name=len(self.shoppers) + 1,
                        start_time=date_time,
                        leave_distribution=self.shopper_leave_distribution)
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
                    str(shopper.location)]
                )

    def create_signal_df(self):
        self.signal_df = pd.DataFrame(self.signal_history, columns=['mac_address', 'date_time', 'store_id'])

    def create_mac_address_df(self):
        mac_address = []
        for shopper in self.shoppers:
            mac_address.append([str(shopper), len(list(set(shopper.locations_visited))), shopper.length_of_stay, shopper.start_time])

        self.mac_address_df = pd.DataFrame(mac_address, columns=['mac_address', 'number_of_shops', 'length_of_stay', 'start_time'])

    def distributions(self):
        self.shopper_leave_distribution = None
