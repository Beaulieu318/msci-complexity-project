import random
import datetime

from tqdm import tqdm_notebook as tqdm

from msci.modelling.abm.environment import Environment
from msci.modelling.abm.shopper import Shopper


def simulate(max_iterations, max_shoppers, environment):
    """
    The simulation iterates through time then through each shopper.
    A list at each time with a dictionary of shoppers location is returned.
    :param max_iterations: (int) The maximum number of time steps.
    :param max_shoppers: (int) The maximum number of shoppers.
    :param environment: (object) The environment object which is the mall.
    :return: (list(dict)) A list of each time step containing a dictionary of the shoppers location.
    """

    shoppers = []
    shoppers_history = []
    signal_history = []

    start_date_time = datetime.datetime(year=2016, month=12, day=22, hour=7)

    # Iterate over time
    for iteration in tqdm(range(max_iterations)):

        # Add new shopper
        if len(shoppers) < max_shoppers:
            if random.randint(0, int(iteration/10)) == 0:
                x_coord = random.randint(55, 300)
                y_coord = random.randint(0, 100)
                shoppers.append(Shopper(name=len(shoppers)+1, start_coordinates=(x_coord, y_coord)),)

        # Make each shopper do something
        for shopper in shoppers:
            shopper.shop(environment)

        # Add shopper history to a list of dictionaries
        shoppers_history.append(
            dict(list([str(shopper), shopper.coordinates] for shopper in shoppers))
        )

        date_time = start_date_time + datetime.timedelta(minutes=1)*iteration

        for shopper in shoppers:
            signal_history.append([
                str(shopper), date_time,
                shopper.coordinates[0],
                shopper.coordinates[1]]
            )

    return shoppers_history, signal_history


def main(max_iterations=100, max_shoppers=10):
    environment = Environment(area=[
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],

        ['-' for _ in range(100)] for _ in range(100)
    ])

    return simulate(max_iterations, max_shoppers, environment)


if __name__ == '__main__':
    main()
