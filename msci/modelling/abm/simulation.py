from environment import Environment
from shopper import Shopper


def simulation(max_iterations, max_shoppers, environment):
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

    # Iterate over time
    for iteration in range(max_iterations):

        # Add new shopper
        if len(shoppers) < max_shoppers:
            shoppers.append(Shopper(name=len(shoppers)+1))

        # Make each shopper do something
        for shopper in shoppers:
            shopper.shop(environment)

        # Add shopper history to a list of dictionaries
        shoppers_history.append(
            dict(list([str(shopper), shopper.coordinates] for shopper in shoppers))
        )

    return shoppers_history


def main(max_iterations=100, max_shoppers=10):
    environment = Environment(area=[
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],

        ['-' for _ in range(100)] for _ in range(100)
    ])

    return simulation(max_iterations, max_shoppers, environment)


if __name__ == '__main__':
    main()
