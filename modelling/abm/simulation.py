from environment import Environment
from shopper import Shopper


def simulation(max_iterations, max_shoppers, environment, environment_history, shoppers):

    # Iterate over time
    for iteration in range(max_iterations):

        # Add new shopper
        if len(shoppers) < max_shoppers:
            shoppers.append(Shopper(name=len(shoppers)+1))

        # Make each shopper do something
        for shopper in shoppers:
            shopper.shop(environment)

            # Change the environment depending on what the shopper did
            # environment.change(shopper)

        environment.output()
        # environment_history.append(environment)


def main():
    max_iterations = 20
    max_shoppers = 3
    environment = Environment(area=[
        ['-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-'],
    ])
    environment_history = []
    shoppers = []

    simulation(max_iterations, max_shoppers, environment, environment_history, shoppers)


if __name__ == '__main__':
    main()
