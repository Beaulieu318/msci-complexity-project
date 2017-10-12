from environment import Environment
from shopper import Shopper
from animation import ShoppersAnimation


def simulation(max_iterations, max_shoppers, environment, shoppers, shoppers_history):
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


def main():
    max_iterations = 1000
    max_shoppers = 10
    environment = Environment(area=[
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],
        # ['-', '-', '-', '-', '-'],

        ['-' for _ in range(100)] for _ in range(100)
    ])

    shoppers = []
    shoppers_history = []

    simulation(max_iterations, max_shoppers, environment, shoppers, shoppers_history)

    anim = ShoppersAnimation(shoppers_history)
    anim.run()


if __name__ == '__main__':
    main()
