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

            # Change the environment depending on what the shopper did
            # environment.change(shopper)

        # environment.output()
        shoppers_history.append(
            dict(list([str(shopper), shopper.coordinates] for shopper in shoppers))
        )


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

    shoppers = []
    shoppers_history = []

    simulation(max_iterations, max_shoppers, environment, shoppers, shoppers_history)

    anim = ShoppersAnimation(shoppers_history)
    anim.run()


if __name__ == '__main__':
    main()
