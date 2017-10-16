import simulation

from animation import ShoppersAnimation


def animate():
    shoppers_history = simulation.main(max_iterations=1000, max_shoppers=10)

    anim = ShoppersAnimation(shoppers_history, area=[50, 50])
    anim.run()


if __name__ == '__main__':
    animate()
