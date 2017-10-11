import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ShoppersAnimation:
    def __init__(self):
        self.fig = plt.figure(figsize=(7, 7))
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, 10), ax.set_xticks([])
        ax.set_ylim(0, 10), ax.set_yticks([])

        self.shoppers_position = [np.array([[0,  0], [1,  1]]), np.array([[3,  3], [4,  4]])]

        self.scat = []
        for shopper in self.shoppers_position:
            self.scat.append(ax.scatter(shopper[0][0], shopper[0][1], marker='1'))

    def update(self, frame_number):
        time.sleep(0.1)
        for shopper_num in range(len(self.shoppers_position)):
            self.scat[shopper_num].set_offsets(self.shoppers_position[shopper_num][frame_number])

    def run(self):
        animation = FuncAnimation(self.fig, self.update, interval=10, frames=2)
        plt.show()


visualisation = ShoppersAnimation()
visualisation.run()
