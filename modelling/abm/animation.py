import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ShoppersAnimation:
    def __init__(self, signals):
        self.signals = signals

        self.fig = plt.figure(figsize=(7, 7))
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, 50), ax.set_xticks([])
        ax.set_ylim(0, 50), ax.set_yticks([])

        self.scat = {}

        for shopper in self.signals[-1].items():
            self.scat[shopper[0]] = ax.scatter(shopper[1][0], shopper[1][1])

    def update(self, frame_number):
        time.sleep(0.1)
        for shopper in self.signals[frame_number].items():
            self.scat[shopper[0]].set_offsets(shopper[1])

    def run(self):
        animation = FuncAnimation(self.fig, self.update, interval=10, frames=len(self.signals))
        plt.show()


# visualisation = ShoppersAnimation(
    # [
    #     {'1': [0,  5], '2': [3,  3]},
    #     {'1': [3, 5], '2': [4, 4]},
    #     {'1': [3, 6], '2': [4, 7]},
    # ]
# )
# visualisation.run()
