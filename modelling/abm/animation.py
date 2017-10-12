import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ShoppersAnimation:
    def __init__(self, signals):
        self.signals = signals

        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        self.ax.set_xlim(0, 100), self.ax.set_xticks([])
        self.ax.set_ylim(0, 100), self.ax.set_yticks([])

        self.scat = {}

        for shopper in self.signals[0].items():
            self.scat[shopper[0]] = self.ax.scatter(shopper[1][0], shopper[1][1])

    def update(self, frame_number):
        for shopper in self.signals[frame_number].items():
            try:
                self.scat[shopper[0]].set_offsets(shopper[1])
            except KeyError:
                self.scat[shopper[0]] = self.ax.scatter(shopper[1][0], shopper[1][1])

    def run(self):
        animation = FuncAnimation(self.fig, self.update, interval=10, frames=len(self.signals))
        plt.show()
