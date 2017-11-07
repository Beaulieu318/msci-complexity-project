import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg

dir_path = os.path.dirname(os.path.realpath(__file__))


class RealShoppersAnimation:
    def __init__(self, signals, interval=0):
        """
        Animates the shoppers.
        :param signals: (list(dict)) a list of times which contain a dictionary of the shoppers location.
        """
        self.signals = signals
        self.interval = interval
        self.scat = {}
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)

    def _initiate_animation(self):
        img = mpimg.imread(dir_path + '/../images/mall_of_mauritius_map.png')
        self.ax.imshow(img[::-1], origin='lower', extent=[-77, 470, -18, 255], alpha=0.1)

        self.ax.set_xlim(0, 350), self.ax.set_xticks([])
        self.ax.set_ylim(0, 200), self.ax.set_yticks([])

        self._initiate_shopper()

    def _initiate_shopper(self):
        for shopper in self.signals[0].items():
            self.scat[shopper[0]] = self.ax.scatter(shopper[1][0], shopper[1][1])

    def _update(self, frame_number):
        for shopper in self.signals[frame_number].items():
            try:
                self.scat[shopper[0]].set_offsets(shopper[1])
            except KeyError:
                self.scat[shopper[0]] = self.ax.scatter(shopper[1][0], shopper[1][1])

    def run(self):
        animation = FuncAnimation(self.fig, self._update, init_func=self._initiate_animation(),
                                  interval=self.interval, frames=len(self.signals))
        return animation
