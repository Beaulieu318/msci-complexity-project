import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ShoppersAnimation:
    def __init__(self, signals, area):
        """
        Animates the shoppers.
        :param signals: (list(dict)) a list of times which contain a dictionary of the shoppers location.
        :param area: (list) The maximum x and y coordinates.
        """
        self.signals = signals
        self.area = area
        self.scat = {}
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)

    def _initiate_shopper(self):
        self.ax.set_xlim(0, self.area[0]), self.ax.set_xticks([])
        self.ax.set_ylim(0, self.area[1]), self.ax.set_yticks([])

        for shopper in self.signals[0].items():
            self.scat[shopper[0]] = self.ax.scatter(shopper[1][0], shopper[1][1])

    def _update(self, frame_number):
        for shopper in self.signals[frame_number].items():
            try:
                self.scat[shopper[0]].set_offsets(shopper[1])
            except KeyError:
                self.scat[shopper[0]] = self.ax.scatter(shopper[1][0], shopper[1][1])

    def run(self):
        animation = FuncAnimation(self.fig, self._update, init_func=self._initiate_shopper(),
                                  interval=10, frames=len(self.signals))
        plt.show()
