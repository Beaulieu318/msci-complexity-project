import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ShoppersAnimation:
    def __init__(self, signals, environment, interval=0, image=None):
        """
        Animates the shoppers.
        :param signals: (list(dict)) a list of times which contain a dictionary of the shoppers location.
        :param area: (list) The maximum x and y coordinates.
        """
        self.signals = signals
        self.environment = environment
        self.interval = interval
        self.scat = {}
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        self.image = image

    def _initiate_animation(self):
        self.ax.set_xlim(0, self.environment.max_x), self.ax.set_xticks([])
        self.ax.set_ylim(0, self.environment.max_y), self.ax.set_yticks([])

        if self.image is not None:
            self.ax.imshow(self.image[::-1], origin='lower', extent=[-77, 470, -18, 255], alpha=0.1)

        self._initiate_wall()
        self._initiate_shopper()

    def _initiate_wall(self):
        x_coord_wall, y_coord_wall = self.environment.find_walls()
        self.scat['wall'] = self.ax.scatter(x_coord_wall, y_coord_wall, marker='x', s=0.5)

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
