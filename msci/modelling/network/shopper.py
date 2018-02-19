import random
from scipy.integrate import quad


class Shopper:
    def __init__(self, name, start_time, leave_distribution):
        """
        Initiates the shopper (agent) with a name and starting coordinates.
        :param name: (str) The name of the shopper.
        """
        self.name = str(name)
        self.location = None
        self.last_location = None
        self.locations_visited = []
        self.shopping = True
        self.start_time = start_time
        self.length_of_stay = 0
        self.iterations = 0

        self.leave_distribution = leave_distribution

    def __str__(self):
        return self.name

    def shop(self, environment, minutes_per_iteration):
        """
        The method called to make the shopper perform an action.
        :param environment: (Object) contains the environment of the shop.
        """
        if self.shopping:

            probability_leaving = self.leave_distribution.cdf(self.length_of_stay)
            # probability_leaving = quad(self.leave_distribution, 0, self.length_of_stay)[0]

            if random.random() < probability_leaving:
                self.leave(environment)
            else:
                self.move(environment)
                # self.look()
                # self.buy()

                self.iterations += 1
                self.length_of_stay = self.iterations * minutes_per_iteration

        return self.location

    def move(self, environment):
        """
        The shopper moves in a direction.
        :param environment: (Object) contains the environment of the shop.
        """
        self.last_location = self.location
        self.location = environment.locations[random.randint(0, len(environment.locations)-1)]

        if self.location != self.last_location:
            environment.move_shopper(self)
            self.locations_visited.append(self.location)

    def leave(self, environment):
        self.last_location = self.location
        self.location = None

        if self.location != self.last_location:
            environment.move_shopper(self)

        self.shopping = False
