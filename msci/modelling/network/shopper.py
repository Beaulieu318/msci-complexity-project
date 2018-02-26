import random
import numpy as np


class Shopper:
    def __init__(self, name, start_time, maximum_length_of_stay):
        """
        Initiates the shopper (agent) with a name and starting coordinates.
        :param name: (str) The name of the shopper.
        """
        self.name = str(name)

        self.location = None
        self.last_location = None
        self.location_index = None
        self.last_location_index = None

        self.locations_visited = []
        self.shopping = True
        self.start_time = start_time
        self.length_of_stay = 0
        self.iterations = 0

        self.maximum_length_of_stay = maximum_length_of_stay

    def __str__(self):
        return self.name

    def shop(self, environment, minutes_per_iteration):
        """
        The method called to make the shopper perform an action.
        :param environment: (Object) contains the environment of the shop.
        :param minutes_per_iteration: (float) number of minutes per iteration.
        """
        if self.shopping:

            # probability_leaving = self.leave_distribution.pdf(self.length_of_stay)
            # probability_leaving = quad(self.leave_distribution, 0, self.length_of_stay)[0]

            if self.maximum_length_of_stay < self.length_of_stay:
                self.leave(environment)
            else:
                self.move(environment, minutes_per_iteration)
                # self.look()
                # self.buy()

                self.iterations += 1
                self.length_of_stay = self.iterations * minutes_per_iteration

        return self.location

    def move(self, environment, minutes_per_iteration):
        """
        The shopper moves in a direction.
        :param environment: (Object) contains the environment of the shop.
        :param minutes_per_iteration: (float) number of minutes per iteration.
        """
        max_speed_meters_per_minute = 83
        max_speed_meters_per_iteration = max_speed_meters_per_minute * minutes_per_iteration

        probability_leaving_shop = 0.05

        self.last_location = self.location
        self.last_location_index = self.location_index

        if self.last_location is None:
            # First shop visited
            self.location_index = random.randint(0, len(environment.locations)-1)
            self.location = environment.locations[self.location_index]

        else:
            # Next shops visited

            # Probability of leaving a shop
            if random.random() < probability_leaving_shop:

                # Max speed limitation
                possible_locations_index = np.where(
                    environment.shop_distance_matrix[self.last_location_index] < max_speed_meters_per_iteration
                )[0]
                self.location_index = np.random.choice(possible_locations_index)
                self.location = environment.locations[self.location_index]

        if self.location != self.last_location:
            environment.move_shopper(self)
            self.locations_visited.append(self.location)

    def leave(self, environment):
        self.last_location = self.location
        self.last_location_index = self.location_index

        self.location = None

        if self.location != self.last_location:
            environment.move_shopper(self)

        self.shopping = False
