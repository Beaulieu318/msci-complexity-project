import random
import numpy as np


class Shopper:
    def __init__(self, name, start_time, maximum_length_of_stay, A=None, pi=None):
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
        self.locations_index_visited = []
        self.shopping = True
        self.start_time = start_time
        self.length_of_stay = 0
        self.iterations = 0

        self.maximum_length_of_stay = maximum_length_of_stay

        self.direction = np.zeros(2)

        self.A = A
        self.pi = pi

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

        probability_leaving_shop = 0.050

        self.last_location = self.location
        self.last_location_index = self.location_index

        if self.last_location is None:
            # First shop visited
            possible_locations_index = range(len(environment.locations))
            self.location_index = np.random.choice(possible_locations_index, p=self.pi / sum(self.pi))
            self.location = environment.locations[self.location_index]

        else:
            # Probability of leaving a shop
            if random.random() < probability_leaving_shop:

                possible_locations_index = list(range(len(environment.locations)))
                possible_locations_index.remove(self.location_index)
                possible_locations_index = np.array(possible_locations_index)

                # Max speed limitation
                # possible_locations_index = possible_locations_index[
                #     np.where(
                #         environment.shop_distance_matrix[self.last_location_index][possible_locations_index] < max_speed_meters_per_iteration
                #     )[0]
                # ]

                # Probabilistic measures
                probabilities = np.ones(len(possible_locations_index))

                # Increase probability for forwards direction
                # r1r2 = np.dot(
                #     environment.shop_direction_matrix[self.last_location_index][possible_locations_index],
                #     self.direction,
                # )
                # probabilities[np.where(r1r2 == 0)] = 0.5
                # probabilities[np.where(r1r2 > 0)] = 0.50
                # probabilities[np.where(r1r2 < 0)] = 0.50

                # Transition probabilities from real data
                probabilities = self.A[self.last_location_index][possible_locations_index]

                # Decrease probability to return to a shop
                # probabilities[np.isin(possible_locations_index, self.locations_index_visited)] *= 0.05
                # probabilities[~np.isin(possible_locations_index, self.locations_index_visited)] *= 0.95

                # Next shops visited
                self.location_index = np.random.choice(possible_locations_index, p=probabilities/sum(probabilities))
                self.location = environment.locations[self.location_index]

        if self.location != self.last_location:
            if self.last_location is not None:
                self.direction = environment.shop_direction_matrix[self.last_location_index][self.location_index]
            environment.move_shopper(self)
            self.locations_visited.append(self.location)
            self.locations_index_visited.append(self.location_index)

    def leave(self, environment):
        self.last_location = self.location
        self.last_location_index = self.location_index

        self.location = None

        if self.location != self.last_location:
            environment.move_shopper(self)

        self.shopping = False
