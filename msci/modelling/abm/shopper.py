import random


class Shopper:
    def __init__(self, name, start_coordinates=(0, 0)):
        """
        Initiates the shopper (agent) with a name and starting coordinates.
        :param name: (str) The name of the shopper.
        :param start_coordinates: (The initial location of the shopper.
        """
        self.name = str(name)
        self.coordinates = start_coordinates
        self.new_coordinates = (0, 0)

    def __str__(self):
        return self.name

    def shop(self, environment):
        """
        The method called to make the shopper perform an action.
        :param environment: (Object) contains the environment of the shop.
        """
        self.move(environment)
        # self.look()
        # self.buy()

        return self.coordinates

    def move(self, environment):
        """
        The shopper moves in a direction.
        :param environment: (Object) contains the environment of the shop.
        """
        can_move = False
        self.new_coordinates = (0, 0)

        while not can_move:
            diff_coordinates = (random.randint(-1, 1), random.randint(-1, 1))
            self.new_coordinates = tuple(sum(x) for x in zip(self.coordinates, diff_coordinates))

            if environment.in_area(self.new_coordinates):
                can_move = environment.is_empty(self.new_coordinates)

        environment.move_shopper(self)
        self.coordinates = self.new_coordinates
