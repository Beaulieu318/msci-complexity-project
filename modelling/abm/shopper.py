import random


class Shopper:
    """
    This call defines the shopper (agent).
    """

    def __init__(self, name, start_coordinates=(0, 0)):
        self.name = str(name)
        self.coordinates = start_coordinates

    def __str__(self):
        return self.name

    def shop(self, environment):
        """
        The shopper can do 1 of three options in each iteration:
         - Look
         - Move
         - Buy

        These are given a distribution.

        :param environment: (Object) This contains the environment of the shop.
        """
        self.move(environment)
        # self.look()
        # self.buy()

        return self.coordinates

    def move(self, environment):
        """
        The shopper moves in a direction.
        """
        can_move = False
        self.new_coordinates = (0, 0)

        while not can_move:
            diff_coordinates = (random.randint(-1, 1), random.randint(-1, 1))
            self.new_coordinates = tuple(sum(x) for x in zip(self.coordinates, diff_coordinates))

            in_area = environment.in_area(self.new_coordinates)
            if in_area:
                can_move = environment.is_empty(self.new_coordinates)

        environment.move_shopper(self)
        self.coordinates = self.new_coordinates
