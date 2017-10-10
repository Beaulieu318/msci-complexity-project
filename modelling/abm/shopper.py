class Shopper:
    """
    This call defines the shopper (agent).
    """

    def __init__(self, start_coordinates=(0, 0)):
        self.coordinates = start_coordinates

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
