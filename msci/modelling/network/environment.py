
class Environment:
    def __init__(self, locations):
        """
        Initiates the environment which is the shopping mall.
        :param locations: (dict) The locations in the shopping mall
        """
        self.locations = locations

    def move_shopper(self, shopper):
        """
        Moves the shopper in the environment.
        :param shopper: (object) The shopper object.
        """

        # Remove shopper from position
        if shopper.last_location is not None:
            last_location_index = self.locations.index(shopper.last_location)
            self.locations[last_location_index].shoppers.remove(shopper)

        # Add shopper to new position
        if shopper.location is not None:
            location_index = self.locations.index(shopper.location)
            self.locations[location_index].shoppers.append(shopper)
