class Environment:
    def __init__(self, area):
        """
        Initiates the environment which is the shopping mall.
        :param area: (list(list)) The shopping mall which has '-' for blank empty locations.
        """
        self.area = area
        self.max_x = len(area[0])
        self.max_y = len(area)

    def in_area(self, coordinates):
        """
        Checks to see if shopper is in the environments area.
        :param coordinates: (tuple) The x and y coordinates of the new location of the shopper.
        :return: (bool) A true/false whether the shopper is in the area.
        """
        x_coord = coordinates[0]
        y_coord = coordinates[1]

        # Coordinates in range of area
        if 0 <= x_coord < len(self.area[0]) and 0 <= y_coord < len(self.area):
            return True
        return False

    def is_empty(self, coordinates):
        """
        Checks to see if the new location is empty.
        :param coordinates: (tuple) The x and y coordinates of the new location of the shopper.
        :return: (bool) A true/false whether the new location is empty.
        """
        x_coord = coordinates[0]
        y_coord = coordinates[1]

        # There is nothing in the location
        contains_nothing = self.area[y_coord][x_coord] == '-'
        if contains_nothing:
            return True
        return False

    def move_shopper(self, shopper):
        """
        Moves the shopper in the environment.
        :param shopper: (object) The shopper object.
        """
        shopper_x_coord = shopper.coordinates[0]
        shopper_y_coord = shopper.coordinates[1]

        # Remove shopper from position
        self.area[shopper_y_coord][shopper_x_coord] = '-'

        shopper_new_x_coord = shopper.new_coordinates[0]
        shopper_new_y_coord = shopper.new_coordinates[1]

        # Add shopper to new position
        self.area[shopper_new_y_coord][shopper_new_x_coord] = shopper

    def find_walls(self):
        """
        Finds the locations of the walls in the area
        :return: (list, list) two list containing the x and y coordinates of the walls
        """
        y_coord = 0

        x_location_of_wall = []
        y_location_of_wall = []
        for y_area in self.area:
            x_coord = 0
            for x_value in y_area:
                if x_value == '#':
                    x_location_of_wall.append(x_coord)
                    y_location_of_wall.append(y_coord)
                x_coord += 1
            y_coord += 1

        return x_location_of_wall, y_location_of_wall

    def output(self):
        """
        Prints a diagram contain the location of the shoppers.
        N.B. This has been superseded by the animation class.
        """
        for y_area in self.area:
            for x__coord in y_area:
                print(str(x__coord), end='')
            print('')
        print('')
