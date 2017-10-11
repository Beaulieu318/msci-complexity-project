class Environment:
    def __init__(self, area):
        self.area = area

    def in_area(self, coordinates):
        x_coord = coordinates[0]
        y_coord = coordinates[1]

        # Coordinates in range of area
        if 0 <= x_coord < len(self.area[0]) and 0 <= y_coord < len(self.area):
            return True
        return False

    def is_empty(self, coordinates):
        x_coord = coordinates[0]
        y_coord = coordinates[1]

        # There is nothing in the location
        contains_nothing = self.area[y_coord][x_coord] == '-'
        if contains_nothing:
            return True
        return False

    def move_shopper(self, shopper):
        shopper_x_coord = shopper.coordinates[0]
        shopper_y_coord = shopper.coordinates[1]

        # Remove shopper from position
        self.area[shopper_y_coord][shopper_x_coord] = '-'

        shopper_new_x_coord = shopper.new_coordinates[0]
        shopper_new_y_coord = shopper.new_coordinates[1]

        # Add shopper to new position
        self.area[shopper_new_y_coord][shopper_new_x_coord] = shopper

    def output(self):
        for y_area in self.area:
            for x__coord in y_area:
                print(str(x__coord), end='')
            print('')
        print('')
