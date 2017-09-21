class Mapper:
    """
    The map class allows a map to be built. The map contains multiple floors and a square area.
    """
    def __init__(self):
        """
        Dictionary which contains the map.
        """
        self.map = dict()
        
    def add_level(self, floor=0):
        """
        Add a floor to the map dictionary
        
        :param floor: (int) The floor which the new floor is on.
        """
        self.map[floor] = []
        
    def add_area(self, floor=0, dimensions=(10, 10)):
        """
        Add a floor area to the floor by setting the dimesions of the square area.
        
        :param floor: (int) The floor which the area is being added to.
        :param dimensions: (tuple(int, int)) The dimesions of the area of the floor (horizontal/x, vertical/y)
        """
        if floor not in self.map:
            level_error_msg = 'floor {} does not exist.'.format(floor)
            raise Exception(level_error_msg)
            
        area = [[['-'] for _ in range(dimensions[0])] for _ in range(dimensions[1])]
        
        self.map[floor] = area
        
    def output(self, floor=0):
        """
        Returns the floor that the map is on.
        
        :param floor: (int) The floor which the area is being outputted
        :return: (List[List[]]) The floor
        """
        for y in range(len(self.map[floor])):
            for x in range(len(self.map[floor][y])):
                if self.map[floor][y][x][-1] == '-':
                    print(self.map[floor][y][x][-1],)
                else:
                    print(self.map[floor][y][x][-1].sign,)
            print()
