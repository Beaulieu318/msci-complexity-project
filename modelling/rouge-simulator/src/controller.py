class controller:
    """
    The map class allows a map to be built. The map contains multiple floors and a square area.
    """
    def __init__(self):
        """
        Dictionary which contains the map.
        """
        self.items = []
        
    def create_map(self, map, floor=0, dimensions=(10, 10)):
        """
        Create the map
        
        :param map: (mapper) The map object which contains the map
        :param floor: (int) The floor which the area is being added to.
        :param dimesions: (tuple(int, int)) The dimesions of the area of the floor (horizontal/x, vertical/y)
        """
        self.current_map = map
        self.current_map.add_level(floor)
        self.current_map.add_area(floor, dimensions)
        
    def add_item(self, item):
        """
        Adds a location to the players location.
        
        :param item: (item) The item which is being added.
        """
        if item.floor == None or item.position == (None, None):
            raise Exception('floor or position has not been set')
        
        if item.floor not in self.current_map.map:
            level_error_msg = 'floor {} does not exist.'.format(item.floor)
            raise Exception(level_error_msg)
            
        max_x = len(self.current_map.map[item.floor][0])
        max_y = len(self.current_map.map[item.floor])
        if item.position[0] > max_x or item.position[1] > max_y:
            raise Exception('The position is not within the area of the floor')
            
        map_position = self.current_map.map[item.floor][item.position[1]][item.position[0]]
        if map_position == ['-']:
            del map_position[-1]
            map_position.append(item)
            self.items.append(item)
        elif not map_position[-1].solid:
            map_position.append(item)
            self.items.append(item)
        else:
            raise Exception('This position is already taken')
        
    def move_item(self, item, diff_level=0, diff_position=(0,0)):
        """
        Moves an item by a floor or an amount.
        
        :param item: (item) The item which is being moved.
        :param diff_level: (int) The floor amount the floor is being changed
        :param diff_position: (tuple(int, int)) The amount the position is being changed.
        :return: () The function way return nothing.
        """        
        if item not in self.items:
            raise Exception('Item does not exist and therefore cannot be moved')
            
        item_index = self.items.index(item)
        item = self.items[item_index]
        new_level = item.floor + diff_level
        new_position = (item.position[0] + diff_position[0], item.position[1] - diff_position[1]) # Check up on this (dodgy!!!)
        
        # Check if we are moving outside the area
        max_x = len(self.current_map.map[item.floor][0]) - 1
        max_y = len(self.current_map.map[item.floor]) - 1
        if new_position[0] < 0 or new_position[0] > max_x or new_position[1] < 0 or new_position[1] > max_y:
            return
        
        old_map_position = self.current_map.map[item.floor][item.position[1]][item.position[0]]
        new_map_position = self.current_map.map[new_level][new_position[1]][new_position[0]]
        
        # Check if move can be made and move to the next position
        moved = False
        if new_map_position == ['-']:
            del new_map_position[-1]
            new_map_position.append(item)
            item.floor = new_level
            item.position = new_position
            moved = True
        elif not new_map_position[-1].solid:
            new_map_position.append(item)
            item.floor = new_level
            item.position = new_position
            moved = True
        else:
            return
            
        # Edit last position if move has been made
        if moved:
            if len(old_map_position) == 1:
                del old_map_position[-1]
                old_map_position.append('-')
            else:
                del old_map_position[-1]
                
    def pickup_item(self, item):
        """
        Allows items to be picked up and put into a bag.
        
        :param item: (item) The item which is being moved.
        :return: () The function way return nothing.
        """
        if item not in self.items:
            raise Exception('Item does not exist and therefore cannot be moved')
            
        if str(item.__class__) != 'src.items.player':
            raise Exception('Item is not a player and therefore cannot pick up an object')
            
        item_index = self.items.index(item)
        item = self.items[item_index]
        position = self.current_map.map[item.floor][item.position[1]][item.position[0]]
        
        # Check if there is another item
        if len(position) == 1:
            return
            
        # Check if item can be picked up
        if position[-2].pickup == False:
            return
            
        # Remove item from shopping list if it is in there
        shopping_list = position[-1].current_shopping_list.shopping_list
        if position[-2] in shopping_list:
            item_shopping_list_index = shopping_list.index(position[-2])
            del shopping_list[item_shopping_list_index]
            
        # Pick item up and remove from map
        position[-1].current_basket.add_item(position[-2])
        del position[-2]
