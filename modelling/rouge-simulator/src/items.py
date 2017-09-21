from .basket import Basket
from .shopping_list import ShoppingList


class Item:
    """
    The items class allows objects to be placed onto the map.
    """
    def __init__(self, name='', sign='@', solid=True, pickup=False):
        self.name = name
        self.sign = sign
        self.solid = solid
        self.pickup = pickup
        self.floor = None
        self.position = (None, None)
        
    def add_position(self, floor=0, position=(0, 0)):
        """
        Adds the location to the item.
        
        :param floor: (int) The floor of the item.
        :param position: (tupple(int, int)) The position of the item.
        """
        self.floor = floor
        self.position = position


class Player(Item):
    """
    This is the person class. They can pick stuff up.
    """
    def __init__(self, name=''):
        Item.__init__(self, name=name, sign='@', solid=True, pickup=False)
        
    def create_basket(self):
        self.current_basket = Basket()
        
    def create_shopping_list(self):
        self.current_shopping_list = ShoppingList()


class Wall(Item):
    """
    The wall class builds a wall from one coordinate to another.
    """
    def __init__(self, name=''):
        Item.__init__(self, name=name, sign='#', solid=True, pickup=False)
