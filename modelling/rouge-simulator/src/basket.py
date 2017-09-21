class Basket:
    """
    Creates the basket
    """
    def __init__(self):
        self.basket = []
        
    def add_item(self, item):
        """
        Adds the item from the location to the basket.
        
        :param item: (item) The item which is to be added.
        """
        self.basket.append(item)
        
    def remove_item(self, item):
        """
        Removes the item from the basket.
        
        :param item: (item) The item which is to be added.
        """
        if item not in self.basket:
            raise Exception('This item is not in the basket')
            
        item_index = self.basket.index(item)
        del self.basket[item_index]
        
    def output(self):
        """
        Prints the basket
        
        :return: (List) The basket
        """
        basket_names = [item.name for item in self.basket]
        print_statement = 'The basket contains: {}'.format(basket_names)
        print(print_statement)
