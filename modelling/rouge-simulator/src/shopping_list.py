class shopping_list:
    """
    Creates the shopping list
    """
    def __init__(self):
        self.shopping_list = []
        
    def add_item(self, item):
        """
        Adds the item from the location to the shopping list.
        
        :param item: (item) The item which is to be added.
        """
        self.shopping_list.append(item)
        
    def remove_item(self, item):
        """
        Removes the item from the shopping list.
        
        :param item: (item) The item which is to be added.
        """
        if item not in self.shopping_list:
            raise Exception('This item is not in the shopping list')
            
        item_index = self.shopping_list.index(item)
        del self.shopping_list[item_index]
        
    def output(self):
        """
        Prints the shopping list.
        
        :return: (List) The shopping list
        """
        shopping_list_names = [item.name for item in self.shopping_list]
        print_statement = 'The shopping list contains: {}'.format(shopping_list_names)
        print(print_statement)