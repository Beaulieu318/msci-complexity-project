class Location:
    def __init__(self, name):
        self.name = name
        self.shoppers = []

    def __str__(self):
        return self.name
