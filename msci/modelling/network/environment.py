import numpy as np


class Environment:
    def __init__(self, locations_df):
        """
        Initiates the environment which is the shopping mall.
        :param locations: (dict) The locations in the shopping mall
        """
        self.locations_df = locations_df
        self.locations = self.locations_df.store_id.as_matrix()

        self.locations_list = np.empty((len(self.locations), 0)).tolist()

        self.shop_distance_matrix = np.zeros((len(self.locations_df), len(self.locations_df)))
        self._create_shop_distance_matrix()

    def move_shopper(self, shopper):
        """
        Moves the shopper in the environment.
        :param shopper: (object) The shopper object.
        """
        # Remove shopper from position
        if shopper.last_location_index is not None:
            self.locations_list[shopper.last_location_index].remove(str(shopper))

        # Add shopper to new position
        if shopper.location_index is not None:
            self.locations_list[shopper.location_index].append(str(shopper))

    def _create_shop_distance_matrix(self):
        distance = lambda p1, p2: np.sqrt(
            (float(p1.centroid[0]) - float(p2.centroid[0])) ** 2 + (float(p1.centroid[1]) - float(p2.centroid[1])) ** 2
        )

        for i in range(self.shop_distance_matrix.shape[0]):
            for j in range(self.shop_distance_matrix.shape[1]):
                self.shop_distance_matrix[i][j] = distance(self.locations_df.iloc[i], self.locations_df.iloc[j])
