import numpy as np
from tqdm import tqdm_notebook as tqdm


class Environment:
    def __init__(self, locations_df):
        """
        Initiates the environment which is the shopping mall.
        :param locations_df: (pd.DataFrame) The locations in the shopping mall
        """
        self.locations_df = locations_df
        self.locations = self.locations_df.store_id.as_matrix()

        self.locations_list = np.empty((len(self.locations), 0)).tolist()
        self.angle_matrix = np.zeros((len(self.locations), len(self.locations)))

        self.shop_distance_matrix = np.zeros((len(self.locations_df), len(self.locations_df)))
        self.shop_direction_matrix = np.zeros((len(self.locations_df), len(self.locations_df), 2))
        self._create_matrices()

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

    def _create_matrices(self):
        def distance(p1, p2):
            return np.sqrt((float(p2.x) - float(p1.x)) ** 2 + (float(p2.y) - float(p1.y)) ** 2)

        def direction(p1, p2):
            return (float(p2.x) - float(p1.x)), (float(p2.y) - float(p1.y))

        for i in tqdm(range(self.shop_distance_matrix.shape[0])):
            for j in range(self.shop_distance_matrix.shape[1]):
                self.shop_distance_matrix[i][j] = distance(self.locations_df.iloc[i], self.locations_df.iloc[j])
                self.shop_direction_matrix[i][j] = direction(self.locations_df.iloc[i], self.locations_df.iloc[j])


    def shop_transition_angles(self):
        xbar = 0
        ybar = 0
        angles = []
        xpos = self.locations_df.x.tolist()
        ypos = self.locations_df.y.tolist()
        for s in range(len(self.locations)):
            x = xpos[s][0] - xbar
            y = ypos[s][1] - ybar
            if y > 0 and x > 0:
                angles[s] = np.arctan(y/x)
            elif y > 0 and x < 0:
                angles[s] = np.pi - np.arctan(y/np.absolute(x))
            elif y < 0 and x > 0:
                angles[s] = 2*np.pi - np.arctan(np.absolute(y)/x)
            else:
                angles[s] = np.pi + np.arctan(y/x)
        for i in range(len(angles)):
            for j in range(len(angles)):
                if abs(angles[i] - angles[j]) < np.pi:
                    if angles[i] - angles[j] > 0:
                        self.angle_matrix[i][j] = 0
                    else:
                        self.angle_matrix[i][j] = 1
                else:
                    self.angle_matrix[i][j] = 0.5


    def realign_transition_matrix(self, shop_names, transition_matrix, initial_probabilities):
        x = shop_names
        y = self.locations

        index = np.argsort(x)
        sorted_x = x[index]
        sorted_index = np.searchsorted(sorted_x, y)

        yindex = np.take(index, sorted_index, mode="clip")
        mask = x[yindex] != y

        result = np.ma.array(yindex, mask=mask)

        return transition_matrix[result][:, result], initial_probabilities[result]
