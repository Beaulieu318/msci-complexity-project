class Shop:
    def __init__(self, name, corners):
        self.name = name
        self.walls = self._make_walls(corners)

    def __str__(self):
        return self.name

    @staticmethod
    def _make_walls(corners):
        points = []
        for num in range(len(corners) - 1):
            dx = corners[num + 1][0] - corners[num][0]
            dy = corners[num + 1][1] - corners[num][1]
            if dx is not 0:
                points_x = range(corners[num][0], corners[num + 1][0], int(dx / abs(dx)))
            else:
                points_x = [corners[num + 1][0]] * abs(dy)
            if dy is not 0:
                points_y = range(corners[num][1], corners[num + 1][1], int(dy / abs(dy)))
            else:
                points_y = [corners[num + 1][1]] * abs(dx)

            for point in list(zip(points_x, points_y)):
                points.append(point)
                points.append(corners[num + 1])

        return points

    def add_walls(self, environment):
        y_coord = 0
        for y in environment.area:
            x_coord = 0
            for _ in y:
                if (x_coord, y_coord) in self.walls:
                    environment.area[y_coord][x_coord] = '#'
                x_coord += 1
            y_coord += 1
