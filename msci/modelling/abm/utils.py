import numpy as np

import matplotlib.path as mpltPath


def find_store_id(sim_signal_df, shops):
    shops_inside = []
    for shop in shops[1:]:
        path = mpltPath.Path(shop.corners)
        inside = path.contains_points(sim_signal_df.xy.tolist())
        shops_inside.append(inside)

    shops_inside = np.array(shops_inside)

    signals_shops = []
    for signal_inside in shops_inside.T:

        shops_visited_index = np.where(signal_inside == True)[0]

        if len(shops_visited_index) == 0:
            shop_visited = np.nan
        elif len(shops_visited_index) == 1:
            shop_visited = str(shops[shops_visited_index[0]])
        else:
            shop_visited = 'more than 1 shop'

        signals_shops.append(shop_visited)

    return signals_shops
