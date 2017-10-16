import pandas as pd

from animation import ShoppersAnimation


def reformat_data():
    shopper_df = pd.read_csv('../../data/bag_mus_12-22-2016.csv')
    shopper_df.date_time = shopper_df.date_time.astype('datetime64[ns]')

    mac_group_df = shopper_df.groupby('mac_address')
    large_mac_df = mac_group_df.size()[mac_group_df.size() > 1000]
    large_shopper_df = shopper_df[shopper_df.mac_address.isin(large_mac_df.index.tolist())]

    large_shopper_df.index = large_shopper_df.date_time

    large_shopper_clean = large_shopper_df[['mac_address','x', 'y']]
    large_shopper_clean['coords'] = list(zip(large_shopper_clean.x, large_shopper_clean.y))
    large_shopper_clean = large_shopper_clean[['mac_address', 'coords']]

    return large_shopper_clean.groupby('date_time').apply(_create_group).tolist()


def _create_group(items):
    values = {}
    for mac, coord in list(zip(items.mac_address.tolist(), items.coords.tolist())):
        values[mac] = coord
    return values


def animate():
    shoppers_history = reformat_data()

    anim = ShoppersAnimation(shoppers_history, area=[300, 300])
    anim.run()


if __name__ == '__main__':
    animate()
