import sys

import Tkinter as tk

import src.controller as ct
import src.map as mp
import src.items as it

# Generate map
current_map = mp.mapper()

# Generate walls
wall_coordinates = zip([4]*5, range(5))
wall_block = it.wall(name='wall')
wall_blocks = [it.wall(name='wall') for _ in range(len(wall_coordinates))]
for wall_coordinate, wall_block in zip(wall_coordinates, wall_blocks):
    wall_block.add_position(floor=0, position=wall_coordinate)

# Generate shopping items
milk = it.item(name='milk', sign='m', solid=False, pickup=True)
milk.add_position(floor=0, position=(1,0))
bread = it.item(name='bread', sign='b', solid=False, pickup=True)
bread.add_position(floor=0, position=(4,5))
shopping_items = [milk, bread]

# Generate player
player = it.player(name='player')
player.add_position(floor=0, position=(2,0))
player.create_basket()
player.create_shopping_list()
player.current_shopping_list.add_item(milk)

# Create controller
current_controller = ct.controller()
current_controller.create_map(map=current_map, floor=0, dimensions=(10, 10))

for wall_block in wall_blocks:
    current_controller.add_item(item=wall_block)

for shopping_item in shopping_items:
    current_controller.add_item(item=shopping_item)
    
current_controller.add_item(item=player)

# Render map with items
current_controller.current_map.output(floor=0)

def key(event):
    """shows key or tk code for the key"""
    if event.keysym == 'Escape':
        root.destroy()
    
    if event.keysym == 'Up':
        current_controller.move_item(item=player, diff_level=0, diff_position=(0,1))
    
    if event.keysym == 'Down':
        current_controller.move_item(item=player, diff_level=0, diff_position=(0,-1))
        
    if event.keysym == 'Left':
        current_controller.move_item(item=player, diff_level=0, diff_position=(-1,0))
        
    if event.keysym == 'Right':
        current_controller.move_item(item=player, diff_level=0, diff_position=(1,0))
        
    if event.char == 'p':
        current_controller.pickup_item(item=player)
    
    # Render changed map
    print
    current_controller.items[-1].current_shopping_list.output()
    current_controller.items[-1].current_basket.output()
    current_controller.current_map.output(floor=0)
    
    
root = tk.Tk()
root.bind_all('<Key>', key)
root.withdraw()
root.mainloop()