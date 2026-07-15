'''Handle external world detection into agent input.'''
import numpy as np
from logic.agents import PreyState
import constants as c
from math import ceil

def prey_get_closest_pred(world):
    '''
    Returns an array of the closest predator which falls within the vision radius for each prey. 
    If there is none or no agent at that index, returns -1 for that idx.
    '''
    pred = world.pred
    prey = world.prey
    closest_preds = np.full(prey.cap, -1)

    for i in np.flatnonzero(world.prey.alive):
        pos_i = prey.pos[i]
        
        r_cells = ceil(c.PREY_VISION_RADIUS/c.CELLSIZE)
        nearby_pred = world.grid.nearby_pred(pos_i, r=r_cells)

        if len(nearby_pred) == 0: # If nothing is nearby skip thru
            continue

        pred_dist_sq = np.sum((pred.pos[nearby_pred] - pos_i)**2, axis=1) # Getting dist sq, could do linalg.norm but it's slower
        
        closest_idx = np.argmin(pred_dist_sq)
        
        if pred_dist_sq[closest_idx] < c.PREY_VISION_RADIUS**2:
            closest_preds[i] = nearby_pred[closest_idx]

    return closest_preds

def prey_get_closest_prey(world):
    '''
    Returns an array of the closest prey which falls within the vision radius for each prey. 
    If there is none or no agent at that index, returns -1 for that idx.
    Note: excludes self as viable option.
    '''
    prey = world.prey
    closest_prey = np.full(prey.cap, -1)

    for i in np.flatnonzero(world.prey.alive):
        pos_i = prey.pos[i]
        
        r_cells = ceil(c.PREY_VISION_RADIUS/c.CELLSIZE)
        nearby_prey = np.array(world.grid.nearby_prey(pos_i, r=r_cells), dtype=np.int32)
        nearby_prey = nearby_prey[nearby_prey != i] # exclude self

        if len(nearby_prey) == 0: # If nothing is nearby skip thru
            continue

        prey_dist_sq = np.sum((prey.pos[nearby_prey] - pos_i)**2, axis=1) # Getting dist sq, could do linalg.norm but it's slower
        
        closest_idx = np.argmin(prey_dist_sq)
        
        if prey_dist_sq[closest_idx] < c.PREY_VISION_RADIUS**2:
            closest_prey[i] = nearby_prey[closest_idx]

    return closest_prey

def prey_get_closest_food(world):
    '''
    Returns an array of the closest food which falls within the vision radius for each prey. 
    If there is none or no agent at that index, returns -1 for that idx.
    '''
    plant = world.plant
    prey = world.prey
    closest_plants = np.full(prey.cap, -1)

    for i in np.flatnonzero(world.prey.alive):
        pos_i = prey.pos[i]
        
        r_cells = ceil(c.PREY_VISION_RADIUS/c.CELLSIZE)
        nearby_plant = np.array(world.grid.nearby_plant(pos_i, r=r_cells), dtype=np.int32)
        nearby_plant = nearby_plant[plant.alive[nearby_plant] == 1] # Only get EXISTING plants
                                                                    # FYI nearby_plant supposedly manages this,
                                                                    # but good in case of stale grid

        if len(nearby_plant) == 0: # If nothing is nearby skip thru
            continue

        plant_dist_sq = np.sum((plant.pos[nearby_plant] - pos_i)**2, axis=1) # Getting dist sq, could do linalg.norm but it's slower
        
        closest_idx = np.argmin(plant_dist_sq)
        
        if plant_dist_sq[closest_idx] < c.PREY_VISION_RADIUS**2:
            closest_plants[i] = nearby_plant[closest_idx]

    return closest_plants