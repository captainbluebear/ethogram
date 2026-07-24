"""Handle all eating behavior."""
import numpy as np
from constants import AGENT_SIZE, PLANT_ENERGY, PREY_ENERGY
from logic.state import WorldState

def step_consumption(world):
    step_consume_prey(world)
    step_consume_pred(world)

def step_consume_prey(world:WorldState):
    '''Define interaction for prey eating.'''
    for i in np.flatnonzero(world.prey.alive):
        for a in world.grid.nearby_plant(world.prey.pos[i]):
            if not world.plant.alive[a]: # avoid plant being eaten by multiple prey (grid only updates per update rate)
                continue

            delta = world.prey.pos[i] - world.plant.pos[a]
            d2 = np.sum(delta**2)
            if d2 < (AGENT_SIZE*2)**2:
                world.plant.alive[a] = False    # die!
                world.plant.growth[a] = 0       # restart growth timer
                world.prey.energy[i] += PLANT_ENERGY

def step_consume_pred(world:WorldState):
    '''Define interaction for predators eating.'''
    for i in np.flatnonzero(world.pred.alive):
        for a in world.grid.nearby_prey(world.pred.pos[i]):
            if not world.prey.alive[a]: # avoid prey being eaten by multiple pred
                continue

            delta = world.pred.pos[i] - world.prey.pos[a]
            d2 = np.sum(delta**2)
            if d2 < (AGENT_SIZE*2)**2:
                world.prey.remove_agent(int(a))      # Handle prey death
                world.pred.energy[i] += PREY_ENERGY