"""Handle all life/death logic, including reproduction."""
import numpy as np
from constants import AGENT_SIZE, REGROWTH_TIME, REPRO_COST, MATE_THRESHOLD, PREY_REFRACTORY_GAP, PRED_REFRACTORY_GAP
from constants import State
from logic.state import WorldState

def step_lifecycle(world:WorldState, dt):
    plant_reproduction(world, dt)
    prey_reproduction(world)
    pred_reproduction(world)
    step_death(world, dt)

def plant_reproduction(world:WorldState, dt): # Handle plants coming back to life
    alive = world.plant.alive
    growth = world.plant.growth
    
    dead = ~alive
    if not dead.any(): # Skip if no dead plants
        return
    
    growth[dead] += dt # Increment time dead

    regrow = dead & (growth >= REGROWTH_TIME)
    
    if not regrow.any():
        return
    # world.plant.pos[regrow] = np.random.rand(regrow.sum(), 2) * [world.width, world.height] # random regrowth location
    alive[regrow] = True
    growth[regrow] = 0

def prey_reproduction(world:WorldState):
    '''Returns a set of all prey that mated.'''
    prey_pos = world.prey.pos
    prey_energy = world.prey.energy
    prey_refractory = world.prey.refractory
    state = world.prey.state
    prey = world.prey
    distsq = (AGENT_SIZE*2)**2
    already_mated = set()

    for p1 in np.flatnonzero(world.prey.alive):
        if p1 in already_mated or state[p1] != State.MATE: # Skip mated agents and agents not looking to mate
            continue

        for p2 in world.grid.nearby_prey(world.prey.pos[p1]):
            # Avoid dead prey reproducing (since grid only updates per update rate), avoid already-mated agents, avoid
            # agents not looking to mate, avoid self
            if not world.prey.alive[p2] or p2 in already_mated or state[p2] != State.MATE or p2 == p1:
                continue

            delta = world.prey.pos[p1] - world.prey.pos[p2]
            d2 = np.sum(delta**2)
            if d2 < distsq:
                prey.add_agent(prey_pos[p1], 0, 0) # Spawn at p1's position
                
                ### Parent Operations ###
                prey_energy[p1] -= REPRO_COST           # Reduce energy after mate
                prey_energy[p2] -= REPRO_COST

                prey_refractory[p1] -= PREY_REFRACTORY_GAP   # Increase refractory period again
                prey_refractory[p2] -= PREY_REFRACTORY_GAP

                already_mated.add(p1)                   # Avoid double mating
                already_mated.add(p2)

def pred_reproduction(world:WorldState):
    '''Returns a set of all pred that mated.'''
    pred_pos = world.pred.pos
    pred_energy = world.pred.energy
    pred_refractory = world.pred.refractory
    state = world.pred.state
    pred = world.pred
    pred_alive = world.pred.alive
    distsq = (AGENT_SIZE*2)**2
    already_mated = set()

    for p1 in np.flatnonzero(world.pred.alive):
        if p1 in already_mated or state[p1] != State.MATE: # Skip mated agents and agents not looking to mate
            continue

        for p2 in world.grid.nearby_pred(pred_pos[p1]):
            # Avoid dead prey reproducing (since grid only updates per update rate), avoid already-mated agents, avoid
            # agents not looking to mate
            if not pred_alive[p2] or p2 in already_mated or state[p2] != State.MATE or p2 == p1:
                continue

            delta = world.pred.pos[p1] - world.pred.pos[p2]
            d2 = np.sum(delta**2)
            if d2 < distsq:
                pred.add_agent(pred_pos[p1], 0, 0) # Spawn at p1's position
                # print("Mated")
                
                ### Parent Operations ###
                pred_energy[p1] -= REPRO_COST           # Reduce energy after mate
                pred_energy[p2] -= REPRO_COST

                pred_refractory[p1] -= PRED_REFRACTORY_GAP   # Increase refractory period again
                pred_refractory[p2] -= PRED_REFRACTORY_GAP

                already_mated.add(p1)                   # Avoid double mating
                already_mated.add(p2)


def step_death(world:WorldState, dt):
    """If energy drops to 0, kill agent."""
    prey_dead = (world.prey.alive) & (world.prey.energy <= 0)
    pred_dead = (world.pred.alive) & (world.pred.energy <= 0)

    if prey_dead.any():
        dead_idx = np.flatnonzero(prey_dead)
        world.prey.alive[dead_idx] = False       # not alive
        world.prey.vel[dead_idx] = 0             # set vel to 0
        world.prey.free_indices.extend(dead_idx) # return index to available list
        world.prey.n_prey -= len(dead_idx)
    
    if pred_dead.any():
        dead_idx = np.flatnonzero(pred_dead)
        world.pred.alive[dead_idx] = False
        world.pred.vel[dead_idx] = 0
        world.pred.free_indices.extend(dead_idx)
        world.pred.n_pred -= len(dead_idx)
