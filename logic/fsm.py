'''Holds movement and sensing logic. Eating logic.'''
import numpy as np
from logic.agents import PreyState, PredState
from logic.detection import prey_get_closest_pred
import logic.movement as m
import constants as c
from math import exp

def step_fsm(world, dt):
    # Handle prey
    prey = world.prey
    raw_prey = update_drive_prey(prey, world)
    update_state_prey(raw_prey, prey)
    apply_behavior_prey(world, prey, dt)

    # Handle pred
    pred = world.pred
    raw_pred = update_drive_pred(pred, world)
    update_state_pred(raw_pred, pred)
    apply_behavior_pred(world, pred, dt)

######################## PREY FUNCTIONS ########################

def update_drive_prey(prey: PreyState, world):
    '''
    Calculate the internal drives (eat, mate) as well as external pressures and return them 'raw' e.g. normalized, but without
    genome weights. Returns (n, N_STATES) ndarray 
    '''
    raw = np.zeros((prey.cap, c.N_STATES), dtype=np.float32)

    # Eat drive
    eat_drive = np.clip(1.0 - (prey.energy/c.MAX_ENERGY), 0.0, 1.0)

    # Mate drive
    # prey.energy - c.MATE_THRESHOLD is the original numerator
    mate_drive = np.clip((prey.energy) / (c.MAX_ENERGY - c.MATE_THRESHOLD), 0.0, 1.0) # Normalized drive from above threshold
    mate_drive = np.where(prey.refractory <= 0, mate_drive, 0.0) # If in refractory period automatically set drive to 0

    # Flee drive
    nearest_pred = prey_get_closest_pred(world)
    has_pred = nearest_pred >= 0
    nearest_dist = np.full(prey.cap, np.inf, dtype=np.float32)
    nearest_dist[has_pred] = np.linalg.norm(prey.pos[has_pred] - world.pred.pos[nearest_pred[has_pred]], axis=1)
    flee_drive = np.where(nearest_dist <= c.PREY_FLEE_THRESHOLD, 1.0 - (nearest_dist/c.PREY_FLEE_THRESHOLD), 0.0)

    # Set all raw drives
    raw[:, c.State.EAT] = eat_drive
    raw[:, c.State.MATE] = mate_drive
    raw[:, c.State.FLEE] = flee_drive
    raw[:, c.State.IDLE] = c.IDLE_BASELINE

    return raw

def update_state_prey(raw: np.ndarray, prey: PreyState):
    weighted_urge = raw*prey.genome # Calculate genome weights
    # print(weighted_urge[0])
    # TODO add hysteresis to avoid rubberbanding/jitter behavior
    updated_states = np.argmax(weighted_urge, axis=1).astype(prey.state.dtype) # Update states to be max value
    prey.state = np.where(prey.alive, updated_states, prey.state) # Only update if prey is alive

def apply_behavior_prey(world, prey: PreyState, dt):
    # Create ID masks for each state. Use them to apply the relevant movement vector
    idle_mask = (prey.state == c.State.IDLE) & prey.alive
    eat_mask = (prey.state == c.State.EAT) & prey.alive
    mate_mask = (prey.state == c.State.MATE) & prey.alive
    flee_mask = (prey.state == c.State.FLEE) & prey.alive

    # Update vel
    update_vels = np.zeros_like(prey.vel)

    idle_vels = m.prey_idle(prey)
    eat_vels = m.prey_move_food(world, prey)
    mate_vels = m.prey_move_mate(world, prey)
    flee_vels = m.prey_flee(world, prey)

    update_vels[idle_mask] = idle_vels[idle_mask]
    update_vels[eat_mask] = eat_vels[eat_mask]
    update_vels[mate_mask] = mate_vels[mate_mask]
    update_vels[flee_mask] = flee_vels[flee_mask]

    snapping = .8 # higher value is faster snap to target
    # prey.vel = update_vels
    # Use the below line when ready. Above is for debugging so that small movement issues can't hide
    prey.vel += (update_vels - prey.vel)*(1-exp(-snapping*dt)) # Exp decay (alternative to LERP)
    
    # Update position
    prey.pos += prey.vel*dt     
    prey.pos[:, 0] %= world.width
    prey.pos[:, 1] %= world.height     # Keep in world bounds

    # Update energy 
    prey.energy -= 0.02*np.linalg.norm(prey.vel, axis=1)*dt # goes down proportional to speed
    # Update refractory
    prey.refractory -= dt

######################## PREDATOR FUNCTIONS ########################

def update_drive_pred(pred: PredState, world):
    '''
    Calculate the internal drives (eat, mate) as well as external pressures and return them 'raw' e.g. normalized, but without
    genome weights. Returns (n, N_STATES) ndarray 
    '''
    raw = np.zeros((pred.cap, c.N_STATES - 1), dtype=np.float32)

    # Eat drive
    eat_drive = np.clip(1.0 - (pred.energy/c.MAX_ENERGY), 0.0, 1.0)

    # Mate drive
    # prey.energy - c.MATE_THRESHOLD is the original numerator
    mate_drive = np.clip((pred.energy) / (c.MAX_ENERGY - c.MATE_THRESHOLD), 0.0, 1.0) # Normalized drive from above threshold
    mate_drive = np.where(pred.refractory <= 0, mate_drive, 0.0) # If in refractory period automatically set drive to 0

    # # Flee drive
    # nearest_pred = prey_get_closest_pred(world)
    # has_pred = nearest_pred >= 0
    # nearest_dist = np.full(prey.cap, np.inf, dtype=np.float32)
    # nearest_dist[has_pred] = np.linalg.norm(prey.pos[has_pred] - world.pred.pos[nearest_pred[has_pred]], axis=1)
    # flee_drive = np.where(nearest_dist <= c.PREY_FLEE_THRESHOLD, 1.0 - (nearest_dist/c.PREY_FLEE_THRESHOLD), 0.0)

    # Set all raw drives
    raw[:, c.State.EAT] = eat_drive
    raw[:, c.State.MATE] = mate_drive
    # raw[:, c.State.FLEE] = flee_drive
    raw[:, c.State.IDLE] = c.IDLE_BASELINE

    return raw

def update_state_pred(raw: np.ndarray, pred: PredState):
    weighted_urge = raw*pred.genome # Calculate genome weights
    # print(weighted_urge[0])
    # TODO add hysteresis to avoid rubberbanding/jitter behavior
    updated_states = np.argmax(weighted_urge, axis=1).astype(pred.state.dtype) # Update states to be max value
    pred.state = np.where(pred.alive, updated_states, pred.state) # Only update if prey is alive

def apply_behavior_pred(world, pred: PredState, dt):
    # Create ID masks for each state. Use them to apply the relevant movement vector
    idle_mask = (pred.state == c.State.IDLE) & pred.alive
    eat_mask = (pred.state == c.State.EAT) & pred.alive
    mate_mask = (pred.state == c.State.MATE) & pred.alive

    # Update vel
    update_vels = np.zeros_like(pred.vel)

    idle_vels = m.pred_idle(pred)
    eat_vels = m.pred_move_food(world, pred)
    mate_vels = m.pred_move_mate(world, pred)

    update_vels[idle_mask] = idle_vels[idle_mask]
    update_vels[eat_mask] = eat_vels[eat_mask]
    update_vels[mate_mask] = mate_vels[mate_mask]

    snapping = .8 # higher value is faster snap to target
    # prey.vel = update_vels
    # Use the below line when ready. Above is for debugging so that small movement issues can't hide
    pred.vel += (update_vels - pred.vel)*(1-exp(-snapping*dt)) # Exp decay (alternative to LERP)
    
    # Update position
    pred.pos += pred.vel*dt     
    pred.pos[:, 0] %= world.width
    pred.pos[:, 1] %= world.height     # Keep in world bounds

    # Update energy 
    pred.energy -= 0.02*np.linalg.norm(pred.vel, axis=1)*dt # goes down proportional to speed
    # Update refractory
    pred.refractory -= dt