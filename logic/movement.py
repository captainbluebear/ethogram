'''Handle all movement definitions, such as what happens when idling/fleeing/finding food/mating.'''
import numpy as np
from logic.agents import PreyState, PredState
from constants import State
from logic.detection import prey_get_closest_food, prey_get_closest_prey, prey_get_closest_pred, pred_get_closest_prey, pred_get_closest_pred

FLEE_FACTOR = 10.0
PREY_FOOD_FACTOR = 50.0
PRED_FOOD_FACTOR = 60.0
MATE_FACTOR = 50.0
WANDER_FACTOR = 40.0

def prey_idle(prey: PreyState):
    return _prey_wander(prey)   # Since I've already implemented a basic wander, it's easier to just call it here under
                                # a different name to avoid confusion.

def prey_move_food(world, prey: PreyState):
    '''Calculate direction vector to food for all prey. If agent DNE, vector is 0. If no food spotted, agent wanders.'''
    plant = world.plant
    vels = np.zeros_like(prey.vel)
    closest_food = prey_get_closest_food(world)
    
    food_seen = closest_food >= 0
    dist = plant.pos[closest_food[food_seen]] - prey.pos[food_seen]
    dist_norm = dist/(np.linalg.norm(dist, axis=1, keepdims=True) + 1e-8)    # Normalize so that speed is constant
    vels[food_seen] = dist_norm*PREY_FOOD_FACTOR

    no_food_seen = (closest_food < 0) & prey.alive # get indices where no food is seen but the agent exists
    vels[no_food_seen] = _prey_wander(prey)[no_food_seen]

    return vels

def prey_move_mate(world, prey: PreyState):
    '''Calculate direction vector to mate for all prey that are alive. If agent DNE, vector is 0. If no mate spotted, agent wanders.'''
    vels = np.zeros_like(prey.vel)
    closest_prey = prey_get_closest_prey(world)

    mask = (closest_prey >= 0) & (prey.state[closest_prey] == State.MATE) # Only grab other mateable prey
    dist = prey.pos[closest_prey[mask]] - prey.pos[mask]
    dist_norm = dist/(np.linalg.norm(dist, axis=1, keepdims=True) + 1e-8)    # Normalize so that speed is constant
    vels[mask] = dist_norm*MATE_FACTOR

    no_prey_seen = (closest_prey < 0) & prey.alive # get indices where no prey is seen but the agent exists
    vels[no_prey_seen] = _prey_wander(prey)[no_prey_seen]

    return vels

def prey_flee(world, prey: PreyState):
    '''Calculate direction vector away from pred for all prey that are alive. If no pred or agent DNE, vector is 0.'''
    pred = world.pred
    vels = np.zeros_like(prey.vel)
    closest_pred = prey_get_closest_pred(world)

    mask = closest_pred >= 0
    vels[mask] = (prey.pos[mask] - pred.pos[closest_pred[mask]])*FLEE_FACTOR # Flee with extra flee!

    return vels

def _prey_wander(prey: PreyState):
    '''
    A helper method for two situations: MATE and EAT states engaged but relevant agents not close enough to be detected. 
    Provides a wander mechanic to search for relevant agent (prey/food) before re-engaging once something is found.
    Note this is modeled off wander steering mechanics.'''
    vels = np.zeros_like(prey.vel)
    angle = np.pi/6
    alive = prey.alive
    v = prey.vel[alive]

    norm = np.linalg.norm(v, axis=1, keepdims=True)
    stationary = (norm < 1e-6).ravel()
    norm_safe = np.where(norm > 1e-6, norm, 1.0)    # avoid division by 0
    n_v = v/norm_safe                               # normalize velocity

    rot = np.random.uniform(-angle, angle, size=alive.sum())

    angles = np.atan2(n_v[:,1], n_v[:,0])
    angles_jittered = angles + rot
    rotated = np.stack([np.cos(angles_jittered),np.sin(angles_jittered)], axis=1)

    # Handle stationary agents - make sure they start at a random angle
    # NOTE not sure if this is necessary. Only needed if initializing from 0 means all newborns or low vel go in same axis
    random_angle = np.random.uniform(0, 2*np.pi, size=alive.sum())
    random_dir = np.stack([np.cos(random_angle), np.sin(random_angle)], axis=1)

    final = np.where(stationary[:, None], random_dir, rotated)

    vels[alive] = final*WANDER_FACTOR   # Scale by some consistent speed

    return vels





def pred_idle(pred: PredState):
    return _pred_wander(pred)   # Since I've already implemented a basic wander, it's easier to just call it here under
                                # a different name to avoid confusion.

def pred_move_food(world, pred: PredState):
    '''Calculate direction vector to food for all prey. If agent DNE, vector is 0. If no food spotted, agent wanders.'''
    prey = world.prey
    vels = np.zeros_like(pred.vel)
    closest_prey = pred_get_closest_prey(world)
    
    prey_seen = closest_prey >= 0
    dist = prey.pos[closest_prey[prey_seen]] - pred.pos[prey_seen]
    dist_norm = dist/np.linalg.norm(dist, axis=1, keepdims=True)    # Normalize so that speed is constant
    vels[prey_seen] = dist_norm*PRED_FOOD_FACTOR

    no_prey_seen = (closest_prey < 0) & pred.alive # get indices where no food is seen but the agent exists
    vels[no_prey_seen] = _pred_wander(pred)[no_prey_seen]

    return vels

def pred_move_mate(world, pred: PredState):
    '''Calculate direction vector to mate for all pred that are alive. If agent DNE, vector is 0. If no mate spotted, agent wanders.'''
    vels = np.zeros_like(pred.vel)
    closest_pred = pred_get_closest_pred(world)

    mask = (closest_pred >= 0) & (pred.state[closest_pred] == State.MATE) # Only grab other mateable agents
    dist = pred.pos[closest_pred[mask]] - pred.pos[mask]
    dist_norm = dist/(np.linalg.norm(dist, axis=1, keepdims=True) + 1e-8)    # Normalize so that speed is constant
    vels[mask] = dist_norm*MATE_FACTOR

    no_prey_seen = (closest_pred < 0) & pred.alive # get indices where no pred is seen but the agent exists
    vels[no_prey_seen] = _pred_wander(pred)[no_prey_seen]

    return vels

def _pred_wander(pred: PredState):
    '''
    A helper method for two situations: MATE and EAT states engaged but relevant agents not close enough to be detected. 
    Provides a wander mechanic to search for relevant agent (prey/food) before re-engaging once something is found.
    Note this is modeled off wander steering mechanics.'''
    vels = np.zeros_like(pred.vel)
    angle = np.pi/6
    alive = pred.alive
    v = pred.vel[alive]

    norm = np.linalg.norm(v, axis=1, keepdims=True)
    stationary = (norm < 1e-6).ravel()
    norm_safe = np.where(norm > 1e-6, norm, 1.0)    # avoid division by 0
    n_v = v/norm_safe                               # normalize velocity

    rot = np.random.uniform(-angle, angle, size=alive.sum())

    angles = np.atan2(n_v[:,1], n_v[:,0])
    angles_jittered = angles + rot
    rotated = np.stack([np.cos(angles_jittered),np.sin(angles_jittered)], axis=1)

    # Handle stationary agents - make sure they start at a random angle
    # NOTE not sure if this is necessary. Only needed if initializing from 0 means all newborns or low vel go in same axis
    random_angle = np.random.uniform(0, 2*np.pi, size=alive.sum())
    random_dir = np.stack([np.cos(random_angle), np.sin(random_angle)], axis=1)

    final = np.where(stationary[:, None], random_dir, rotated)

    vels[alive] = final*WANDER_FACTOR   # Scale by some consistent speed

    return vels