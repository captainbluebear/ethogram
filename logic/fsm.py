'''Holds movement and sensing logic. Eating logic.'''
import numpy as np
from logic.agents import PreyState
from logic.state import WorldState
from logic.detection import prey_get_closest_pred
import logic.movement as m
import constants as c
from math import exp

def step_fsm_prey(world, dt):
    prey = world.prey
    # print(dt)
    raw = update_drive_prey(prey, world)
    update_state_prey(raw, prey)
    apply_behavior_prey(world, prey, dt)

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


# def step_movement_predator(world, dt):
#     '''Define how predators move'''
#     prey_pos = world.prey.pos
#     pred_pos = world.pred.pos
#     pred_vel = world.pred.vel
#     pred_dir = world.pred.dir

#     for idx in np.flatnonzero(world.pred.alive):
#         idx_pos = pred_pos[idx]
#         idx_dir = pred_dir[idx]
#         seen_prey = vision_cone_predator(world, idx)
    
#         # get closest prey location and move towards it 
#         if seen_prey.size > 0:
#             dists = np.linalg.norm(prey_pos[seen_prey] - idx_pos, axis=1)
#             closest = seen_prey[np.argmin(dists)]

#             dir = prey_pos[closest] - idx_pos
#             dir /= np.linalg.norm(dir) + 1e-8 # normalize (1e-8 prevents div by 0 if prey and pred occupy same space)

#             # Fix directional
#             turn_rate = 5 # radians/sec
#             alpha = 1 - np.exp(-turn_rate*dt) # framerate independent turning
#             pred_dir[idx] = (1 - alpha)*idx_dir + alpha*dir
#             pred_dir[idx] /= np.linalg.norm(pred_dir[idx]) + 1e-8

#             # Fix velocity
#             curr_spd = np.linalg.norm(pred_vel[idx])
#             target_vel = pred_dir[idx] * np.maximum(curr_spd, 33)
#             pred_vel[idx] = (1 - alpha)*pred_vel[idx] + alpha*target_vel
 
#         else:
#             # Find mate
#             if world.pred.energy[idx] >= REPRO_THRESHOLD:
#                 closest_pred = None
#                 best_d = float("inf") 
#                 for p in world.grid.nearby_pred(idx_pos):
#                     if p == idx:
#                         continue
#                     dx = pred_pos[idx][0] - pred_pos[p][0]
#                     dy = pred_pos[idx][1] - pred_pos[p][1]
#                     d2 = dx**2 + dy**2
#                     if d2 < best_d:
#                         best_d = d2
#                         closest_pred = p
                    
#                 if closest_pred is not None:
#                     delta = prey_pos[closest_pred] - pred_pos[idx] 
#                     if np.linalg.norm(delta) == 0:
#                         print("delta is 0 (mate)")
#                     dir = delta / (np.linalg.norm(delta) + 1e-8) # direction vector AWAY from avg pred location
#                     world.pred.vel[idx] += 30*dt*dir
#                 else:
#                     # Random movement
#                     noise = np.random.uniform(-1, 1, size=2)
#                     pred_vel[idx] += 20.0 * noise * dt
#             else:
#                 # Random movement
#                 noise = np.random.uniform(-1, 1, size=2)
#                 pred_vel[idx] += 20.0 * noise * dt

#     # limit on velocity overaccelerating
#     speed = np.linalg.norm(pred_vel, axis=1, keepdims=True) + 1e-8
#     pred_vel = pred_vel / speed * np.minimum(speed, SPEED_CAP)


#     # Correct dir
#     direction = np.linalg.norm(pred_vel, axis=1, keepdims=True)
#     desired_dir = pred_vel / direction
#     pred_dir = ((1 - .15) * pred_dir + .15 * desired_dir)
#     pred_dir /= np.linalg.norm(pred_dir, axis=1, keepdims=True) + 1e-8
    
#     # Update agent position (move)
#     pred_pos += pred_vel * dt

#     # Keep in world bounds
#     pred_pos[:, 0] %= world.width
#     pred_pos[:, 1] %= world.height

#     world.pred.energy -= 0.01*(speed.squeeze() - 1e-8)*dt


# def step_movement_prey(world: WorldState, dt):
#     '''Define how prey move'''
#     prey_pos = world.prey.pos
#     pred_pos = world.pred.pos
#     prey_vel = world.prey.vel
#     prey_dir = world.prey.dir

#     for idx in np.flatnonzero(world.prey.alive):    
#         idx_pos = prey_pos[idx]
        
#         # Predator escape
#         closest_pred = None
#         best_d = float("inf")
#         for p in world.grid.nearby_pred(idx_pos):
#             dx = prey_pos[idx][0] - pred_pos[p][0]
#             dy = prey_pos[idx][1] - pred_pos[p][1]
#             d2 = dx**2 + dy**2
#             if d2 < best_d:
#                 best_d = d2
#                 closest_pred = p
            
#         if closest_pred is not None: # Run from predator
#             delta = prey_pos[idx] - pred_pos[closest_pred]
#             if np.linalg.norm(delta) == 0:
#                 print("delta is 0")
#             dir = delta / np.linalg.norm(delta) + 1e-8 # direction vector AWAY from avg pred location
#             world.prey.vel[idx] += 40*dt*dir
#         else: # PLANT MOMENT
#             closest_plant = None
#             best_d = 1e9
#             for p in world.grid.nearby_plant(idx_pos):
#                 dx = prey_pos[idx][0] - world.plant.pos[p][0]
#                 dy = prey_pos[idx][1] - world.plant.pos[p][1]
#                 d2 = dx**2 + dy**2
#                 if d2 < best_d:
#                     best_d = d2
#                     closest_plant = p
                
#             if closest_plant is not None:
#                 delta = world.plant.pos[closest_plant] - prey_pos[idx]
#                 if np.linalg.norm(delta) == 0:
#                     print("delta is 0 (plant)")
#                 dir = delta / np.linalg.norm(delta) + 1e-8
#                 prey_vel[idx] += 30.0*dt*dir
#             else: 
#                 # Find mate
#                 if world.prey.energy[idx] >= REPRO_THRESHOLD:
#                     closest_prey = None
#                     best_d = float("inf")
#                     for p in world.grid.nearby_prey(idx_pos):
#                         if p == idx:
#                             continue
#                         dx = prey_pos[idx][0] - prey_pos[p][0]
#                         dy = prey_pos[idx][1] - prey_pos[p][1]
#                         d2 = dx**2 + dy**2
#                         if d2 < best_d:
#                             best_d = d2
#                             closest_prey = p
                        
#                     if closest_prey is not None:
#                         delta = prey_pos[closest_prey] - prey_pos[idx] 
#                         dir = delta / (np.linalg.norm(delta) + 1e-8) # direction vector AWAY from avg pred location
#                         world.prey.vel[idx] += 30*dt*dir
#                     else:
#                         noise = np.random.uniform(-1, 1, size=2)
#                         prey_vel[idx] += 30.0 * noise * dt
#                 else: # Default wander
#                     noise = np.random.uniform(-1, 1, size=2)
#                     prey_vel[idx] += 30.0 * noise * dt
    
#     # limit on velocity overaccelerating
#     speed = np.linalg.norm(prey_vel, axis=1, keepdims=True) + 1e-8
#     prey_vel = prey_vel / speed * np.minimum(speed, SPEED_CAP) #SPEED_CAP

#     # Correct dir
#     direction = np.linalg.norm(prey_vel, axis=1, keepdims=True)
#     desired_dir = prey_vel / direction
#     prey_dir = ((1 - .15) * prey_dir + .15 * desired_dir)
#     prey_dir /= np.linalg.norm(prey_dir, axis=1, keepdims=True) + 1e-8
    
#     # Update agent position (move)
#     prey_pos += prey_vel * dt

#     # Keep in world bounds
#     prey_pos[:, 0] %= world.width
#     prey_pos[:, 1] %= world.height

#     world.prey.energy -= 0.01*(speed.squeeze() - 1e-8)*dt