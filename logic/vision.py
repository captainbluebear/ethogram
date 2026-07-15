'''Handle vision logic and detection.'''
import numpy as np
from constants import AGENT_SIZE, CELLSIZE
from logic.state import WorldState


def rotate_vector(v, theta):
    '''Rotate 2D vector v by theta radians'''
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])

local_ray_length = 70
local_cone_angle = 270
local_num_rays = 16
# Preallocate a fixed buffer at module level - avoids per-call allocation
RAY_OFFSETS = np.linspace(-local_cone_angle/2, local_cone_angle/2, local_num_rays, dtype=np.float32)
_NEARBY_BUFFER = np.zeros((500, 6), dtype=np.float32)

def vision_raycast_prey(world, agent_idx, cone_angle=local_cone_angle, 
                        ray_length=local_ray_length, num_rays=local_num_rays):
    pos_i = world.prey.pos[agent_idx]
    dir_i = world.prey.dir[agent_idx]
    heading = np.arctan2(dir_i[1], dir_i[0])
    r_cells = int(np.ceil(ray_length / CELLSIZE))

    # --- Fill buffer directly instead of building list of arrays ---
    n = 0

    nearby_pred = world.grid.nearby_pred(pos_i, r=r_cells)
    if nearby_pred:
        k = len(nearby_pred)
        _NEARBY_BUFFER[n:n+k, :2] = world.pred.pos[nearby_pred]
        _NEARBY_BUFFER[n:n+k, 2] = 1; _NEARBY_BUFFER[n:n+k, 3] = 0; _NEARBY_BUFFER[n:n+k, 4] = 0
        _NEARBY_BUFFER[n:n+k, 5] = nearby_pred
        n += k

    nearby_prey = world.grid.nearby_prey(pos_i, r=r_cells)
    if nearby_prey:
        k = len(nearby_prey)
        _NEARBY_BUFFER[n:n+k, :2] = world.prey.pos[nearby_prey]
        _NEARBY_BUFFER[n:n+k, 2] = 0; _NEARBY_BUFFER[n:n+k, 3] = 1; _NEARBY_BUFFER[n:n+k, 4] = 0
        _NEARBY_BUFFER[n:n+k, 5] = nearby_prey
        n += k

    nearby_plants = world.grid.nearby_plant(pos_i, r=r_cells)
    if nearby_plants:
        k = len(nearby_plants)
        _NEARBY_BUFFER[n:n+k, :2] = world.plant.pos[nearby_plants]
        _NEARBY_BUFFER[n:n+k, 2] = 0; _NEARBY_BUFFER[n:n+k, 3] = 0; _NEARBY_BUFFER[n:n+k, 4] = 1
        _NEARBY_BUFFER[n:n+k, 5] = nearby_plants
        n += k

    if n == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0] * num_rays

    nearby = _NEARBY_BUFFER[:n]  # view, no copy

    # --- Vectorized ray math (unchanged) ---
    ray_angles = heading + RAY_OFFSETS
    dirs = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=1)

    d = nearby[:, :2] - pos_i
    dist_sq = d[:, 0]**2 + d[:, 1]**2

    cross = np.abs(d[None,:,0] * dirs[:,1,None] - d[None,:,1] * dirs[:,0,None]) <= 5
    dot   = d[None,:,0] * dirs[:,0,None] + d[None,:,1] * dirs[:,1,None] > 0
    mag   = dist_sq[None,:] < ray_length**2
    mask  = cross & dot & mag  # (num_rays, N)

    # --- Argmin over masked distances per ray ---
    # Set non-hit distances to inf, then argmin across entities
    masked_dist = np.where(mask, dist_sq[None,:], np.inf)  # (num_rays, N)
    best = np.argmin(masked_dist, axis=1)                   # (num_rays,)
    hit = masked_dist[np.arange(num_rays), best] < np.inf  # (num_rays,) bool

    inputs = []
    for i in range(num_rays):
        if not hit[i]:
            inputs += [0.0, 0.0, 0.0, 0.0, 0.0]
            continue
        entity = nearby[best[i]]
        agent_type = entity[2:5]
        dist_norm = 1.0 - (np.sqrt(masked_dist[i, best[i]]) / ray_length)
        mate_drive = float(world.prey.mate_drive[int(entity[5])]) if agent_type[1] == 1 else 0.0
        inputs += [float(agent_type[0]), float(agent_type[1]), float(agent_type[2]),
                   float(dist_norm), mate_drive]

    return inputs
# def vision_raycast_prey(world: WorldState, 
#                         agent_idx, cone_angle=local_cone_angle, ray_length=local_ray_length, num_rays=local_num_rays):
#     pos_i = world.prey.pos[agent_idx]
#     dir_i = world.prey.dir[agent_idx]
#     heading = np.arctan2(dir_i[1], dir_i[0])

#     # --- Gather nearby entities (unchanged) ---
#     nearby_agents = []
#     r_cells = int(np.ceil(ray_length / CELLSIZE))

#     nearby_pred = np.array(world.grid.nearby_pred(pos_i, r=r_cells))
#     if len(nearby_pred) > 0:
#         pos = world.pred.pos[nearby_pred]
#         type = np.tile([1, 0, 0], (len(nearby_pred), 1))
#         nearby_agents.append(np.hstack([pos, type, nearby_pred.reshape(-1, 1)]))

#     nearby_prey = np.array(world.grid.nearby_prey(pos_i, r=r_cells))
#     if len(nearby_prey) > 0:
#         pos = world.prey.pos[nearby_prey]
#         type = np.tile([0, 1, 0], (len(nearby_prey), 1))
#         nearby_agents.append(np.hstack([pos, type, nearby_prey.reshape(-1, 1)]))

#     nearby_plants = np.array(world.grid.nearby_plant(pos_i, r=r_cells))
#     if len(nearby_plants) > 0:
#         pos = world.plant.pos[nearby_plants]
#         type = np.tile([0, 0, 1], (len(nearby_plants), 1))
#         nearby_agents.append(np.hstack([pos, type, nearby_plants.reshape(-1, 1)]))

#     if not nearby_agents:
#         return [0, 0, 0, 0.0, 0.0] * num_rays

#     nearby = np.vstack(nearby_agents)  # (N, 6)

#     # --- Vectorized ray casting ---
#     ray_angles = heading - cone_angle/2 + np.arange(num_rays) * (cone_angle / num_rays)
#     dirs = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=1)  # (num_rays, 2)

#     d = nearby[:, :2] - pos_i          # (N, 2) - relative positions
#     dist_sq = d[:, 0]**2 + d[:, 1]**2  # (N,)

#     # Broadcast all rays against all entities simultaneously
#     # cross/dot/mag: each is (num_rays, N)
#     cross = np.abs(d[None,:,0] * dirs[:,1,None] - d[None,:,1] * dirs[:,0,None]) <= 5
#     dot   = d[None,:,0] * dirs[:,0,None] + d[None,:,1] * dirs[:,1,None] > 0
#     mag   = dist_sq[None,:] < ray_length**2
#     mask  = cross & dot & mag  # (num_rays, N)

#     inputs = []
#     for i in range(num_rays):
#         hits = np.where(mask[i])[0]
#         if len(hits) == 0:
#             inputs += [0.0, 0.0, 0.0, 0.0, 0.0]
#             continue

#         closest = hits[np.argmin(dist_sq[hits])]
#         entity = nearby[closest]

#         agent_type = entity[2:5]
#         dist_norm = 1.0 - (np.sqrt(dist_sq[closest]) / ray_length)

#         mate_drive = 0.0
#         if agent_type[1] == 1:  # PREY
#             mate_drive = world.prey.mate_drive[int(entity[5])]

#         inputs += [float(agent_type[0]), float(agent_type[1]), float(agent_type[2]), float(dist_norm), float(mate_drive)]

#     return inputs
            

# def vision_cone_predator(world: WorldState, agent_idx, cone_angle=local_cone_angle, ray_length=local_ray_length):
#     '''
#     Return a list of sensed prey.

#     :param world: world instance
#     :param agent_idx: predator in consideration
#     :param num_rays: number of vision rays
#     :param cone_angle: angle of vision cone
#     :param ray_length: length of rays in pixels
#     '''    
#     pos_i = world.pred.pos[agent_idx]
#     dir_i = world.pred.dir[agent_idx]
#     prey_pos = world.prey.pos

#     r_cells = int(np.ceil(ray_length / CELLSIZE)) # choose r according to ray length
#     nearby_prey = np.fromiter(world.grid.nearby_prey(pos_i, r=r_cells), dtype=np.intp) # TODO make nearby_prey non-generator
#     if nearby_prey.size == 0:
#         return nearby_prey # returns empty ndarray
    
#     to_prey = prey_pos[nearby_prey] - pos_i # get vectors
#     dists = np.linalg.norm(to_prey, axis=1, keepdims=True) # get euclidian length
#     to_prey_norm = to_prey/(dists + 1e-8) # normalize vectors for dir
    
#     dots = to_prey_norm @ dir_i # take dots aka cos(θ), where θ is the angle between them. We'll check this angle using np.cos

#     in_vision = (dots >= np.cos(cone_angle/2)) & (dists.squeeze() < ray_length) # cosine is decreasing
    
#     return nearby_prey[in_vision]


# Debug      
# def draw_rays(world, idx, cone_angle=cone_angle, ray_length=ray_length, num_rays=num_rays):
#     pos_i = world.prey.pos[idx]
#     dir_i = world.prey.dir[idx]

#     # Draw the two edge lines of the cone
#     half_angle = cone_angle / 2 * np.pi / 180
#     left_edge  = rotate_vector(dir_i, -half_angle) * ray_length
#     right_edge = rotate_vector(dir_i,  half_angle) * ray_length

#     arcade.draw_line(pos_i[0], pos_i[1],
#                      pos_i[0] + left_edge[0],  pos_i[1] + left_edge[1],
#                      arcade.color.YELLOW, 1)
#     arcade.draw_line(pos_i[0], pos_i[1],
#                      pos_i[0] + right_edge[0], pos_i[1] + right_edge[1],
#                      arcade.color.YELLOW, 1)
    
#     # Draw the arc closing the cone
#     angle_of_dir = np.arctan2(dir_i[1], dir_i[0]) * 180 / np.pi
#     arcade.draw_arc_outline(pos_i[0], pos_i[1],
#                             ray_length * 2, ray_length * 2,
#                             arcade.color.YELLOW,
#                             angle_of_dir - cone_angle / 2,
#                             angle_of_dir + cone_angle / 2,
#                             1)

# Highlight seen prey
    # seen = vision_cone_predator(world, predator_idx)
    # for prey_idx in seen:
    #     p = world.prey.pos[prey_idx]
    #     arcade.draw_circle_outline(p[0], p[1], 5, arcade.color.RED, 2)

def draw_rays(world, idx, cone_angle=local_cone_angle, ray_length=local_ray_length, num_rays=local_num_rays):
    pos_i = world.prey.pos[idx]
    dir_i = world.prey.dir[idx]
    heading = np.arctan2(dir_i[1], dir_i[0])

    cone_rad = cone_angle * np.pi / 180

    for i in range(num_rays):
        angle = heading - cone_rad/2 + i * (cone_rad / num_rays)
        dir_ray = np.array([np.cos(angle), np.sin(angle)])
        end = pos_i + dir_ray * ray_length

        # Default: draw ray as dim white (no hit)
        ray_color = (80, 80, 80)
        hit_pos = None

        # Check nearby agents — same logic as vision_raycast_prey
        r_cells = int(np.ceil(ray_length / CELLSIZE))
        parts = []

        nearby_pred = np.array(world.grid.nearby_pred(pos_i, r=r_cells))
        if len(nearby_pred) > 0:
            pos  = world.pred.pos[nearby_pred]
            type = np.tile([1, 0, 0], (len(nearby_pred), 1))
            idxs = nearby_pred.reshape(-1, 1)
            parts.append(np.hstack([pos, type, idxs]))

        nearby_prey = np.array(world.grid.nearby_prey(pos_i, r=r_cells))
        if len(nearby_prey) > 0:
            pos  = world.prey.pos[nearby_prey]
            type = np.tile([0, 1, 0], (len(nearby_prey), 1))
            idxs = nearby_prey.reshape(-1, 1)
            parts.append(np.hstack([pos, type, idxs]))

        nearby_plants = np.array(world.grid.nearby_plant(pos_i, r=r_cells))
        if len(nearby_plants) > 0:
            pos  = world.plant.pos[nearby_plants]
            type = np.tile([0, 0, 1], (len(nearby_plants), 1))
            idxs = nearby_plants.reshape(-1, 1)
            parts.append(np.hstack([pos, type, idxs]))

        if parts:
            nearby = np.vstack(parts)
            d = nearby[:, :2] - pos_i
            cross = np.abs(d[:,0]*dir_ray[1] - d[:,1]*dir_ray[0]) <= 5
            dot   = d[:,0]*dir_ray[0] + d[:,1]*dir_ray[1] > 0
            mag   = d[:,0]**2 + d[:,1]**2 < ray_length**2
            filtered = nearby[cross & dot & mag]

            if len(filtered) > 0:
                d_f = filtered[:, :2] - pos_i
                dists_sq = d_f[:,0]**2 + d_f[:,1]**2
                min_idx  = np.argmin(dists_sq)
                closest  = filtered[min_idx]
                hit_pos  = closest[:2]

                is_pred, is_prey, is_plant = int(closest[2]), int(closest[3]), int(closest[4])
                if is_pred:
                    ray_color = arcade.color.RED
                elif is_prey:
                    ray_color = arcade.color.GREEN
                elif is_plant:
                    ray_color = arcade.color.DARK_GREEN

        # Draw ray up to hit point or full length
        ray_end = hit_pos if hit_pos is not None else end
        arcade.draw_line(pos_i[0], pos_i[1],
                         ray_end[0], ray_end[1],
                         ray_color, 1)

        # Draw a dot at the hit point
        if hit_pos is not None:
            arcade.draw_circle_filled(hit_pos[0], hit_pos[1], 3, ray_color)

    # Draw cone edges
    half_rad = cone_rad / 2
    left_edge  = np.array([np.cos(heading - half_rad), np.sin(heading - half_rad)]) * ray_length
    right_edge = np.array([np.cos(heading + half_rad), np.sin(heading + half_rad)]) * ray_length

    arcade.draw_line(pos_i[0], pos_i[1],
                     pos_i[0] + left_edge[0],  pos_i[1] + left_edge[1],
                     arcade.color.YELLOW, 1)
    arcade.draw_line(pos_i[0], pos_i[1],
                     pos_i[0] + right_edge[0], pos_i[1] + right_edge[1],
                     arcade.color.YELLOW, 1)

    # Draw arc
    angle_of_dir = heading * 180 / np.pi
    arcade.draw_arc_outline(pos_i[0], pos_i[1],
                            ray_length * 2, ray_length * 2,
                            arcade.color.YELLOW,
                            angle_of_dir - cone_angle / 2,
                            angle_of_dir + cone_angle / 2,
                            1)

'''
    angles = np.linspace(-cone_angle/2, cone_angle/2, num_rays) * np.pi/180
    rays = [rotate_vector(dir_i, a) for a in angles]

    if nearby_prey:
        for ray in rays:
            for prey_index in nearby_prey:
            # check if ray is within 3 points of object
                delta = world.prey.pos[prey_index] - pos_i
                proj_length = np.dot(delta, ray) # how far along the ray our dot is

                if 0 < proj_length < ray_length: # get sideways error
                    # the vector down from prey_index to the ray which is perpendicular. norm gives length
                    perpendicular = np.linalg.norm(delta - proj_length*ray)

                    if perpendicular < AGENT_SIZE:
                        hits.append(prey_index)
    
    hits = list(set(hits))
    return hits
    '''