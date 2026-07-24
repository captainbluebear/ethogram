import numpy as np
from collections import defaultdict
from constants import *
from random import random

class PreyState:
    pos: np.ndarray
    vel: np.ndarray
    dir: np.ndarray
    alive: np.ndarray
    energy: np.ndarray
    n_prey: int

    __slots__ = ("cap", "n_prey", "alive", "free_indices", 
                 "pos", "vel", "energy", 
                 "mate_drive", "eat_drive", "refractory",
                 "genome", "state")
                #thirst? age, fertility, strength, stealth
    
    def __init__(self, n, width, height):
        self.n_prey = n
        self.cap = int(n * POP_ARRAY_FAC) # total capacity of array (not starting population)

        self.alive = np.zeros((self.cap), dtype=np.bool)
        self.alive[:n] = np.ones(n, dtype=np.bool)              # Total population
        self.free_indices = list(range(n, self.cap))            # maintain a list of free indices
                                                                # alive and free_indices are complements to reduce computation

        self.pos = np.zeros((self.cap, 2), dtype=np.float32)
        self.pos[:n] = np.random.rand(n, 2) * [width, height]   # X,Y position in world

        self.vel = np.zeros((self.cap, 2), dtype=np.float32)    # Speed vector

        self.energy = np.zeros((self.cap), dtype=np.float32)
        self.energy[:n] = PREY_START_ENERGY              # Agent energy

        # Drive to reproduce and eat. Initalized to 0
        self.mate_drive = np.zeros((self.cap, 1), dtype=np.float32)
        self.eat_drive = np.zeros((self.cap, 1), dtype=np.float32)

        self.refractory = np.full(self.cap, PREY_REFRACTORY_START, dtype=np.float32)      # Reproduction cooldown timer (avoid spam)

        self.state = np.full(self.cap, State.IDLE, dtype=np.uint8)
        self.genome = np.ones((self.cap, N_STATES), dtype=np.float32)   # behavior weights for primitive evolution

    def add_agent(self, pos, parent_a_idx, parent_b_idx):
        if len(self.free_indices) == 0:
            print("Err: No more free prey indices.")
            return 
        
        self.n_prey += 1
        i = self.free_indices.pop()
        self.alive[i] = True
        self.pos[i] = pos
        self.vel[i] = 0
        self.energy[i] = PREY_START_ENERGY
        self.mate_drive[i] = 0
        self.eat_drive[i] = 0
        self.refractory[i] = 0
        self.state[i] = State.IDLE
        # print(f"Prey pop: {self.n_prey}")
        # TODO Handle crossover mutation generation here

    def remove_agent(self, idx: int):
        self.n_prey -= 1
        self.alive[idx] = False
        self.free_indices.append(idx)


class PredState:
    pos: np.ndarray
    vel: np.ndarray
    dir: np.ndarray
    alive: np.ndarray
    energy: np.ndarray
    n_pred: int
    __slots__ = ("cap", "n_pred", "alive", "free_indices", 
                 "pos", "vel", "energy", 
                 "mate_drive", "eat_drive", "refractory",
                 "genome", "state")
                #thirst? age, fertility, strength, stealth
    
    
    def __init__(self, n, width, height):
        self.n_pred = n
        self.cap = int(n * POP_ARRAY_FAC) # total capacity of array (not starting population)

        self.alive = np.zeros((self.cap), dtype=np.bool)
        self.alive[:n] = np.ones(n, dtype=np.bool)              # Total population
        self.free_indices = list(range(n, self.cap))            # maintain a list of free indices
                                                                # alive and free_indices are complements to reduce computation

        self.pos = np.zeros((self.cap, 2), dtype=np.float32)
        self.pos[:n] = np.random.rand(n, 2) * [width, height]   # X,Y position in world

        self.vel = np.zeros((self.cap, 2), dtype=np.float32)    # Speed vector

        self.energy = np.zeros((self.cap), dtype=np.float32)
        self.energy[:n] = PRED_START_ENERGY                     # Agent energy

        # Drive to reproduce and eat. Initalized to 0
        self.mate_drive = np.zeros((self.cap, 1), dtype=np.float32)
        self.eat_drive = np.zeros((self.cap, 1), dtype=np.float32)

        self.refractory = np.full(self.cap, PRED_REFRACTORY_START, dtype=np.float32)      # Reproduction cooldown timer (avoid spam)

        self.state = np.full(self.cap, State.IDLE, dtype=np.uint8)
        self.genome = np.ones((self.cap, N_STATES-1), dtype=np.float32)   # behavior weights for primitive evolution

    def add_agent(self, pos, parent_a_idx, parent_b_idx):
        if len(self.free_indices) == 0:
            print("Err: No more free pred indices.")
            return 
        
        self.n_pred += 1
        i = self.free_indices.pop()
        self.alive[i] = True
        self.pos[i] = pos
        self.vel[i] = 0
        self.energy[i] = PRED_START_ENERGY
        self.mate_drive[i] = 0
        self.eat_drive[i] = 0
        self.refractory[i] = 0
        self.state[i] = State.IDLE
        print(f"Pred pop: {self.n_pred}")
        # TODO Handle crossover mutation generation here

    def remove_agent(self, idx: int):
        self.n_pred -= 1
        self.alive[idx] = False
        self.free_indices.append(idx)

class PlantState:
    pos: np.ndarray
    alive: np.ndarray
    growth: np.ndarray

    __slots__ = ("pos", "alive", "growth")
    
    def __init__(self, n, width, height):
        self.pos = np.random.rand(n, 2) * [width, height]
                
        self.alive = np.ones(n, dtype=np.bool) # start as "alive" (grown)
        self.growth = np.zeros(n, dtype=np.float32) # time spent growing

###### Spatial Grid Handling ######
class Grid:
    def __init__(self, width, height, cellsize, prey:PreyState, pred:PredState, plant:PlantState):
        self.cellsize = cellsize
        self.gw = width//cellsize + 1
        self.gh = height//cellsize + 1
        self.n_cells = self.gw*self.gh
        self.update_grid(prey, pred, plant)
    
    def _build_csr(self, state): # Storage by cell idx and 
        idx = np.flatnonzero(state.alive)
        if idx.size == 0:
            empty_offsets = np.zeros(self.n_cells + 1, dtype=np.int64)
            return empty_offsets, np.empty(0, dtype=np.int64)
        
        cells = np.floor_divide(state.pos[idx], self.cellsize).astype(np.int64)
        np.clip(cells[:, 0], 0, self.gw - 1, out=cells[:, 0])   # In-place clipping for bounds
        np.clip(cells[:, 1], 0, self.gh - 1, out=cells[:, 1])
        cell_id = cells[:, 0] * self.gh + cells[:, 1]           # Flatten cells into a single ID - same cell = same ID

        order = np.argsort(cell_id, kind='stable')      # Sort by cell_id to group same ID idxs together
        sorted_idx = idx[order]
        sorted_cells = cell_id[order]
        
        counts = np.bincount(sorted_cells, minlength=self.n_cells)  # Count how many times each cell appears in sorted_cells
        offsets = np.zeros(self.n_cells+1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        return offsets, sorted_idx

    def update_grid(self, prey: PreyState, pred: PredState, plant: PlantState):
        self.prey_offset, self.prey_flat = self._build_csr(prey)
        self.pred_offset, self.pred_flat = self._build_csr(pred)
        self.plant_offset, self.plant_flat = self._build_csr(plant)

    def _nearby(self, offset, flat, agent_pos, r):
        cx = int(agent_pos[0]//self.cellsize)
        cy = int(agent_pos[1]//self.cellsize)
        slices = []
        x0, x1 = max(cx-r, 0), min(cx+r, self.gw-1) # Grab range within radius (plus wrapping)
        y0, y1 = max(cy-r, 0), min(cy+r, self.gh-1)
        
        for x in range(x0, x1+1):
            base = x*self.gh    # Calculate the grid fmla
            start = offset[base+y0]
            end = offset[base+y1+1]
            if start < end:
                slices.append(flat[start:end])
        
        if not slices:
            return np.empty(0, dtype=np.int64) # should never happen - returns self
        return np.concatenate(slices)

    def nearby_prey(self, agent_pos, r=1):
        return self._nearby(self.prey_offset, self.prey_flat, agent_pos, r)
    
    def nearby_pred(self, agent_pos, r=1):
        return self._nearby(self.pred_offset, self.pred_flat, agent_pos, r)

    def nearby_plant(self, agent_pos, r=1):
        return self._nearby(self.plant_offset, self.plant_flat, agent_pos, r)