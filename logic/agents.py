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

class Grid:
    def __init__(self, width, height, cellsize, prey:PreyState, pred:PredState, plant:PlantState):
        self.cellsize = cellsize
        self.grid_prey = defaultdict(list)      # Holds idxs of prey in their respective grid cells
        self.grid_pred = defaultdict(list)      # Holds idxs of preds...
        self.grid_plant = defaultdict(list)     # Holdx idxs of plants...
        self.grid_height = height//cellsize + 1
        self.grid_width = width//cellsize + 1
        self.update_grid(prey, pred, plant)
    
    def update_grid(self, prey:PreyState, pred:PredState, plant:PlantState):
        self.grid_prey.clear()
        self.grid_pred.clear()
        self.grid_plant.clear()

        for i in np.flatnonzero(prey.alive):
            self.grid_prey[(int(prey.pos[i][0]//self.cellsize), 
                            int(prey.pos[i][1]//self.cellsize))].append(i)
        
        for i in np.flatnonzero(pred.alive):
            self.grid_pred[(int(pred.pos[i][0]//self.cellsize), 
                            int(pred.pos[i][1]//self.cellsize))].append(i)
        
        for i in np.flatnonzero(plant.alive):
            self.grid_plant[(int(plant.pos[i][0]//self.cellsize), 
                            int(plant.pos[i][1]//self.cellsize))].append(i)
    
    def nearby_prey(self, agent_pos, r=1) -> list[int]:
        '''
        Given an agent's (x,y) position as returned by agent.pos[agent_idx], 
        grab prey in the neighboring cells (up to given radius) as well as self.
        '''
        prey = []
        cx = int(agent_pos[0]//self.cellsize)
        cy = int(agent_pos[1]//self.cellsize)
        for x in range(-r, r+1):
            for y in range(-r, r+1):
                cell = self.grid_prey.get((cx + x, cy + y))
                if cell:
                    prey.extend(cell)
        return prey

    def nearby_pred(self, agent_pos, r=1) -> list[int]:
        '''
        Given an agent's (x,y) position as returned by agent.pos[agent_idx], 
        grab prey in 8 directly neighboring cells and self.
        '''
        pred = []
        cx = int(agent_pos[0]//self.cellsize)
        cy = int(agent_pos[1]//self.cellsize)
        for x in range(-r, r+1):
            for y in range(-r, r+1):
                cell = self.grid_pred.get((cx + x, cy + y))
                if cell:
                    pred.extend(cell)
        return pred

    def nearby_plant(self, agent_pos, r=1) -> list[int]:
        '''
        Given an agent's (x,y) position as returned by agent.pos[agent_idx], 
        grab prey in 8 directly neighboring cells and self.
        '''
        plant = []
        cx = int(agent_pos[0]//self.cellsize)
        cy = int(agent_pos[1]//self.cellsize)
        for x in range(-r, r+1):
            for y in range(-r, r+1):
                plant.extend(self.grid_plant.get((cx + x, cy + y), ()))
                cell = self.grid_plant.get((cx + x, cy + y))
                if cell:
                    plant.extend(cell)
        return plant