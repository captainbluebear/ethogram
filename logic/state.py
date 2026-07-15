from constants import CELLSIZE
from logic.agents import *

class WorldState:
    '''Create and hold core world/simulation state.
        - predators, prey, plants
        - width, height
        - grid system
    '''
    prey: PreyState
    pred: PredState
    width: float
    height: float
    grid: Grid

    def __init__(self, n_prey, n_predators, n_plants, width, height):
        self.n_prey = n_prey
        self.n_pred = n_predators
        self.n_plant = n_plants
        self.width = width
        self.height = height

        self.prey = PreyState(self.n_prey, width, height)
        self.pred = PredState(self.n_pred, width, height)
        self.plant = PlantState(self.n_plant, width, height)

        self.grid = Grid(self.width, self.height, CELLSIZE, self.prey, self.pred, self.plant) # Spatial Partitioning