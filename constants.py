from enum import IntEnum

class State(IntEnum):
    IDLE = 0
    EAT = 1
    MATE = 2
    FLEE = 3

N_STATES = len(State)
IDLE_BASELINE = 0.1     # Idle value for states (to avoid craziness when all other drives are low)

##### GENERAL SIMULATION VARIABLES
CELLSIZE = 25           # Size of grid cells
AGENT_SIZE = 4          # Size of agent
POP_ARRAY_FAC = 6       # How much larger the initial numpy arrays should be from the actual init size.


MAX_ENERGY = 60         # Maximum energy an agent can hold
MATE_THRESHOLD = 30     # Energy threshold for mating
REPRO_COST = 5          # Energy taken when reproducing
REFRACTORY_START = 10   # Time in seconds before reproduction is allowed; this is the base value when agents spawn
REFRACTORY_GAP = 2      # Time in seconds which gets added after a successful reproduction

##### PREY VARIABLES #####
PREY_START_ENERGY = 10      # How much energy prey starts with
PREY_VISION_RADIUS = 75     # Radius of detection in pixels
PREY_FLEE_THRESHOLD = 15    # Pixel radius at which flee desire starts

PREY_ENERGY = 5            # Amount of energy gained from consuming prey

##### PREDATOR VARIABLES #####
PRED_START_ENERGY = 10
PRED_VISION_RADIUS = 75     # Radius of detection in pixels

##### PLANT VARIABLES #####
REGROWTH_TIME = 15      # Time for plants to regrow
PLANT_ENERGY = 2        # Amount of energy gained from consuming plant




##### GLOBAL CONSTANTS FOR REFERENCING #####
PRED = 0
PREY = 1
PLANT = 2


ENERGY_BOUND = 40
ACCEL_BOUND = 6
MAX_SPEED = 50 # px/s   # Top speed of agents
MAX_TURN_SPEED = 10 # Multiplier for how quickly an agent can turn

# METABOLISM_RATE = 0.5
# MOVE_COST = 0.00005
# TURN_COST = 0.05
# REPRODUCE_FITNESS_BONUS = 1
# FEEDING_FITNESS_BONUS = 0.01

# EAT_THRESHOLD  = 0.1 # should act on moderate desire