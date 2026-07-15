# Environmental evolution simulation for agent-based models.
This is an ongoing personal project born from a desire to explore my interest in wildlife and ecology through the lens of computer science. By combining the field of artificial intelligence with ecological simulation, I have recreated predator-prey interaction in a two-dimensional space. This project aims to create an accurate representation of a natural environment while allowing certain parameters to be changed, resulting in an experiment which allows us to see how modifying certain conditions in a natural environment may affect the creatures within it.

This project is written in Python using the [Arcade API](https://api.arcade.academy/en/stable/) as the visual framework.

## Current Behavioral Implementations
Certain aspects of this simulation are more true-to-life than others. At the moment, certain behaviors have been modeled:
- Basic movement
- Feeding behavior
- Life/Death and Reproduction

### Movement
Movement is currently a conditional model which swaps states (search for food, flee, reproduce) depending on the energy level of the agent and (for prey) whether a predator is nearby. 

Linear interpolation used for turning.

### Feeding
Predator feed upon prey; prey feed upon plants; plants simply exist. In this model, predators gain a significant boost of energy upon consuming a prey, and prey gain a small amount of energy from eating a single plant.

### Life/Death and Reproduction
The lifecycle of an agent is energy-dependent. If its energy drops to zero, it will perish. If it reaches a certain age, it will also perish. To reproduce, two agents of the same type must meet and both be above the energy threshold for reproduction. They will lose a certain amount of energy upon reproducing, preventing an infinite cycle of children. However, if their energy is high enough the agents may be able to produce more than one child, simulating the growth in an abundant environment.

## Instructions
TBD 


## Future Iterations:
- Modify the energy cost of movement to be velocity dependent.
- Add an energy cap. Continuously eating will not provide a benefit past a certain threshold.
- Implement age and aging.
- Swap to KD-tree instead of a grid system
- Implement vision cones for predators and prey instead of a radius
- Transfer off pure Python into Godot or Unity engine for a more visually appealing graphic and greater control over environmental factors, as well as added complexity in the form of 3D landscape.

## Resources
https://youtu.be/N3tRFayqVtk?si=F-QSEgo3tdW-sWCz

https://gama-platform.org/wiki/PredatorPrey




