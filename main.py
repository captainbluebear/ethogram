"""
Handle drawing agents and holding main function - centralized Arcade logic
"""
import arcade
import numpy as np
from logic.state import WorldState
from logic.consumption import step_consumption
from logic.lifecycle import step_lifecycle
from constants import *
from logic.fsm import step_fsm

import cProfile
import pstats

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 500
WINDOW_TITLE = "🌿 Ethogram"
NUM_PREY = 80
NUM_PREDS = 1
NUM_PLANTS = 1000
SIM_DT = 0.05 # 20 Hz sim

class GameView(arcade.Window):
    """Main application class."""

    def __init__(self, world):
        super().__init__(world.width, world.height, WINDOW_TITLE) # Call parent class to set up window
        self.world = world
        self.accumulator = 0.0
        self.background_color = arcade.csscolor.DIM_GRAY

        # Sprite list for prey/pred
        self.prey_list = arcade.SpriteList()
        for i in range(world.prey.cap):
            sprite = arcade.SpriteCircle(radius=AGENT_SIZE, color=arcade.color.GREEN)
            sprite.agent_index = i # type: ignore
            self.prey_list.append(sprite)

        self.pred_list = arcade.SpriteList()
        for i in range(world.pred.cap):
            sprite = arcade.SpriteCircle(radius=AGENT_SIZE, color=arcade.color.FIREBRICK)
            sprite.agent_index = i # type: ignore
            self.pred_list.append(sprite)
        
        self.plant_list = arcade.SpriteList()
        for i in range(world.n_plant):
            sprite = arcade.SpriteCircle(radius=AGENT_SIZE, color=arcade.color.PLUM)
            sprite.agent_index = i # type: ignore
            self.plant_list.append(sprite)

        # User Interaction
        self.selected_type = None
        self.selected_id = None

        # Grid lines
        self.grid_lines = arcade.shape_list.ShapeElementList()
        for x in range(0, self.world.width, CELLSIZE):
            self.grid_lines.append(
                arcade.shape_list.create_line(x, 0, x, self.world.height, arcade.color.DARK_GRAY)
            )
        for y in range(0, self.world.height, CELLSIZE):
            self.grid_lines.append(
                arcade.shape_list.create_line(0, y, self.world.width, y, arcade.color.DARK_GRAY)
            )

    # ---------- PREY ----------
    PREY_COLORS = {
        State.IDLE:  (40, 200, 40),      # Green
        State.EAT:   (170, 255, 40),     # Lime
        State.FLEE:  (40, 220, 255),     # Cyan
        State.MATE:  (120, 255, 170),    # Mint
    }

    # ---------- PREDATORS ----------
    PRED_COLORS = {
        State.IDLE:  (170, 40, 40),      # Dark Red
        State.EAT:  (255, 50, 50),      # Bright Red
        State.MATE:  (255, 80, 200),     # Magenta
    }

    def on_update(self, delta_time):    
        self.accumulator += delta_time
        steps = 0
        while self.accumulator >= SIM_DT:
            steps += 1
            # Sim calculations
            step_fsm(self.world, SIM_DT)
            step_consumption(self.world)
            step_lifecycle(self.world, SIM_DT)
            self.world.grid.update_grid(self.world.prey, self.world.pred,self.world.plant)            
            self.accumulator -= SIM_DT


        if steps > 1:
            print(f"catch-up: ran {steps} steps this frame")

        # Update sprite location
        for i, sprite in enumerate(self.prey_list): 
            sprite.center_x = self.world.prey.pos[i, 0]
            sprite.center_y = self.world.prey.pos[i, 1]

        for i, sprite in enumerate(self.pred_list): 
            sprite.center_x = self.world.pred.pos[i, 0]
            sprite.center_y = self.world.pred.pos[i, 1]

        for i, sprite in enumerate(self.plant_list):
            sprite.center_x = self.world.plant.pos[i, 0]
            sprite.center_y = self.world.plant.pos[i, 1]

        # Update alive
        for i, sprite in enumerate(self.prey_list):
            if not self.world.prey.alive[i]:
                sprite.visible = False
            else:
                sprite.visible = True

        for i, sprite in enumerate(self.pred_list):
            if not self.world.pred.alive[i]:
                sprite.visible = False
            else:
                sprite.visible = True
        
        for i, sprite in enumerate(self.plant_list):
            if not self.world.plant.alive[i]:
                sprite.visible = False
            else:
                sprite.visible = True

        for i, sprite in enumerate(self.prey_list):
            if self.world.prey.alive[i]:
                sprite.color = self.PREY_COLORS[self.world.prey.state[i]]

        for i, sprite in enumerate(self.pred_list):
            if self.world.pred.alive[i]:
                sprite.color = self.PRED_COLORS[self.world.pred.state[i]]

    def setup(self):
        """Set up the game here. Call this function to restart the game."""
        pass

    def on_draw(self):
        """Render the screen."""

        # The clear method should always be called at the start of on_draw.
        # It clears the whole screen to whatever the background color is
        # set to. This ensures that you have a clean slate for drawing each
        # frame of the game.
        self.clear()

        # Render spritelist
        self.plant_list.draw()
        self.prey_list.draw()
        self.pred_list.draw()

        
        ##### DEBUG DRAWINGS #####
        # Draw grid
        self.grid_lines.draw()

        # Render agent information (top left)
        energy = 0
        speed = 0
        state = 0
        if self.selected_type == "prey":
            alive = self.world.prey.alive[self.selected_id]
            if not alive:
                self.selected_type = None
                self.selected_id = None
                return
            energy = self.world.prey.energy[self.selected_id]
            speed = np.linalg.norm(self.world.prey.vel[self.selected_id])
            state = self.world.prey.state[self.selected_id]

        elif self.selected_type == "pred":
            alive = self.world.pred.alive[self.selected_id]
            if not alive:
                self.selected_type = None
                self.selected_id = None
                return
            energy = self.world.pred.energy[self.selected_id]
            speed = np.linalg.norm(self.world.pred.vel[self.selected_id])
            state = self.world.pred.state[self.selected_id]

        panel_width = 220
        panel_height = 120
        arcade.draw_lrbt_rectangle_filled(5, 5 + panel_width, (self.height - 5) - panel_height, self.height - 5,
                                         (50, 50, 50, 180))  # RGBA

        text = (f"Type: {self.selected_type} \nIndex: {self.selected_id} \nEnergy: {energy:.2f} \n"
                f"Speed: {speed:.2f} \nState: {State(state).name}")
        arcade.draw_text(text, 10, self.height - 20, arcade.color.WHITE, 14, multiline=True, width=300)

        if self.selected_type and self.selected_id is not None:
            if self.selected_type == "prey":
                sprite = self.prey_list[self.selected_id]
            elif self.selected_type == "pred":
                sprite = self.pred_list[self.selected_id]
            else:
                sprite = self.plant_list[self.selected_id]

            arcade.draw_circle_outline(
                sprite.center_x,
                sprite.center_y,
                AGENT_SIZE + 4,
                arcade.color.YELLOW,
                2
            )
        
        # Temporarily perma print index 0 for debugging purposes
        # zero = self.prey_list[0]
        # arcade.draw_circle_outline(
        #         zero.center_x,
        #         zero.center_y,
        #         AGENT_SIZE + 4,
        #         arcade.color.YELLOW,
        #         2
        #     )
        
        

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        if button != arcade.MOUSE_BUTTON_LEFT:
            return

        # Check prey first
        clicked = arcade.get_sprites_at_point((x, y), self.prey_list)
        if clicked:
            sprite = clicked[0]
            self.selected_type = "prey"
            self.selected_id = sprite.agent_index
            return

        # Then predators
        clicked = arcade.get_sprites_at_point((x, y), self.pred_list)
        if clicked:
            sprite = clicked[0]
            self.selected_type = "pred"
            self.selected_id = sprite.agent_index
            return

        # If nothing clicked → deselect
        self.selected_type = None
        self.selected_id = None



def main():
    """Main function"""
    world = WorldState(NUM_PREY, NUM_PREDS, NUM_PLANTS, WINDOW_WIDTH, WINDOW_HEIGHT) 
    window = GameView(world)
    window.setup()

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        arcade.run()
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.dump_stats('profile_output.prof')   # for snakeviz, if you want the flamegraph later
        stats.print_stats(30)   


if __name__ == "__main__":
    main()