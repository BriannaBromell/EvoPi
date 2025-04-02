"""
Adapted to use Pygame community edition
https://www.reddit.com/r/pygame/comments/1112q10/pygame_community_edition_announcement/

"""
#main.py
#--- Imports (top level) ---
import os
import gc
import json
import math
import pickle
import queue 
import asyncio
import random
import threading
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pygame
import pygame_gui
import OpenGL

#--- Imports (Modular, local) ---
from graphics import GraphicsRenderer
import user_interface
from game_gc import clean_game_state # garbage collection script
from game_state import save_game_state, load_game_state #save and load functionality - uses data collection function save_current_state
from game_clock import GameClock #game clock for days and seasons
from genetics import Genome, Gene
from food import FoodManager, Food
from organism import Organism, SpatialGrid, food_grid

ray_cast_executor = ThreadPoolExecutor(max_workers=4)
# --- Initialize Pygame ---
pygame.init() 
pygame.font.init()
game_clock = GameClock()
last_season = 0

# --- Display/Screen setup ---
    # Get Display Information 
display_info = pygame.display.Info()
screen_width_full = display_info.current_w
screen_height_full = display_info.current_h
    # Calculate Window Dimensions (1/2 of display) ---
screen_width = screen_width_full // 2
screen_height = screen_height_full // 2


# --- Debug Flag --- draws extra logic visuals 
debug = True  # Debug mode ON for requested debug feedback
debug_fov_mode = "arc" # "full", "arc", or "none" - control FOV drawing style

  

# --- OpenGL ---
# Screen dimensions dynamically set
display_surface = pygame.display.set_mode((screen_width, screen_height), pygame.OPENGL | pygame.DOUBLEBUF)  
graphics_renderer = GraphicsRenderer(screen_width, screen_height)

# --- Font & UI Setup --- UI must come after screen initialization
name_font = pygame.font.SysFont("Segoe UI Emoji", 18)
leaderboard_font = pygame.font.SysFont("Segoe UI Emoji", 24)
info_font = pygame.font.SysFont("Segoe UI Emoji", 16)
selected_organism = None  # Add this with other game state variables
pygame.display.set_caption("Organism Hunting Simulation")

# --- Game Parameters ---
#Configuration Load
from config import (
    screen_width,
    screen_height,
    num_food_clumps,
    food_per_clump,
    clump_radius,
    food_size,
    foodcell_energy_value,
    foodcell_branch_complexity,
    organism_size,
    base_organism_size,
    num_rays,
    min_mating_energy_trigger,
    max_mating_energy_trigger,
    debug,
    debug_fov_mode,
#colors
    white,
    black,
    red,
    green,
    brown,
    blue,
    yellow,
    orange,
    mate_seeking_color,
    light_grey
)
#Food distribution
num_food = num_food_clumps * food_per_clump
food_spawn_radius = min(screen_width, screen_height) * 0.8

num_organisms = 10  # Initial organism count


# --- Incremental Food Generation Parameters ---
    # Base time between food particles when density is below threshold
base_food_generation_interval = 0.05  # Seconds per particle at low density
    # How much the interval decreases per excess food item above threshold
density_speedup_rate = 0.10  
    # Density level (number of food items) where accelerated growth begins
speedup_density_threshold = num_food * 0.15  

# --- Seasonal Food Respawn Parameters ---
seasonal_respawn_interval = 30  # Seconds between seasonal respawn attempts
seasonal_respawn_chance = 0.2  # 70% chance of respawn each season

# --- Naming Prefixes and Suffixes --- Legacy
#carnivore_name_prefixes = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
#carnivore_name_suffixes = ["Predator", "Hunter", "Seeker", "Crawler", "Provoker", "Aggressor", "Stalker", "Ranger"]
#name_prefixes = ["Geo", "Bio", "Eco", "Evo", "Hydro", "Pyro", "Chrono", "Astro", "Cosmo", "Terra"]
#name_suffixes = ["Bot", "Mite", "Pod", "Worm", "Beast", "Fly", "Drake", "Wing", "Crawler", "Shaper"]



# --- Game Functions ---

def generate_food_particle(food_list):
    """Generates food near existing clusters or randomly if no food exists."""
    if food_list:  # Only attempt clustering if food exists
        # Choose random existing food item as cluster center
        cluster_center = random.choice(food_list).position
        # Generate position within clump radius of the cluster center
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, clump_radius)
        x = cluster_center[0] + math.cos(angle) * radius
        y = cluster_center[1] + math.sin(angle) * radius
    else:  # Fallback to random spawn if no food exists
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, food_spawn_radius)
        x = screen_width/2 + math.cos(angle)*radius
        y = screen_height/2 + math.sin(angle)*radius
    return Food((x, y))

def incremental_food_generation(food_list):
    """Generates food particles with density-accelerated spawning near clusters."""
    global food_generation_timer, food_generation_interval

    food_generation_timer -= 1/60

    if food_generation_timer <= 0:
        # Generate new food particle near existing clusters
        new_food = generate_food_particle(food_list)
        food_list.append(new_food)

        # Calculate density-based interval adjustment
        current_density = len(food_list)
        if current_density > speedup_density_threshold:
            # Accelerate spawning proportional to excess density
            density_over_threshold = current_density - speedup_density_threshold
            interval_reduction = density_over_threshold * density_speedup_rate
            food_generation_interval = max(0.05,  # Prevent excessive speed
                base_food_generation_interval - interval_reduction)
        else:
            # Use base interval when below density threshold
            food_generation_interval = base_food_generation_interval

        food_generation_timer = food_generation_interval



def generate_organisms():
    """Generates initial organisms."""
    organisms = []
    for _ in range(num_organisms):
        pos = (random.uniform(0, screen_width), random.uniform(0, screen_height))
        organisms.append(Organism(pos))
    return organisms

def ray_casting_thread_function(organisms, food_list_snapshot, ray_cast_results_queue):
    """Function for the ray casting thread."""
    """Use synchronized food list snapshot"""
    organism_ray_data = {}
    for organism in organisms:
        organism_data = {
            'position': organism.position.copy(),
            'direction': organism.direction,
            'name': organism.name
        }
        # Use synchronized snapshot instead of copying within thread
        detected_food_list = organism.cast_rays_for_food_threaded(food_list_snapshot, organism_data)
        detected_mate_list = organism.cast_rays_for_mate_threaded(organisms, organism_data)
        organism_ray_data[organism] = {
            'food_list': detected_food_list,
            'mate_list': detected_mate_list
        }
    ray_cast_results_queue.put(organism_ray_data)
# --- Save game functionality --- exports expandable/defined data to game_state.py for pickling
def save_current_state():
    game_data = {
        'organisms': organisms,
        'food_list': food_list,
        'game_clock': game_clock.get_state(),
        'last_season': last_season,
        'seasonal_timer': seasonal_timer
        
    }
    save_game_state(game_data)

# --- Food Generation Timers ---
food_generation_timer = base_food_generation_interval
food_generation_interval = base_food_generation_interval

# --- Food Seasonal Respawn ---
seasonal_timer = seasonal_respawn_interval
async def staggered_spawn(new_food, food_list):
    """Spawn food in batches over multiple frames."""
    batch_size = len(new_food) // 8  #Smaller batch sizes (1/8th of total)
    for i in range(0, len(new_food), batch_size):
        food_list.extend(new_food[i:i + batch_size])
        food_grid.update(food_list)  # Update spatial grid incrementally
        await asyncio.sleep(0.016)  # Yield to main thread

# --- Main Game Loop ---
if __name__ == '__main__':
    # ==== Phase 1 - Initialization ====
    # Initialize core game systems and resources
    # ==================================
    
    # Pygame and graphics setup
    pygame.init()
    pygame.font.init()
    display_surface = pygame.display.set_mode((screen_width, screen_height), pygame.OPENGL | pygame.DOUBLEBUF)
    graphics_renderer = GraphicsRenderer(screen_width, screen_height)

    # Food system initialization
    food_manager = FoodManager(screen_width, screen_height)
    # Game systems setup
    game_clock = GameClock()
    last_season = 0
    
    # UI system initialization
    user_interface.UI_MANAGER, user_interface.VISION_BUTTON = user_interface.init_ui(
        screen_width, 
        screen_height
    )
    pygame.display.set_caption("Organism Hunting Simulation")

    # Spatial partitioning systems
    #food_grid = SpatialGrid(cell_size=150)  # Optimized food queries
    organism_grid = SpatialGrid(cell_size=200)  # Optimized organism queries

    # ==== Phase 2 - Game State Loading ====
    # Load saved state or generate initial state
    # ========================================
    loaded_state = load_game_state()
    if loaded_state:
        # Restore from save file
        organisms = loaded_state['organisms']
        food_list = loaded_state['food_list']
        game_clock = GameClock(loaded_state['game_clock'])
        # Preserve thread pool and caches
        Food.initialize_pool()
        for food in food_list:
            if not hasattr(food, 'cached_surface'):
                food.__setstate__(food.__getstate__())  # Rebuild surfaces
    else:
        # Create new game state
        organisms = generate_organisms()
        food_list = food_manager.generate_food(food_spawn_radius)
        game_clock = GameClock()

    # Timer initialization
    food_generation_timer = base_food_generation_interval
    seasonal_timer = seasonal_respawn_interval

    # ==== Phase 3 - Main Loop Setup ====
    # Prepare runtime variables and systems
    # ====================================
    clock = pygame.time.Clock()
    FPS = 60
    running = True
    ray_cast_results_queue = queue.Queue()  # For async ray casting results
    FRAME_COUNTER = 0
    GC_INTERVAL = 10  # Garbage collection interval

    # ==== Phase 4 - Game Loop Execution ====
    # Core game loop running at target FPS
    # =======================================
    while running:
        FRAME_COUNTER += 1

        # ==== Phase 4.1 - Spatial System Update ====
        # Update spatial partitioning grids
        # ==========================================

        food_grid.update(food_list)
        organism_grid.update(organisms)
        # ==== Phase 4.2 - Memory Management ====
        # Perform garbage collection at intervals
        # =======================================
        if FRAME_COUNTER % GC_INTERVAL == 0:
            gc.collect()  # Python garbage collection
            organisms, food_list = clean_game_state(organisms, food_list)  # Game-specific cleanup

        # ==== Phase 4.3 - Frame Data Reset ====
        # Initialize frame-specific containers
        # ======================================
        food_eaten_this_frame = []
        organisms_alive = []
        organisms_children = []
        food_to_remove_frame = []

        # ==== Phase 4.4 - Time Management ====
        # Update game clock and seasonal system
        # =====================================
        delta_time = clock.tick(FPS) / 1000.0
        game_clock.update(delta_time)
        current_season, current_day, current_year = game_clock.get_season_day_year()
        total_days = game_clock.get_total_days()

        # ==== Phase 4.5 - Input Handling ====
        # Process system and user input events
        # ====================================
        for event in pygame.event.get():
            # Window closure handling
            if event.type == pygame.QUIT:
                save_current_state()
                running = False

            # UI event processing
            user_interface.UI_MANAGER.process_events(event)

            # Vision toggle button
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == user_interface.VISION_BUTTON:
                        debug = not debug
                        debug_fov_mode = "arc" if debug else "none"
                        user_interface.VISION_BUTTON.set_text(
                            f'Creature Vision: {"On" if debug else "Off"}'
                        )
            
            # Organism selection handling
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                selected_organism = None
                for org in organisms:
                    distance = math.hypot(org.position[0] - mouse_pos[0], org.position[1] - mouse_pos[1])
                    if distance < organism_size * 2:
                        selected_organism = org
                        break

        # ==== Phase 4.6 - Rendering Prep ====
        # Clear screen and update food
        # =============================
        display_surface.fill(black)
        incremental_food_generation(food_list)
        Food.draw_all(food_list, display_surface)

        # ==== Phase 4.7 - Async Processing ====
        # Start parallel ray casting tasks
        # ======================================
        if FRAME_COUNTER % 2 == 0:  # Process ray casting every other frame
            # Take synchronized snapshot AFTER food removal
            food_list_snapshot = list(food_list)
            organism_list_snapshot = list(organisms)
            future = ray_cast_executor.submit(
                ray_casting_thread_function,
                organism_list_snapshot,
                food_list_snapshot,
                ray_cast_results_queue
            )
        # ==== Phase 4.8 - Organism Simulation ====
        # Update all organism behaviors and states
        # ========================================
        # Process ray casting results
        ray_cast_results = ray_cast_results_queue.get() if not ray_cast_results_queue.empty() else {}

        # Update each organism
        for organism in organisms:
            organism.ray_cast_results = ray_cast_results.get(organism, {})
            result = organism.update(food_list, organisms)
            
            # Track alive organisms
            if organism.energy > 0:
                organisms_alive.append(organism)
            
            # Handle interaction results
            if isinstance(result, Food):
                food_to_remove_frame.append(result)
                food_eaten_this_frame.append(result)
            elif isinstance(result, Organism):
                organisms_children.append(result)

        # Batch process movement
        if organisms_alive:
            Organism.batch_update_positions(organisms_alive)

        # ==== Phase 4.9 - Organism Rendering ====
        # Draw all visible organisms
        # ==========================
        for organism in organisms_alive:
            # Debug visualization
            if debug:
                if organism.current_goal == "food" and organism.ray_cast_results.get('food'):
                    closest_food_pos = organism.ray_cast_results['food'].position
                    pygame.draw.line(display_surface, yellow, 
                                   (int(organism.position[0]), int(organism.position[1])), 
                                   (int(closest_food_pos[0]), int(closest_food_pos[1])), 2)
                if organism.current_goal == "mate_seeking" and organism.ray_cast_results.get('mate'):
                    closest_mate_pos = organism.ray_cast_results['mate'].position
                    pygame.draw.line(display_surface, red, 
                                   (int(organism.position[0]), int(organism.position[1])), 
                                   (int(closest_mate_pos[0]), int(closest_mate_pos[1])), 2)
            
            # Actual organism drawing
            organism.draw(display_surface, selected_organism)

        # ==== Phase 4.10 - Food Management ====
        # Update food state and seasonal respawn
        # ======================================
        # Remove eaten food
        for eaten_food in food_to_remove_frame:
            if eaten_food in food_list:
                food_list.remove(eaten_food)
        food_grid.update(food_list)

        # Add new organisms
        organisms_alive.extend(organisms_children)
        organisms = organisms_alive

        # Seasonal food respawn
        if current_season != last_season:
            if random.random() < seasonal_respawn_chance:
                new_food = food_manager.generate_food(food_spawn_radius)
                asyncio.run(staggered_spawn(new_food, food_list))
                if debug:
                    print(f"Season {current_season} began - respawned {len(new_food)} food")
            last_season = current_season

        # ==== Phase 4.11 - UI Rendering ====
        # Update and draw user interface
        # ===============================
        user_interface.UI_MANAGER.update(delta_time)
        user_interface.UI_MANAGER.draw_ui(display_surface)
        
        # Leaderboard and organism info
        user_interface.draw_leaderboard(
            display_surface,
            organisms,
            current_season,
            current_day,
            current_year,
            leaderboard_font, 
            info_font          
        )        
        if selected_organism:
            user_interface.draw_organism_info(
                display_surface,
                selected_organism,
                info_font,     
                screen_width    
            )

        # ==== Phase 4.12 - Final Rendering ====
        # Push frame to display
        # =====================
        graphics_renderer.render(display_surface)
        pygame.display.flip()

    # ==== Phase 5 - Cleanup ====
    # Shutdown systems and release resources
    # ======================================
    print("Game loop ended, cleaning up...")
    ray_cast_executor.shutdown(wait=True, cancel_futures=True) 
    graphics_renderer.cleanup()
    if Food._matrix_pool is not None:
        Food._matrix_pool.shutdown(wait=True, cancel_futures=True) 
    pygame.font.quit()
    pygame.quit()
    print("Game exited successfully.")