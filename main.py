"""
Adapted to use Pygame community edition
https://www.reddit.com/r/pygame/comments/1112q10/pygame_community_edition_announcement/

"""
#main.py
#--- Imports (top level) ---
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
import user_interface
from game_gc import clean_game_state # garbage collection script
from game_state import save_game_state, load_game_state #save and load functionality - uses data collection function save_current_state
from game_clock import GameClock #game clock for days and seasons
from genetics import Genome, Gene
from graphics import GraphicsRenderer

ray_cast_executor = ThreadPoolExecutor(max_workers=4)
# --- Initialize Pygame ---
pygame.init() 
pygame.font.init()
game_clock = GameClock()
last_season = 0

# --- Debug Flag --- draws extra logic visuals 
debug = True  # Debug mode ON for requested debug feedback
debug_fov_mode = "arc" # "full", "arc", or "none" - control FOV drawing style

# --- Display/Screen setup ---
    # Get Display Information 
display_info = pygame.display.Info()
screen_width_full = display_info.current_w
screen_height_full = display_info.current_h
    # Calculate Window Dimensions (1/2 of display) ---
screen_width = screen_width_full // 2
screen_height = screen_height_full // 2
  

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

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
brown = (139, 69, 19)
blue = (0, 0, 255)
yellow = (255, 255, 0)
orange = (255, 165, 0)  
mate_seeking_color = red
light_grey = (100, 100, 100, 50) # Define light grey for FOV outline, with alpha for transparency

# --- Game Parameters ---
#Food shape and energy value calculations
"""Complexity → Number of Cells → Total Energy 🔄"""
foodcell_branch_complexity = 2   # Branch length range: 1-3 cells, Controls visual branching complexity
foodcell_energy_value = 2   # Energy per matrix cell
"""
FOOD ENERGY CALCULATION 
-------------------------------------
Formula: Energy = (1 + 8×branch_length) × foodcell_energy_value
Where:
- 1 = Central core cell (always present)
- 8 = Branch directions (N/S/E/W + diagonals)
- branch_length = random.randint(1, foodcell_branch_complexity + 1)
                  (1-3 cells with current settings)
+---------------+-----------------+-----------------+-----------------+
|               | Minimum Energy     | Maximum Energy  | Average Energy  |
+---------------+-----------------+-----------------+-----------------+
| Branch Length   | 1 cell             | 3 cells                   | 2 cells         |
| Total Cells         | 1 + 8×1 = 9   | 1 + 8×3 = 25         | 1 + 8×2 = 17   |
| Total Energy      | 9 × 3 = 27    | 25 × 3 = 75         | 17 × 3 = 51     |
+---------------+-----------------+-----------------+-----------------+
"""


#Food distribution
num_food_clumps = 25
food_per_clump = 8
clump_radius = 30
num_food = num_food_clumps * food_per_clump
food_size = 5
food_spawn_radius = min(screen_width, screen_height) * 0.8

num_organisms = 20  # Initial organism count
organism_size = 10
base_organism_size = 8
num_rays = 3
min_mating_energy_trigger = 400
max_mating_energy_trigger = 600

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

# --- Helper Functions ---
def get_distance_np(positions1, positions2):
    """Calculates Euclidean distance between two arrays of points using NumPy."""
    positions1 = np.array(positions1)
    positions2 = np.array(positions2)
    return np.sqrt(np.sum((positions1 - positions2)**2, axis=1))

def normalize_angle(angle):
    """Keeps angle within 0 to 360 degrees."""
    return angle % 360

#v1.01 batch food rendering with dirty rectangles & spatial partitioning for food
class SpatialGrid:
    def __init__(self, cell_size=100):
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        self._prev_positions = {}

    def update(self, entities):
        """Update entity positions in grid (2x faster than brute force)"""
        new_grid = defaultdict(list)
        for entity in entities:
            cell = self._get_cell(entity.position)
            new_grid[cell].append(entity)
            # Store previous position for delta checks
            self._prev_positions[entity] = entity.position
        self.grid = new_grid

    def query_radius(self, position, radius):
        """Get entities within radius using grid lookup (O(1) vs O(n))"""
        cx, cy = self._get_cell(position)
        search_radius = int(math.ceil(radius / self.cell_size))
        
        results = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                cell = (cx + dx, cy + dy)
                results.extend(self.grid.get(cell, []))
        return results

    def _get_cell(self, position):
        return (int(position[0] // self.cell_size), 
                int(position[1] // self.cell_size))


# --- Food Class ---
class Food:
    # Class-level cache for shared surfaces (key: hash of shape_matrix tuple)
    shape_cache = {}  # Add this line
    # Class-level pool for matrix generation
    _matrix_pool = None  # Initialize as None
    #v1.01 Class-level batch rendering system
    _batch_surface = None
    _dirty_rects = []

    @staticmethod
    def initialize_pool():
        if Food._matrix_pool is None:
            Food._matrix_pool = ThreadPoolExecutor()  # Use threads instead of processes
    @staticmethod
    def generate_branching_matrix_async():
        """Generate a branching matrix in a separate process."""
        Food.initialize_pool()  # Ensure the pool is initialized
        size = 5 + 2 * foodcell_branch_complexity
        if size % 2 == 0:
            size += 1  # Ensure odd symmetry
        matrix = np.zeros((size, size), dtype=int)
        center = size // 2
        matrix[center][center] = 2  # Core

        # Generate randomized branches
        directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        for dx, dy in directions:
            branch_length = random.randint(1, foodcell_branch_complexity + 1)
            x, y = center, center
            for _ in range(branch_length):
                x += dx
                y += dy
                if 0 <= x < size and 0 <= y < size:
                    matrix[x][y] = 1
        return matrix
    #v1.01 batch food rendering with dirty rectangles & spatial partitioning for food
    @staticmethod
    def update_batch_surface(screen_width, screen_height):
        """Initialize batch surface once"""
        if Food._batch_surface is None:
            Food._batch_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
            Food._batch_surface.fill((0, 0, 0, 0))


    def __init__(self, position, shape_matrix=None):
        self.position = np.array(position, dtype=float)
        
        if shape_matrix is None:
            # Generate matrix asynchronously
            future = self._matrix_pool.submit(self.generate_branching_matrix_async)  # Use submit instead of apply_async
            self.shape_matrix = future.result()  # Get the result from the future
        else:
            self.shape_matrix = np.array(shape_matrix)
        
        # Calculate energy value
        self.energy_value = self.calculate_energy()
        
        # Cache handling
        shape_hash = hash(tuple(map(tuple, self.shape_matrix)))
        if shape_hash not in Food.shape_cache:
            Food.shape_cache[shape_hash] = self._create_cached_surface()
        self.cached_surface = Food.shape_cache[shape_hash]
    def calculate_energy(self):
        """Calculate energy based on number of active cells in matrix"""
        return np.count_nonzero(self.shape_matrix) * foodcell_energy_value  # 10 energy per cell
    def generate_branching_matrix(self):
        """Generates a food structure matrix with branches radiating from the center using global foodcell_branch_complexity."""
        size = 5 + 2 * foodcell_branch_complexity
        if size % 2 == 0:
            size += 1  # Ensure odd symmetry
        matrix = np.zeros((size, size), dtype=int)
        center = size // 2
        matrix[center][center] = 2  # Core

        # Generate randomized branches
        directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        for dx, dy in directions:
            branch_length = random.randint(1, foodcell_branch_complexity + 1)
            x, y = center, center
            for _ in range(branch_length):
                x += dx
                y += dy
                if 0 <= x < size and 0 <= y < size:
                    matrix[x][y] = 1
        return matrix


    def _create_cached_surface(self):
        """Internal method to generate a surface for the shape_matrix."""
        matrix_height, matrix_width = self.shape_matrix.shape
        cell_size = int(food_size * 0.95)
        
        surf_width = matrix_width * cell_size
        surf_height = matrix_height * cell_size
        surface = pygame.Surface((surf_width, surf_height), pygame.SRCALPHA)
        
        # Draw the shape
        for row in range(matrix_height):
            for col in range(matrix_width):
                cell_value = self.shape_matrix[row, col]
                if cell_value == 0:
                    continue
                x = col * cell_size
                y = row * cell_size
                if cell_value == 1:
                    pygame.draw.circle(surface, green, (x, y), int(food_size * 0.5))
                elif cell_value == 2:
                    pygame.draw.circle(surface, brown, (x, y), int(food_size * 0.8))
        return surface

    #batch food rendering
    @staticmethod
    def draw_all(food_list, surface):
        """Batch draw with dirty rectangle tracking and safe initialization"""
        """
            v1.01 Dirty rectangles 
            v1.02 Added surface validation
        """
        # Early exit if no food to draw
        if not food_list:
            return
        
        # Initialize or update batch surface if needed
        if (Food._batch_surface is None or 
            Food._batch_surface.get_size() != surface.get_size()):
            Food.update_batch_surface(surface.get_width(), surface.get_height())
        
        # Clear previous frame's dirty areas
        for rect in Food._dirty_rects:
            Food._batch_surface.fill((0, 0, 0, 0), rect)
        
        # Track new dirty rectangles
        new_dirty = []
        for food in food_list:
            rect = food.cached_surface.get_rect(
                center=(int(food.position[0]), int(food.position[1]))
            )
            Food._batch_surface.blit(food.cached_surface, rect.topleft)
            new_dirty.append(rect)
        
        Food._dirty_rects = new_dirty
        surface.blit(Food._batch_surface, (0, 0))

    def __getstate__(self):
            """Save position and legacy shape_matrix for compatibility."""
            return {
                'position': self.position.tolist(),
                'shape_matrix': self.shape_matrix.tolist()  # Preserve for old saves
            }
    def __setstate__(self, state):
        """Load state, prioritizing global complexity over saved shapes."""
        self.position = np.array(state['position'], dtype=float)
        
        # Regenerate matrix using current global setting
        self.shape_matrix = self.generate_branching_matrix()
        
        # Recalculate energy value after loading
        self.energy_value = self.calculate_energy()  # Add this line
        
        # Rebuild surface cache
        shape_hash = hash(tuple(map(tuple, self.shape_matrix)))
        if shape_hash not in Food.shape_cache:
            Food.shape_cache[shape_hash] = self._create_cached_surface()
        self.cached_surface = Food.shape_cache[shape_hash]


# --- Pre-caching Food Shapes ---
def pre_cache_food_shapes():
    """Pre-generate common food shapes to avoid runtime overhead."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(Food.generate_branching_matrix_async) for _ in range(20)]
        for future in futures:
            matrix = future.result()
            Food((0, 0), matrix)  # Create a dummy Food object to cache the surface

# --- Organism Class ---
class Organism:
    def __init__(self, position, generation_number=1, direction=None, genome=None, 
                 energy=None, name=None, current_goal=None, base_color=None, 
                 organism_size=None, lifespan=None, age=None):

        # Initialize genome first
        self.genome = genome if genome is not None else Genome()


        #Memory config(config file in the future)
        self.memory_grid = defaultdict(lambda: (0.0, 0.0))  # (cell_x, cell_y): (density_score, last_seen_time)
        self.memory_decay_rate = 0.98  # Slower decay (from 0.95)
        self.memory_cell_size = 75  # More granular cells (from 150)
        self.current_cell_target = None
        self.target_persistence = 0    
        #Memory system
        self.last_memory_check = pygame.time.get_ticks()  # Initialize memory check timer
        self.current_cell_target = None  # Initialize current navigation target    
        # Dynamically set traits from genome
        for trait_name in self.genome.genes:
            setattr(self, trait_name, self.genome.get_trait(trait_name))
        # Existing position/direction initialization
        self.position = np.array(position, dtype=float)
        self.direction = direction if direction is not None else random.uniform(0, 360)
        # Modified energy calculation using metabolism
        self.energy = energy if energy is not None else 100 * self.metabolism
        # Set color from genome
        self.base_color = base_color if base_color is not None else self.genome.get_color()

        self.generation_number = generation_number
        self.name = name if name is not None else self.generate_name()
        self.targeted_food = None
        self.current_goal = current_goal if current_goal is not None else "food"
        self.age = age if age is not None else 0
        self.last_direction_change_time = pygame.time.get_ticks()
        self.wander_turn_interval = random.uniform(1000, 3000)
        self.ray_cast_results = {}
    # Vectorized Organism Updates:
    @staticmethod
    def batch_update_positions(organisms):
        if not organisms:
            return
        
        # Vectorized position update
        directions = np.array([org.direction for org in organisms])
        speeds = np.array([org.speed for org in organisms])
        positions = np.array([org.position for org in organisms])
        
        # Calculate movement vectors
        radians = np.radians(directions)
        offsets = np.stack([
            np.cos(radians) * speeds,
            -np.sin(radians) * speeds
        ], axis=1)
        
        # Update positions and energy
        new_positions = (positions + offsets) % [screen_width, screen_height]
        for i, org in enumerate(organisms):
            org.position = new_positions[i]
            org.energy -= org.metabolism * 0.01 * org.speed

#--- Organis Naming conventions ---
    def generate_name(self):
        """Generates a pronounceable name with capitalization and single hyphen."""
        syllables = self._load_syllables()
        num_syllables = random.randint(4, 5)  # 4 or 5 syllables

        name_parts = [random.choice(syllables) for _ in range(num_syllables)]

        # Capitalize the first syllable
        name_parts[0] = name_parts[0].capitalize()

        # Intuitively capitalize another syllable (language theory inspired)
        if num_syllables > 2:
            # Capitalize a syllable roughly in the middle or toward the end
            capitalize_index = random.randint(num_syllables // 2, num_syllables - 2)
            name_parts[capitalize_index] = name_parts[capitalize_index].capitalize()

            # Join the syllables with a single hyphen before the second capitalized part
            first_part = "".join(name_parts[:capitalize_index])
            second_part = "".join(name_parts[capitalize_index:])
            return f"{first_part}-{second_part}-({self.generation_number})"
        else:
            # If no second capitalization, join all syllables without hyphens
            return "".join(name_parts) + f"-({self.generation_number})"

    def _load_syllables(self):
        """Loads syllables from config/syllables.json, creates if needed."""
        config_dir = Path("config")
        syllables_path = config_dir / "syllables.json"

        try:
            # Create config directory if needed
            config_dir.mkdir(exist_ok=True)

            # Create syllables.json if it doesn't exist
            if not syllables_path.exists():
                with open(syllables_path, "w") as f:
                    json.dump({"syllables": [
                        "va", "mor", "fin", "thor", "ly", "dra", "gla", "vis", "nox", "zen",
                        "pyr", "thos", "rel", "vyn", "zyl", "quor", "myr", "jex", "fex", "wix",
                        "lox", "rex", "vex", "zax", "plox", "trix", "blor", "grak", "zind", "vorth",
                        "quil", "mord", "flax", "nix", "quiv", "jolt", "brel", "dorn", "fyr", "glyn",
                        "hax", "jorm", "krel", "lorn", "myn", "phex", "qyr", "ryn", "syx", "tyr",
                        "vrox", "wyx", "zorn", "plix", "trax", "blyn", "drax", "frol", "grix", "hurn",
                        "sa", "krax", "lynn", "morth", "prox", "quyl", "ryx", "syl", "tron", "vix",
                        "wyn", "zorth", "plax", "tryn", "blix", "dron", "fyx", "glynx", "hurm", "jyx",
                        "kryl", "lyx", "morx", "pryx", "quorn", "rynox", "sylx", "trox", "vlyn", "wyrm",
                        "nes", "plorn", "tryx", "blorn", "drix", "va", "nes", "sa", "bri", "an", "na", "el", "i", "za", "beth",
                        "char", "lotte", "so", "phi", "a", "em", "ma", "ol", "iv", "ia",
                        "no", "ah", "liam", "jac", "son", "eth", "an", "hen", "ry",
                        "will", "am", "ben", "ja", "min", "a", "vi", "a", "dan",
                        "iel", "mat", "thew", "chris", "to", "pher", "and", "rew", "jos",
                        "eph", "jam", "es", "rob", "ert", "mic", "hael", "dav", "id",
                        "ste", "ven", "ge", "org", "ed", "ward", "pat", "rick", "al",
                        "ex", "an", "der", "nic", "ho", "las", "sam", "uel", "jon", "ath",
                        "an", "tim", "o", "thy", "vin", "cent", "lau", "ren", "tay",
                        "lor", "mad", "i", "son", "hay", "ley", "han", "nah", "paig",
                        "e", "mor", "gan", "ash", "ley", "kait", "lyn", "court", "ney",
                        "shel", "by", "whit", "ney", "steph", "anie", "am", "ber", "rach",
                        "el", "au", "drey", "nat", "alie", "sab", "ri", "na", "cam", "eron"
                    ]}, f, indent=4)  # Expanded additional syllables here
            with open(syllables_path, "r") as f:
                data = json.load(f)
                return data["syllables"]

        except (FileNotFoundError, json.JSONDecodeError, OSError) as e: #add os error to catch permission errors.
            print(f"Error loading/creating syllables: {e}. Using default syllables.")
            return ["kra", "mor", "fin", "thor", "ly", "dra", "gla", "vis", "nox", "zen", "pyr", "thos", "rel", "vyn", "zyl", "quor", "myr", "jex", "fex", "wix", "lox", "rex", "vex", "zax", "plox", "trix"]  # Default syllables
    def cast_rays_for_food_threaded(self, food_list, organism_data):
        """Spatial grid-optimized food detection (5x faster)"""
        organism_pos = organism_data['position']
        organism_direction = organism_data['direction']
        
        # Use spatial grid to limit checks
        nearby_food = food_grid.query_radius(organism_pos, self.sight_range)
        if not nearby_food:
            return [] # Return empty list if no food

        num_rays_val = num_rays
        start_angle = organism_direction - self.sight_fov / 2
        angles = normalize_angle(start_angle + np.arange(num_rays_val) * (self.sight_fov / (num_rays_val - 1) if num_rays_val > 1 else 0) )
        radians = np.radians(angles)

        ray_ends_x = organism_pos[0] + np.cos(radians) * self.sight_range
        ray_ends_y = organism_pos[1] - np.sin(radians) * self.sight_range
        ray_ends = np.stack([ray_ends_x, ray_ends_y], axis=-1)

        food_positions = np.array([food.position for food in food_list])
        distances_to_food = get_distance_np(organism_pos[np.newaxis, :], food_positions)

        # --- Angle to Food Calculation - RE-EXAMINE THIS ---
        angles_to_food = -np.degrees(np.arctan2(food_positions[:, 1] - organism_pos[1], food_positions[:, 0] - organism_pos[0]))
        # --- Angle Difference Calculation - RE-EXAMINE THIS ---
        angle_diffs = (angles_to_food - organism_direction) % 360
        angle_differences = np.where(angle_diffs > 180, 360 - angle_diffs, angle_diffs)

        food_in_fov_mask = (distances_to_food <= self.sight_range) & (angle_differences <= self.sight_fov / 2)
        food_in_fov_indices = np.where(food_in_fov_mask)[0]

        detected_food = [] # List to store food in FOV with distances
        if food_in_fov_indices.size > 0:
            for index in food_in_fov_indices:
                detected_food.append({'food': food_list[index], 'distance': distances_to_food[index]}) # Store food object and distance
        return detected_food # Return list of detected food


    def cast_rays_for_mate_threaded(self, organisms, organism_data):
        """Casts rays to detect mates within FOV and range and returns all in FOV with distances."""
        organism_pos = organism_data['position']
        organism_direction = organism_data['direction']

        potential_mates = [org for org in organisms if org != self and org.energy > 0]
        if not potential_mates:
            return [] # Return empty list if no mates

        num_rays_val = num_rays
        start_angle = organism_direction - self.sight_fov / 2
        angles = normalize_angle(start_angle + np.arange(num_rays_val) * (self.sight_fov / (num_rays_val - 1) if num_rays_val > 1 else 0))
        radians = np.radians(angles)

        ray_ends_x = organism_pos[0] + np.cos(radians) * self.sight_range
        ray_ends_y = organism_pos[1] - np.sin(radians) * self.sight_range
        ray_ends = np.stack([ray_ends_x, ray_ends_y], axis=-1)

        mate_positions = np.array([mate.position for mate in potential_mates])
        distances_to_mates = get_distance_np(organism_pos[np.newaxis, :], mate_positions)

        angles_to_mates = -np.degrees(np.arctan2(mate_positions[:, 1] - organism_pos[1], mate_positions[:, 0] - organism_pos[0]))
        angle_diffs = (angles_to_mates - organism_direction) % 360
        angle_differences = np.where(angle_diffs > 180, 360 - angle_diffs, angle_diffs)

        mates_in_fov_mask = (distances_to_mates <= self.sight_range) & (angle_differences <= self.sight_fov / 2)
        mates_in_fov_indices = np.where(mates_in_fov_mask)[0]

        detected_mates = [] # List to store mates in FOV with distances
        if mates_in_fov_indices.size > 0:
            for index in mates_in_fov_indices:
                detected_mates.append({'mate': potential_mates[index], 'distance': distances_to_mates[index]}) # Store mate object and distance
        return detected_mates # Return list of detected mates


    def hunt_food(self, closest_food):
        """Move towards the closest food."""
        if closest_food:
            angle_to_food_radians = math.atan2(closest_food.position[1] - self.position[1], closest_food.position[0] - self.position[0])
            angle_to_food_degrees = math.degrees(angle_to_food_radians)
            angle_to_food_degrees = -angle_to_food_degrees # NEGATE angle_to_food_degrees - TEST FOR REVERSAL
            angle_diff = normalize_angle(angle_to_food_degrees - self.direction)
            if angle_diff > 180:
                angle_diff -= 360
            turn_rate = 10  # Degrees per frame - Keep moderate turn rate for now
            self.direction += max(-turn_rate, min(turn_rate, angle_diff))
            self.direction = normalize_angle(self.direction)
            self.move_forward()

    def hunt_mate(self, closest_mate):
        """Move towards the closest mate."""
        if closest_mate:
            angle_to_mate_radians = math.atan2(closest_mate.position[1] - self.position[1], closest_mate.position[0] - self.position[0])
            angle_to_mate_degrees = math.degrees(angle_to_mate_radians)
            angle_to_mate_degrees = -angle_to_mate_degrees # NEGATE angle_to_mate_degrees - APPLY SAME FIX AS HUNT_FOOD
            # Smooth turning towards mate - same logic as hunt_food
            angle_diff = normalize_angle(angle_to_mate_degrees - self.direction)
            if angle_diff > 180:
                angle_diff -= 360
            turn_rate = 10  # Degrees per frame, ADJUSTED turn_rate to match hunt_food for now
            self.direction += max(-turn_rate, min(turn_rate, angle_diff))  # Clamp turn rate
            self.direction = normalize_angle(self.direction)
            self.move_forward()

    def move_forward(self):
        """Move organism forward based on speed and adjust energy expenditure based on size and speed."""
        # Baseline energy expenditure for size 10 and speed 1
        baseline_size = base_organism_size
        baseline_speed = 1
        # Baseline energy cost per frame
        baseline_energy_cost = self.metabolism / 10 #0.1  

        # Calculate energy expenditure based on size and speed
        size_factor = self.size / baseline_size
        speed_factor = self.speed / baseline_speed

        # Adjust energy expenditure based on size and speed
        energy_cost = baseline_energy_cost * size_factor * speed_factor

        # Move the organism
        radians = math.radians(self.direction)
        dx = math.cos(radians) * self.speed
        dy = -math.sin(radians) * self.speed
        self.position += np.array([dx, dy])
        self.position[0] %= screen_width  # Wrap around screen using NumPy modulo
        self.position[1] %= screen_height

        # Deduct energy based on the calculated energy cost
        self.energy -= energy_cost

    def mate(self, other_parent):
        """Sexual reproduction with another organism using genetic inheritance"""
        # Energy check - REQUIRED TO PREVENT OVERPRODUCTION
        half_mating_energy = min_mating_energy_trigger * 0.5
        if (self.energy < half_mating_energy or 
            other_parent.energy < half_mating_energy):
            return None

        # Energy cost - CRUCIAL FOR BALANCE
        energy_cost = half_mating_energy * 0.75  # 75% of half trigger
        self.energy -= energy_cost
        other_parent.energy -= energy_cost

        # Create child genome
        child_genome = Genome(parent1=self.genome, parent2=other_parent.genome)
        
        # Create child with genetic traits
        child = Organism(
                position=((self.position[0] + other_parent.position[0])/2,
                (self.position[1] + other_parent.position[1])/2),
                generation_number=max(self.generation_number, other_parent.generation_number) + 1,
                genome=child_genome,
                base_color=child_genome.get_color()  # Use genome's color method
            )

        # Share memories
        for cell, (density, _) in other_parent.memory_grid.items():
            current_density = self.memory_grid.get(cell, (0, 0))[0]
            if density > current_density:
                self.memory_grid[cell] = (density, pygame.time.get_ticks()/1000)
        
        return child
    def update(self, food_list, organisms):
        """Update organism behavior based on current goal. Ray casting results are expected to be pre-calculated."""
        #Update Memory
        self._update_food_memory(food_list)
        #Update Age
        self.age += 1/60
        if self.energy <= 0:
            return False
        if self.age >= self.lifespan:
            self.energy = 0
            if debug:
                print(f"{self.name}: Died of old age at {int(self.age)} seconds (lifespan: {int(self.lifespan)} seconds).")
            return False

        if self.current_goal == "mate_seeking":
            if self.energy < (min_mating_energy_trigger*0.75):
                if debug:
                    print(f"{self.name}: Energy dropped below mating threshold ({int(self.energy)} < {min_mating_energy_trigger}). Switching to food seeking.")
                self.current_goal = "food"
                return self.update(food_list, organisms)  # Re-run update to immediately process new goal

            detected_mates_list = self.ray_cast_results.get('mate_list', []) # Get list of detected mates
            if detected_mates_list:
                # Find closest mate
                closest_mate_info = min(detected_mates_list, key=lambda item: item['distance']) # Find mate with min distance
                closest_mate = closest_mate_info['mate']
                self.hunt_mate(closest_mate)
            else:
                self.move_forward()  # Keep moving in mate seeking, no wandering

        elif self.current_goal == "food":
            detected_food_list = self.ray_cast_results.get('food_list', []) # Get list of detected food
            if detected_food_list:
                closest_food_info = min(detected_food_list, key=lambda item: item['distance'])
                closest_food = closest_food_info['food']

                if self.targeted_food is None or self.targeted_food != closest_food: # Target new food only if no target or target changed
                    self.targeted_food = closest_food # Lock on to closest food
                self.hunt_food(self.targeted_food) # Hunt the *targeted* food

            else:
                self.targeted_food = None # Clear target if no food detected
                self.current_goal = "wander"
                return self.update(food_list, organisms)  # Re-run update for wander
        elif self.current_goal == "wander":
            # --- State Transition Stability ---
            if self.current_cell_target:  # Continue existing navigation
                self._navigate_to_cell(self.current_cell_target)
                
            # Only check for new targets every 2 seconds
            if pygame.time.get_ticks() - self.last_memory_check > 2000:
                best_cell = self._find_most_attractive_cell()
                self.last_memory_check = pygame.time.get_ticks()  # Update the last check time
                if best_cell:
                    self.current_cell_target = best_cell

            # Require 3 consecutive frames of food detection
            detected_food_list = self.ray_cast_results.get('food_list', []) # Check for food while wandering
            if len(detected_food_list) >= 3:  # More stable detection
                closest_food_info = min(detected_food_list, key=lambda item: item['distance'])
                self.targeted_food = closest_food_info['food']
                self.current_goal = "food"
                self.current_cell_target = None
                return

            # Fallback to random wandering if no memory target
            if not self.current_cell_target:
                current_time = pygame.time.get_ticks()
                if current_time - self.last_direction_change_time >= self.wander_turn_interval:
                    self.direction += random.uniform(-90, 90)  # Wider wander turns
                    self.direction = normalize_angle(self.direction)
                    self.last_direction_change_time = current_time
                    self.wander_turn_interval = random.uniform(1000, 3000)  # reset interval
                self.move_forward()

        # Check for eating food (remains the same)
        eaten_food = None  # Track eaten food item
        for food in food_list:
            if get_distance_np(self.position[np.newaxis, :], food.position[np.newaxis, :])[0] < organism_size / 2 + food_size / 2:  # NumPy distance check
                self.eat_food(food)
                eaten_food = food  # Mark food as eaten
                break  # Eat only one food per frame

        if eaten_food:
            return eaten_food  # Return eaten food to remove from list

        # --- Energy-dependent Mating Logic (Corrected Child Handling) ---
        mating_probability = 0.0
        if self.energy >= min_mating_energy_trigger:
            if self.energy >= max_mating_energy_trigger:
                mating_probability = 1.0
            else:
                probability_range = 1.0 - 0.01
                energy_range = max_mating_energy_trigger - min_mating_energy_trigger
                energy_over_min = self.energy - min_mating_energy_trigger
                mating_probability = 0.01 + (energy_over_min / energy_range) * probability_range
        else:
            mating_probability = 0.0

        if random.random() < mating_probability:
            nearby_mates = []
            for other_org in organisms:
                if other_org != self and other_org.energy > 0 and get_distance_np(self.position[np.newaxis, :], other_org.position[np.newaxis, :])[0] < organism.size * 3:  # NumPy distance check
                    nearby_mates.append(other_org)

            if nearby_mates:
                mate = random.choice(nearby_mates)
                child = self.mate(mate)
                if child:
                    if debug: # Debug print to confirm child creation and return
                        print(f"{self.name}: Successfully mated with {mate.name} and created child: {child.name}")
                    self.current_goal = "food"  # After mating, seek food
                    return child  # Return child organism
            else:
                self.current_goal = "mate_seeking"

        return False  # No child created, no food eaten, organism is still alive
    def _update_food_memory(self, food_list):
        now = pygame.time.get_ticks() / 1000  # Current time in seconds
        visible_cells = set()
        
        # Get cells in FOV using spatial grid
        nearby_food = food_grid.query_radius(self.position, self.sight_range)
        cell_density = defaultdict(int)
        
        for food in nearby_food:
            cell_x = int(food.position[0] // self.memory_cell_size)
            cell_y = int(food.position[1] // self.memory_cell_size)
            cell_density[(cell_x, cell_y)] += 1
            visible_cells.add((cell_x, cell_y))
        
        # Update memory with decay
        for cell in visible_cells:
            density = cell_density[cell] / (self.memory_cell_size**2)  # Density per pixel²
            self.memory_grid[cell] = (density, now)
        
        # Decay old memories
        for cell in list(self.memory_grid.keys()):
            if cell not in visible_cells:
                old_density, last_time = self.memory_grid[cell]
                time_diff = now - last_time
                decayed_density = old_density * (self.memory_decay_rate ** time_diff)
                if decayed_density < 0.01:
                    del self.memory_grid[cell]
                else:
                    self.memory_grid[cell] = (decayed_density, last_time)

    def eat_food(self, food):
        """Organism eats food and gains energy proportional to complexity"""
        self.energy += food.energy_value

    def _find_most_attractive_cell(self):
        now = pygame.time.get_ticks() / 1000
        current_cell = (
            int(self.position[0] // self.memory_cell_size),
            int(self.position[1] // self.memory_cell_size)
        )
        
        best_score = -1
        best_cell = None
        
        # Iterate through all cells in memory
        for (cell_x, cell_y), (density, last_seen) in self.memory_grid.items():
            age = now - last_seen
            distance = math.hypot(
                (cell_x - current_cell[0]) * self.memory_cell_size,
                (cell_y - current_cell[1]) * self.memory_cell_size
            )
            
            # Calculate score with persistence bonus
            score = density * math.exp(-age / 300) * (1 - distance / (self.sight_range * 1.5))
            
            # Add persistence bonus if revisiting the same cell
            if (cell_x, cell_y) == self.current_cell_target:
                score *= 1.5  # Bonus for revisiting the same cell
            
            # Track the best cell
            if score > best_score:
                best_score = score
                best_cell = (cell_x, cell_y)
        
        return best_cell


    def _navigate_to_cell(self, target_cell):
        if self.current_cell_target != target_cell:
            self.target_persistence = 0
            self.current_cell_target = target_cell
            
        self.target_persistence += 1
        
        # Only recalculate direction every 10 frames
        if self.target_persistence % 10 == 0:
            target_pos = (
                (target_cell[0] + random.uniform(0.3, 0.7)) * self.memory_cell_size,  # Fixed closing parenthesis
                (target_cell[1] + random.uniform(0.3, 0.7)) * self.memory_cell_size   # Fixed closing parenthesis
            )
            angle = math.degrees(math.atan2(target_pos[1]-self.position[1], 
                                  target_pos[0]-self.position[0]))
            self.direction = normalize_angle(-angle)
        
        self.move_forward()

    def draw(self, surface):
        """Draw the organism and debug rays - Optimized Version."""
        # --- 1. Draw Base Organism (Circle) ---
        pygame.draw.circle(surface, self.base_color, (int(self.position[0]), int(self.position[1])), int(self.size))

        # --- 2. Draw White Ring if Selected ---
        if selected_organism == self:
            pygame.draw.circle(surface, white, (int(self.position[0]), int(self.position[1])), int(self.size) + 5, 2)  # White ring around selected organism

        # --- 3. Mate Seeking Indicator ---
        if self.current_goal == "mate_seeking":
            pygame.draw.circle(surface, yellow, (int(self.position[0]), int(self.position[1])), int(self.size) + 3, 2)

        # --- 4. Direction Indicator and Eyes ---
        # --- 4.1. Calculate Head Position (Direction Indicator End) ---
        head_x = self.position[0] + math.cos(math.radians(self.direction)) * self.size
        head_y = self.position[1] - math.sin(math.radians(self.direction)) * self.size
        pygame.draw.line(surface, white, (int(self.position[0]), int(self.position[1])), (int(head_x), int(head_y)), 3)

        # --- 4.2. Calculate Eye Positions (Inward and Closer) ---
        eye_offset_distance = self.size / 1.5  # Reduced offset for inward eyes
        eye_radius = self.size / 2
        pupil_radius = eye_radius / 2
        pupil_offset_distance = eye_radius / 3

        # Angles for eye positioning relative to organism's center, not head point
        left_eye_angle_radians = math.radians(self.direction + 90)
        right_eye_angle_radians = math.radians(self.direction - 90)

        # Eye positions now offset from organism center, pulling them inwards
        left_eye_x = self.position[0] + math.cos(left_eye_angle_radians) * eye_offset_distance
        left_eye_y = self.position[1] - math.sin(left_eye_angle_radians) * eye_offset_distance
        right_eye_x = self.position[0] + math.cos(right_eye_angle_radians) * eye_offset_distance
        right_eye_y = self.position[1] - math.sin(right_eye_angle_radians) * eye_offset_distance
        # Pupil offset remains relative to direction for 'looking' effect
        pupil_x_offset = math.cos(math.radians(self.direction)) * pupil_offset_distance
        pupil_y_offset = -math.sin(math.radians(self.direction)) * pupil_offset_distance

        left_pupil_x = left_eye_x + pupil_x_offset
        left_pupil_y = left_eye_y + pupil_y_offset
        right_pupil_x = right_eye_x + pupil_x_offset
        right_pupil_y = right_eye_y + pupil_y_offset

        eye_color = white
        pupil_color = black
        pygame.draw.circle(surface, eye_color, (int(left_eye_x), int(left_eye_y)), int(eye_radius))  # Left Eyeball
        pygame.draw.circle(surface, eye_color, (int(right_eye_x), int(right_eye_y)), int(eye_radius))  # Right Eyeball
        pygame.draw.circle(surface, pupil_color, (int(left_pupil_x), int(left_pupil_y)), int(pupil_radius))  # Left Pupil
        pygame.draw.circle(surface, pupil_color, (int(right_pupil_x), int(right_pupil_y)), int(pupil_radius))  # Right Pupil


        # --- 5. Name Display ---
        name_surface = name_font.render(self.name, True, white)
        name_rect = name_surface.get_rect(center=(int(self.position[0]), int(self.position[1] - self.size - 30)))
        surface.blit(name_surface, name_rect)

        # --- 6. Energy and Age Display ---
        energy_age_text = f"⚡{int(self.energy)}|⌛{int(self.age)}/{int(self.lifespan)}"
        energy_age_surface = name_font.render(energy_age_text, True, white)
        energy_age_rect = energy_age_surface.get_rect(center=(int(self.position[0]), int(self.position[1] - self.size - 10)))
        surface.blit(energy_age_surface, energy_age_rect)

        # Organism class (inside the draw method)
        # --- 7. Debug FOV Drawing (Optimized) ---
        if debug and debug_fov_mode != "none":  # Only draw FOV if the FOV button is active
            start_angle_deg = self.direction - self.sight_fov / 2
            end_angle_deg = self.direction + self.sight_fov / 2
            start_ray_rad = math.radians(start_angle_deg) # Calculate radians once
            end_ray_rad = math.radians(end_angle_deg)   # Calculate radians once

            fov_start_point = (self.position[0] + math.cos(start_ray_rad) * self.sight_range, self.position[1] - math.sin(start_ray_rad) * self.sight_range)
            fov_end_point = (self.position[0] + math.cos(end_ray_rad) * self.sight_range, self.position[1] - math.sin(end_ray_rad) * self.sight_range)

            fov_arc_points = [] # Initialize once outside the if/elif blocks
            num_fov_segments = 6 # Number of segments - keep consistent

            # FOV Arc Points Calculation - Moved out of if/elif for reuse
            for i in range(num_fov_segments + 1):
                angle_deg = start_angle_deg + (self.sight_fov / num_fov_segments) * i
                angle_rad = math.radians(angle_deg) # Calculate radians inside loop
                x = self.position[0] + math.cos(angle_rad) * self.sight_range
                y = self.position[1] - math.sin(angle_rad) * self.sight_range
                fov_arc_points.append((x, y))

            if debug_fov_mode == "full": # Draw full FOV cone
                pygame.draw.line(surface, light_grey, (int(self.position[0]), int(self.position[1])), (int(fov_start_point[0]), int(fov_start_point[1])), 1)
                pygame.draw.line(surface, light_grey, (int(self.position[0]), int(self.position[1])), (int(fov_end_point[0]), int(fov_end_point[1])), 1)
                pygame.draw.lines(surface, light_grey, False, fov_arc_points, 1) # Draw FOV arc (reusing calculated points)

            elif debug_fov_mode == "arc": # Draw only the FOV arc
                pygame.draw.lines(surface, light_grey, False, fov_arc_points, 1) # Draw FOV arc (reusing calculated points)
        # Organism class (inside the draw method)
        # --- 8. Debug Rays Drawing ---
        # Draw rays to detected food (green)
        detected_food_list = self.ray_cast_results.get('food_list', [])
        for food_info in detected_food_list:
            food = food_info['food']
            pygame.draw.line(surface, green, (int(self.position[0]), int(self.position[1])), (int(food.position[0]), int(food.position[1])), 1)

        # Draw rays to detected mates (yellow)
        detected_mate_list = self.ray_cast_results.get('mate_list', [])
        for mate_info in detected_mate_list:
            mate = mate_info['mate']
            pygame.draw.line(surface, yellow, (int(self.position[0]), int(self.position[1])), (int(mate.position[0]), int(mate.position[1])), 1)    
    def __getstate__(self):
        """Automatically capture all instance attributes except those derived from genome"""
        state = self.__dict__.copy()
        
        # Remove genome-derived traits (they're stored in the genome object)
        for trait in self.genome.genes.keys():
            state.pop(trait, None)
        
        # Convert numpy arrays to lists for serialization
        state['position'] = self.position.tolist()
        
        # Keep these special attributes that aren't genome-derived
        keep_attrs = ['genome', 'position', 'direction', 'energy', 'generation_number',
                     'name', 'current_goal', 'base_color', 'age', 'last_direction_change_time',
                     'wander_turn_interval', 'targeted_food', 'ray_cast_results']
        
        return {k: v for k, v in state.items() if k in keep_attrs}
    def __setstate__(self, state):
        """Restore state dynamically"""
        # Restore basic attributes
        self.__dict__.update(state)
        
        # Convert position back to numpy array
        self.position = np.array(state['position'], dtype=float)
        
        # Restore genome-derived traits
        if hasattr(self, 'genome'):
            for trait_name in self.genome.genes:
                setattr(self, trait_name, self.genome.get_trait(trait_name))
        
        # Handle missing attributes from older versions
        defaults = {
            'ray_cast_results': {},
            'targeted_food': None,
            'wander_turn_interval': random.uniform(1000, 3000),
            'memory_cell_size': 75,  # Added missing memory attributes
            'memory_decay_rate': 0.98,
            'memory_grid': defaultdict(lambda: (0.0, 0.0)),
            'current_cell_target': None,
            'target_persistence': 0,
            'last_memory_check': pygame.time.get_ticks()
        }
        for attr, default in defaults.items():
            if not hasattr(self, attr):
                setattr(self, attr, default)
# --- Game Functions ---
def generate_food():
    """Generates food items in clumps seasonally."""
    food_objects = []
    for _ in range(num_food_clumps):
        clump_center_angle = random.uniform(0, 2 * math.pi)
        clump_center_radius = random.uniform(0, food_spawn_radius)
        clump_center_x = screen_width / 2 + math.cos(clump_center_angle) * clump_center_radius
        clump_center_y = screen_height / 2 + math.sin(clump_center_angle) * clump_center_radius
        clump_center_pos = (clump_center_x, clump_center_y)

        for _ in range(food_per_clump):
            food_angle = random.uniform(0, 2 * math.pi)
            food_radius = random.uniform(0, clump_radius)
            food_x = clump_center_pos[0] + math.cos(food_angle) * food_radius
            food_y = clump_center_pos[1] + math.sin(food_angle) * food_radius
            food_pos = (food_x, food_y)
            food_objects.append(Food(food_pos))
    return food_objects

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

def ray_casting_thread_function(organisms, food_list, ray_cast_results_queue):
    """Function for the ray casting thread."""
    # Create a copy of food_list to prevent concurrent modification issues
    food_list_copy = list(food_list)  # Snapshot of current food items
    organism_ray_data = {}
    for organism in organisms:
        organism_data = {
            'position': organism.position.copy(),
            'direction': organism.direction,
            'name': organism.name
        }
        # Use the copied food list for detection
        detected_food_list = organism.cast_rays_for_food_threaded(food_list_copy, organism_data)
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
    batch_size = len(new_food) // 4
    for i in range(0, len(new_food), batch_size):
        food_list.extend(new_food[i:i + batch_size])
        await asyncio.sleep(0)  # Yield to main thread

# --- Main Game Loop ---
if __name__ == '__main__':
    # Initialize Pygame and display first
    pygame.init()
    pygame.font.init()
    display_surface = pygame.display.set_mode((screen_width, screen_height), pygame.OPENGL | pygame.DOUBLEBUF)
    graphics_renderer = GraphicsRenderer(screen_width, screen_height)

    # Initialize the ProcessPoolExecutor AFTER display setup
    Food.initialize_pool()  # <-- Moved here
    pre_cache_food_shapes()

    # Initialize UI and other game components
    game_clock = GameClock()
    last_season = 0
    user_interface.UI_MANAGER, user_interface.VISION_BUTTON = user_interface.init_ui(
        screen_width, 
        screen_height
    )
    #user_interface.init_ui(leaderboard_font, info_font, screen_width, screen_height)
    pygame.display.set_caption("Organism Hunting Simulation")
    #v1.01 batch food rendering with dirty rectangles & spatial partitioning for food
    food_grid = SpatialGrid(cell_size=150)
    organism_grid = SpatialGrid(cell_size=200)

    # Load or generate initial game state
    loaded_state = load_game_state()
    if loaded_state:
        organisms = loaded_state['organisms']
        food_list = loaded_state['food_list']
        game_clock = GameClock(loaded_state['game_clock'])
    else:
        organisms = generate_organisms()
        food_list = generate_food()  # This will now work because the pool is initialized
        game_clock = GameClock()

    # Initialize food generation timers
    food_generation_timer = base_food_generation_interval
    food_generation_interval = base_food_generation_interval

    # Initialize seasonal food respawn timer
    seasonal_timer = seasonal_respawn_interval

    # Main game loop
    clock = pygame.time.Clock()
    FPS = 60
    running = True
    ray_cast_results_queue = queue.Queue()

    FRAME_COUNTER = 0
    GC_INTERVAL = 10  # Only run GC every 10 frames

    while running:
        FRAME_COUNTER += 1
        # Update spatial grids first
        food_grid.update(food_list)
        organism_grid.update(organisms)
        
        # Garbage collection Run interval
        if FRAME_COUNTER % GC_INTERVAL == 0:
            gc.collect() #python garbage collection
            #game garbage collection
            organisms, food_list = clean_game_state(organisms, food_list)
        food_eaten_this_frame = []
        organisms_alive = []
        organisms_children = []
        food_to_remove_frame = []

        # ---Handle time delta---
        delta_time = clock.tick(FPS) / 1000.0
        game_clock.update(delta_time)
        current_season, current_day = game_clock.get_season_day()
        total_days = game_clock.get_total_days()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_current_state()
                running = False
        # ---Process UI events first
            user_interface.UI_MANAGER.process_events(event)            

            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == user_interface.VISION_BUTTON:
                        # Toggle debug modes
                        debug = not debug
                        debug_fov_mode = "arc" if debug else "none"
                        user_interface.VISION_BUTTON.set_text(
                            f'Creature Vision: {"On" if debug else "Off"}'
                        )
            
            # Handle organism selection
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                selected_organism = None
                for org in organisms:
                    distance = math.hypot(org.position[0] - mouse_pos[0], org.position[1] - mouse_pos[1])
                    if distance < organism_size * 2:
                        selected_organism = org
                        break
        # Clear screen after handling events
        display_surface.fill(black)
        # Food update and drawing first after events and clear
        incremental_food_generation(food_list)
        Food.draw_all(food_list, display_surface)
        # Start ray casting (uses workers)
        future = ray_cast_executor.submit(ray_casting_thread_function,
                                        organisms, food_list, 
                                        ray_cast_results_queue)
        # Organism update and drawing
        if not ray_cast_results_queue.empty():
            ray_cast_results = ray_cast_results_queue.get()
        else:
            ray_cast_results = {}  # Default empty results
        for organism in organisms:
            organism.ray_cast_results = ray_cast_results.get(organism, {})
            eaten_food_item_or_child = organism.update(food_list, organisms)
            if organism.energy > 0:
                organisms_alive.append(organism)
            if isinstance(eaten_food_item_or_child, Food):
                food_to_remove_frame.append(eaten_food_item_or_child)
                food_eaten_this_frame.append(eaten_food_item_or_child)
            elif isinstance(eaten_food_item_or_child, Organism):
                organisms_children.append(eaten_food_item_or_child)

        if organisms_alive:
            Organism.batch_update_positions(organisms_alive)

        for organism in organisms_alive:
            if debug:
                if organism.current_goal == "food" and organism.ray_cast_results.get('food'):
                    closest_food_pos = organism.ray_cast_results['food'].position
                    pygame.draw.line(display_surface, yellow, (int(organism.position[0]), int(organism.position[1])), (int(closest_food_pos[0]), int(closest_food_pos[1])), 2)
                if organism.current_goal == "mate_seeking" and organism.ray_cast_results.get('mate'):
                    closest_mate_pos = organism.ray_cast_results['mate'].position
                    pygame.draw.line(display_surface, red, (int(organism.position[0]), int(organism.position[1])), (int(closest_mate_pos[0]), int(closest_mate_pos[1])), 2)
                organism.draw(display_surface)
            else:
                organism.draw(display_surface)

        for eaten_food in food_to_remove_frame:
            if eaten_food in food_list:
                food_list.remove(eaten_food)

        organisms_alive.extend(organisms_children)
        organisms = organisms_alive

        if current_season != last_season:
            if random.random() < seasonal_respawn_chance:
                new_food = generate_food()
                asyncio.run(staggered_spawn(new_food, food_list))
                if debug:
                    print(f"Season {current_season} began - respawned {len(new_food)} food")
            last_season = current_season
        # ---Handle UI last---
        # Update UI
        user_interface.UI_MANAGER.update(delta_time)
        # Draw UI elements
        user_interface.UI_MANAGER.draw_ui(display_surface)
        user_interface.draw_leaderboard(
            display_surface,
            organisms,
            current_season,
            current_day,
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
        graphics_renderer.render(display_surface)
        pygame.display.flip()

    # Clean up OpenGL resources on exit
    print("Game loop ended, cleaning up...")
    graphics_renderer.cleanup()
    pygame.quit()

    # Close the ProcessPoolExecutor after the game loop ends
    if Food._matrix_pool is not None:
        Food._matrix_pool.shutdown(wait=True)  # Shutdown the ProcessPoolExecutor
    print("Game exited successfully.")