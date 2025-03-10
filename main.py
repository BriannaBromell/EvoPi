#--- Imports (top level) ---
import math
import random
import pickle
import numpy as np
import threading
#--- Imports and/or Installs (external package dependencies) ---
#Add additional modules by name to ensure_dependencies function call along with pygame and requests
from import_dependencies import ensure_dependencies 
try:
    ensure_dependencies(['pygame', 'requests', 'OpenGL']) #Expandable dependencies module list
    import pygame
    import requests
    print("All imports succeeded. Game can continue.")
except ImportError as e:
    print(f"Critical import error: {e}. Game cannot continue.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
#--- Imports (Modular, local) ---
import queue # Import queue for thread-safe communication
from user_interface import init_ui, draw_leaderboard, draw_organism_info # Game UI
from game_gc import clean_game_state # garbage collection script
from game_state import save_game_state, load_game_state #save and load functionality - uses data collection function save_current_state
from game_clock import GameClock #game clock for days and seasons

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
screen = pygame.display.set_mode((screen_width, screen_height), pygame.OPENGL | pygame.DOUBLEBUF)  
display_surface = pygame.Surface((screen_width, screen_height))
# OpenGL Initialization
from OpenGL.GL import *
glEnable(GL_TEXTURE_2D)
glViewport(0, 0, screen_width, screen_height)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, screen_width, screen_height, 0, -1, 1)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
# Create OpenGL texture
texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screen_width, screen_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

# --- Font & UI Setup --- UI must come after screen initialization
name_font = pygame.font.SysFont("Segoe UI Emoji", 18)
leaderboard_font = pygame.font.SysFont("Segoe UI Emoji", 24)
info_font = pygame.font.SysFont("Segoe UI Emoji", 16)
selected_organism = None  # Add this with other game state variables
init_ui(leaderboard_font, info_font, screen_width, screen_height)
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
light_grey = (100, 100, 100, 30) # Define light grey for FOV outline, with alpha for transparency

# --- Game Parameters ---
num_organisms = 20  # Increased organism count as requested
num_food_clumps = 25
food_per_clump = 8
clump_radius = 30
num_food = num_food_clumps * food_per_clump
food_size = 5
organism_size = 10
ray_length = 120
ray_fov = 120
num_rays = 3
food_spawn_radius = min(screen_width, screen_height) * 0.8
min_mating_energy_trigger = 400
max_mating_energy_trigger = 600

# --- Density-Dependent Food Growth Parameters ---
base_food_growth_rate = 0.7
density_threshold_for_slow_growth = num_food * 0.5
slow_growth_factor = 0.2

# --- Incremental Food Generation Parameters ---
base_food_generation_interval = 0.5  # Seconds per particle at low density
food_density_speedup_factor = 0.1  # Reduction in interval per food item over threshold - adjust for speed
max_food_density_for_speedup = num_food * 0.5  # Density above which speedup starts

# --- Seasonal Food Respawn Parameters ---
seasonal_respawn_interval = 30  # Seconds between seasonal respawn attempts
seasonal_respawn_chance = 0.7  # 70% chance of respawn each season

# --- Naming Prefixes and Suffixes ---
#carnivore_name_prefixes = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
#carnivore_name_suffixes = ["Predator", "Hunter", "Seeker", "Crawler", "Provoker", "Aggressor", "Stalker", "Ranger"]
name_prefixes = ["Geo", "Bio", "Eco", "Evo", "Hydro", "Pyro", "Chrono", "Astro", "Cosmo", "Terra"]
name_suffixes = ["Bot", "Mite", "Pod", "Worm", "Beast", "Fly", "Drake", "Wing", "Crawler", "Shaper"]

# --- Helper Functions ---
def get_distance_np(positions1, positions2):
    """Calculates Euclidean distance between two arrays of points using NumPy."""
    positions1 = np.array(positions1)
    positions2 = np.array(positions2)
    return np.sqrt(np.sum((positions1 - positions2)**2, axis=1))

def normalize_angle(angle):
    """Keeps angle within 0 to 360 degrees."""
    return angle % 360


# --- Food Class ---
class Food:
    # Class-level cache for shared surfaces (key: hash of shape_matrix tuple)
    shape_cache = {}

    def __init__(self, position, shape_matrix=None):
        self.position = np.array(position, dtype=float)
        
        # Set shape_matrix (use default if None)
        if shape_matrix is None:
            self.shape_matrix = np.array([
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            ])
        else:
            self.shape_matrix = np.array(shape_matrix)
        
        # Check cache for existing surface
        shape_hash = hash(tuple(map(tuple, self.shape_matrix)))  # Hash the matrix
        if shape_hash not in Food.shape_cache:
            # Generate and cache new surface
            Food.shape_cache[shape_hash] = self._create_cached_surface()
        self.cached_surface = Food.shape_cache[shape_hash]

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

    @staticmethod
    def draw_all(food_list, surface):
        """Batch draw all food items in a single pass."""
        for food in food_list:
            rect = food.cached_surface.get_rect(center=(int(food.position[0]), int(food.position[1])))
            surface.blit(food.cached_surface, rect.topleft)
    def __getstate__(self):
        """Return state values to be pickled."""
        return {'position': self.position.tolist(),
                'shape_matrix': self.shape_matrix.tolist()} # Save shape_matrix to state
    def __setstate__(self, state):
        """Restore state from pickled values."""
        self.position = np.array(state.get('position', [0, 0]), dtype=float)
        shape_matrix_list = state.get('shape_matrix') # Get shape_matrix from state
        if shape_matrix_list is not None: # Check if shape_matrix was in saved state
            self.shape_matrix = np.array(shape_matrix_list) # Restore shape_matrix
        else: # If no shape_matrix in state, use default (for older saves or new Food objects without shape)
            self.shape_matrix = np.array([
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0], # Center brown dot (stem)
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            ])
        self.cached_surface = self.create_cached_surface() 


# --- Organism Class ---
class Organism:
    def __init__(self, position, generation_number=1, direction=None, strength=None, speed=None, sight_radius=None, energy=None, name=None, current_goal=None, base_color=None, organism_size=None, lifespan=None, age=None):
        self.position = np.array(position, dtype=float)  # Store position as NumPy array
        self.direction = direction if direction is not None else random.uniform(0, 360)
        self.strength = strength if strength is not None else random.uniform(1, 3)
        self.speed = speed if speed is not None else 1.0 #random.uniform(1, 2) #organism speed range
        self.sight_radius = sight_radius if sight_radius is not None else ray_length
        self.energy = energy if energy is not None else 100
        self.generation_number = generation_number
        self.name = name if name is not None else self.generate_name()
        self.targeted_food = None # Initialize targeted_food
        self.current_goal = current_goal if current_goal is not None else "food"
        self.base_color = base_color if base_color is not None else self.calculate_color_from_attributes() # Call color function here
        self.organism_size = organism_size if organism_size is not None else random.uniform(8, 15)
        
        self.lifespan = lifespan if lifespan is not None else random.uniform(60 * 1, 60 * 2) # default lifespan is 1-2 minutes (60 seconds times low range to 60 seconds times high range)
        self.age = age if age is not None else 0
        self.last_direction_change_time = pygame.time.get_ticks()  # For wander smooth turn
        self.wander_turn_interval = random.uniform(1000, 3000)  # Wander turn interval milliseconds
        self.ray_cast_results = {} # Store ray cast results


    @staticmethod
    def batch_update_positions(organisms):
        """Vectorized position update"""
        directions = np.array([org.direction for org in organisms])
        speeds = np.array([org.speed for org in organisms])
        positions = np.array([org.position for org in organisms])
        
        radians = np.radians(directions)
        offsets = np.column_stack((
            np.cos(radians) * speeds,
            -np.sin(radians) * speeds
        ))
        
        new_positions = (positions + offsets) % [screen_width, screen_height]
        
        # Update individual organisms
        for i, org in enumerate(organisms):
            org.position = new_positions[i]
            org.energy -= 0.1
    def generate_name(self):
        """Generates a name using prefix, suffix and generation number."""
        prefix = random.choice(name_prefixes)
        suffix = random.choice(name_suffixes)
        return f"{prefix}{suffix}-{random.randint(1000, 9999)}-({self.generation_number})"
    def cast_rays_for_food_threaded(self, food_list, organism_data):
        """Casts rays to detect food within FOV and range and returns all in FOV with distances."""
        organism_pos = organism_data['position']
        organism_direction = organism_data['direction']

        if not food_list:
            return [] # Return empty list if no food

        num_rays_val = num_rays
        start_angle = organism_direction - ray_fov / 2
        angles = normalize_angle(start_angle + np.arange(num_rays_val) * (ray_fov / (num_rays_val - 1) if num_rays_val > 1 else 0) )
        radians = np.radians(angles)

        ray_ends_x = organism_pos[0] + np.cos(radians) * ray_length
        ray_ends_y = organism_pos[1] - np.sin(radians) * ray_length
        ray_ends = np.stack([ray_ends_x, ray_ends_y], axis=-1)

        food_positions = np.array([food.position for food in food_list])
        distances_to_food = get_distance_np(organism_pos[np.newaxis, :], food_positions)

        # --- Angle to Food Calculation - RE-EXAMINE THIS ---
        angles_to_food = -np.degrees(np.arctan2(food_positions[:, 1] - organism_pos[1], food_positions[:, 0] - organism_pos[0]))
        # --- Angle Difference Calculation - RE-EXAMINE THIS ---
        angle_diffs = (angles_to_food - organism_direction) % 360
        angle_differences = np.where(angle_diffs > 180, 360 - angle_diffs, angle_diffs)

        food_in_fov_mask = (distances_to_food <= ray_length) & (angle_differences <= ray_fov / 2)
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
        start_angle = organism_direction - ray_fov / 2
        angles = normalize_angle(start_angle + np.arange(num_rays_val) * (ray_fov / (num_rays_val - 1) if num_rays_val > 1 else 0))
        radians = np.radians(angles)

        ray_ends_x = organism_pos[0] + np.cos(radians) * ray_length
        ray_ends_y = organism_pos[1] - np.sin(radians) * ray_length
        ray_ends = np.stack([ray_ends_x, ray_ends_y], axis=-1)

        mate_positions = np.array([mate.position for mate in potential_mates])
        distances_to_mates = get_distance_np(organism_pos[np.newaxis, :], mate_positions)

        angles_to_mates = -np.degrees(np.arctan2(mate_positions[:, 1] - organism_pos[1], mate_positions[:, 0] - organism_pos[0]))
        angle_diffs = (angles_to_mates - organism_direction) % 360
        angle_differences = np.where(angle_diffs > 180, 360 - angle_diffs, angle_diffs)

        mates_in_fov_mask = (distances_to_mates <= ray_length) & (angle_differences <= ray_fov / 2)
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
        """Move organism forward based on speed."""
        radians = math.radians(self.direction)
        dx = math.cos(radians) * self.speed
        dy = -math.sin(radians) * self.speed
        self.position += np.array([dx, dy])
        self.position[0] %= screen_width  # Wrap around screen using NumPy modulo
        self.position[1] %= screen_height
        self.energy -= 0.1

    def mate(self, other_parent):
        """Sexual reproduction with another organism using enhanced DNA evolution"""
        half_mating_energy = min_mating_energy_trigger * 0.5
        per_parent_energy_cost = half_mating_energy * 0.5

        if debug:
            print(f"{self.name}: Attempting to mate with {other_parent.name}")

        if self.energy < half_mating_energy or other_parent.energy < half_mating_energy:
            if debug:
                print(f"    {self.name}: Mating failed - Insufficient energy (Self: {int(self.energy)}, Mate: {int(other_parent.energy)})")
            return None

        # Deduct energy from both parents
        self.energy -= per_parent_energy_cost
        other_parent.energy -= per_parent_energy_cost
        if debug:
            print(f"    {self.name}: Energy after mating cost - Self: {int(self.energy)}, Mate: {int(other_parent.energy)}")

        # DNA Configuration - Easily expandable
        inheritable_traits = [
            'strength', 'speed', 'sight_radius',
            'organism_size', 'lifespan', 'base_color'
        ]

        mutation_config = {
            # Format: trait: (mutation_rate, min_value, max_value, mutation_strength)
            'strength': (0.15, 0.5, 3.0, 0.2),
            'speed': (0.15, 0.5, 2.5, 0.15),
            'sight_radius': (0.1, 50, 200, 0.1),
            'organism_size': (0.1, 1, 50, 0.15),
            'lifespan': (0.1, 60, 600, 0.1),
            'base_color': (0.05, None, None, 20)  # Color mutation in RGB space
        }

        # Genetic Crossover - Blend parent DNA
        child_dna = {}
        for trait in inheritable_traits:
            # Random inheritance with potential blending
            inheritance_type = random.choice(['average', 'parent1', 'parent2'])

            if inheritance_type == 'average':
                if trait == 'base_color':
                    # Average RGB components for base_color
                    parent1_color = np.array(getattr(self, trait))
                    parent2_color = np.array(getattr(other_parent, trait))
                    child_dna[trait] = tuple(( (parent1_color + parent2_color) / 2 ).astype(int)) # Component-wise average and convert to tuple
                else:
                    child_dna[trait] = (getattr(self, trait) + getattr(other_parent, trait)) / 2
            else:
                child_dna[trait] = getattr(self, trait) if inheritance_type == 'parent1' else getattr(other_parent, trait)

        # Genetic Mutation
        for trait, (rate, min_val, max_val, strength) in mutation_config.items():
            if random.random() < rate:
                if trait == 'base_color':
                    # Mutate color by adjusting RGB values
                    mutated_color = [
                        min(255, max(0, channel + random.randint(-strength, strength)))
                        for channel in child_dna[trait]
                    ]
                    child_dna[trait] = tuple(mutated_color)
                else:
                    # Apply percentage-based mutation
                    mutation = 1 + random.uniform(-strength, strength)
                    child_dna[trait] = min(max_val, max(min_val, child_dna[trait] * mutation))

        # Create offspring
        child_position = (
            (self.position[0] + other_parent.position[0]) / 2,
            (self.position[1] + other_parent.position[1]) / 2
        )

        child = Organism(
            position=child_position,
            generation_number=max(self.generation_number, other_parent.generation_number) + 1,
            **{trait: child_dna[trait] for trait in inheritable_traits}
        )

        if debug:
            print(f"    {self.name}: Child '{child.name}' created with DNA: {child_dna}")

        return child
        """Sexual reproduction with another organism using enhanced DNA evolution"""
        half_mating_energy = min_mating_energy_trigger * 0.5
        per_parent_energy_cost = half_mating_energy * 0.5
        
        if debug:
            print(f"{self.name}: Attempting to mate with {other_parent.name}")
            
        if self.energy < half_mating_energy or other_parent.energy < half_mating_energy:
            if debug:
                print(f"    {self.name}: Mating failed - Insufficient energy (Self: {int(self.energy)}, Mate: {int(other_parent.energy)})")
            return None

        # Deduct energy from both parents
        self.energy -= per_parent_energy_cost
        other_parent.energy -= per_parent_energy_cost
        if debug:
            print(f"    {self.name}: Energy after mating cost - Self: {int(self.energy)}, Mate: {int(other_parent.energy)})")

        # DNA Configuration - Easily expandable
        inheritable_traits = [
            'strength', 'speed', 'sight_radius', 
            'organism_size', 'lifespan', 'base_color'
        ]
        
        mutation_config = {
            # Format: trait: (mutation_rate, min_value, max_value, mutation_strength)
            'strength': (0.15, 0.5, 3.0, 0.2),
            'speed': (0.15, 0.5, 2.5, 0.15),
            'sight_radius': (0.1, 50, 200, 0.1),
            'organism_size': (0.1, 5, 20, 0.15),
            'lifespan': (0.1, 60, 600, 0.1),
            'base_color': (0.05, None, None, 20)  # Color mutation in RGB space
        }

        # Genetic Crossover - Blend parent DNA
        child_dna = {}
        for trait in inheritable_traits:
            # Random inheritance with potential blending
            inheritance_type = random.choice(['average', 'parent1', 'parent2'])
            
            if inheritance_type == 'average':
                child_dna[trait] = (getattr(self, trait) + getattr(other_parent, trait)) / 2
            else:
                child_dna[trait] = getattr(self, trait) if inheritance_type == 'parent1' else getattr(other_parent, trait)

        # Genetic Mutation
        for trait, (rate, min_val, max_val, strength) in mutation_config.items():
            if random.random() < rate:
                if trait == 'base_color':
                    # Mutate color by adjusting RGB values
                    mutated_color = [
                        min(255, max(0, channel + random.randint(-strength, strength)))
                        for channel in child_dna[trait]
                    ]
                    child_dna[trait] = tuple(mutated_color)
                else:
                    # Apply percentage-based mutation
                    mutation = 1 + random.uniform(-strength, strength)
                    child_dna[trait] = min(max_val, max(min_val, child_dna[trait] * mutation))

        # Create offspring
        child_position = (
            (self.position[0] + other_parent.position[0]) / 2,
            (self.position[1] + other_parent.position[1]) / 2
        )
        
        child = Organism(
            position=child_position,
            generation_number=max(self.generation_number, other_parent.generation_number) + 1,
            **{trait: child_dna[trait] for trait in inheritable_traits}
        )

        if debug:
            print(f"    {self.name}: Child '{child.name}' created with DNA: {child_dna}")
            
        return child
    def update(self, food_list, organisms):
        """Update organism behavior based on current goal. Ray casting results are expected to be pre-calculated."""
        self.age += 1/60

        if self.energy <= 0:
            return False
        if self.age >= self.lifespan:
            self.energy = 0
            if debug:
                print(f"{self.name}: Died of old age at {int(self.age)} seconds (lifespan: {int(self.lifespan)} seconds).")
            return False

        if self.current_goal == "mate_seeking":
            if self.energy < (min_mating_energy_trigger*0.5):
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
            current_time = pygame.time.get_ticks()
            if current_time - self.last_direction_change_time >= self.wander_turn_interval:
                self.direction += random.uniform(-90, 90)  # Wider wander turns
                self.direction = normalize_angle(self.direction)
                self.last_direction_change_time = current_time
                self.wander_turn_interval = random.uniform(1000, 3000)  # reset interval

            self.move_forward()
            detected_food_list = self.ray_cast_results.get('food_list', []) # Check for food while wandering
            if detected_food_list:
                closest_food_info = min(detected_food_list, key=lambda item: item['distance'])
                closest_food = closest_food_info['food']
                self.targeted_food = closest_food # Target food immediately when detected while wandering
                self.current_goal = "food"
                return self.update(food_list, organisms)  # Re-run update to switch to food seeking


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
                if other_org != self and other_org.energy > 0 and get_distance_np(self.position[np.newaxis, :], other_org.position[np.newaxis, :])[0] < organism_size * 3:  # NumPy distance check
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


    def eat_food(self, food):
        """Organism eats food and gains energy."""
        self.energy += 50
        # if self.energy > 500:
        # self.energy = 500

    def draw(self, surface):
        """Draw the organism and debug rays - Optimized Version."""
        # --- 1. Draw Base Organism (Circle) ---
        pygame.draw.circle(surface, self.base_color, (int(self.position[0]), int(self.position[1])), organism_size)

        # --- 2. Mate Seeking Indicator ---
        if self.current_goal == "mate_seeking":
            pygame.draw.circle(surface, red, (int(self.position[0]), int(self.position[1])), organism_size + 3, 2)

        # --- 3. Direction Indicator and Eyes ---
        # --- 3.1. Calculate Head Position (Direction Indicator End) ---
        head_x = self.position[0] + math.cos(math.radians(self.direction)) * organism_size
        head_y = self.position[1] - math.sin(math.radians(self.direction)) * organism_size
        pygame.draw.line(surface, white, (int(self.position[0]), int(self.position[1])), (int(head_x), int(head_y)), 3)

        # --- 3.2. Calculate Eye Positions (Inward and Closer) ---
        eye_offset_distance = organism_size / 1.5  # Reduced offset for inward eyes
        eye_radius = organism_size / 2
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


        # --- 4. Name Display ---
        name_surface = name_font.render(self.name, True, white)
        name_rect = name_surface.get_rect(center=(int(self.position[0]), int(self.position[1] - organism_size - 30)))
        surface.blit(name_surface, name_rect)

        # --- 5. Energy and Age Display ---
        energy_age_text = f"⚡{int(self.energy)}|⌛{int(self.age)}/{int(self.lifespan)}"
        energy_age_surface = name_font.render(energy_age_text, True, white)
        energy_age_rect = energy_age_surface.get_rect(center=(int(self.position[0]), int(self.position[1] - organism_size - 10)))
        surface.blit(energy_age_surface, energy_age_rect)

        # --- 6. Debug FOV Drawing (Optimized) ---
        if debug and debug_fov_mode != "none":
            start_angle_deg = self.direction - ray_fov / 2
            end_angle_deg = self.direction + ray_fov / 2
            start_ray_rad = math.radians(start_angle_deg) # Calculate radians once
            end_ray_rad = math.radians(end_angle_deg)   # Calculate radians once

            fov_start_point = (self.position[0] + math.cos(start_ray_rad) * ray_length, self.position[1] - math.sin(start_ray_rad) * ray_length)
            fov_end_point = (self.position[0] + math.cos(end_ray_rad) * ray_length, self.position[1] - math.sin(end_ray_rad) * ray_length)

            fov_arc_points = [] # Initialize once outside the if/elif blocks
            num_fov_segments = 6 # Number of segments - keep consistent

            # FOV Arc Points Calculation - Moved out of if/elif for reuse
            for i in range(num_fov_segments + 1):
                angle_deg = start_angle_deg + (ray_fov / num_fov_segments) * i
                angle_rad = math.radians(angle_deg) # Calculate radians inside loop
                x = self.position[0] + math.cos(angle_rad) * ray_length
                y = self.position[1] - math.sin(angle_rad) * ray_length
                fov_arc_points.append((x, y))

            if debug_fov_mode == "full": # Draw full FOV cone
                pygame.draw.line(surface, light_grey, (int(self.position[0]), int(self.position[1])), (int(fov_start_point[0]), int(fov_start_point[1])), 1)
                pygame.draw.line(surface, light_grey, (int(self.position[0]), int(self.position[1])), (int(fov_end_point[0]), int(fov_end_point[1])), 1)
                pygame.draw.lines(surface, light_grey, False, fov_arc_points, 1) # Draw FOV arc (reusing calculated points)

            elif debug_fov_mode == "arc": # Draw only the FOV arc
                pygame.draw.lines(surface, light_grey, False, fov_arc_points, 1) # Draw FOV arc (reusing calculated points)


        # --- 7. Debug Rays Drawing ---
        if debug: # Draw debug rays
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
        """Return state values to be pickled - convert NumPy arrays to lists."""
        state = {
            'position': self.position.tolist(),
            'direction': self.direction,
            'strength': self.strength,
            'speed': self.speed,
            'sight_radius': self.sight_radius,
            'energy': self.energy,
            'generation_number': self.generation_number,
            'name': self.name,
            'current_goal': self.current_goal,
            'base_color': self.base_color,
            'organism_size': self.organism_size,
            'lifespan': self.lifespan,
            'age': self.age,
            'last_direction_change_time': self.last_direction_change_time,  # Save wander timer state
            'wander_turn_interval': self.wander_turn_interval,  # Save wander timer state
            'targeted_food': self.targeted_food, # ADD THIS LINE - Save targeted_food
        }
        return state

    def calculate_color_from_attributes(self):
        """Calculates organism color based on its attributes, creating a spectrum."""
        # --- Normalize attributes to 0-255 range ---
        # Example normalization (adjust ranges as needed based on your attribute distributions)
        normalized_strength = int(self.strength / 3.0 * 255) # Strength max is approx 3
        normalized_speed = int(self.speed / 2.0 * 255)      # Speed max is approx 2
        normalized_sight = int(self.sight_radius / ray_length * 255) # Sight radius max is ray_length

        # --- Map attributes to RGB components ---
        red_component = normalized_strength
        green_component = normalized_speed
        blue_component = 255 - normalized_sight  # Invert sight to get different spectrum range

        # --- Ensure components are within 0-255 ---
        red_component = max(0, min(red_component, 255))
        green_component = max(0, min(green_component, 255))
        blue_component = max(0, min(blue_component, 255))


        return (red_component, green_component, blue_component) # Return RGB tuple
    def __setstate__(self, state):
        """Restore state from pickled values - convert lists to NumPy arrays."""
        self.position = np.array(state.get('position', [random.uniform(0, screen_width), random.uniform(0, screen_height)]), dtype=float)
        self.direction = state.get('direction', random.uniform(0, 360))
        self.strength = state.get('strength', random.uniform(1, 3))
        self.speed = state.get('speed', random.uniform(1, 2))
        self.sight_radius = state.get('sight_radius', ray_length)
        self.energy = state.get('energy', 100)
        self.generation_number = state.get('generation_number', 1)
        self.name = state.get('name', self.generate_name())
        self.current_goal = state.get('current_goal', "food")
        self.base_color = state.get('base_color', self.calculate_color_from_attributes()) # Default to attribute-based color if not in state
        self.organism_size = state.get('organism_size', 8)
        self.lifespan = state.get('lifespan', random.uniform(60 * 2, 60 * 5))
        self.age = state.get('age', 0)
        self.last_direction_change_time = state.get('last_direction_change_time', pygame.time.get_ticks())  # Load wander timer state, default to current time
        self.wander_turn_interval = state.get('wander_turn_interval', random.uniform(1000, 3000))  # Load wander timer interval, default to random
        self.targeted_food = state.get('targeted_food', None) # ADD THIS LINE - Load targeted_food, default to None


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

def generate_food_particle():
    """Generates a single food particle at a random position within the spawn radius."""
    angle = random.uniform(0, 2 * math.pi)
    radius = random.uniform(0, food_spawn_radius)
    x = screen_width / 2 + math.cos(angle) * radius
    y = screen_height / 2 + math.sin(angle) * radius
    food_pos = (x, y)
    return Food(food_pos)

def incremental_food_generation(food_list):
    """Incrementally generates food particles based on density."""
    global food_generation_timer, food_generation_interval

    food_generation_timer -= 1/60

    if food_generation_timer <= 0:
        food_list.append(generate_food_particle())

        food_density = len(food_list)
        if food_density > max_food_density_for_speedup:
            density_over_threshold = food_density - max_food_density_for_speedup
            speedup_reduction = density_over_threshold * food_density_speedup_factor
            food_generation_interval = max(0.1, base_food_generation_interval - speedup_reduction)
        else:
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
    organism_ray_data = {}
    for organism in organisms:
        organism_data = {'position': organism.position.copy(), 'direction': organism.direction, 'name': organism.name} # copy position to avoid main thread modification during ray cast, include name
        detected_food_list = organism.cast_rays_for_food_threaded(food_list, organism_data) # Get list of food
        detected_mate_list = organism.cast_rays_for_mate_threaded(organisms, organism_data) # Get list of mates
        organism_ray_data[organism] = {'food_list': detected_food_list, 'mate_list': detected_mate_list} # Store lists

    ray_cast_results_queue.put(organism_ray_data) # Put results in queue

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

# --- Game Loading Initialization ---
loaded_state = load_game_state()
if loaded_state:
    organisms = loaded_state['organisms']
    food_list = loaded_state['food_list']
    game_clock = GameClock(loaded_state['game_clock'])
else:
    organisms = generate_organisms()
    food_list = generate_food()
    game_clock = GameClock()

# --- Food Generation Timers ---
food_generation_timer = base_food_generation_interval
food_generation_interval = base_food_generation_interval

# --- Seasonal Respawn Timer ---
seasonal_timer = seasonal_respawn_interval

# --- Main Game Loop ---
clock = pygame.time.Clock()
FPS = 60
running = True
last_season = 0
last_gc_day = -1
food_to_remove = []  # Initialize food_to_remove list outside the loop
ray_cast_results_queue = queue.Queue() # Initialize queue for ray casting results

while running:
    food_eaten_this_frame = []  # Foods eaten in current frame, reset each frame
    organisms_alive = []
    organisms_children = [] # Initialize organisms_children list here, EACH FRAME
    food_to_remove_frame = []  # Foods to remove in current frame, reset each frame


    # Handle time delta
    delta_time = clock.tick(FPS)/1000.0  # Get seconds since last frame
    game_clock.update(delta_time)
    # Get current time state
    current_season, current_day = game_clock.get_season_day()
    total_days = game_clock.get_total_days()

    # --- Garbage Collection ---
    if total_days != last_gc_day:
        organisms, food_list = clean_game_state(organisms, food_list)
        last_gc_day = total_days

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            save_current_state()
            running = False
    # --- Mouse click tracking ---
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                selected_organism = None
                
                # Organism selection (checking clicks)
                for org in organisms:
                    distance = math.hypot(org.position[0]-mouse_pos[0], org.position[1]-mouse_pos[1])
                    if distance < organism_size:
                        selected_organism = org
                        break
    #screen.fill(black)
    display_surface.fill(black)  # Instead of screen.fill(black)

    # --- Food Update and Drawing ---
    incremental_food_generation(food_list)
    #Food.draw_all(food_list, screen)
    Food.draw_all(food_list, display_surface) 

    # --- Start Ray Casting Thread ---
    ray_casting_thread = threading.Thread(target=ray_casting_thread_function, args=(organisms, food_list, ray_cast_results_queue))
    ray_casting_thread.start()

    # --- Organism Update and Drawing ---
    ray_cast_results = ray_cast_results_queue.get() # Block until results are available

    # Collect all updates first
    for organism in organisms:
        organism.ray_cast_results = ray_cast_results.get(organism, {}) # Retrieve results from queue
        eaten_food_item_or_child = organism.update(food_list, organisms)  # Update returns eaten food item or child
        if organism.energy > 0: #Organism is alive
            organisms_alive.append(organism)
        if eaten_food_item_or_child == False: #Organism died this frame, or no child created, or no food eaten
            pass # do nothing
        elif isinstance(eaten_food_item_or_child, Food):
            food_to_remove_frame.append(eaten_food_item_or_child) # Remove food item from food list
            food_eaten_this_frame.append(eaten_food_item_or_child) # Track food eaten this frame
        elif isinstance(eaten_food_item_or_child, Organism):
            organisms_children.append(eaten_food_item_or_child) # Add child organism to organism list

    # Batch update positions
    if organisms_alive:  # Only update positions if there are alive organisms
        Organism.batch_update_positions(organisms_alive)

    # Draw organisms/food after position updates
    for organism in organisms_alive:
        # --- Debug Drawing ---
        if debug:
            if organism.current_goal == "food" and organism.ray_cast_results.get('food'):
                closest_food_pos = organism.ray_cast_results['food'].position
                pygame.draw.line(display_surface, yellow, (int(organism.position[0]), int(organism.position[1])), (int(closest_food_pos[0]), int(closest_food_pos[1])), 2)
            if organism.current_goal == "mate_seeking" and organism.ray_cast_results.get('mate'):
                closest_mate_pos = organism.ray_cast_results['mate'].position
                pygame.draw.line(display_surface, red, (int(organism.position[0]), int(organism.position[1])), (int(closest_mate_pos[0]), int(closest_mate_pos[1])), 2)
            organism.draw(display_surface) # Draw organism and FOV (in debug mode)
        else:
            organism.draw(display_surface) # Draw organism (no FOV)

    # --- Process Food Consumption ---
    for eaten_food in food_to_remove_frame:
        if eaten_food in food_list:
            food_list.remove(eaten_food)

    # --- Process New Children ---
    organisms_alive.extend(organisms_children) # EXTEND organisms_alive list with children created this frame
    # --- Update Organism List ---
    organisms = organisms_alive #Update to only living organisms

    # --- Seasonal Food Respawn  ---
    if current_season != last_season:
        if random.random() < seasonal_respawn_chance:
            new_food = generate_food()
            food_list.extend(new_food)
            if debug:
                print(f"Season {current_season} began - respawned {len(new_food)} food")
        last_season = current_season

    # --- Leaderboard ---
        # --- Season/day display & top 3 creatures (leaderboard) ---     
    draw_leaderboard(display_surface, organisms, current_season, current_day)
    if selected_organism:
        draw_organism_info(display_surface, selected_organism)



    # Convert Pygame Surface to OpenGL texture
    texture_data = pygame.image.tostring(display_surface, "RGBA", True)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screen_width, screen_height, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

    # Clear and draw texture
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
# --- OpenGL ---
    # With this (flip texture vertically):
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)  # Changed from (0,0)
    glVertex2f(0, 0)
    glTexCoord2f(1, 1)  # Changed from (1,0)
    glVertex2f(screen_width, 0)
    glTexCoord2f(1, 0)  # Changed from (1,1)
    glVertex2f(screen_width, screen_height)
    glTexCoord2f(0, 0)  # Changed from (0,1)
    glVertex2f(0, screen_height)
    glEnd()

    pygame.display.flip()

pygame.quit()