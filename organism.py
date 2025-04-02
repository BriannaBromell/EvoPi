# organism.py
import os
import json
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pygame


from genetics import Genome
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
    light_grey,
#font
    name_font
)
# SpatialGrid needs to be defined before Organism
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
# --- Helper Functions ---
def get_distance_np(positions1, positions2):
    """Calculates Euclidean distance between two arrays of points using NumPy."""
    positions1 = np.array(positions1)
    positions2 = np.array(positions2)
    return np.sqrt(np.sum((positions1 - positions2)**2, axis=1))

def normalize_angle(angle):
    """Keeps angle within 0 to 360 degrees."""
    return angle % 360

# Food grid needs to be defined but will be set when imported
food_grid = SpatialGrid(cell_size=150)  # Optimized food queries

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
        self.food_not_detected_counter=0
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
    def cast_rays_for_food_threaded(self, food_list_snapshot, organism_data):
        """Corrected vectorized ray casting with proper Pygame coordinate handling"""
        organism_pos = organism_data['position']
        organism_dir = organism_data['direction']
        
        # Get nearby food using spatial grid
        nearby_food = food_grid.query_radius(organism_pos, self.sight_range)
        
        # Convert to set for O(1) lookups
        snapshot_set = set(food_list_snapshot)
        valid_food = [food for food in nearby_food if food in snapshot_set]
        
        if not valid_food:
            return []

        # Vectorize all position calculations
        food_positions = np.array([f.position for f in valid_food])
        org_x, org_y = organism_pos
        
        # Calculate directional vectors (PYGAME Y-AXIS FIX)
        dx = food_positions[:, 0] - org_x
        dy = org_y - food_positions[:, 1]  # Inverted for Pygame's coordinate system
        
        # Vectorized distance calculations
        distances = np.hypot(dx, dy)
        
        # Correct angle calculation (0° = right, clockwise rotation)
        angles_to_food = np.degrees(np.arctan2(dy, dx)) % 360
        
        # Angle difference calculation
        angle_diffs = (angles_to_food - organism_dir) % 360
        angle_differences = np.minimum(angle_diffs, 360 - angle_diffs)
        
        # Create combined mask
        fov_mask = (distances <= self.sight_range) & (angle_differences <= self.sight_fov / 2)
        
        # Get sorted results by distance
        detected_indices = np.argsort(distances[fov_mask])
        detected_distances = distances[fov_mask][detected_indices]
        detected_food_objects = [valid_food[i] for i in np.where(fov_mask)[0][detected_indices]]
        
        return [{'food': f, 'distance': d} for f, d in zip(detected_food_objects, detected_distances)]

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
            turn_rate = 25  # Degrees per frame - Keep moderate turn rate for now
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
            """Food detection uses a counter for persistence"""
            detected_food_list = self.ray_cast_results.get('food_list', []) # Get list of detected food
            if detected_food_list:
                # Reset counter immediately upon detection
                self.food_not_detected_counter = 0  # <-- ADD THIS LINE
                closest_food_info = min(detected_food_list, key=lambda item: item['distance'])
                closest_food = closest_food_info['food']

                if self.targeted_food is None or self.targeted_food != closest_food: # Target new food only if no target or target changed
                    self.targeted_food = closest_food # Lock on to closest food
                self.hunt_food(self.targeted_food) # Hunt the *targeted* food

            else:
                # Require consecutive misses before switching
                self.food_not_detected_counter += 1
                if self.food_not_detected_counter >= 5:  # 5 frames of no detection
                    self.targeted_food = None
                    self.current_goal = "wander"
                    self.food_not_detected_counter = 0
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
                if other_org != self and other_org.energy > 0 and get_distance_np(self.position[np.newaxis, :], other_org.position[np.newaxis, :])[0] < self.size * 3:  # NumPy distance check
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

    def draw(self, surface, selected_organism=None):
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

            #fov_start_point = (self.position[0] + math.cos(start_ray_rad) * self.sight_range, self.position[1] - math.sin(start_ray_rad) * self.sight_range)
            fov_start_point = (
                self.position[0] + math.cos(math.radians(start_angle_deg)) * self.sight_range,
                self.position[1] + math.sin(math.radians(start_angle_deg)) * self.sight_range  # Positive Y direction
            )

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
                     'wander_turn_interval', 'targeted_food', 'ray_cast_results', 'food_not_detected_counter']
        
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
            'last_memory_check': pygame.time.get_ticks(),
            'food_not_detected_counter': 0
        }
        for attr, default in defaults.items():
            if not hasattr(self, attr):
                setattr(self, attr, default)
