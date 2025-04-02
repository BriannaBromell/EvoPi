# food.py
import os, json
import pygame
import pygame.gfxdraw
import random
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
    debug_fov_mode
)
green = (0, 255, 0)
brown = (139, 69, 19)

"""Complexity → Number of Cells → Total Energy 🔄"""
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
class Food:
    # Class-level cache for shared surfaces (key: hash of shape_matrix tuple)
    shape_cache = {}
    # Class-level pool for matrix generation
    _matrix_pool = None  # Initialize as None
    #v1.01 Class-level batch rendering system
    _batch_surface = None
    _dirty_rects = []

    @classmethod
    def initialize_pool(cls):
        if cls._matrix_pool is None:
            cls._matrix_pool = ThreadPoolExecutor()  # Use threads instead of processes
    
    @classmethod
    def generate_branching_matrix_async(cls):
        """Generate a branching matrix in a separate process."""
        cls.initialize_pool()  # Ensure the pool is initialized
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
            'shape_matrix': self.shape_matrix.tolist(), #Preserve for old saves
            'energy_value': self.energy_value,
            'shape_hash': hash(tuple(map(tuple, self.shape_matrix)))  # Save the hash for cache lookup
            }
    def __setstate__(self, state):
        """Load state, prioritizing global complexity over saved shapes."""
        self.position = np.array(state['position'], dtype=float)
        self.shape_matrix = np.array(state['shape_matrix'])
        self.energy_value = state['energy_value']
        # Try to reuse cached surface if available otherwise Rebuild surface cache
        shape_hash = state.get('shape_hash', hash(tuple(map(tuple, self.shape_matrix))))
        if shape_hash in Food.shape_cache:
            self.cached_surface = Food.shape_cache[shape_hash]
        else:
            self.cached_surface = self._create_cached_surface()
            Food.shape_cache[shape_hash] = self.cached_surface

class FoodManager:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.food_list = []
        # Initialize food systems
        self.initialize_food_systems()
        
        
    def initialize_food_systems(self):
        """Initialize all food-related systems including pool and shape caching"""
        Food.initialize_pool()
        self.pre_cache_food_shapes()
    # --- Pre-caching Food Shapes ---
    def pre_cache_food_shapes(self):
        """Pre-generate common food shapes to avoid runtime overhead."""
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(Food.generate_branching_matrix_async) for _ in range(20)]
            for future in futures:
                matrix = future.result()
                Food((0, 0), matrix)  # Create dummy Food object to cache the surface
    def generate_food(self, food_spawn_radius):
        """Generates food items in clumps seasonally."""
        food_objects = []
        for _ in range(num_food_clumps):
            clump_center_angle = random.uniform(0, 2 * math.pi)
            clump_center_radius = random.uniform(0, food_spawn_radius)
            clump_center_x = self.screen_width / 2 + math.cos(clump_center_angle) * clump_center_radius
            clump_center_y = self.screen_height / 2 + math.sin(clump_center_angle) * clump_center_radius
            clump_center_pos = (clump_center_x, clump_center_y)

            for _ in range(food_per_clump):
                food_angle = random.uniform(0, 2 * math.pi)
                food_radius = random.uniform(0, clump_radius)
                food_x = clump_center_pos[0] + math.cos(food_angle) * food_radius
                food_y = clump_center_pos[1] + math.sin(food_angle) * food_radius
                food_pos = (food_x, food_y)
                food_objects.append(Food(food_pos))
        
        self.food_list.extend(food_objects)
        return food_objects