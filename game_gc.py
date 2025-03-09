import pygame
import numpy as np

def clean_game_state(organisms, food_list):
    """Cleans game state by removing dead organisms and invalid food"""
    # Remove dead organisms (energy <= 0 or age >= lifespan)
    alive_organisms = [
        org for org in organisms 
        if org.energy > 0 and org.age < org.lifespan
    ]
    
    # Clean organisms' targeted food references
    for org in alive_organisms:
        if org.targeted_food and org.targeted_food not in food_list:
            org.targeted_food = None
            
    # Remove any food with invalid positions (sanity check)
    valid_food = [
        f for f in food_list
        if 0 <= f.position[0] <= pygame.display.Info().current_w and
           0 <= f.position[1] <= pygame.display.Info().current_h
    ]
    
    # Clean potential floating references in ray_cast_results
    for org in alive_organisms:
        if hasattr(org, 'ray_cast_results'):
            org.ray_cast_results.clear()
    
    return alive_organisms, valid_food