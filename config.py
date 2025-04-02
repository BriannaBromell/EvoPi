import pygame

def init_config():
    """Initialize pygame and return configuration dictionary"""
    if not pygame.get_init():
        pygame.init()
    
    try:
        display_info = pygame.display.Info()
        screen_width = display_info.current_w // 2
        screen_height = display_info.current_h // 2
    except:
        screen_width = 960
        screen_height = 540
    # Color definitions
    colors = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'brown': (139, 69, 19),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'orange': (255, 165, 0),
        'mate_seeking_color': (255, 0, 0),  # Same as red
        'light_grey': (100, 100, 100, 50)  # With alpha
    }
    return {
        'screen_width': screen_width,
        'screen_height': screen_height,
        'num_food_clumps': 25,
        'food_per_clump': 8,
        'clump_radius': 30,
        'food_size': 5,
        'foodcell_energy_value': 2,
        'foodcell_branch_complexity': 2,
        'organism_size': 1,
        'base_organism_size': 8,
        'num_rays': 3,
        'min_mating_energy_trigger': 400,
        'max_mating_energy_trigger': 600,
        # Debug settings
        'debug': True,
        'debug_fov_mode': "arc",
        # Colors
        **colors  # Unpacks all color definitions
    }

# Initialize config
config = init_config()

# Create lowercase variables
screen_width = config['screen_width']
screen_height = config['screen_height']
num_food_clumps = config['num_food_clumps']
food_per_clump = config['food_per_clump']
clump_radius = config['clump_radius']
food_size = config['food_size']
foodcell_energy_value = config['foodcell_energy_value']
foodcell_branch_complexity = config['foodcell_branch_complexity']
organism_size = config['organism_size']
base_organism_size = config['base_organism_size']
num_rays = config['num_rays']
min_mating_energy_trigger = config['min_mating_energy_trigger']
max_mating_energy_trigger = config['max_mating_energy_trigger']
debug = config['debug']
debug_fov_mode = config['debug_fov_mode']

#Font
name_font = pygame.font.SysFont("Segoe UI Emoji", 18)


# Create color variables
white = config['white']
black = config['black']
red = config['red']
green = config['green']
brown = config['brown']
blue = config['blue']
yellow = config['yellow']
orange = config['orange']
mate_seeking_color = config['mate_seeking_color']
light_grey = config['light_grey']