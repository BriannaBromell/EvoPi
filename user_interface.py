# user_interface.py
import pygame
import pygame_gui

# Global UI manager and button reference
UI_MANAGER = None
VISION_BUTTON = None

def init_ui(screen_width, screen_height):
    """Initialize UI elements with default fonts"""
    global UI_MANAGER, VISION_BUTTON
    
    # Initialize UI manager with basic settings
    UI_MANAGER = pygame_gui.UIManager(
        (screen_width, screen_height),
        theme_path='theme.json'  # Point to your theme file
    )
    
    # Create vision button
    VISION_BUTTON = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((10, screen_height - 50), (200, 40)),
        text='Creature Vision: On',
        manager=UI_MANAGER
    )
    
    return UI_MANAGER, VISION_BUTTON

def draw_leaderboard(surface, organisms, current_season, current_day, leaderboard_font, info_font):
    """Draws leaderboard with custom fonts"""
    # Season/Day display
    season_text = leaderboard_font.render(f"Season {current_season} | Day {current_day}", True, (255,255,255))
    season_rect = season_text.get_rect(topleft=(10, 10))
    surface.blit(season_text, season_rect)

    # Leaderboard elements
    lb_header = leaderboard_font.render("Leaderboard (Top 3 Energy)", True, (255,255,255))
    population_text = leaderboard_font.render(f"Population: {len(organisms)}", True, (255,255,255))
    
    # Position elements with spacing
    y_offset = season_rect.bottom + 15
    surface.blit(lb_header, (10, y_offset))
    y_offset += lb_header.get_height() + 5
    surface.blit(population_text, (10, y_offset))
    y_offset += population_text.get_height() + 15

    # Top 3 organisms
    sorted_organisms = sorted(organisms, key=lambda org: org.energy, reverse=True)
    for i, org in enumerate(sorted_organisms[:3]):
        entry_text = leaderboard_font.render(f"{i+1}. {org.name}: {int(org.energy)}", True, (255,255,255))
        surface.blit(entry_text, (10, y_offset))
        y_offset += entry_text.get_height() + 8

def draw_organism_info(surface, organism, info_font, screen_width):
    """Draws organism stats panel with custom info font"""
    if not organism:
        return

    # Core non-genetic traits and genome indicator
    traits = [
        f"Name: {organism.name}",
        f"Generation: {organism.generation_number}",
        f"Energy: {int(organism.energy)}",
        f"Age: {int(organism.age)}s",
        f"Goal: {organism.current_goal}",
        f"🧬 Genome:"
    ]

    # Dynamically add genetic traits
    for trait_name in sorted(organism.genome.genes.keys()):
        value = getattr(organism, trait_name)
        if trait_name == 'lifespan':
            formatted = f"{int(value)}s"
        elif isinstance(value, float):
            formatted = f"{value:.2f}"
        else:
            formatted = f"{int(value)}"
        traits.append(f"  {trait_name.capitalize()}: {formatted}")

    # Panel setup
    padding = 10
    line_height = info_font.get_height()
    max_width = 250
    
    max_text_width = max(info_font.size(trait)[0] for trait in traits)
    panel_width = min(max(max_text_width + padding*2, 250), screen_width//3)
    content_height = len(traits) * (line_height + 5) + padding * 2
    
    panel = pygame.Surface((panel_width, content_height), pygame.SRCALPHA)
    panel.fill((30, 30, 30, 220))
    pygame.draw.rect(panel, (255,255,255), (0, 0, panel_width, content_height), 2)

    # Draw traits
    y_pos = padding
    for trait in traits:
        text_surface = info_font.render(trait, True, (255,255,255))
        panel.blit(text_surface, (padding, y_pos))
        y_pos += line_height + 5

    surface.blit(panel, (screen_width - panel_width - 10, 10))