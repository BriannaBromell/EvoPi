# user_interface.py
import pygame

def init_ui(leaderboard_font, info_font, screen_width, screen_height):
    global LB_FONT, INFO_FONT, SCREEN_WIDTH, SCREEN_HEIGHT
    LB_FONT = leaderboard_font
    INFO_FONT = info_font
    SCREEN_WIDTH = screen_width
    SCREEN_HEIGHT = screen_height

def draw_leaderboard(surface, organisms, current_season, current_day):
    """Draws all UI elements with proper spacing"""
    # Season/Day display
    season_text = LB_FONT.render(f"Season {current_season} | Day {current_day}", True, (255,255,255))
    season_rect = season_text.get_rect(topleft=(10, 10))
    surface.blit(season_text, season_rect)

    # Leaderboard elements
    lb_header = LB_FONT.render("Leaderboard (Top 3 Energy)", True, (255,255,255))
    population_text = LB_FONT.render(f"Population: {len(organisms)}", True, (255,255,255))
    
    # Position elements with spacing
    y_offset = season_rect.bottom + 15
    surface.blit(lb_header, (10, y_offset))
    y_offset += lb_header.get_height() + 5
    surface.blit(population_text, (10, y_offset))
    y_offset += population_text.get_height() + 15

    # Top 3 organisms
    sorted_organisms = sorted(organisms, key=lambda org: org.energy, reverse=True)
    for i, org in enumerate(sorted_organisms[:3]):
        entry_text = LB_FONT.render(f"{i+1}. {org.name}: {int(org.energy)}", True, (255,255,255))
        surface.blit(entry_text, (10, y_offset))
        y_offset += entry_text.get_height() + 8

def draw_organism_info(surface, organism):
    """Draws organism stats panel"""
    if not organism:
        return

    traits = [
        f"Name: {organism.name}",
        f"Generation: {organism.generation_number}",
        f"Energy: {int(organism.energy)}",
        f"Age: {int(organism.age)}s",
        f"Lifespan: {int(organism.lifespan)}s",
        f"Speed: {organism.speed:.2f}",
        f"Strength: {organism.strength:.2f}",
        f"Sight Radius: {int(organism.sight_radius)}",
        f"Size: {int(organism.organism_size)}",
        f"Goal: {organism.current_goal}"
    ]
    
    # Panel setup
    padding = 10
    line_height = INFO_FONT.get_height()
    panel_width = 250
    content_height = len(traits) * (line_height + 5) + padding * 2
    
    panel = pygame.Surface((panel_width, content_height), pygame.SRCALPHA)
    panel.fill((30, 30, 30, 220))
    pygame.draw.rect(panel, (255,255,255), (0, 0, panel_width, content_height), 2)

    # Draw traits
    y_pos = padding
    for trait in traits:
        text_surface = INFO_FONT.render(trait, True, (255,255,255))
        panel.blit(text_surface, (padding, y_pos))
        y_pos += line_height + 5

    # Position panel
    surface.blit(panel, (SCREEN_WIDTH - panel_width - 10, 10))

class ToggleButton:
    def __init__(self, x, y, width, height, text, font, initial_state=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.state = initial_state
        self.color = (100, 100, 100)  # Grey color for the button
        self.text_color = (255, 255, 255)  # White color for the text

    def draw(self, surface):
        # Draw the button rectangle
        pygame.draw.rect(surface, self.color, self.rect)
        # Draw the button text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)

    def toggle(self):
        self.state = not self.state
        self.color = (0, 255, 0) if self.state else (100, 100, 100)  # Green when active, grey when inactive