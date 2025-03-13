import pygame
from OpenGL.GL import *

class GraphicsRenderer:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.texture = self._initialize_opengl()

    def _initialize_opengl(self):
        """Initialize OpenGL settings and create a texture."""
        glEnable(GL_TEXTURE_2D)
        glViewport(0, 0, self.screen_width, self.screen_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.screen_width, self.screen_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Create OpenGL texture
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.screen_width, self.screen_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        return texture

    def render(self, surface):
        """Convert Pygame Surface to OpenGL texture and render it."""
        texture_data = pygame.image.tostring(surface, "RGBA", True)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.screen_width, self.screen_height, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        
        # Clear and draw texture
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Draw the texture as a quad
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(self.screen_width, 0)
        glTexCoord2f(1, 0); glVertex2f(self.screen_width, self.screen_height)
        glTexCoord2f(0, 0); glVertex2f(0, self.screen_height)
        glEnd()

    def cleanup(self):
        """Clean up OpenGL resources."""
        glDeleteTextures([self.texture])