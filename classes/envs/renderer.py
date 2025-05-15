import pygame
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Tuple, Optional, List, Any, Union
from abc import ABC, abstractmethod

from classes.helper import *
from classes.envs import create_atari_env
from ocatari.utils import draw_label, draw_arrow

UPSCALE_FACTOR = 4


def get_human_renderer(config):
    if config.montezuma_first_room:
        return MontezumaFirstRoomRender(config, ret_image=False)
    return BoxRenderer(config, ret_image=False)


def get_image_renderer(config):
    if config.montezuma_first_room:
        return MontezumaFirstRoomRender(config, ret_image=True)
    return ColoredBoxRenderer(config, ret_image=True)


class BaseRenderer(ABC):
    """
    Base class for game renderers that all games should inherit from.
    """
    @abstractmethod
    def render(self, obj_list):
        """
        Visualize the given ObjList `obj_list`.
        
        Return either None or a 2d array representing the rendered image.
        """
        pass


class BoxRenderer(BaseRenderer):
    def __init__(self, config, ret_image=False):
        self.config = config
        self.ret_image = ret_image
        self.rendering_initialized = False
        self.image_size = (160, 210)

    def _initialize_rendering(self):
        pygame.init()
        self.window_size = (self.image_size[0] * UPSCALE_FACTOR,
                            self.image_size[1] * UPSCALE_FACTOR
                            )  # render with higher res
        self.label_font = pygame.font.SysFont('Pixel12x10', 16)
        if self.ret_image:
            self.window = pygame.Surface(self.window_size)
        else:
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        self.rendering_initialized = True

    def render(self, obj_list):
        # Prepare screen if not initialized
        if not self.rendering_initialized:
            self._initialize_rendering()

        # Clear the window
        self.window.fill((0, 0, 0))

        # Init overlay surface
        overlay_surface = pygame.Surface(self.window_size)
        overlay_surface.set_colorkey((0, 0, 0))

        # For each object, render its position and velocity vector
        for obj in obj_list:
            # if game_object is None:
            #     continue

            x, y = obj.x, obj.y
            w, h = obj.w, obj.h

            if x == np.nan:
                continue

            # Object velocity
            dx = obj.velocity_x
            dy = obj.velocity_y

            # Transform to upscaled screen resolution
            x *= UPSCALE_FACTOR
            y *= UPSCALE_FACTOR
            w *= UPSCALE_FACTOR
            h *= UPSCALE_FACTOR
            dx *= UPSCALE_FACTOR
            dy *= UPSCALE_FACTOR

            # Compute center coordinates
            x_c = x + w // 2
            y_c = y + h // 2

            # Draw an 'X' at object center
            pygame.draw.line(overlay_surface,
                             color=(255, 255, 255),
                             width=2,
                             start_pos=(x_c - 4, y_c - 4),
                             end_pos=(x_c + 4, y_c + 4))
            pygame.draw.line(overlay_surface,
                             color=(255, 255, 255),
                             width=2,
                             start_pos=(x_c - 4, y_c + 4),
                             end_pos=(x_c + 4, y_c - 4))

            # Draw bounding box
            pygame.draw.rect(overlay_surface,
                             color=(255, 255, 255),
                             rect=(x, y, w, h),
                             width=2)

            # Draw object category label (optional with value)
            label = obj.obj_type
            draw_label(self.window,
                       label,
                       position=(x, y + h + 4),
                       font=self.label_font)

            # Draw object orientation
            # if game_object.orientation is not None:
            #     draw_orientation_indicator(overlay_surface, game_object.orientation.value, x_c, y_c, w, h)

            # Draw velocity vector
            if dx != 0 or dy != 0:
                draw_arrow(overlay_surface,
                           start_pos=(float(x_c), float(y_c)),
                           end_pos=(x_c + 2 * dx, y_c + 2 * dy),
                           color=(100, 200, 255),
                           width=2)

        self.window.blit(overlay_surface, (0, 0))

        if self.ret_image:
            frame = pygame.surfarray.array3d(self.window)
            frame = np.rot90(frame, k=-1)
            frame = np.fliplr(frame)
            return frame
        else:
            frameskip = 1
            self.clock.tick(
                60 // frameskip)  # limit FPS to avoid super fast movement
            pygame.display.flip()
            pygame.event.pump()
            

class MontezumaFirstRoomRender(BaseRenderer):
    def __init__(self, config, ret_image=False):
        self.config = config
        self.atari_env = create_atari_env(config, 'MontezumaRevenge')
        self.atari_env.reset()
        self.arr = self.atari_env.env.get_ram()
        self.ret_image = ret_image

    def render(self, obj_list: ObjList) -> Optional[Any]:
        """
        Renders the current state of the imagined environment.

        Args:
            obj_list: List of objects to render
            ret_image: If True, returns the rendered image instead of displaying

        Returns:
            Optional rendered image if ret_image is True
        """
        for i in range(0, 128):
            self.atari_env.env.set_ram(i, self.arr[i])

        try:
            self.atari_env.env.set_ram(42, obj_list[0].x + 1)
            self.atari_env.env.set_ram(43, 255 + 53 - obj_list[0].y)
        except:
            pass

        # skulls = obj_list.get_objs_by_obj_type('skull')
        # if len(skulls) > 0:
        #     self.atari_env.env.set_ram(47, skulls[0].x - 32)
        #     self.atari_env.env.set_ram(46, 406 - skulls[0].y)

        # if len(obj_list.get_objs_by_obj_type('key')) > 0:
        #     self.atari_env.env.set_ram(49, 4)
        # else:
        #     self.atari_env.env.set_ram(49, 0)
        #     if len(obj_list.get_objs_by_obj_type('barrier')) == 2:
        #         self.atari_env.env.set_ram(65, 16) # have key hud
        #     elif obj_list.get_objs_by_obj_type('barrier')[0].x < 100:
        #         self.atari_env.env.set_ram(28, 117) # right barrier gone
        #     else:
        #         self.atari_env.env.set_ram(26, 117) # left barrier gone

        self.atari_env.env.step(0)

        if self.ret_image:
            return self.atari_env.env._env.render()
        else:
            # Need to do this to get real bounding boxes
            for game_object in self.atari_env.env.objects:
                if game_object.category.lower() == 'player':
                    game_object.xy = (obj_list[0].x, obj_list[0].y)
            self.atari_env.env.render()


class ColoredBoxRenderer(BoxRenderer):
    def __init__(self, config, ret_image=False):
        super().__init__(config, ret_image)
        # Define a color mapping for different object categories
        if config.task.startswith('Montezuma'):
            self.color_map = {
                'player': (255, 0, 0),       # Red
                'skull': (255, 255, 255),  # White
                'ladder': (101, 67, 33),      # Dark Brown
                'key': (255, 255, 0),    # Yellow
                'platform': (128, 128, 128),# Gray
                'zone': (255, 165, 0),      # Orange
                'wall': (0, 100, 0),         # Dark Green
                'ball': (0, 0, 255),      # Blue
                'default': (255, 255, 255)  # White
            }
        else:
            self.color_map = {
                'player': (144, 238, 144),    # Light Green
                'ball': (255, 255, 255),          # White
                'enemy': (255, 165, 0),        # Orange
                'wall': (128, 128, 128),          # dark gray
                'zone': (128, 128, 128),          # dark gray
                'default': (139, 69, 19),          # BRown
            }

    def render(self, obj_list):
        # Prepare screen if not initialized
        if not self.rendering_initialized:
            self._initialize_rendering()

        # Clear the window
        if self.config.task.startswith('Montezuma'):
            self.window.fill((0, 0, 0))
        else:
            self.window.fill((139, 69, 19))  # Brown color

        # Init overlay surface
        overlay_surface = pygame.Surface(self.window_size)
        overlay_surface.set_colorkey((0, 0, 0))

        # Separate player from other objects
        player_obj = None
        other_objs = []

        # First pass: separate player and other objects
        for obj in obj_list:
            if obj.obj_type.lower() == 'player':
                player_obj = obj
            else:
                other_objs.append(obj)

        # Render all non-player objects first
        for obj in other_objs:
            if obj.x == np.nan:
                continue

            x, y = obj.x, obj.y
            w, h = obj.w, obj.h

            # Object velocity
            dx = obj.velocity_x
            dy = obj.velocity_y

            # Transform to upscaled screen resolution
            x *= UPSCALE_FACTOR
            y *= UPSCALE_FACTOR
            w *= UPSCALE_FACTOR
            h *= UPSCALE_FACTOR
            dx *= UPSCALE_FACTOR
            dy *= UPSCALE_FACTOR

            # Compute center coordinates
            x_c = x + w // 2
            y_c = y + h // 2

            # Get color based on object category
            color = self.color_map.get(obj.obj_type.lower(), self.color_map['default'])

            # Special rendering for ladders
            if obj.obj_type.lower() == 'ladder':
                # Draw the main ladder rectangle
                pygame.draw.rect(overlay_surface,
                               color=color,
                               rect=(x, y, w, h))
                
                # Draw horizontal stripes
                stripe_height = 4  # Height of each stripe
                stripe_spacing = 8  # Space between stripes
                stripe_color = (0, 0, 0)  # Black stripes
                
                # Calculate number of stripes based on height
                num_stripes = int(h / (stripe_height + stripe_spacing))
                
                # Draw stripes
                for i in range(num_stripes):
                    stripe_y = y + i * (stripe_height + stripe_spacing)
                    pygame.draw.rect(overlay_surface,
                                   color=stripe_color,
                                   rect=(x, stripe_y, w, stripe_height))
            else:
                # Draw filled box with the category color for non-ladder objects
                pygame.draw.rect(overlay_surface,
                               color=color,
                               rect=(x, y, w, h))

            # Draw an 'X' at object center
            pygame.draw.line(overlay_surface,
                           color=(255, 255, 255),
                           width=2,
                           start_pos=(x_c - 4, y_c - 4),
                           end_pos=(x_c + 4, y_c + 4))
            pygame.draw.line(overlay_surface,
                           color=(255, 255, 255),
                           width=2,
                           start_pos=(x_c - 4, y_c + 4),
                           end_pos=(x_c + 4, y_c - 4))

            # Draw object category label
            label = obj.obj_type
            draw_label(self.window,
                      label,
                      position=(x, y + h + 4),
                      font=self.label_font)

            # Draw velocity vector
            if dx != 0 or dy != 0:
                draw_arrow(overlay_surface,
                         start_pos=(float(x_c), float(y_c)),
                         end_pos=(x_c + 2 * dx, y_c + 2 * dy),
                         color=(100, 200, 255),
                         width=2)

        # Render player last (on top of everything)
        if player_obj is not None:
            x, y = player_obj.x, player_obj.y
            w, h = player_obj.w, player_obj.h

            # Object velocity
            dx = player_obj.velocity_x
            dy = player_obj.velocity_y

            # Transform to upscaled screen resolution
            x *= UPSCALE_FACTOR
            y *= UPSCALE_FACTOR
            w *= UPSCALE_FACTOR
            h *= UPSCALE_FACTOR
            dx *= UPSCALE_FACTOR
            dy *= UPSCALE_FACTOR

            # Compute center coordinates
            x_c = x + w // 2
            y_c = y + h // 2

            # Draw player box
            pygame.draw.rect(overlay_surface,
                           color=self.color_map['player'],
                           rect=(x, y, w, h))

            # Draw an 'X' at player center
            pygame.draw.line(overlay_surface,
                           color=(255, 255, 255),
                           width=2,
                           start_pos=(x_c - 4, y_c - 4),
                           end_pos=(x_c + 4, y_c + 4))
            pygame.draw.line(overlay_surface,
                           color=(255, 255, 255),
                           width=2,
                           start_pos=(x_c - 4, y_c + 4),
                           end_pos=(x_c + 4, y_c - 4))

            # Draw player label
            draw_label(self.window,
                      'player',
                      position=(x, y + h + 4),
                      font=self.label_font)

            # Draw velocity vector
            if dx != 0 or dy != 0:
                draw_arrow(overlay_surface,
                         start_pos=(float(x_c), float(y_c)),
                         end_pos=(x_c + 2 * dx, y_c + 2 * dy),
                         color=(100, 200, 255),
                         width=2)

        self.window.blit(overlay_surface, (0, 0))

        if self.ret_image:
            frame = pygame.surfarray.array3d(self.window)
            frame = np.rot90(frame, k=-1)
            frame = np.fliplr(frame)
            return frame
        else:
            frameskip = 1
            self.clock.tick(60 // frameskip)  # limit FPS to avoid super fast movement
            pygame.display.flip()
            pygame.event.pump()
