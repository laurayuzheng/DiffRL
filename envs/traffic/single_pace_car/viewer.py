import pygame
from highway_env.envs.common.graphics import EnvViewer, ObservationGraphics

from highway_env.envs.common.action import ActionType, DiscreteMetaAction, ContinuousAction
from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics

class PaceCarEnvViewer(EnvViewer):

    def display(self, mean_speed) -> None:
        """Display the road and vehicles on a pygame window."""
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)

        RoadGraphics.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(self.agent_surface, (0, self.config["screen_height"]))
                else:
                    self.screen.blit(self.agent_surface, (self.config["screen_width"], 0))

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        ObservationGraphics.display(self.env.observation_type, self.sim_surface)

        # render mean speed text;
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render('Mean Speed: {:.2f} m/s'.format(mean_speed), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (self.config['screen_width'] // 2, self.config['screen_height'] - 20)
        self.sim_surface.blit(text, text_rect)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / "highway-env_{}.png".format(self.frame)))
            self.frame += 1
