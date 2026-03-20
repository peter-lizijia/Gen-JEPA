import collections
import os
import warnings

import cv2
import gymnasium as gym
import numpy as np

with warnings.catch_warnings():
    # Filter out DeprecationWarnings raised from pkg_resources
    warnings.filterwarnings("ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning)
    import pygame

import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gymnasium import spaces
from pymunk.vec2d import Vec2d

from .pymunk_override import DrawOptions

RENDER_MODES = ["rgb_array"]
if os.environ.get("MUJOCO_GL") != "egl":
    RENDER_MODES.append("human")


def pymunk_to_shapely(body, shapes):
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class PushTEnv(gym.Env):
    """
    ## Description

    PushT environment.

    The goal of the agent is to push the block to the goal zone. The agent is a circle and the block is a tee shape.

    ## Action Space

    The action space is continuous and consists of two values: [x, y]. The values are in the range [0, 512] and
    represent the target position of the agent.

    ## Observation Space

    If `obs_type` is set to `state`, the observation space is a 5-dimensional vector representing the state of the
    environment: [agent_x, agent_y, block_x, block_y, block_angle]. The values are in the range [0, 512] for the agent
    and block positions and [0, 2*pi] for the block angle.

    If `obs_type` is set to `environment_state_agent_pos` the observation space is a dictionary with:
    - `environment_state`: 16-dimensional vector representing the keypoint locations of the T (in [x0, y0, x1, y1, ...]
        format). The values are in the range [0, 512]. See `get_keypoints` for a diagram showing the location of the
        keypoint indices.
    - `agent_pos`: A 2-dimensional vector representing the position of the robot end-effector.

    If `obs_type` is set to `pixels`, the observation space is a 224x224 RGB image of the environment.

    ## Rewards

    Reward is a constant 0.0. Success/failure flags are returned in `info`.

    ## Success Criteria

    The environment is considered solved if the block is at least 95% in the goal zone.

    ## Starting State

    The block starts on the right side of the road and must be pushed to the left goal area.

    ## Episode Termination

    The episode terminates on success (>=95% overlap with goal) or failure (red light + block enters the road).

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PushTEnv<gym_pusht/PushT-v0>>>>>
    ```

    * `obs_type`: (str) The observation type. Can be either `state`, `keypoints`, `pixels` or `pixels_agent_pos`.
      Default is `state`.

    * `block_cog`: (tuple) The center of gravity of the block if different from the center of mass. Default is `None`.

    * `damping`: (float) The damping factor of the environment if different from 0. Default is `None`.

    * `observation_width`: (int) The width of the observed image. Default is `224`.

    * `observation_height`: (int) The height of the observed image. Default is `224`.

    * `visualization_width`: (int) The width of the visualized image. Default is `224`.

    * `visualization_height`: (int) The height of the visualized image. Default is `224`.

    ## Reset Arguments

    Passing the option `options["reset_to_state"]` will reset the environment to a specific state.

    > [!WARNING]
    > For legacy compatibility, the inner fonctionning has been preserved, and the state set is not the same as the
    > the one passed in the argument.

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0")
    >>> state, _ = env.reset(options={"reset_to_state": [0.0, 10.0, 20.0, 30.0, 1.0]})
    >>> state
    array([ 0.      , 10.      , 57.866196, 50.686398,  1.      ],
          dtype=float32)
    ```

    ## Version History

    * v0: Original version

    ## References

    * TODO:
    """

    metadata = {"render_modes": RENDER_MODES, "render_fps": 10}
    LIGHT_GREEN = "GREEN"
    LIGHT_RED = "RED"

    def __init__(
        self,
        obs_type="state",
        render_mode="rgb_array",
        block_cog=None,
        damping=None,
        observation_width=224,
        observation_height=224,
        visualization_width=224,
        visualization_height=224,
        road_rect=(236, 308, 5, 506),
        goal_rect=(24, 128, 200, 312),
        spawn_right=(340, 440, 200, 312),
        light_durations=(1.0, 1.5),
        stop_line_margin=12.0,
    ):
        super().__init__()
        # Observations
        self.obs_type = obs_type

        # Rendering
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Initialize spaces
        self._initialize_observation_space()
        self.action_space = spaces.Box(low=0, high=512, shape=(2,), dtype=np.float32)

        # Physics
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["render_fps"]
        self.dt = 0.01
        self.step_dt = 1.0 / self.control_hz
        self.block_cog = block_cog
        self.damping = damping
        self.road_rect = self._validate_rect("road_rect", road_rect)
        self.goal_rect = self._validate_rect("goal_rect", goal_rect)
        self.spawn_right = self._validate_rect("spawn_right", spawn_right)
        self.light_durations = np.asarray(light_durations, dtype=np.float64)
        if self.light_durations.size == 0 or np.any(self.light_durations <= 0):
            raise ValueError("light_durations must be a non-empty sequence of positive values.")
        self.stop_line_margin = float(stop_line_margin)
        if self.stop_line_margin <= 0:
            raise ValueError(f"stop_line_margin must be > 0, got {stop_line_margin}")
        self.light = self.LIGHT_GREEN
        self.remaining_steps = 1

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

        self.teleop = None
        self._last_action = None

        self.success_threshold = 0.95  # 95% coverage

    def _initialize_observation_space(self):
        if self.obs_type == "state":
            # [agent_x, agent_y, block_x, block_y, block_angle]
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0]),
                high=np.array([512, 512, 512, 512, 2 * np.pi]),
                dtype=np.float64,
            )
        elif self.obs_type == "environment_state_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "environment_state": spaces.Box(
                        low=np.zeros(16),
                        high=np.full((16,), 512),
                        dtype=np.float64,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([512, 512]),
                        dtype=np.float64,
                    ),
                },
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([512, 512]),
                        dtype=np.float64,
                    ),
                }
            )
        else:
            raise ValueError(
                f"Unknown obs_type {self.obs_type}. Must be one of [pixels, state, environment_state_agent_pos, "
                "pixels_agent_pos]"
            )

    def _get_coverage(self):
        goal_body = self.get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        return intersection_area / goal_area

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._last_action = action
        for _ in range(n_steps):
            # Step PD control
            # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
            acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * self.dt

            # Step physics
            self.space.step(self.dt)

        # Compute termination flags.
        coverage = self._get_coverage()
        is_success = coverage > self.success_threshold
        is_fail = self.light == self.LIGHT_RED and self._is_block_on_road()
        terminated = is_success or is_fail
        reward = 0.0

        if not terminated:
            self._advance_light()

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = is_success
        info["is_fail"] = is_fail
        info["coverage"] = coverage

        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup()

        if options is not None and options.get("reset_to_state") is not None:
            state = np.array(options.get("reset_to_state"))
        else:
            state = self._sample_initial_state()
        self._set_state(state)
        self._reset_light()

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = False
        info["is_fail"] = False

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _draw(self):
        # Create a screen
        screen = pygame.Surface((512, 512))
        screen.fill((255, 255, 255))
        draw_options = DrawOptions(screen)
        road_color = pygame.Color(94, 182, 94) if self.light == self.LIGHT_GREEN else pygame.Color(212, 91, 91)

        # Draw traffic road (light color encodes traffic signal).
        self._draw_world_rect(screen, draw_options.surface, self.road_rect, color=road_color, width=0)
        stop_x = self.road_rect[1] + self.stop_line_margin
        stop_line_a = pymunk.pygame_util.to_pygame(Vec2d(stop_x, self.road_rect[2]), draw_options.surface)
        stop_line_b = pymunk.pygame_util.to_pygame(Vec2d(stop_x, self.road_rect[3]), draw_options.surface)
        pygame.draw.line(screen, pygame.Color("DimGray"), stop_line_a, stop_line_b, 2)

        # Draw goal pose
        goal_body = self.get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            goal_points = [pymunk.pygame_util.to_pygame(point, draw_options.surface) for point in goal_points]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(screen, pygame.Color("LightGreen"), goal_points)

        # Draw agent and block
        self.space.debug_draw(draw_options)
        return screen

    def _get_img(self, screen, width, height, render_action=False):
        img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
        img = cv2.resize(img, (width, height))
        render_size = min(width, height)
        if render_action and self._last_action is not None:
            action = np.array(self._last_action)
            coord = (action / 512 * [height, width]).astype(np.int32)
            marker_size = int(8 / 96 * render_size)
            thickness = int(1 / 96 * render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        return img

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        screen = self._draw()  # draw the environment on a screen

        if self.render_mode == "rgb_array":
            return self._get_img(screen, width=width, height=height, render_action=visualize)
        elif self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((512, 512))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(
                screen, screen.get_rect()
            )  # copy our drawings from `screen` to the visible window
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"] * int(1 / (self.dt * self.control_hz)))
            pygame.display.update()
        else:
            raise ValueError(self.render_mode)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def teleop_agent(self):
        teleop_agent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return teleop_agent(act)

    def get_obs(self):
        if self.obs_type == "state":
            agent_position = np.array(self.agent.position)
            block_position = np.array(self.block.position)
            block_angle = self.block.angle % (2 * np.pi)
            return np.concatenate([agent_position, block_position, [block_angle]], dtype=np.float64)

        if self.obs_type == "environment_state_agent_pos":
            return {
                "environment_state": self.get_keypoints(self._block_shapes).flatten(),
                "agent_pos": np.array(self.agent.position),
            }

        pixels = self._render()
        if self.obs_type == "pixels":
            return pixels
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": pixels,
                "agent_pos": np.array(self.agent.position),
            }

    @staticmethod
    def get_goal_pose_body(pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = int(1 / (self.dt * self.control_hz))
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
            "light": self.light,
            "remaining_steps": int(self.remaining_steps),
            "road_rect": self.road_rect.copy(),
            "goal_rect": self.goal_rect.copy(),
        }
        return info

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    @staticmethod
    def _validate_rect(name, rect):
        if len(rect) != 4:
            raise ValueError(f"{name} must have 4 values (x_min, x_max, y_min, y_max).")
        x_min, x_max, y_min, y_max = [float(v) for v in rect]
        if not (0 <= x_min < x_max <= 512 and 0 <= y_min < y_max <= 512):
            raise ValueError(f"{name} must satisfy 0 <= x_min < x_max <= 512 and 0 <= y_min < y_max <= 512.")
        return np.array([x_min, x_max, y_min, y_max], dtype=np.float64)

    def _sample_initial_state(self):
        block_x = self.np_random.uniform(self.spawn_right[0], self.spawn_right[1])
        block_y = self.np_random.uniform(self.spawn_right[2], self.spawn_right[3])
        block_angle = self.np_random.uniform(-0.08, 0.08)
        agent_x = np.clip(block_x + 62 + self.np_random.uniform(-8, 8), 20, 492)
        agent_y = np.clip(block_y + self.np_random.uniform(-18, 18), 20, 492)
        return np.array([agent_x, agent_y, block_x, block_y, block_angle], dtype=np.float64)

    def _sample_light_steps(self):
        duration = float(self.np_random.choice(self.light_durations))
        return max(1, int(np.round(duration / self.step_dt)))

    def _reset_light(self):
        self.light = self.LIGHT_GREEN if self.np_random.random() < 0.5 else self.LIGHT_RED
        self.remaining_steps = self._sample_light_steps()

    def _advance_light(self):
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            self.light = self.LIGHT_RED if self.light == self.LIGHT_GREEN else self.LIGHT_GREEN
            self.remaining_steps = self._sample_light_steps()

    def _get_block_aabb(self):
        left = min(shape.bb.left for shape in self.block.shapes)
        right = max(shape.bb.right for shape in self.block.shapes)
        bottom = min(shape.bb.bottom for shape in self.block.shapes)
        top = max(shape.bb.top for shape in self.block.shapes)
        return np.array([left, right, bottom, top], dtype=np.float64)

    @staticmethod
    def _aabb_intersects_rect(aabb, rect):
        return not (
            aabb[1] <= rect[0]
            or aabb[0] >= rect[1]
            or aabb[3] <= rect[2]
            or aabb[2] >= rect[3]
        )

    def _is_block_on_road(self):
        return self._aabb_intersects_rect(self._get_block_aabb(), self.road_rect)

    @staticmethod
    def _draw_world_rect(screen, surface, rect, color, width=0):
        x_min, x_max, y_min, y_max = rect
        world_points = [
            Vec2d(x_min, y_min),
            Vec2d(x_max, y_min),
            Vec2d(x_max, y_max),
            Vec2d(x_min, y_max),
        ]
        pixel_points = [pymunk.pygame_util.to_pygame(point, surface) for point in world_points]
        pygame.draw.polygon(screen, color, pixel_points, width=width)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = self.damping if self.damping is not None else 0.0
        self.teleop = False

        # Add walls
        walls = [
            self.add_segment(self.space, (5, 506), (5, 5), 2),
            self.add_segment(self.space, (5, 5), (506, 5), 2),
            self.add_segment(self.space, (506, 5), (506, 506), 2),
            self.add_segment(self.space, (5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone
        spawn_center = (
            (self.spawn_right[0] + self.spawn_right[1]) / 2.0,
            (self.spawn_right[2] + self.spawn_right[3]) / 2.0,
        )
        self.agent = self.add_circle(self.space, (spawn_center[0] + 60.0, spawn_center[1]), 15)
        self.block, self._block_shapes = self.add_tee(self.space, spawn_center, 0)
        goal_center = (
            (self.goal_rect[0] + self.goal_rect[1]) / 2.0,
            (self.goal_rect[2] + self.goal_rect[3]) / 2.0,
        )
        self.goal_pose = np.array([goal_center[0], goal_center[1], 0.0])  # x, y, theta (in radians)
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

    def _set_state(self, state):
        self.agent.position = list(state[:2])
        # Setting angle rotates with respect to center of mass, therefore will modify the geometric position if not
        # the same as CoM. Therefore should theoretically set the angle first. But for compatibility with legacy data,
        # we do the opposite.
        self.block.position = list(state[2:4])
        self.block.angle = state[4]

        # Run physics to take effect
        self.space.step(self.dt)

    @staticmethod
    def add_segment(space, a, b, radius):
        # TODO(rcadene): rename add_segment to make_segment, since it is not added to the space
        shape = pymunk.Segment(space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")  # https://htmlcolorcodes.com/color-names
        return shape

    @staticmethod
    def add_circle(space, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        space.add(body, shape)
        return body

    @staticmethod
    def add_tee(space, position, angle, scale=30, color="LightSlateGray", mask=None):
        if mask is None:
            mask = pymunk.ShapeFilter.ALL_MASKS()
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.angle = angle
        body.position = position
        body.friction = 1
        space.add(body, shape1, shape2)
        return body, [shape1, shape2]

    @staticmethod
    def get_keypoints(block_shapes):
        """Get a (8, 2) numpy array with the T keypoints.

        The T is composed of two rectangles each with 4 keypoints.

        0───────────1
        │           │
        3───4───5───2
            │   │
            │   │
            │   │
            │   │
            7───6
        """
        keypoints = []
        for shape in block_shapes:
            for v in shape.get_vertices():
                v = v.rotated(shape.body.angle)
                v = v + shape.body.position
                keypoints.append(np.array(v))
        return np.row_stack(keypoints)
