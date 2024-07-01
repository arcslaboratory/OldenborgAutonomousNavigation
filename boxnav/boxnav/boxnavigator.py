from argparse import ArgumentParser, Namespace
from enum import Enum, IntEnum
from math import atan2, cos, degrees, inf, radians, sin
from pathlib import Path
from random import choice, random, randrange, uniform
from time import sleep
from typing import Callable
from time import sleep

from celluloid import Camera
from matplotlib import pyplot as plt
from matplotlib.patches import Arrow, Wedge, Rectangle

from ue5osc import NUM_TEXTURES, Communicator, TexturedSurface

from .box import Pt, Box
from .boxenv import BoxEnv
from boxnav.environments import oldenborg_boxes as boxes


# TODO: consider change from ROTATE_LEFT and ROTATE_RIGHT to ROTATE_CCW and ROTATE_CW
class Action(Enum):
    NO_ACTION = -1
    FORWARD = 0
    BACKWARD = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3
    TELEPORT = 4

    def __str__(self) -> str:
        return self.name


class Navigator(Enum):
    PERFECT = 0
    WANDERING = 1
    TELEPORTING = 2
    VISION = 3

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def argparse(value: str) -> str:
        try:
            return Navigator[value.upper()]  # type: ignore
        except KeyError:
            return value


class ImageExtension(Enum):
    PNG = "png"
    JPG = "jpg"

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def argparse(value: str) -> str:
        try:
            return ImageExtension[value.upper()]  # type: ignore
        except KeyError:
            return value


class ImageQuality(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    HIGHEST = 4

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def argparse(value: str) -> str:
        try:
            return ImageQuality[value.upper()]  # type: ignore
        except KeyError:
            return value


def add_box_navigator_arguments(parser: ArgumentParser) -> None:
    starting_box = boxes[0]
    initial_x = starting_box.left + starting_box.width / 2
    initial_y = starting_box.lower + 50

    parser.add_argument(
        "--initial_position",
        type=Pt,
        default=Pt(initial_x, initial_y),
        help="Set the starting position of the agent",
    )
    parser.add_argument(
        "--initial_rotation",
        type=float,
        default=radians(90),
        help="Set the starting rotation of the agent",
    )
    parser.add_argument(
        "--translation_increment",
        type=float,
        default=120.0,
        help="Determines how far to move forward each step.",
    )

    parser.add_argument(
        "--rotation_increment",
        type=float,
        default=radians(10),
        help="Determines how much to rotate by for each step.",
    )

    parser.add_argument(
        "--distance_threshold",
        type=int,
        default=75,
        help="Determines how close the robot has to be to the target to activate the next one.",
    )

    parser.add_argument(
        "--direction_threshold",
        type=int,
        default=radians(12),
        help="Determines how close the robot has to be to the target to activate the next one.",
    )

    parser.add_argument(
        "--animation_extension",
        type=str,
        help="Output format for animation (no animation created if not provided).",
    )

    parser.add_argument(
        "--ue", action="store_true", help="Connect and send command to Unreal Engine."
    )

    parser.add_argument(
        "--py_port", type=int, default=7001, help="Python OSC server port."
    )

    parser.add_argument(
        "--ue_port", type=int, default=7447, help="Unreal Engine OSC server port."
    )

    parser.add_argument(
        "--image_directory",
        type=str,
        help="Directory in which images should be saved (no images saved otherwise).",
    )

    parser.add_argument(
        "--image_resolution",
        type=str,
        default="244x244",
        help="Set resolution of images as ResXxResY.",
    )

    parser.add_argument(
        "--image_quality",
        type=ImageQuality.argparse,
        default=ImageQuality.LOW,
        help="Set quality of images in the range 1 to 4.",
    )

    parser.add_argument(
        "--image_extension",
        type=ImageExtension.argparse,
        default=ImageExtension.PNG,
        help="Output format for saved images (png or jpg).",
    )

    parser.add_argument(
        "--randomize_interval",
        type=int,
        default=inf,
        help="Randomizes the texture of the walls, floors, and ceilings every N actions.",
    )

    parser.add_argument(
        "--teleport_box_size",
        type=int,
        default=100,
        help="Size of the teleport box, indicating how far teleport navigator could teleport",
    )


class BoxNavigator:
    def __init__(
        self,
        env: BoxEnv,
        args: Namespace,
        vision_callback: Callable[[str], Action] | None = None,
    ) -> None:
        self.env = env
        self.initial_position = args.initial_position
        self.env_distances = [
            Pt.distance(self.initial_position, self.env.boxes[0].target)
        ]
        for i in range(len(self.env.boxes) - 1):
            self.env_distances.append(
                Pt.distance(
                    self.env.boxes[i].target,
                    self.env.boxes[i + 1].target,
                )
            )
        self.initial_rotation = args.initial_rotation
        self.final_target = self.env.boxes[-1].target

        # TODO: find appropriate values for these
        self.target_threshold = args.distance_threshold
        self.target_half_wedge = args.direction_threshold / 2.0
        self.translation_increment = args.translation_increment
        self.rotation_increment = args.rotation_increment
        self.is_stuck_threshold = 10

        self.teleport_box_size = args.teleport_box_size

        self.generating_animation = args.animation_extension is not None
        if self.generating_animation:
            self.animation_extension = args.animation_extension
            self.animation_scale = 300
            fig, self.axis = plt.subplots()
            self.camera = Camera(fig)

        match args.navigator:
            case Navigator.PERFECT:
                self.__compute_action_navigator = self.__compute_action_correct

            case Navigator.WANDERING:
                self.possible_actions = [
                    Action.FORWARD,
                    Action.ROTATE_LEFT,
                    Action.ROTATE_RIGHT,
                ]
                # TODO: find appropriate value for this
                self.chance_of_random_action = 0.25
                self.__compute_action_navigator = self.__compute_action_wandering

            case Navigator.TELEPORTING:
                self.__compute_action_navigator = self.__compute_action_teleporting

            case Navigator.VISION:
                assert args.ue, "Vision navigator requires sync_with_ue."
                assert vision_callback, "Vision navigator requires vision_callback."

                self.vision_callback = vision_callback
                self.__compute_action_navigator = self.__compute_action_vision

            case _:
                raise NotImplementedError("Unknown navigator type.")

        self.sync_with_ue = args.ue
        if self.sync_with_ue:
            assert args.py_port, "Syncing UE requires py_port."
            assert args.ue_port, "Syncing UE requires ue_port."
            assert args.image_resolution, "Syncing UE requires image_resolution."
            assert args.image_quality, "Syncing UE requires image_quality."
            assert args.randomize_interval, "Syncing UE requires randomize_interval."

            self.ue = Communicator("127.0.0.1", args.ue_port, args.py_port)
            self.image_resolution = args.image_resolution
            self.image_quality = int(args.image_quality)
            self.randomize_interval = args.randomize_interval

            self.image_directory = None
            if args.image_directory:
                assert args.image_extension, "Saving images requires image_extension."

                self.image_directory = Path(args.image_directory).resolve()
                self.image_directory.mkdir(parents=True, exist_ok=True)

                self.image_extension = args.image_extension
                self.images_saved = 0

        self.num_resets = 0
        self.trial_num = 0

        # Initial delay is 1s so that UE can get warmed up
        self.image_delay = 3

        # All other member variables are initialized in reset()
        self.reset()

    def reset(self) -> None:
        self.position = self.initial_position
        self.rotation = self.initial_rotation
        self.target = self.env.boxes[0].target
        self.previous_target = self.position

        self.num_actions_executed = 0
        self.num_resets += 1
        self.trial_num += 1

        self.is_stuck_counter = 0

        if self.__compute_action_navigator == self.__compute_action_teleporting:
            self.current_box = self.env.boxes[0]
            self.target_inside = False
            self.dominant_direction = self.determine_direction_to_target()
            self.update_anchors()
            self.set_teleport_box()

        # self.stuck = False  # Can only be True in unreal wrapper
        # self.previous_target = self.position
        # self.current_box = self.env.boxes[0]  # Start in the first box
        # self.dominant_direction = self.determine_direction_to_target(self.target)
        # self.anchor_1 = self.rotation_anchor(self.target, self.current_box)[0]
        # self.anchor_2 = self.rotation_anchor(self.target, self.current_box)[1]

        if self.sync_with_ue:
            try:
                self.ue.set_resolution(self.image_resolution)
                self.ue.set_quality(self.image_quality)
                self.ue.set_raycast_length(self.translation_increment)

            except TimeoutError:
                self.ue.close_osc()
                print("Check if UE packaged game is running.")
                raise SystemExit

            # Reset z coordinates with a 2s delay to allow the reset before moving on
            self.ue.reset(2)

            self.__sync_ue_rotation()
            self.__sync_ue_position()

        self.__update_animation()

    def display(self) -> None:
        # Plot agent as a red circle
        self.axis.plot(self.position.x, self.position.y, "ro")

        # Plot agent's heading as a red wedge
        wedge_lo = degrees(self.rotation - self.target_half_wedge)
        wedge_hi = degrees(self.rotation + self.target_half_wedge)
        position = self.position.xy()
        wedge = Wedge(position, self.animation_scale, wedge_lo, wedge_hi, color="red")
        self.axis.add_patch(wedge)

        # Plot target as a green circle
        self.axis.plot(self.target.x, self.target.y, "go")

        # Plot line to target as a green arrow
        dxy = (self.target - self.position).normalized() * self.animation_scale
        arrow = Arrow(self.position.x, self.position.y, dxy.x, dxy.y, color="g")
        self.axis.add_patch(arrow)

        # For teleport navigator, want to display teleporting box
        if self.__compute_action_navigator == self.__compute_action_teleporting:
            self.draw_current_teleporting_box()  # Draw the rectangle
            self.axis.plot(self.teleport_box_ll.x, self.teleport_box_ll.y, "mx")
            # self.axis.plot(self.anchor_1.x, self.anchor_1.y, "mx")
            # self.axis.plot(self.anchor_2.x, self.anchor_2.y, "mx")

    def save_animation(self, filename: str, progress_bar_callback=None) -> None:
        animation = self.camera.animate()
        animation.save(filename, progress_callback=progress_bar_callback)

    def __update_animation(self) -> None:
        if self.generating_animation:
            self.env.display(self.axis)
            self.display()
            self.axis.invert_xaxis()
            self.camera.snap()

    def at_final_target(self) -> bool:
        return Pt.distance(self.position, self.final_target) < self.target_threshold

    def is_stuck(self) -> bool:
        return self.is_stuck_counter >= self.is_stuck_threshold

    def execute_action(self, action: Action) -> bool:
        "Execute the given action."

        if not self.__move_is_possible(action):
            return False

        match action:
            case Action.FORWARD:
                self.__action_translate(Action.FORWARD)
            case Action.BACKWARD:
                self.__action_translate(Action.BACKWARD)
            case Action.ROTATE_LEFT:
                self.__action_rotate(Action.ROTATE_LEFT)
            case Action.ROTATE_RIGHT:
                self.__action_rotate(Action.ROTATE_RIGHT)
            case Action.TELEPORT:
                self.__action_teleport()
            case _:
                raise NotImplementedError("Unknown action.")

        self.num_actions_executed += 1

        if self.generating_animation:
            self.__update_animation()

        return True

    def execute_navigator_action(self) -> tuple[Action, Action]:
        "Return the action taken by the navigator and the 'perfect' action."
        self.__update_target_if_necessary()

        # Compute the correct action
        action_correct = self.__compute_action_correct()

        # Save the image (needed for vision navigator and dataset generation)
        if self.sync_with_ue and self.image_directory:
            # Negate the angle because Unreal uses a left-hand coordinate system
            signed_angle_to_target = -self.__compute_signed_angle_to_target()
            angle = f"{signed_angle_to_target:+.2f}".replace(".", "p")

            self.images_saved += 1
            self.latest_image_filepath = f"{self.image_directory}/{self.trial_num:03}_{self.images_saved:06}_{angle}.{self.image_extension}"

            # Lower the delay after the first image since UE is warmed up
            self.ue.save_image(self.latest_image_filepath, delay=self.image_delay)
            self.image_delay = 0.25

        # Randomize the texture of the walls, floors, and ceilings
        if self.sync_with_ue:
            if (
                self.num_actions_executed != 0
                and self.randomize_interval != inf
                and self.num_actions_executed % self.randomize_interval == 0
            ):
                random_surface = choice(list(TexturedSurface))
                self.ue.set_texture(random_surface, randrange(NUM_TEXTURES))

        # Loop until we have an executable or until "stuck"
        while True:
            if self.is_stuck_counter >= self.is_stuck_threshold:
                return Action.NO_ACTION, action_correct

            # Compute *potential* navigator action
            action_navigator = self.__compute_action_navigator()

            # Check if navigator action is possible (rotation should not reset stuck counter)
            if action_navigator in [Action.FORWARD, Action.BACKWARD]:
                if self.__move_is_possible(action_navigator):
                    self.is_stuck_counter = 0
                    break
                else:
                    self.is_stuck_counter += 1
            else:
                break

        # Execute the action
        self.execute_action(action_navigator)
        print("Action taken: ", action_navigator)

        return action_navigator, action_correct

    def __move_is_possible(self, action: Action) -> bool:
        # Rotations are always possible
        if action in [Action.ROTATE_LEFT, Action.ROTATE_RIGHT, Action.TELEPORT]:
            return True

        sign = -1 if action == Action.BACKWARD else 1

        new_x = self.position.x + sign * self.translation_increment * cos(self.rotation)
        new_y = self.position.y + sign * self.translation_increment * sin(self.rotation)
        possible_new_position = Pt(new_x, new_y)

        # Translations are only possible if we end up in a box
        return len(self.env.get_boxes_enclosing_point(possible_new_position)) > 0

    def __compute_signed_angle_to_target(self) -> float:
        heading_vector = Pt(cos(self.rotation), sin(self.rotation)).normalized()
        target_vector = (self.target - self.position).normalized()
        return Pt.angle_between(heading_vector, target_vector)

    def __sync_ue_position(self) -> None:
        try:
            self.ue.set_location_xy(self.position.x, self.position.y, delay=0.1)

        except TimeoutError:
            self.ue.close_osc()
            print("Could not sync position with UE.")
            raise SystemExit

    def __sync_ue_rotation(self) -> None:
        try:
            # NOTE: see README for more information
            # Conversion from BoxNav to Unreal: Unreal = 180 - BoxNav
            # However, we need to maintain the same angle in both so that positions need not be converted
            # unreal_yaw = 180.0 - degrees(self.rotation)
            # self.ue.set_yaw(unreal_yaw)
            self.ue.set_yaw(degrees(self.rotation))

        except TimeoutError:
            self.ue.close_osc()
            print("Could not sync rotation with UE.")
            raise SystemExit

    def action_translate(self, direction: Action) -> None:
        self.__action_translate(direction)

    def __action_translate(self, direction: Action) -> None:
        sign = -1 if direction == Action.BACKWARD else 1

        new_x = self.position.x + sign * self.translation_increment * cos(self.rotation)
        new_y = self.position.y + sign * self.translation_increment * sin(self.rotation)
        possible_new_position = Pt(new_x, new_y)

        self.position = possible_new_position

        if self.sync_with_ue:
            self.__sync_ue_position()

    def __action_rotate(self, direction: Action) -> None:
        # NOTE: Unreal uses a left-handed coordinate system, so we use the opposite of
        # the "normal" Cartesian coordinate system. Specifically, we consider positive
        # rotations to be clockwise.
        sign = 1 if direction == Action.ROTATE_RIGHT else -1

        self.rotation += sign * self.rotation_increment

        if self.sync_with_ue:
            self.__sync_ue_rotation()

    def determine_direction_to_target(self) -> str:
        """Determine the 'direction' to the target based on changes in coordinates."""

        # Calculate the change in coordinates
        delta_x = self.target.x - self.previous_target.x
        delta_y = self.target.y - self.previous_target.y

        # Determine the dominant change
        if abs(delta_x) > abs(delta_y):
            if delta_x > 0:
                dominant_direction = "left"
            else:
                dominant_direction = "right"
        else:
            if delta_y > 0:
                dominant_direction = "up"
            else:
                dominant_direction = "down"

        return dominant_direction

    def set_teleport_box(self):
        # Generate the encompassing points of the current box:
        self.dominant_direction = self.determine_direction_to_target()

        # Calculate lower left and upper right points based on direction
        if self.dominant_direction == "left":
            self.teleport_box_ll = Pt(self.position.x, self.current_box.lower)
            self.teleport_box_ur = Pt(
                self.position.x + self.teleport_box_size, self.current_box.upper
            )
        elif self.dominant_direction == "right":
            self.teleport_box_ll = Pt(
                self.position.x - self.teleport_box_size, self.current_box.lower
            )
            self.teleport_box_ur = Pt(self.position.x, self.current_box.upper)
        elif self.dominant_direction == "up":
            self.teleport_box_ll = Pt(self.current_box.left, self.position.y)
            self.teleport_box_ur = Pt(
                self.current_box.right, self.position.y + self.teleport_box_size
            )
        elif self.dominant_direction == "down":
            self.teleport_box_ll = Pt(
                self.current_box.left, self.position.y - self.teleport_box_size
            )
            self.teleport_box_ur = Pt(self.current_box.right, self.position.y)

        # Create box using these two Pts
        teleport_box_ul = Pt(self.teleport_box_ll.x, self.teleport_box_ur.y)
        self.teleport_box = Box(
            self.teleport_box_ll, teleport_box_ul, self.teleport_box_ur, None
        )

        self.target_inside = (
            True if self.teleport_box.point_is_inside(self.target) else False
        )

    def rotation_anchor(self, current_target: Pt, current_box: Box) -> list[Pt]:
        width_half = current_box.width / 2
        height_half = current_box.height / 2

        if self.dominant_direction in ["left", "right"]:
            # Calculate the distance from the center to the left and right sides of the box
            anchor_1 = Pt(current_target.x, current_target.y - height_half)
            anchor_2 = Pt(current_target.x, current_target.y + height_half)

        elif self.dominant_direction in ["up", "down"]:
            # Calculate the distance from the center to the top and bottom sides of the box
            anchor_1 = Pt(current_target.x - width_half, current_target.y)
            anchor_2 = Pt(current_target.x + width_half, current_target.y)

        # TODO: Do we need this else statement?
        else:
            # Default to using the current target
            self.anchor_1 = Pt(current_target.x, current_target.y)
            self.anchor_2 = Pt(current_target.x, current_target.y)

        return [Pt(anchor_1.x, anchor_1.y), Pt(anchor_2.x, anchor_2.y)]

    def update_anchors(self) -> None:
        self.anchor_1 = self.rotation_anchor(self.target, self.current_box)[0]
        self.anchor_2 = self.rotation_anchor(self.target, self.current_box)[1]

    def random_rotation_to_target(self, anchor_1: Pt, anchor_2: Pt) -> float:
        random_x = uniform(anchor_1.x, anchor_2.x)
        random_y = uniform(anchor_1.y, anchor_2.y)
        angle = atan2(random_y - self.position.y, random_x - self.position.x)
        return angle

    def __action_teleport(self) -> Action:
        # TODO: shorten the next 8 lines
        # Want random pt within this box
        x = uniform(self.teleport_box_ll.x, self.teleport_box_ur.x)
        y = uniform(self.teleport_box_ll.y, self.teleport_box_ur.y)
        possible_new_position = Pt(x, y)

        # Prevent from teleporting out of bounds
        while len(self.env.get_boxes_enclosing_point(possible_new_position)) == 0:
            x = uniform(self.teleport_box_ll.x, self.teleport_box_ur.x)
            y = uniform(self.teleport_box_ll.y, self.teleport_box_ur.y)
            possible_new_position = Pt(x, y)

        self.position = possible_new_position

        # Set random rotation in direction of target
        random_angle = self.random_rotation_to_target(self.anchor_1, self.anchor_2)
        self.rotation = random_angle

        if self.sync_with_ue:
            self.__sync_ue_position()
            self.__sync_ue_rotation()

        # If target isn't in the box, update the box
        # otherwise, don't update box until target reached
        if not self.target_inside:
            self.set_teleport_box()

    def draw_current_teleporting_box(self) -> None:
        if self.num_actions_executed != 1:
            self.axis.add_patch(
                Rectangle(
                    self.teleport_box_ll.xy(),
                    self.teleport_box.width,
                    self.teleport_box.height,
                    angle=self.teleport_box.angle_degrees,
                    facecolor="yellow",
                    alpha=0.6,  # Transparency level of the rectangle
                )
            )

    def __compute_action_correct(self) -> Action:
        signed_angle_to_target = self.__compute_signed_angle_to_target()

        # Already facing in approximately the correct direction (within the wedge)
        if abs(signed_angle_to_target) < self.target_half_wedge:
            action = Action.FORWARD

        # NOTE: Unreal uses a left-handed coordinate system, so we use the opposite of
        # the "normal" Cartesian coordinate system. Specifically, we consider positive
        # rotations to be clockwise.
        elif signed_angle_to_target < 0:
            action = Action.ROTATE_LEFT
        else:
            action = Action.ROTATE_RIGHT

        return action

    def __compute_action_wandering(self) -> Action:
        # TODO: we can probably do much better than this
        if random() < self.chance_of_random_action:
            return choice(self.possible_actions)
        else:
            return self.__compute_action_correct()

    def __compute_action_teleporting(self) -> Action:
        return Action.TELEPORT

    def __compute_action_vision(self) -> Action:
        # What do we pass to the callback?
        # - number of resets
        # - number of actions executed
        # - ...
        return self.vision_callback(self.latest_image_filepath)

    def __update_target_if_necessary(self) -> None:
        assert not self.at_final_target(), "Already at final target."

        surrounding_boxes = self.env.get_boxes_enclosing_point(self.position)
        distance_to_target = Pt.distance(self.position, self.target)

        if distance_to_target < self.target_threshold:
            self.previous_target = self.target

            # This will throw an exception if the boxes do not properly overlap
            self.target = surrounding_boxes[-1].target

            # Update variables specific to teleport
            if self.__compute_action_navigator == self.__compute_action_teleporting:
                self.current_box = surrounding_boxes[-1]
                self.dominant_direction = self.determine_direction_to_target()
                self.target_inside = False
                self.update_anchors()

    def get_percent_through_env(self) -> float:
        last_box = self.env.get_boxes_enclosing_point(self.position)[-1]
        progress = sum(d for d in self.env_distances[: self.env.boxes.index(last_box)])

        progress += self.env_distances[self.env.boxes.index(last_box)] - Pt.distance(
            self.position,
            last_box.target,
        )

        return (progress / sum(self.env_distances)) * 100
