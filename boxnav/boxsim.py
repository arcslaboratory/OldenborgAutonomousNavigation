from argparse import ArgumentParser, Namespace
from math import radians
from pathlib import Path, PurePath
from time import sleep

import enlighten
from subprocess import run as sprun
from os import chdir

from boxnav.box import Pt
from boxnav.boxenv import BoxEnv
from boxnav.boxnavigator import BoxNavigator, Navigator, add_box_navigator_arguments
from boxnav.environments import oldenborg_boxes as boxes


def check_path(directory: str) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    # Check if directory is empty
    if len(list(Path(path).iterdir())) != 0:
        raise ValueError(f"Directory {path} is not empty.")


def simulate(args: Namespace) -> None:
    """Create and update the box environment and run the navigator."""

    box_env = BoxEnv(boxes)

    # TODO: move to constructor
    starting_box = boxes[0]
    initial_x = starting_box.left + starting_box.width / 2
    initial_y = starting_box.lower + 50
    initial_position = Pt(initial_x, initial_y)
    initial_rotation = radians(90)

    # TODO: use context manager for UE connection?
    agent = BoxNavigator(box_env, initial_position, initial_rotation, args)

    pbar_manager = enlighten.get_manager()
    actions_pbar = pbar_manager.counter(total=args.max_total_actions, desc="   Actions")

    for _ in range(args.max_total_actions):
        # Check for stopping conditions
        if agent.is_stuck() or agent.at_final_target():
            num_actions = agent.num_actions_executed

            if agent.at_final_target():
                print(f"Agent reached final target in {num_actions} actions.")
            else:
                print(f"Agent was stuck after {num_actions} actions.")

            if args.stop_after_one_trial:
                break

            agent.reset()

        # We are not using the return value when generating a dataset
        action_taken, _ = agent.execute_navigator_action()
        print("Action taken:", action_taken)

        # Navigation progress is based on the percentage of the environment navigated
        actions_pbar.update()

    if args.ue:
        agent.ue.close_osc()

    actions_pbar.close()

    print("Simulation complete.")

    if args.auto_upload:
        chdir("..")
        chdir("learning")
        sprun(
            [
                "python",
                "upload_data.py",
                PurePath(agent.image_directory).stem,
                args.auto_upload,
                (
                    "Automatically uploading data from "
                    + PurePath(agent.image_directory).stem
                    + " run."
                ),
                agent.image_directory,
            ]
        )
        chdir("..")
        chdir("boxnav")

    if args.animation_extension:
        # Generate a unique filename (don't overwrite previous animations)
        num = 1
        while True:
            output_filename = f"output_{num:02}.{args.animation_extension}"

            if not Path(output_filename).exists():
                break
            num += 1

        # The total number of animation frames depends on the number of actions taken
        # We have two cases:
        # 1. We completed all args.max_total_actions actions
        # 2. We stopped early because we reached the final target (or got stuck) after one trial
        total_actions = args.max_total_actions
        if args.stop_after_one_trial:
            total_actions = agent.num_actions_executed

        animation_pbar = pbar_manager.counter(total=total_actions, desc="    Frames")
        agent.save_animation(output_filename, lambda i, n: animation_pbar.update())
        animation_pbar.close()
        print(f"Animation saved to {output_filename}...", end=" ", flush=True)

    pbar_manager.stop()


def main():
    """Parse arguments and run simulation."""

    argparser = ArgumentParser("Navigate around a box environment.")

    argparser.add_argument(
        "navigator",
        type=Navigator.argparse,
        choices=list(Navigator),
        help="Navigator to run.",
    )

    add_box_navigator_arguments(argparser)

    # The following two arguments are specific to data generation

    argparser.add_argument(
        "--max_total_actions",
        type=int,
        default=10,
        help="Maximum total allowed actions across all trials.",
    )

    argparser.add_argument(
        "--stop_after_one_trial",
        action="store_true",
        help="Stop after one time through the environment (for debugging).",
    )

    argparser.add_argument(
        "--auto_upload",
        type=str,
        help="Automatically upload data to wandb after running boxsim. Provide wandb project name (existing or new).",
    )

    args = argparser.parse_args()

    if args.image_directory:
        args.ue = True

    if args.image_directory:
        check_path(args.image_directory)

    print("Starting simulation.")
    simulate(args)


if __name__ == "__main__":
    main()
