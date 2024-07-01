import pathlib
import platform
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
from math import radians
from pathlib import Path, PurePath

import enlighten

import wandb
from fastai.callback.wandb import WandbCallback
from fastai.vision.learner import load_learner
from utils import y_from_filename  # noqa: F401 (needed for fastai load_learner)
from subprocess import run as sprun
from time import sleep

from boxnav.box import Pt
from boxnav.boxenv import BoxEnv
from boxnav.boxnavigator import (
    Action,
    BoxNavigator,
    Navigator,
    add_box_navigator_arguments,
)
from boxnav.environments import oldenborg_boxes as boxes


@contextmanager
def set_posix_windows():
    # NOTE: This is a workaround for a bug in fastai/pathlib on Windows
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def check_path(directory: str) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    # Check if directory is empty
    if len(list(Path(path).iterdir())) != 0:
        raise ValueError(f"Directory {path} is not empty.")


fastai_to_boxnav = {
    "left": Action.ROTATE_LEFT,
    "right": Action.ROTATE_RIGHT,
    "forward": Action.FORWARD,
}

action_prev = Action.NO_ACTION


def inference_func(model, image_file: str):
    global action_prev

    action_now, action_index, action_probs = model.predict(image_file)

    action_now = fastai_to_boxnav[action_now]

    # Prevent cycling actions (e.g., left followed by right)
    right_left = action_now == Action.ROTATE_LEFT and action_prev == Action.ROTATE_RIGHT
    left_right = action_now == Action.ROTATE_RIGHT and action_prev == Action.ROTATE_LEFT
    if right_left or left_right:
        action_index = action_probs.argsort()[1]
        action_now = fastai_to_boxnav[model.dls.vocab[action_index]]

    # TODO: Maybe log with loguru
    # action_prob = action_probs[action_index]
    # print(f"Moving {action_to_take} with probability {action_prob:.2f}")

    action_prev = action_now
    return action_now


def parse_args():
    arg_parser = ArgumentParser("Track performance of trained networks.")

    # Wandb configuration
    arg_parser.add_argument("wandb_name", help="Name of run inference results.")
    arg_parser.add_argument("wandb_project", help="Wandb project name.")
    arg_parser.add_argument("wandb_notes", help="Wandb run description.")
    arg_parser.add_argument("wandb_model", help="Path to the model to evaluate.")

    arg_parser.add_argument("output_dir", help="Directory to store saved images.")

    arg_parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of times to run model through environment",
    )
    arg_parser.add_argument(
        "--max_actions",
        type=int,
        default=10,
        help="Maximum number of actions to take.",
    )
    add_box_navigator_arguments(arg_parser)

    return arg_parser.parse_args()


def main():
    args = parse_args()

    args.navigator = Navigator.VISION
    args.ue = True
    args.image_directory = args.output_dir

    wandb_entity = "arcslaboratory"
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name
    wandb_notes = args.wandb_notes
    wandb_model = args.wandb_model

    run = wandb.init(
        job_type="inference",
        entity=wandb_entity,
        name=wandb_name,
        project=wandb_project,
        notes=wandb_notes,
    )

    if run is None:
        raise Exception("wandb.init() failed")

    # Download the fastai learner
    artifact = run.use_artifact(f"{wandb_model}:latest", type="model")
    model_dir = artifact.download()
    model_filename = Path(model_dir) / (wandb_model + ".pkl")

    # Load the learner and its model
    # TODO: this doesn't load the "best" model, but the last one
    # We should probably also download the weights and load them manually
    if platform.system() == "Windows":
        with set_posix_windows():
            model = load_learner(model_filename)
    else:
        model = load_learner(model_filename)

    # TODO: temporary fix? (we might remove callback on training side)
    model.remove_cb(WandbCallback)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if output directory is empty
    if any(output_dir.iterdir()):
        print("Output directory is not empty. Aborting.")
        return

    if args.output_dir:
        args.ue = True

    if args.output_dir:
        check_path(args.output_dir)

    print("Starting inference.")

    box_env = BoxEnv(boxes)

    starting_box = boxes[0]
    initial_x = starting_box.left + starting_box.width / 2
    initial_y = starting_box.lower + 50
    initial_position = Pt(initial_x, initial_y)
    initial_rotation = radians(90)

    # TODO: use context manager for UE connection?
    agent = BoxNavigator(
        box_env,
        initial_position,
        initial_rotation,
        args,
        vision_callback=partial(inference_func, model),
    )

    pbar_manager = enlighten.get_manager()
    trials_pbar = pbar_manager.counter(
        total=args.num_trials,
        desc="Trials and Avg. Completion",
        color="yellow",
        leave=True,
    )
    navigation_pbar = trials_pbar.add_subcounter("green", count=0)
    navigation_counter = 0

    inference_data = []
    average_actions = []

    for _ in range(args.num_trials):
        total_actions_taken, correct_action_taken = 0, 0
        forward_count, rotate_left_count, rotate_right_count = 0, 0, 0
        incorrect_left_count, incorrect_right_count = 0, 0

        for i in range(args.max_actions):
            try:
                executed_action, correct_action = agent.execute_navigator_action()

            except Exception as e:
                print(e)
                break

            total_actions_taken += 1
            correct_action_taken += 1 if executed_action == correct_action else 0
            if (
                executed_action == Action.ROTATE_LEFT
                and correct_action == Action.ROTATE_RIGHT
            ):
                incorrect_left_count += 1
            elif (
                executed_action == Action.ROTATE_RIGHT
                and correct_action == Action.ROTATE_LEFT
            ):
                incorrect_right_count += 1

            match executed_action:
                case Action.FORWARD:
                    forward_count += 1
                case Action.ROTATE_LEFT:
                    rotate_left_count += 1
                case Action.ROTATE_RIGHT:
                    rotate_right_count += 1

            if agent.get_percent_through_env() >= 99.0:
                average_actions.append(total_actions_taken)
                print("Agent reached final target.")
                break

            elif agent.is_stuck():
                print("Agent is stuck.")
                break

            navigation_pbar.count = navigation_counter + (
                int(agent.get_percent_through_env()) / 100
            )
            trials_pbar.count = _ + 1
            navigation_pbar.update(incr=0)

        navigation_counter += int(agent.get_percent_through_env()) / 100

        run_data = [
            agent.get_percent_through_env(),
            total_actions_taken,
            correct_action_taken,
            forward_count,
            rotate_left_count,
            rotate_right_count,
            incorrect_left_count,
            incorrect_right_count,
        ]
        inference_data.append(run_data)

        agent.reset()

    final_metrics = (
        "\n\nCompleted "
        + str(100 * (navigation_pbar.count / args.num_trials))
        + "% on average across "
        + str(args.num_trials)
        + " trial(s)"
    )

    if len(average_actions) > 0:
        final_metrics += (
            "with the agent taking "
            + str(sum(average_actions) / len(average_actions))
            + " actions on average to finish across "
            + str(len(average_actions))
            + " trial(s).\n\n"
        )

    else:
        final_metrics += ".\n\n"

    agent.ue.close_osc()
    trials_pbar.close()
    pbar_manager.stop()

    # Implement new table
    table_cols = [
        "Percent through environment",
        "Total Actions Taken",
        "Correct Actions Taken",
        "Forward Action Taken",
        "Rotate Left Action Taken",
        "Rotate Right Action Taken",
        "Incorrect Left Taken",
        "Incorrect Right Taken",
    ]
    inference_data_table = wandb.Table(columns=table_cols, data=inference_data)
    run.log({"Inference Data": inference_data_table})

    video_name = PurePath(agent.image_directory).stem
    filelist_name = video_name + "_filelist.txt"

    image_files = sorted(agent.image_directory.glob("*.png"))
    with open(filelist_name, "w") as file_out:
        for file in image_files:
            file_out.write(f"file '{Path(file)}'\n")

    sprun(
        [
            "ffmpeg",
            # "-f" is the format argument and "concat" specifies that the format is to concatenate multiple files
            "-f",
            "concat",
            # "-safe" is the argument of whether to check the input paths for safety and "0" says they don't need to be checked
            "-safe",
            "0",
            # "-i" is the input file argument and "filelist_name" is the file containing the paths to the images that will be concatenated
            "-i",
            filelist_name,
            # "-c:v" sets the codec and specifies it is for video stream and "libx264" specifies the codec to encode the video stream
            # "-c:v",
            # "libx264",
            # # "-pix_fmt" specifies the pixel format for the output video and "yuv420p" is a pixel format using YUV color space and 4:2:0 chroma subsampling
            # "-pix_fmt",
            # "yuv420p",
            # the following specifies where the output video will be saved
            agent.image_directory / (video_name + ".mp4"),
            # agent.image_directory / (video_name),
        ]
    )

    print(final_metrics)


if __name__ == "__main__":
    main()
