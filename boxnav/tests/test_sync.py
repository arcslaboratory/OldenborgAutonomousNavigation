from argparse import ArgumentParser
from math import radians
from sys import argv
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

box_env = BoxEnv(boxes)

argparser = ArgumentParser("Navigate around a box environment.")

add_box_navigator_arguments(argparser)
argparser.add_argument(
    "navigator",
    type=Navigator.argparse,
    choices=list(Navigator),
    help="Navigator to run.",
)
argparser.add_argument("delay", type=float, help="Delay between actions.")

argv.append("--ue")

args = argparser.parse_args()

agent = BoxNavigator(box_env, args)


action_sequence = [
    Action.FORWARD,
    Action.FORWARD,
    Action.FORWARD,
    Action.FORWARD,
    Action.FORWARD,
    Action.ROTATE_LEFT,
    Action.ROTATE_LEFT,
    Action.ROTATE_LEFT,
    Action.FORWARD,
    Action.FORWARD,
]

for action in action_sequence:
    print("-" * 64)
    print("Action:", action)

    if args.delay != float("inf"):
        sleep(args.delay)
    else:
        _ = input("Press Enter to continue... ")

    agent.execute_action(action)

    print("BoxNav position:", agent.position)
    print("Unreal position:", agent.ue.get_location())

    print(f"BoxNav rotation: {agent.rotation:0.03}")
    print(f"Unreal rotation: {radians(agent.ue.get_rotation()[2]):0.03}")


agent.ue.close_osc()

if args.animation_extension:
    agent.save_animation("test_animation.gif")

print("Done!")
