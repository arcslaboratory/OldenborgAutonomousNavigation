from argparse import ArgumentParser

from . import Communicator


def main():
    """Argument Parser that verifies that the image path is getting passed in and
    optional ability to set ip and ports."""
    parser = ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP Address")
    parser.add_argument("--ue_port", type=int, default=7447, help="UE server port.")
    parser.add_argument("--py_port", type=int, default=7001, help="Python server port.")
    parser.add_argument("--resolution", type=str, default="1280x720", help="Resolution")
    # parser.add_argument("--save_image", type=str, help="Image path")
    parser.add_argument("--reset", action="store_true", help="Reset to start position")
    parser.add_argument(
        "--set_location",
        type=float,
        nargs=2,
        help="Set (x,y) location of Unreal Camera",
    )
    parser.add_argument(
        "--set_yaw", type=float, help="Set the rotation (on xy-plane) of Unreal Camera"
    )
    parser.add_argument(
        "--get_location",
        action="store_true",
        help="Return current location of Unreal Camera",
    )
    parser.add_argument(
        "--get_rotation",
        action="store_true",
        help="Return current (roll, pitch, yaw) of Unreal Camera",
    )
    parser.add_argument(
        "--set_texture",
        type=int,
        nargs=2,
        help="Set the texture of walls/floors/ceilings to a different material: (object, material) where object can be 0, 1, or 2",
    )

    args = parser.parse_args()

    with Communicator(args.ip, args.ue_port, args.py_port) as osc_communicator:
        # TODO: Execute arguments in order... (probably need to manually reference argv)

        if args.reset:
            osc_communicator.reset()

        if args.set_location:
            osc_communicator.set_location_xy(args.set_location[0], args.set_location[1])

        if args.set_yaw:
            osc_communicator.set_yaw(args.set_yaw)

        if args.get_location:
            print(osc_communicator.get_location())

        if args.get_rotation:
            print(osc_communicator.get_rotation())

        if args.resolution:
            osc_communicator.set_resolution(args.resolution)

        if args.set_texture:
            osc_communicator.set_texture(args.set_texture[0], args.set_texture[1])


# Calling main function
if __name__ == "__main__":
    main()
