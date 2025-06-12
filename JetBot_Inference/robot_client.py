# Import necessary modules
import os
import queue
import threading
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from jetbot import Camera, Robot
from PIL import Image

from rpc import RPCClient

# Establish connection with the RPC server
server = RPCClient('127.0.0.1', 8033)
server.connect()

# Define a class to represent the robot
class Bot:
    def __init__(self, mtx, dist):
        # Initialize the robot and camera instances
        self.robot = Robot()
        self.camera = Camera.instance(width=224, height=224)
        self.mtx = mtx  # Camera matrix for calibration
        self.dist = dist  # Distortion coefficients for calibration

    def calibrate_image(self, image_filename):
        # Load the image and apply camera calibration
        image = cv2.imread(str(image_filename))
        h, w = image.shape[:2]  # Get image dimensions
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 0, (w,h))  # Get optimal camera matrix
        dst = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)  # Apply un-distortion

        # Crop the image to the valid region of interest (ROI)
        x, y, w, h = roi  # Extract ROI coordinates
        dst = dst[y:y+h, x:x+w]  # Crop the image using ROI
        cv2.imwrite(image_filename, dst)  # Save the calibrated image

    def provide_image(self, image_filename):
        # Capture image from camera, save it, and calibrate
        image_arr = self.camera.value  # Get current camera frame
        image = Image.fromarray(image_arr)
        image.save(image_filename)
        self.calibrate_image(image_filename)  # Calibrate the saved image
        return image_filename

    def execute_command(self, action_to_take, speed):
        # Execute movement commands based on action
        if action_to_take == "forward":
            self.robot.forward(speed = speed)
        elif action_to_take == "left":
            self.robot.left(speed = speed)
        elif action_to_take == "right":
            self.robot.right(speed = speed)  #
        else:
            raise ValueError(f"Unknown action: {action_to_take}")  # Raise error for unknown actions

    def stop(self):
        self.robot.stop()  # Stop all robot movements

def calibrate_camera():
    # Calibrate the camera using chessboard images
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # Termination criteria for corner refinement
    objp = np.zeros((6*9,3), np.float32) * 56  # Prepare object points in the world space
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)  # Define the grid of object points

    objpoints = []  # List to store object points from all images
    imgpoints = []  # List to store image points from all images
    path = os.path.dirname(os.path.abspath(__file__)) + "/camera_calibration/good_images"  # Path to calibration images
    files = Path(path).glob('*.png')  # List of all image files in the directory

    for file in files:
        # Process each image file for camera calibration
        img = cv2.imread(str(file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        retval, corners = cv2.findChessboardCorners(gray, (9,6), None)  # Find chessboard corners
        if retval:
            objpoints.append(objp)  # Add object points to the list
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)  # Refine corner positions
            imgpoints.append(corners2)  # Add refined image points to the list
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners2, retval)

    # Perform camera calibration to get camera matrix and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist  # Return the camera matrix and distortion coefficients

def parse_args():
    # Parse command-line arguments
    arg_parser = ArgumentParser("Track performance of trained networks.")  # Initialize argument parser
    arg_parser.add_argument("--output_dir", default = "./model_input_images", help="Directory to store saved images.")  # Argument for output directory
    arg_parser.add_argument(
        "--max_actions",
        type=int,
        default=150,
        help="Maximum number of actions to take.",
    )
    arg_parser.add_argument(
        "--speed",
        type=float,
        default=2.5,
        help="Robot speed.",
    )
    arg_parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Number of seconds the action lasts.",
    )
    return arg_parser.parse_args()  # Return parsed arguments

def keyboard_kill_switch(q: queue.Queue, done: threading.Event):
    # Monitor for user input to stop the robot
    while not done.is_set():
        user_input = input("Enter x to stop the robot at any point: \n")  # Prompt user to stop the robot
        if user_input == 'x':
            q.put_nowait('x')  # Add stop signal to queue
            done.set()  # Signal that the stop event is set

def main():
    args = parse_args()
    mtx, dist  = calibrate_camera()  # Calibrate the camera and get matrix and distortion coefficients
    bot = Bot(mtx, dist)  # Initialize the Bot object with calibration data
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

    done = threading.Event()  # Event to signal when to stop the robot
    q = queue.Queue()  # Queue to handle stop signals
    look_thread = threading.Thread(target=keyboard_kill_switch, args=(q, done))  # Start thread to monitor user input
    look_thread.start()  # Start the thread

    for action_step in range(args.max_actions):
        image_filename = f"{output_dir}/{action_step:04}.png"  # Generate filename for the image
        bot.execute_command(server.model_run(bot.provide_image(image_filename)), args.speed)  # Execute the command from server model
        time.sleep(args.duration)
        bot.stop()  # Stop the robot after each action
        if not q.empty():
            # Check if stop signal is in the queue
            print("Stopping the robot...")
            bot.stop()  # Stop the robot
            break

    server.disconnect()  # Disconnect from the server

if __name__ == "__main__":
    main()
