import ntpath
import os
from pathlib import Path

import cv2
import numpy as np

# Termination criteria for the iterative algorithm used to refine the corner points
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points: These represent the 3D coordinates of the chessboard corners in the real world.
# Assume the chessboard is fixed on the Z=0 plane. The points are generated in a 6x9 grid pattern,
# and each square on the chessboard is assumed to be 56 units in size.
objp = np.zeros((6*9,3), np.float32)*56
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = [] # store the 3D real-world points for each calibration image.
imgpoints = [] # store the 2D image plane points that correspond to the objpoints.

# Define the path to the directory containing the calibration images
path = os.path.dirname(os.path.abspath(__file__)) + "/good_images"
# Get a list of all PNG image files in the directory
files = Path(path).glob('*.png')

# Loop over each image file to detect chessboard corners and store the points
for file in files:
    img = cv2.imread(str(file))
    # Convert the image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners in the image
    retval, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If corners are found, refine their positions and add them to the lists
    if retval:
        objpoints.append(objp)
        # Refine the corner positions
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw the chessboard corners on the image for visualization
        cv2.drawChessboardCorners(img, (9,6), corners2, retval)

# Perform camera calibration to obtain the camera matrix, distortion coefficients, rotation vectors, and translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the obtained camera matrix, distortion coefficients, rotation vectors, and translation vectors
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# Repeat the calibration for all images, applying the undistortion using the obtained parameters
files_all = Path(path).glob('*.png')

# Loop over each image file to undistort and save the corrected images
for file in files_all:
    img = cv2.imread(str(file))
    h, w = img.shape[:2]

    # Obtain the optimal new camera matrix to undistort the image
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    # Undistort the image using the camera matrix and distortion coefficients
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image to remove any black edges resulting from the undistortion
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Save the undistorted image with the same filename in the "calibrated_images" folder
    name = ntpath.basename(file)
    cv2.imwrite(path + "/calibrated_images/" + name, dst)

# Compute the total reprojection error to evaluate the accuracy of the calibration
mean_error = 0
for i in range(len(objpoints)):
    # Project the 3D points back onto the image plane
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

    # Calculate the error between the detected points and the reprojected points
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)

    # Accumulate the error
    mean_error += error

# Print the mean reprojection error as an indication of the calibration accuracy
print("total error: {}".format(mean_error/len(objpoints)))
