import numpy as np
import cv2
import glob
import os 
import matplotlib.pyplot as plt 
## Code Reference https://learnopencv.com/camera-calibration-using-opencv/
## and CV Assignment 2

CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = [] 
img_path = []
 
objp = np.zeros((1, 6*9, 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
images = glob.glob('IMAGES//*.jpg')
images = sorted(images)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners2)
        print(fname," read successfully")
        img_path.append(fname)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()
 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)
print(ret)
# The camera matrix indicates that the focal lengths along the x and y axes are approximately 3133.08593 and 3102.22097 pixels respectively, and the principal point is approximately at coordinates (1342.82433, 1253.34178) pixels. Skew is 0
print(np.array(rvecs).reshape(20,1,3))
# The above matrix has 20 rows correspoding to the rotation vector of the 20 images selected for camera calibration. 
print(np.array(tvecs).reshape(20,1,3))
# The above matrix has 20 rows correspoding to the translation vector of the 20 images selected for camera calibration. 
counter = 0
for fname in images:
    counter+=1
    img = cv2.imread(fname)
    height,width = img.shape[:2]
    camMatrixNew,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(width,height),1,(width,height)) 
    imgUndist = cv2.undistort(img,mtx,dist,None,camMatrixNew)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[1].imshow(imgUndist)
    plt.show()
    if(counter==5):
        break
# When we correct for radial distortion, we are essentially applying an inverse transformation to the image to counteract the distortion introduced by the lens. This process involves re-mapping pixels in the image based on the distortion coefficients. Camera calibration aims to model the distortion accurately, but it may not be perfect. Small errors or inaccuracies in the calibration process can lead to discrepancies in the undistorted image, causing straight lines to appear slightly different from the original scene.
err_list = []
reprojected_list = []
for i, path in enumerate(img_path):
    reprojected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    reprojected_list.append(reprojected_points)
    error = cv2.norm(imgpoints[i],  reprojected_points, cv2.NORM_L2)/len(reprojected_points)
    err_list.append(error)
    print(f"Re-projection error for {path}: {error}")  

mean_value = np.mean(err_list)
std_deviation = np.std(err_list)
print("Mean Value: ", mean_value)
print("Std Deviation: ", std_deviation)
plt.bar(img_path, err_list)
plt.xticks(rotation=90)
plt.xlabel('Image')
plt.ylabel('Error')
plt.title('Reprojection Error Map')
plt.show()
for i, path in enumerate(img_path):
    img = cv2.imread(path)
    img = cv2.drawChessboardCorners(img, CHECKERBOARD, reprojected_list[i], True)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
# Absolute norm is used to calculate the distance between the actual point and projected point.
for i, path in enumerate(img_path):
    R, _ = cv2.Rodrigues(rvecs[i])
    normal_vector = R[:, 2]
    print(f"Plane normal for {path}:", normal_vector)