import open3d as o3d
import numpy as np
import glob
import os
import matplotlib.pyplot as plt 
import cv2
pcd = o3d.io.read_point_cloud(r"CV-A2-calibration\lidar_scans\frame_1473.pcd")
pcd_points = np.asarray(pcd.points)
o3d.visualization.draw_geometries([pcd])
pcd_list = glob.glob('CV-A2-calibration\lidar_scans\*.pcd')
pcd_list = sorted(pcd_list)
print(pcd_list)
normal_list = []
offset_list = []

for path in pcd_list:
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    _, _, V = np.linalg.svd(points)
    normal = V[-1]
    offset = -np.dot(normal, centroid)
    normal_list.append(normal)
    offset_list.append(offset)

# print(np.array(normal_list).shape)
parameter_list = glob.glob('CV-A2-calibration\camera_parameters\*.jpeg')
parameter_list = sorted(parameter_list)
print(parameter_list)
camera_normal_list = []

for path in parameter_list:
    normal_path = f"{path}\\camera_normals.txt"
    with open(normal_path,'r') as f:
        temp = []
        for line in f:
            temp.append(eval(line.strip()))
    camera_normal_list.append(temp)
print(np.array(camera_normal_list).shape)
## to calculate translation matrix we first get the normal with respect to camera
thetac = np.array(camera_normal_list)
temp = thetac.T @ thetac
temp_inv = np.linalg.inv(temp)
temp2 = temp_inv @ thetac.T
translation_matrix = -1*(temp2 @ np.array(offset_list))
print(translation_matrix)
translation_matrix = translation_matrix.reshape((3,1))
## to calculate rotation matrix we do
temp = thetac.T @ np.array(normal_list)
U, S, V = np.linalg.svd(temp)
rotation_matrix = -1*(V.T @ U.T)
print(rotation_matrix.shape)
print(np.linalg.det(rotation_matrix))
intrinsic_parameter = np.array([[6.353664855742439386e+02, 0.000000000000000000e+00, 6.433965876009681324e+02],
                       [0.000000000000000000e+00, 6.261901730718112731e+02, 3.880747880982142988e+02],
                       [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
for i, path in enumerate(pcd_list):
    img_name = parameter_list[i].split("\\")[-1]
    print(img_name)
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    camera_points = []
    for point in points:
        temp = point @ rotation_matrix + translation_matrix
        camera_points.append(temp)
    break
for i, path in enumerate(pcd_list):
    img_name = parameter_list[i].split("\\")[-1]
    rmatrix = []
    with open(f"{parameter_list[i]}//rotation_matrix.txt", 'r') as f:
        for line in f:
            rmatrix.append([eval(x) for x in line.split()])
    
    tmatrix = []
    with open(f"{parameter_list[i]}//translation_vectors.txt", 'r') as f:
        for line in f:
            tmatrix.append(eval(line.strip()))
    print(parameter_list[i])
    print(img_name)
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    camera_points = []
    for point in points:
        temp = rotation_matrix @ point.reshape((3,1)) + translation_matrix
        camera_points.append(temp)
    camera_points = np.array(camera_points)
    point2d, _ = cv2.projectPoints(np.array(camera_points), np.array(rmatrix), np.array(tmatrix), intrinsic_parameter, None)
    print(point2d.shape)
    img_path = f"CV-A2-calibration\camera_images\{img_name}"
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.scatter(point2d[:, 0, 0], point2d[:, 0, 1], c='r', marker='o', label='Projected Points')
    # plt.gca().invert_yaxis()  # Invert y-axis to match OpenCV's coordinate system
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Checkerboard with Projected 3D Points')
    plt.legend()
    plt.show()

# No note all points are within the boundary, only in few cases. Although the orientation of the points matches the orintentaion of the chessboards.
from mpl_toolkits.mplot3d import Axes3D

err_list = []
img_list = []
for i in range(38):
    lidar_normal = np.array(normal_list[i])
    camera_normal = camera_normal_list[i]
    transformed_lidar_normal = np.array(rotation_matrix) @ np.array(lidar_normal)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, lidar_normal[0], lidar_normal[1], lidar_normal[2], arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, camera_normal[0], camera_normal[1],camera_normal[2], arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, transformed_lidar_normal[0], transformed_lidar_normal[1], transformed_lidar_normal[2], arrow_length_ratio=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot of Points')
    plt.show()

    if(i==4):
        break
    cosine_distance = np.dot(lidar_normal.T, transformed_lidar_normal) / (np.linalg.norm(lidar_normal) * np.linalg.norm(transformed_lidar_normal))
    err_list.append(cosine_distance)
    img_list.append(parameter_list[i].split("\\")[-1])

mean_value = np.mean(err_list)
std_deviation = np.std(err_list)
print("Mean Value: ", mean_value)
print("Std Deviation: ", std_deviation)

plt.bar(img_list, err_list)
plt.xticks(rotation=90)
plt.xlabel('Image')
plt.ylabel('Error')
plt.title('Reprojection Error Map')
plt.show()