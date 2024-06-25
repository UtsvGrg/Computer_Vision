import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import copy
img1 = cv.imread(r"panaroma_generation\1.jpg")
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

img2 = cv.imread(r"panaroma_generation\2.jpg")
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1,None)
key_img1 = copy.deepcopy(img1)
key_img1 = cv.drawKeypoints(gray1,kp1,key_img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(key_img1)
plt.axis('off')
plt.show()  
print(len(kp1))
print(des1.shape)
kp2, des2 = sift.detectAndCompute(gray2,None)
key_img2 = copy.deepcopy(img2)
key_img2 = cv.drawKeypoints(gray2,kp2,key_img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(key_img2)
plt.axis('off')
plt.show()
print(len(kp2))
print(des2.shape)
# The keypoints are located arounf distinctive features such as corners and edges. Also comapring the two images we can see the same points being depicted irrespective the different orientation and scale of the images. Second being more far away and with other another angle.
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good_matches = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(len(good_matches))
matched_image = cv.drawMatches(img1, kp1, img2, kp2, good_matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(14, 6))
plt.imshow(matched_image)
plt.axis('off')
plt.show()
# They are total 243 good matches found by brute force algorithm. I have depicted 50 of them in the above image.
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(len(good_matches))
matched_image = cv.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(14, 6))
plt.imshow(matched_image)
plt.axis('off')
plt.show()
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

homography_matrix, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
print(homography_matrix)
height, width, _ = img2.shape
transformed_img1 = cv.warpPerspective(img1, homography_matrix, (width, height))
result = np.concatenate((transformed_img1, img1), axis=1)

plt.figure(figsize=(14, 6))
plt.imshow(result)
plt.axis('off')
plt.show()
# The image1 perspective changed to the form of image2. Look how it looks identical.
result = np.concatenate((transformed_img1, img2), axis=1)
plt.figure(figsize=(14, 6))
plt.imshow(result)
plt.axis('off')
plt.show()
# The image on the left is transformed image 1 and on the right is image2. Look how is the perspective of both the image the same.

stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
img_list = []
for i in range(1,9):
    temp = cv.imread(f"panaroma_generation\{i}.jpg")
    temp = cv.cvtColor(temp, cv.COLOR_BGR2RGB)
    img_list.append(temp)
status, panorama = stitcher.stitch(img_list)
if status == cv.Stitcher_OK:
    plt.figure(figsize=(14, 6))
    plt.imshow(panorama)
    plt.axis('off')
    plt.show()
else:
    print("Error during stitching:", status)