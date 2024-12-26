from camera import Camera
from feature_track import FeatureSLAM
from point_cloud import Points
import numpy as np
import cv2

cam = Camera()
rgb_intrinsics, depth_intrinsics = cam.get_intrinsics()
slam = FeatureSLAM(rgb_intrinsics, depth_intrinsics)
pc = Points(depth_intrinsics)

while(1):
    show_rgb_img, show_depth_img, color_img, depth_img = cam.get_frames()

    # Process the RGB frame to track features
    rgb_img, pose_list, matches = slam.process_frame(color_img)

    # Combine RGB and Depth images for display
    combined_img = np.hstack((rgb_img, show_depth_img))

    # If matches are found, render 3D points
    if matches is not None and len(matches) > 0:
        points_3d = pc.get_3d_points(matches, depth_img)
        pc.view(points_3d, pose_list)

    # Display the combined image
    cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
    cv2.imshow('RealSense', combined_img)
    cv2.waitKey(1)
