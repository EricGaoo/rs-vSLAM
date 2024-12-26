import numpy as np
import pyrealsense2 as rs
from BA import BundleAdjustment 
import cv2

class FeatureSLAM:
    def __init__(self, rgb_intrinsics, depth_intrinsics):
        self.rgb_K = np.array([
            [rgb_intrinsics.fx, 0, rgb_intrinsics.ppx],
            [0, rgb_intrinsics.fy, rgb_intrinsics.ppy],
            [0, 0, 1]
        ])
        self.rgb_Kinv = np.linalg.inv(self.rgb_K)
        self.depth_intrinsics = depth_intrinsics
        self.depth_K = np.array([
            [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
            [0, depth_intrinsics.fy, depth_intrinsics.ppy],
            [0, 0, 1]
        ])
        self.depth_Kinv = np.linalg.inv(self.depth_K)
        self.orb = cv2.ORB_create(nfeatures = 5000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        self.current_pose = np.eye(4)
        self.keyframe_poses = []
        self.prev_pt = None

        self.bundle_adjustment = BundleAdjustment(rgb_intrinsics)

        self.EP_CON_THRESH = 1
        self.REPROJ_THRESH = 10

    def filter_match_distances(self, matches, max_dis=50, min_dis=5):
        return [
            (pt1, pt2) for pt1, pt2 in matches 
            if (min_dis < np.linalg.norm(np.array(pt1) - np.array(pt2)) < max_dis)
        ]

    def get_matches(self, rgb_frame):
        features = cv2.goodFeaturesToTrack(np.mean(rgb_frame, axis=2).astype(np.uint8), 5000, qualityLevel=0.008, minDistance=9)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
        kps, des = self.orb.compute(rgb_frame, kps)

        matches = []
        if self.prev_pt is not None:
            knn_matches = self.bf.knnMatch(des, self.prev_pt['des'], k=2)
            for m, n in knn_matches:
                if m.distance < 0.7 * n.distance:
                    kp1 = self.prev_pt['kps'][m.trainIdx].pt  # From previous frame
                    kp2 = kps[m.queryIdx].pt                  # From current frame
                    matches.append((kp1, kp2))
        matches = np.array(matches)
        matches = self.filter_match_distances(matches)
        self.prev_pt = {'kps': kps, 'des': des}
        return matches

    def normalize_points(self, pts):
        pts_normalized = (pts - np.array([self.rgb_K[0, 2], self.rgb_K[1, 2]])) / np.array([self.rgb_K[0, 0], self.rgb_K[1, 1]])
        return pts_normalized

    def epipolar_constraint(self, pts1, pts2, matches, E):
        lines1 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, E)
        lines2 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 1, E)

        lines1 = lines1.reshape(-1, 3)  # Epipolar lines corresponding to points in image 1
        lines2 = lines2.reshape(-1, 3)  # Epipolar lines corresponding to points in image 2

        errors_epipolar1 = np.abs(lines1[:, 0] * pts1[:, 0] + lines1[:, 1] * pts1[:, 1] + lines1[:, 2]) / np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)
        errors_epipolar2 = np.abs(lines2[:, 0] * pts2[:, 0] + lines2[:, 1] * pts2[:, 1] + lines2[:, 2]) / np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)

        valid_idx = np.where((errors_epipolar1 < self.EP_CON_THRESH) & (errors_epipolar2 < self.EP_CON_THRESH))[0]
        pts1 = pts1[valid_idx]
        pts2 = pts2[valid_idx]
        matches = [matches[i] for i in valid_idx]  # Apply the valid index filter to the matches list

        return pts1, pts2, matches

    def get_reprojection_error(self, R, t, pts1, pts2):
        points_4d_hom = cv2.triangulatePoints(
            np.eye(3, 4), np.hstack((R, t)), pts1.T, pts2.T
        )
        points_3d = points_4d_hom[:3] / points_4d_hom[3]

        proj_pts1, _ = cv2.projectPoints(points_3d.T, np.zeros((3, 1)), np.zeros((3, 1)), self.rgb_K, None)
        proj_pts2, _ = cv2.projectPoints(points_3d.T, R, t, self.rgb_K, None)

        errors1 = np.linalg.norm(proj_pts1.squeeze() - pts1, axis=1)
        errors2 = np.linalg.norm(proj_pts2.squeeze() - pts2, axis=1)
        mean_error = (errors1.mean() + errors2.mean()) / 2
        return mean_error, points_3d
    
    def get_poses(self, matches):
        if len(matches) < 8:
            print("Not enough Matches")
            return

        pts1 = np.array([pt1 for pt1, pt2 in matches])
        pts2 = np.array([pt2 for pt1, pt2 in matches])

        pts1 = self.normalize_points(pts1)
        pts2 = self.normalize_points(pts2)

        E, mask = cv2.findEssentialMat(
        pts1, pts2, self.rgb_K, method=cv2.RANSAC, prob=0.999, threshold=1.0, maxIters=1000
        )

        if E is None:
            print("Essential Matrix computation failed.")
            return None

        inlier_matches = [matches[i] for i in range(len(matches)) if mask.ravel()[i] == 1]
        pts1 = np.array([pt1 for pt1, pt2 in inlier_matches])
        pts2 = np.array([pt2 for pt1, pt2 in inlier_matches])

        pts1, pts2, matches = self.epipolar_constraint(pts1, pts2, matches, E)

        if len(pts1) < 5 or len(pts2) < 5:
            print("Not enough points for pose recovery.")
            return

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.rgb_K)

        mean_error, proj_3d= self.get_reprojection_error(R, t, pts1, pts2)

        if mean_error: # < self.REPROJ_ERROR
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t.squeeze()
            print("Original R:", R)
            print("Original t:", t)
            self.current_pose = self.current_pose @ np.linalg.inv(pose)
            
            # Add the pose with ID 1
            self.bundle_adjustment.add_pose(1, np.array(R), np.array(t.squeeze()))

            # Add points
            for i in range(len(proj_3d)):
                pt_3d = proj_3d[:, i]
                self.bundle_adjustment.add_point(i+2, pt_3d)
            
            # Optimize
            self.bundle_adjustment.optimize(10)

            # Retrieve the optimized pose, not points
            optimized_pose = self.bundle_adjustment.get_pose_matrix(1)
            print("Optimized Pose:", optimized_pose)
            
            # You might want to update current_pose with the optimized pose
            self.current_pose = self.current_pose @ np.linalg.inv(optimized_pose)
            self.keyframe_poses.append(self.current_pose)

        else:
            print(f"Reprojection error too high: {mean_error}")
            return None

    def process_frame(self, rgb_frame):
        matches = self.get_matches(rgb_frame)
        self.get_poses(matches)
        if matches is not None:
            for pt1, pt2 in matches:
                u1, v1 = map(lambda x: int(round(x)), pt1)
                u2, v2 = map(lambda x: int(round(x)), pt2)
                cv2.circle(rgb_frame, (u1, v1), 5, (0, 255, 0), 1)
                cv2.line(rgb_frame, (u1, v1), (u2, v2), color=(0, 0, 255), thickness=1)

        return rgb_frame, self.keyframe_poses, matches