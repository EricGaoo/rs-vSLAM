import numpy as np
import pyrealsense2 as rs
import pangolin
import OpenGL.GL as gl
from multiprocessing import Process, Queue

class Points:
    def __init__(self, depth_intrinsics):
        self.intrinsics = depth_intrinsics
        self.window_name = "3D Point Cloud"
        self.accumulated_points = []
        self.accumulated_poses = []
        self.q = Queue()
        self.viewer = Process(target=self.display_thread, args=(self.q,))
        self.viewer.daemon = True
        self.viewer.start()

    def get_3d_points(self, matches, depth_img):
        points_3d = []

        for pt1, _ in matches:
            x, y = int(pt1[0]), int(pt1[1])
            depth = depth_img[y, x]*0.001
            if depth > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
                points_3d.append(point_3d)
        return np.array(points_3d)

    def display_thread(self, q):
        pangolin.CreateWindowAndBind(self.window_name, 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(424, 240, 420, 420, 212, 120, 0.2, 100),  # Match width/2 and height/2
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY)
        )
        handler = pangolin.Handler3D(scam)
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -424.0 / 240.0)  # Aspect ratio from 424x240
        dcam.SetHandler(handler)

        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)

            if self.accumulated_points:
                gl.glPointSize(2)
                gl.glColor3f(1.0, 0.0, 0.0)  # Red points
                pangolin.DrawPoints(np.vstack(self.accumulated_points))  # Combine all points into one array

            if self.accumulated_poses:
                for pose in self.accumulated_poses:
                    R = pose[:3, :3]
                    t = pose[:3, 3].reshape(-1, 1)
                    self.draw_pose(R, t)

            if not q.empty():
                points_data = q.get()

                if points_data is not None:
                    points_3d, poses = points_data

                    if points_3d is not None and len(points_3d) > 0:
                        self.accumulated_points.append(points_3d)  # Add new points to the list

                    if poses is not None:
                        self.accumulated_poses.extend(poses)  # Add new poses to the list

            pangolin.FinishFrame()

    def draw_pose(self, R, t):
        
        gl.glLineWidth(2)
        gl.glColor3f(0.0, 0.0, 1.0)  # Blue rectangle

        width, height = 0.2, 0.1  # Dimensions of the rectangle
        rectangle = np.array([
            [0, 0, 0],  # Bottom-left
            [width, 0, 0],  # Bottom-right
            [width, height, 0],  # Top-right
            [0, height, 0],  # Top-left
            [0, 0, 0]  # Close the rectangle
        ]).T

        transformed_rectangle = (R @ rectangle + t.reshape(3, 1)).T
        for i in range(len(transformed_rectangle) - 1):
            line_points = np.array([transformed_rectangle[i], transformed_rectangle[i + 1]])
            pangolin.DrawLine(line_points)

    #points_3d comes from get_3d_points(self, matches, depth_img)
    def view(self, points_3d, pose_list):
        if pose_list:

            if points_3d is None or points_3d.size == 0:
                print("No points to transform.")
                return
            # Assume the pose corresponds to the current points_3d
            pose = pose_list[-1]  # Last pose (4x4 matrix)
            R = pose[:3, :3]      # Extract rotation matrix
            t = pose[:3, 3]       # Extract translation vector

            # Transform points from local camera frame to global frame
            transformed_points = (R @ points_3d.T).T + t  # Apply R and t to each point
            self.q.put((transformed_points, pose_list))
        else:
            self.q.put((points_3d, pose_list))

