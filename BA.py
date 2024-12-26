import numpy as np
import g2o

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, intrinsics):
        super().__init__()
        self.cam = intrinsics
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, R, t, fixed=False):
        R = np.array(R, dtype=np.float64)
        t = np.array(t)

        quart = g2o.Quaternion(R)

        sbacam = g2o.SBACam(quart, t)
        sbacam.set_cam(self.cam.fx, self.cam.fy, self.cam.ppx, self.cam.ppy, 0)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id)   # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3) 

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexPointXYZ()
        v_p.set_id(point_id)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id))
        edge.set_vertex(1, self.vertex(pose_id))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id).estimate()
    
    def get_pose_matrix(self, pose_id):
        pose_est = self.get_pose(pose_id)
        print(pose_est)
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = pose_est.rotation().matrix()
        pose_matrix[:3, 3] = pose_est.translation()
        return pose_matrix
