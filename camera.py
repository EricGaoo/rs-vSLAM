import pyrealsense2 as rs
import numpy as np
import cv2

class Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(config)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        laser_power_range = depth_sensor.get_option_range(rs.option.laser_power)
        depth_sensor.set_option(rs.option.laser_power, laser_power_range.max)

        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

    def apply_rs_filters(self, frame):
        frame = self.spatial.process(frame)
        frame = self.temporal.process(frame)
        return frame

    def get_intrinsics(self):
        depth_profile = self.profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        
        rgb_profile = self.profile.get_stream(rs.stream.color)
        rgb_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        return rgb_intrinsics, depth_intrinsics
    
    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        depth_frame = self.apply_rs_filters(depth_frame)
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        show_rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_normalized = cv2.normalize(
            depth_image, 
            None, 
            0, 
            255, 
            cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U
        )
        show_depth_image = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        return show_rgb_image, show_depth_image, color_image, depth_image
