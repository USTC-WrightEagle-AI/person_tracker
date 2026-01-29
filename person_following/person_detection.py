#person detection &  person pose publishing & rotating robot to find person

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import open3d as o3d

import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import PointStamped, TransformStamped, Twist
from std_msgs.msg import Header
from tf.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_multiply

# Global controller class for robot motion control
class RobotController:
    def __init__(self):
        # ROS publisher for robot velocity control
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.is_searching = False
        self.last_person_x = None
        self.last_seen_side = None  # "left", "right", "center"
        self.image_center_x = 320  # 640 / 2
        self.search_direction = None
        self.search_start_time = None
        self.max_search_time = 5.0  # Maximum search time (seconds)
        self.frame_center_x = 320  # Image center x-coordinate (640 / 2)
        
    def rotate_to_search(self):
        """Rotate the robot to search for a person"""
        if not self.is_searching:
            self.is_searching = True
            self.search_start_time = time.time()
            print("Starting rotation to search for a person...")
        
        twist = Twist()
        if self.last_person_x is not None:
            if self.last_person_x < self.frame_center_x - 50:
                twist.angular.z = 0.3
                print("Rotating left to search...")
            else:
                twist.angular.z = -0.3
                print("Rotating right to search...")
        else:
            twist.angular.z = 0.3
            print("No prior detection, rotating left by default...")
        
        self.vel_pub.publish(twist)
        
        if time.time() - self.search_start_time > self.max_search_time:
            print(f"Search timeout ({self.max_search_time}s), stopping rotation")
            self.stop_rotation()
            return False
        return True
    
    def stop_rotation(self):
        """Stop robot rotation"""
        twist = Twist()
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)
        if self.is_searching:
            print("Rotation stopped")
            self.is_searching = False
    
    def reset_search(self):
        """Reset search state"""
        self.is_searching = False
        self.search_start_time = None


def initialize_realsense():
    """Initialize RealSense camera"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Startup failed: {e}")
        config.disable_all_streams()
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)
        profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    
    return pipeline, align, depth_scale, intrinsics


def get_median_depth_in_roi(depth_frame, depth_scale, x1, y1, x2, y2):
    """
    Compute the median valid depth value inside a bounding box region.
    Returns depth in meters, or None if no valid depth is found.
    """
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width - 1, int(x2))
    y2 = min(height - 1, int(y2))
    
    depth_data = np.asanyarray(depth_frame.get_data())
    roi = depth_data[y1:y2, x1:x2]
    
    roi_meters = roi.astype(float) * depth_scale
    valid_depths = roi_meters[roi_meters > 0.1]
    
    if len(valid_depths) == 0:
        return None
    
    return np.median(valid_depths)


def get_3d_coordinates(depth_frame, depth_scale, intrinsics, pixel_x, pixel_y, depth_value=None):
    """
    Convert 2D pixel coordinates into 3D world coordinates (meters).
    """
    if (pixel_x < 0 or pixel_y < 0 or 
        pixel_x >= intrinsics.width or 
        pixel_y >= intrinsics.height):
        return None
    
    try:
        if depth_value is None:
            depth = depth_frame.get_distance(int(pixel_x), int(pixel_y))
        else:
            depth = depth_value
            
        if depth <= 0:
            return None
        
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth)
        return point
    except RuntimeError:
        return None


def depth_to_points(depth, intrinsic):
    """Convert depth image into 3D point cloud"""
    K = intrinsic
    Kinv = np.linalg.inv(K)
    height, width = depth.shape
    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)
    coord = coord.astype(np.float32)
    coord = coord[None]
    D = depth[:, :, None, None]
    pts3D = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    return pts3D[0, :, :, :3, 0]


def get_body_center_from_keypoints(keypoints):
    """
    Estimate a more accurate body center from human keypoints.
    Returns (x, y) or None.
    """
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    
    valid_points = []
    
    if keypoints[LEFT_SHOULDER][2] > 0.1 and keypoints[RIGHT_SHOULDER][2] > 0.1:
        shoulder_center = (
            (keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2,
            (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2
        )
        valid_points.append(shoulder_center)
    
    if keypoints[LEFT_HIP][2] > 0.1 and keypoints[RIGHT_HIP][2] > 0.1:
        hip_center = (
            (keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0]) / 2,
            (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2
        )
        valid_points.append(hip_center)
    
    if not valid_points:
        return None
    
    center_x = sum(p[0] for p in valid_points) / len(valid_points)
    center_y = sum(p[1] for p in valid_points) / len(valid_points)
    
    return (center_x, center_y)


def transform_point_with_matrix(point, transform_matrix):
    """Apply a 4x4 homogeneous transformation matrix to a 3D point"""
    point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
    transformed_point = np.dot(transform_matrix, point_homogeneous)
    return transformed_point[:3]


def transform_point_to_base_link(point_camera, transformation_matrix):
    """
    Transform a point from camera frame to base_link frame.
    """
    try:
        rospy.init_node('camera_to_base_transform', anonymous=True)
    except:
        pass
    
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    try:
        # First compute the person position in left_arm_base_link frame,
        # then compensate with an offset to approximate base_link coordinates.
        
        tf_buffer.can_transform("left_arm_base_link", "left_gripper_link", rospy.Time.now(), rospy.Duration(1.0))
        gripper_to_base_transform = tf_buffer.lookup_transform("left_arm_base_link", "left_gripper_link", rospy.Time(0))
        
        translation = np.array([
            gripper_to_base_transform.transform.translation.x,
            gripper_to_base_transform.transform.translation.y,
            gripper_to_base_transform.transform.translation.z
        ])
        
        rotation = np.array([
            gripper_to_base_transform.transform.rotation.x,
            gripper_to_base_transform.transform.rotation.y,
            gripper_to_base_transform.transform.rotation.z,
            gripper_to_base_transform.transform.rotation.w
        ])
        
        rotation_matrix = quaternion_matrix(rotation)
        gripper_to_base_matrix = np.identity(4)
        gripper_to_base_matrix[:3, :3] = rotation_matrix[:3, :3]
        gripper_to_base_matrix[:3, 3] = translation

        combined_matrix = np.dot(gripper_to_base_matrix, transformation_matrix)
        point_base = transform_point_with_matrix(point_camera, combined_matrix)
        
        # Manual offset correction (temporary workaround)
        point_base[1] += 0.3

        return point_base
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
            tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
        rospy.logerr("TF transformation failed: %s", e)
        return None
