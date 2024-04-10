#!/usr/bin/env python3

import numpy as np
# import pyrealsense2 as rs
import cv2
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import ros_numpy
import cv_bridge
from sensor_msgs.msg import PointField
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
# from grounding_sam import GroundingSam
import std_msgs
import threading
import torch
import transformations
from scipy.spatial.transform import Rotation as R

import std_msgs.msg
from src.utils import GoalAdjuster

class PixelToCoord:

    def __init__(self):
        #initialize ros node
        rospy.init_node('pixel_to_coord', anonymous=True)
        self.depth_image = None
        # self.sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw/compressed', CompressedImage, self.depth_callback)
        self.sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.sub1 = rospy.Subscriber('/Odometry', Odometry, self.odom_callback)
        self.sub2 = rospy.Subscriber('/llm_object', std_msgs.msg.String, self.llm_callback)
        self.pub = rospy.Publisher('object_detected', PointCloud2, queue_size=10)
        self.pub_single = rospy.Publisher('single_point', PointStamped, queue_size=10)
        self.pub_pose = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=10)
        self.goal_sub = rospy.Subscriber("/adjusted_goal", PoseStamped, self.adjust_goal_callback)
        self.pixel_mask = None
        self.current_object = None
        self.current_goal = None
        self.intrinsics = {
        'width': 1280,  # Width of the image
        'height': 720,  # Height of the image
        'ppx': 631.8179931640625,  # Principal point in x
        'ppy': 375.0325622558594,  # Principal point in y
        'fx': 634.3491821289062,  # Focal length in x
        'fy': 632.8595581054688   # Focal length in y
        }  
        self.Ry90 = np.asarray([[ 0,  0,  1, 0.0],
                                [ 0,  1,  0, 0.0],
                                [-1,  0,  0, 0.0],
                                [ 0,  0,  0, 1.0]])
        self.Rx90 = np.asarray([[ 1,  0,  0, 0.0],
                                [ 0,  0, 1, 0.0],
                                [ 0,  -1,  0, 0.0],
                                [ 0,  0,  0, 1.0]])
        self.Rz90 = np.asarray([[ 0,  1,  0, 0.0],
                                [-1,  0,  0, 0.0],
                                [ 0,  0,  1, 0.0],
                                [ 0,  0,  0, 1.0]])
        self.Rx270 = np.asarray([[ 1,  0,  0, 0.0],
                                [ 0,  -1,  0, 0.0],
                                [ 0,  0,  -1, 0.0],
                                [ 0,  0,  0, 1.0]])
        self.cam_to_lidar = np.asarray([[  0,   0,   1, 0.0],
                                        [ -1,   0,   0, 0.0],
                                        [  0,  -1,   0, 0.0],
                                        [  0,   0,   0, 1.0]])
        self.odom_data = Odometry()

        self.goal_adjuster = GoalAdjuster()


    #subscriber to the depth image
    def depth_callback(self, data):
        self.depth_image = ros_numpy.numpify(data)        

    #subscriber to the odometry
    def odom_callback(self, data):
        self.odom_data = data

    def llm_callback(self, data):
        self.current_object = data.data

    def adjust_goal_callback(self, data):
        self.current_goal = data

    def odometry_to_transformation_matrix(self):
        # Extract the position and orientation from the odometry message
        position = self.odom_data.pose.pose.position
        orientation = self.odom_data.pose.pose.orientation
        # Create a transformation matrix from the position and orientation
        transformation = transformations.quaternion_matrix([orientation.w, orientation.x, orientation.y, orientation.z])
        transformation[:3, 3] = [position.x, position.y, position.z]
        # print('Transformation matrix:', transformation)
        return transformation
        
    # Function to convert depth and pixel location to 3D point
    def depth_to_3d(self, depth, row, col, intrinsics):
        # Convert from pixel position and depth to camera-centric coordinates
        x = (col - intrinsics['ppx']) / intrinsics['fx'] * depth
        y = (row - intrinsics['ppy']) / intrinsics['fy'] * depth
        z = depth
        return [x, y, z]


    def extract_3d_points(self, depth_image, intrinsics):
        # Read the pixel mask
        pixel_mask = cv2.imread('/home/sebastian/catkin_ws/src/octopub/src/outputs/mask.png', cv2.IMREAD_UNCHANGED)

        # Find unique mask values (excluding 0 if it's the background)
        unique_masks = np.unique(pixel_mask)
        unique_masks = unique_masks[unique_masks != 0]  # Remove background if 0 is considered background
        # Prepare a list to hold points for each mask
        points_3d_per_mask = []

        # Iterate over each unique mask
        for mask_value in unique_masks:
            # Initialize a list to hold 3D points for the current mask
            points_3d = []
            # Find rows and columns where the current mask is located
            rows, cols = np.where(pixel_mask == mask_value)

            for row, col in zip(rows, cols):
                depth = depth_image[row, col] / 1000  # Convert from mm to meters
                if depth > 0:  # Check if depth is valid
                    # Convert depth to 3D point
                    point_3d = self.depth_to_3d(depth, row, col, intrinsics)
                    points_3d.append(point_3d)

            # Add the list of 3D points for the current mask to the overall list
            points_3d_per_mask.append(points_3d)

        # Convert the list of lists into a 3D numpy array if necessary
        # This part depends on how you want to structure your 3D array;
        # here's a simple approach to create a list of numpy arrays:
        points_3d_per_mask = [np.array(mask_points) for mask_points in points_3d_per_mask]

        return points_3d_per_mask

    
    #function to publish the 3D points as a pointcloud2 message
    def publish_pointcloud2(self, points_3d_array):
        # Create a PointCloud2 message
        msg = PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'camera_init'
        msg.height = 1
        msg.width = points_3d_array.shape[0]
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = np.asarray(points_3d_array, dtype=np.float32).tostring()  # Convert (N, 3) directly to bytes
        self.pub.publish(msg)

    #function to publish the single 3D point as a geometry_msgs/PointStamped message
    def publish_single_point(self, single_point):
        one_point = PointStamped()
        one_point.header.stamp = rospy.Time.now()
        one_point.header.frame_id = 'camera_init'
        one_point.point.x = single_point[0]
        one_point.point.y = single_point[1]
        one_point.point.z = 0.0 # single_point[2]
        self.pub_single.publish(one_point)

    def send_nav_goal(self, point, quat):
        print(point)
        print(quat)
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "camera_init"
        goal.pose.position.x = point[0]
        goal.pose.position.y = point[1]
        goal.pose.position.z = point[2]
        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]   
        goal.pose.orientation.w = quat[3]
        new_goal = self.goal_adjuster.adjust_goal(goal)
        self.pub_pose.publish(new_goal)
    

    def turn_90_degrees_z(self):
        # Your current quaternion [w, x, y, z]
        current_quat = [self.odom_data.pose.pose.orientation.w, 
                        self.odom_data.pose.pose.orientation.x, 
                        self.odom_data.pose.pose.orientation.y, 
                        self.odom_data.pose.pose.orientation.z]

        # current_quat = [0.0, 0.0, 0.0, 1.0]

        # Create a rotation object from the current quaternion
        current_rotation = R.from_quat(current_quat)

        # Define a rotation of 90 degrees around the z-axis
        rotation_90_degrees_z = R.from_euler('z', 90, degrees=True)

        # Apply this rotation to the current rotation
        new_rotation = current_rotation * rotation_90_degrees_z

        # Convert the new rotation back to a quaternion [w, x, y, z]
        new_quat = new_rotation.as_quat()  # Returns [x, y, z, w] for SciPy

        new_quat = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

        current_xyz = [self.odom_data.pose.pose.position.x,
                       self.odom_data.pose.pose.position.y, 
                       self.odom_data.pose.pose.position.z]

        # Return the new orientation as a quaternion
        return new_quat, current_xyz

    def convert_to_coords(self, transformation):
        
        points_3d = self.extract_3d_points(self.depth_image, self.intrinsics)
        points = []
        for i in range(len(points_3d)):
            #convert points to homogenous coordinates
            # temp_points = np.hstack((points_3d[i], np.ones((points_3d[i].shape[0], 1))))
            points_3d[i] = np.dot(self.cam_to_lidar[:3, :3], points_3d[i].T).T + self.cam_to_lidar[:3, 3]
            points_3d[i] = np.dot(transformation[:3,:3], points_3d[i].T).T + transformation[:3, 3]
            points_mean = np.mean(points_3d[i], axis=0)
            points.append(points_mean)
        #collapse the list of arrays into a single array
        all_points = np.concatenate(points_3d, axis=0)        

        return all_points, points_3d, points