#!/usr/bin/env python
import tf_conversions
from geometry_msgs.msg import TransformStamped
import rospy
import tf2_ros
import tf
import geometry_msgs.msg
import math
from nav_msgs.msg import Odometry


class Tf2Broadcaster:
    def __init__(self):
        # rospy.Subscriber("/move_base/GlobalPlanner/plan", nav_msgs.msg.Path, self.plan_callback)
        rospy.Subscriber("/Odometry", Odometry, self.odom_callback)
        rospy.Subscriber("/cmd_vel", geometry_msgs.msg.Twist, self.vel_callback)
        self.br = tf2_ros.TransformBroadcaster()
        self.received_data = geometry_msgs.msg.TransformStamped()
        self.received_data.header.stamp = rospy.Time.now()
        self.received_data.header.frame_id = "/camera_init"
        self.received_data.child_frame_id = "/base_link"
        self.received_data.transform.translation.x = 0.0
        self.received_data.transform.translation.y = 0.0
        self.received_data.transform.translation.z = 0.0
        self.received_data.transform.rotation.x = 0.0
        self.received_data.transform.rotation.y = 0.0
        self.received_data.transform.rotation.z = 0.0
        self.received_data.transform.rotation.w = 1.0

        self.broadcaster = tf2_ros.StaticTransformBroadcaster()


    def vel_callback(self, msg):
        print("vel_callback: ",
               msg.linear.x, "\n              ",
               msg.linear.y, "\n              ",
               msg.linear.z, "\n              ",
               msg.angular.x, "\n              ",
               msg.angular.y, "\n              ",
               msg.angular.z)

    def odom_callback(self, msg):
        self.received_data.header.stamp = rospy.Time.now()
        self.received_data.header.frame_id = "/camera_init"
        self.received_data.child_frame_id = "/base_link"
        self.received_data.transform.translation.x = msg.pose.pose.position.x
        self.received_data.transform.translation.y = msg.pose.pose.position.y
        self.received_data.transform.translation.z = msg.pose.pose.position.z
        self.received_data.transform.rotation.x = msg.pose.pose.orientation.x
        self.received_data.transform.rotation.y = msg.pose.pose.orientation.y
        self.received_data.transform.rotation.z = msg.pose.pose.orientation.z
        self.received_data.transform.rotation.w = msg.pose.pose.orientation.w

    def broadcast_static_transform(self):
        
        
        # Define the static transform
        static_transformStamped = TransformStamped()
        
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "base_link"
        static_transformStamped.child_frame_id = "camera_init"
        
        # Position of the camera_link frame relative to the body frame
        static_transformStamped.transform.translation.x = - 0.08 # Update with actual value
        static_transformStamped.transform.translation.y = 0.0 # Update with actual value
        static_transformStamped.transform.translation.z = 0.0 # Update with actual value
        
        # Orientation of the camera_link frame relative to the body frame
        # Use Euler angles to define the rotation: roll, pitch, yaw
        quat = tf_conversions.transformations.quaternion_from_euler(0.0, 0.0, 0.0) # Update with actual values (in radians)
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]
        
        # Broadcast the static transform
        self.broadcaster.sendTransform(static_transformStamped)

if __name__ == '__main__':
    rospy.init_node('tf2_broadcaster')
    tf2_broadcaster = Tf2Broadcaster()
    tf2_broadcaster.broadcast_static_transform()  # This will now work correctly
    print("Broadcasting static transform")

    while not rospy.is_shutdown():
        tf2_broadcaster.received_data.header.stamp = rospy.Time.now()
        tf2_broadcaster.br.sendTransform(tf2_broadcaster.received_data)
        # rospy.sleep(0.1)
