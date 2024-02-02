#!/usr/bin/env python3 
import rospy
import tf2_ros
import tf_conversions
import tf
from tf2_ros import Buffer, TransformListener
import geometry_msgs.msg
import nav_msgs.msg
import math


class Tf2Broadcaster:
    def __init__(self):
        # rospy.Subscriber("/move_base/GlobalPlanner/plan", nav_msgs.msg.Path, self.plan_callback)
        rospy.Subscriber("/cmd_vel", geometry_msgs.msg.Twist, self.vel_callback)
        self.br = tf2_ros.TransformBroadcaster()
        self.received_data = geometry_msgs.msg.TransformStamped()
        self.received_data.header.stamp = rospy.Time.now()
        self.received_data.header.frame_id = "/map"
        self.received_data.child_frame_id = "/base_link"
        self.received_data.transform.translation.x = 0.0
        self.received_data.transform.translation.y = 0.0
        self.received_data.transform.translation.z = 0.0
        self.received_data.transform.rotation.x = 0.0
        self.received_data.transform.rotation.y = 0.0
        self.received_data.transform.rotation.z = 0.0
        self.received_data.transform.rotation.w = 1.0

        self.vel_data = geometry_msgs.msg.Twist()
        self.vel_data.linear.x = 0.0
        self.vel_data.linear.y = 0.0
        self.vel_data.linear.z = 0.0
        self.vel_data.angular.x = 0.0
        self.vel_data.angular.y = 0.0
        self.vel_data.angular.z = 0.0
        self.timestamp = rospy.Time.now()
        self.rotation = 0.0
        self.count = 1
    
    # def plan_callback(self, msg):
    #     self.received_data.header.stamp = rospy.Time.now()
    #     self.received_data.header.frame_id = "/map"
    #     self.received_data.child_frame_id = "/base_link"
    #     self.received_data.transform.translation.x = msg.poses[10].pose.position.x
    #     self.received_data.transform.translation.y = msg.poses[10].pose.position.y
    #     self.received_data.transform.translation.z = 0.0
        # self.received_data.transform.rotation.x = msg.poses[10].pose.orientation.x
        # self.received_data.transform.rotation.y = msg.poses[10].pose.orientation.y
        # self.received_data.transform.rotation.z = msg.poses[10].pose.orientation.z
        # self.received_data.transform.rotation.w = msg.poses[10].pose.orientation.w
        
    def vel_callback(self, msg):
        rospy.loginfo("vel_callback number: %s", self.count)
        self.timestamp = rospy.Time.now()
        self.vel_data.linear.x = msg.linear.x
        self.vel_data.linear.y = msg.linear.y
        self.vel_data.linear.z = msg.linear.z
        self.vel_data.angular.x = msg.angular.x
        self.vel_data.angular.y = msg.angular.y
        self.vel_data.angular.z = msg.angular.z
        self.count += 1

    def transform_linear_velocity(self, vx, vy, theta):
        Vx = math.cos(theta) * vx - math.sin(theta) * vy
        Vy = math.sin(theta) * vx + math.cos(theta) * vy
        return Vx, Vy
    
    def get_pose(self, old_vel, old_time, new_time):
        curent_quaternion = (
            self.received_data.transform.rotation.x,
            self.received_data.transform.rotation.y,
            self.received_data.transform.rotation.z,
            self.received_data.transform.rotation.w
        )
        
        dt = new_time - old_time
        # rospy.loginfo("dt: %s", dt.to_sec())
        theta = tf.transformations.euler_from_quaternion(curent_quaternion)
        angle = (theta[2] + old_vel.angular.z*dt.to_sec())
        Vx, Vy = self.transform_linear_velocity(old_vel.linear.x, old_vel.linear.y, angle)
        self.received_data.transform.translation.x += Vx * dt.to_sec()
        self.received_data.transform.translation.y += Vy * dt.to_sec()
        self.received_data.transform.translation.z += old_vel.linear.z * dt.to_sec()

        quaternion_change = tf.transformations.quaternion_about_axis(old_vel.angular.z * dt.to_sec(), (0, 0, 1))

        new_quaternion = tf.transformations.quaternion_multiply(curent_quaternion, quaternion_change)

        self.received_data.transform.rotation.x = new_quaternion[0]
        self.received_data.transform.rotation.y = new_quaternion[1]
        self.received_data.transform.rotation.z = new_quaternion[2]
        self.received_data.transform.rotation.w = new_quaternion[3]

if __name__ == '__main__':
    rospy.init_node('tf2_broadcaster')
    tf2_broadcaster = Tf2Broadcaster()

    while not rospy.is_shutdown():
        tf2_broadcaster.get_pose(tf2_broadcaster.vel_data, tf2_broadcaster.timestamp, rospy.Time.now())

        tf2_broadcaster.received_data.header.stamp = rospy.Time.now()
        tf2_broadcaster.br.sendTransform(tf2_broadcaster.received_data)
        rospy.loginfo("tf2_broadcaster")

        rospy.sleep(0.05)