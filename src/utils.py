#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import numpy as np
 
class GoalAdjuster:
    def __init__(self):
        # rospy.init_node('goal_adjuster')
        self.map_sub = rospy.Subscriber("/projected_map", OccupancyGrid, self.map_callback)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.map_data = None
 
    def map_callback(self, data):
        self.map_data = data
 
    def goal_callback(self, goal):
        if self.map_data is None:
            rospy.loginfo("Map data not received yet.")
            return
        adjusted_goal = self.adjust_goal(goal)
        self.goal_pub.publish(adjusted_goal)
 
    def adjust_goal(self, goal):
        map_grid = np.array(self.map_data.data).reshape((self.map_data.info.height, self.map_data.info.width))
        x, y = self.world_to_map(goal.pose.position.x, goal.pose.position.y)
        # Check if the current goal is in a free space
        if map_grid[y, x] == 0:
            rospy.loginfo("Goal is in a free space.")
            return goal
        rospy.loginfo("Goal is in an obstacle, adjusting...")
        for radius in range(1, 20):  # Increase the search radius incrementally
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if map_grid[y + i, x + j] == 0:
                        rospy.loginfo("Adjusted goal found.")
                        return self.map_to_world(x + j*5, y + i*5, goal)
 
        rospy.loginfo("No suitable adjusted goal found. Returning original goal.")
        return goal
 
    def world_to_map(self, wx, wy):
        mx = int((wx - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        my = int((wy - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        return mx, my
 
    def map_to_world(self, mx, my, original_goal):
        wx = mx * self.map_data.info.resolution + self.map_data.info.origin.position.x
        wy = my * self.map_data.info.resolution + self.map_data.info.origin.position.y
        adjusted_goal = PoseStamped()
        adjusted_goal.header.frame_id = original_goal.header.frame_id
        adjusted_goal.pose.position.x = wx
        adjusted_goal.pose.position.y = wy
        adjusted_goal.pose.orientation = original_goal.pose.orientation
        return adjusted_goal
 
if __name__ == '__main__':
    try:
        GoalAdjuster()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass