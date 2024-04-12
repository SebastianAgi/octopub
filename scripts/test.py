#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import random

if __name__ == "__main__":

    print("Waiting to initialize node")
    #print available topics
    rospy.init_node('test_node')
    print("Node initialized", flush=True)

    pub = rospy.Publisher("test_topic", String, queue_size=10)

    import time

    while not rospy.is_shutdown():
        # print("Hello World", flush=True)
        pub.publish(f"Seb does big poos in the toilet {random.randint(0, 100)} times per day")
        time.sleep(0.01)