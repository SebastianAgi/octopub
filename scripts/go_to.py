import numpy as np
import rospy

import sys
sys.path.append('/home/sebastian/catkin_ws/src/octopub')

from src.grounding_sam import GroundingSam
from src.pixel_to_coord import PixelToCoord


def main():
    print("Starting the main function")
    ptc = PixelToCoord()
    print("PixelToCoord object created")
    gs = GroundingSam()
    print("GroundingSam object created")

    count = 0

    #initialize an empty array for accumulating points
    accumulated_points = np.empty((0,3), dtype=float)  # Assuming points are 3D, adjust the shape as necessary
    print("waiting for object")
    while not rospy.is_shutdown():

        #conditional for specifying an objec to localize (only activate when enter is pressed)
        if count == 0:
            if ptc.current_object != None:
                input_str = ptc.current_object
            else: 
                continue
            # input_str = input("Enter the object to localize: ")
            if input_str == '':
                break
            print("the object to go to: ", input_str)
        while gs.realsense_image is None:
            pass
        transformation = ptc.odometry_to_transformation_matrix()

        try:
            masks = gs.main(str(input_str))

        except:
            print(input_str,'not detected')
            if count == 4:
                count = 0
                print('Object not found')
                continue
            print('Will look around for the object')
            quat, xyz = ptc.turn_90_degrees_z()
            print(quat, xyz)
            ptc.send_nav_goal(xyz, quat)
            count += 1
            continue

        all_points, point_3d, points = ptc.convert_to_coords( transformation)
        accumulated_points = np.vstack((accumulated_points, all_points))
        ptc.publish_pointcloud2(accumulated_points)
        for i in range(len(points)):
            ptc.publish_single_point(points[i])
        print('Published point cloud')
        ptc.publish_single_point(points[0])
        ptc.send_nav_goal(points[0], [0.0, 0.0, 0.0, 1.0])
        
        ptc.current_object = None

if __name__ == "__main__":
    main()