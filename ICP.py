import open3d as o3d
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R

# Hypothetical rotations around x, y, z axes in degrees
# roll_estimate = 90  # replace with your estimate
# pitch_estimate = 0  # replace with your estimate
# yaw_estimate = 90  # replace with your estimate if it appears to be a 90-degree rotation

# # Convert to radians
# roll_rad = np.deg2rad(roll_estimate)
# pitch_rad = np.deg2rad(pitch_estimate)
# yaw_rad = np.deg2rad(yaw_estimate)

# # Create a rotation object using scipy
# r = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad])

# # Convert to a 3x3 rotation matrix
# rotation_matrix = r.as_matrix()

# # Hypothetical translation estimates in x, y, z
# translation_estimate = [0.05, 0, -0.1]  # replace with your estimates

# # Construct the 4x4 transformation matrix
# trans_init1 = np.eye(4)  # Start with an identity matrix
# trans_init1[:3, :3] = rotation_matrix  # Insert the rotation matrix
# trans_init1[:3, 3] = translation_estimate  # Insert the translation vector

# print(trans_init1)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
                                    #   zoom=0.4459,
                                    #   front=[0.9288, -0.2951, -0.2242],
                                    #   lookat=[1.6784, 2.0612, 1.4451],
                                    #   up=[-0.3402, -0.9189, -0.1996])
print('test')
demo_icp_pcds = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
rs_pc = o3d.io.read_point_cloud("/home/sebastianae/Desktop/calibration_data/ouster/third_attempt/filtered_points.pcd")
source = rs_pc
target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
ouster_pc = o3d.io.read_point_cloud("/home/sebastianae/Desktop/calibration_data/depth_camera/third_attempt/filtered_points.pcd")
target = ouster_pc
# threshold = 0.001

# P^(depthcamera)
# P^(ouster)
# T^(depthcamera)_{ouster}
# P^(depthcaemra) = T^(depthcamera}_{ouster} P^(ouster) -> /aligned_cloud

# 
trans_init = np.asarray([[ 0,-1,0,  0.04039775],
                         [ 0,0,-1,  0.05563903],
                         [ 1,0,0,  0.07162135],
                         [ 0.0,         0.0,         0.0,         1.0    ]])

# trans_init = np.asarray([[ 0.01998463,  0.00150283,  0.99979911,  0.07162135],
#                          [-0.99939423,  0.00132105,  0.01998485, -0.04039775],
#                          [-0.00129142, -0.999998,    0.00152892, -0.05563903],
#                          [ 0.0,         0.0,         0.0,         1.0    ]])
print(trans_init)
draw_registration_result(source, target, trans_init)

print("Initial alignment")
threshold = 0.5
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
print(evaluation)

print("Apply point-to-point ICP")#
threshold = 0.5
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)

threshold = 0.1
trans_init = reg_p2p.transformation
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)

threshold = 0.05
trans_init = reg_p2p.transformation
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)


# ############## Rotation ################

# #transform with the values of [[ 0.99939423 -0.00132105 -0.01998485  0.04039775] [ 0.00129142  0.999998   -0.00152892  0.05563903] [ 0.01998463  0.00150283  0.99979911  0.07162135] [ 0.          0.          0.          1.        ]]

# transform = np.asarray([[ 0.99939423, -0.00132105, -0.01998485,  0.04039775],
#                        [ 0.00129142,  0.999998,   -0.00152892,  0.05563903],
#                        [ 0.01998463,  0.00150283,  0.99979911,  0.07162135],
#                        [ 0.0,         0.0,         0.0,         1.0    ]])


# transform = np.asarray([[-0.00129142, -0.999998,    0.00152892, -0.05563903],
#                         [ 0.99939423, -0.00132105, -0.01998485,  0.04039775],
#                         [ 0.01998463,  0.00150283,  0.99979911,  0.07162135],
#                         [ 0.0,         0.0,         0.0,         1.0    ]])

# Rz_90 = np.asarray([[0, -1, 0, 0],
#                     [1, 0, 0, 0],
#                     [0, 0, 1, 0],
#                     [0, 0, 0, 1]])

# Rx_90 = np.asarray([[1, 0, 0, 0],
#                     [0, 0, -1, 0],
#                     [0, 1, 0, 0],
#                     [0, 0, 0, 1]])

# Ry_90 = np.asarray([[0, 0, 1, 0],
#                     [0, 1, 0, 0],
#                     [-1, 0, 0, 0],
#                     [0, 0, 0, 1]])

# Rz_180 = np.asarray([[-1, 0, 0, 0],
#                     [0, -1, 0, 0],
#                     [0, 0, 1, 0],
#                     [0, 0, 0, 1]])

# trans_rot = np.dot(Rz_90, np.dot(Rz_180 , transform))

# print('transformation rotated: ', trans_rot)

R = np.asarray([[0, 0, -1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1]])

