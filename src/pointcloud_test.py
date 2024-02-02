import open3d as o3d
 
 
# Read the .pcd file
pcd = o3d.io.read_point_cloud("/home/sebastian/catkin_ws/src/FAST_LIO/PCD/scans.pcd")

# Delete points that have a z value more and less than 5 and -5 respectively
pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=(-50, -50, -5), max_bound=(50, 50, 5)))
 
# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])