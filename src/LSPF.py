import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.core as o3c

#Least Squares Plane Fitting from pointcloud

# Read the .pcd file
pcd = o3d.io.read_point_cloud("/home/sebastian/catkin_ws/test_pcd.pcd")

number_of_points = np.asarray(pcd.points).shape[0]
print("number of points: ", number_of_points)

# #visualize the pointcloud
# o3d.visualization.draw_geometries([pcd])

def plane_fitting(point, n):
    #point: point in pointcloud that is chosen
    #n: number of points to be used for plane fitting
    #returns: normal vector of plane

    #From given point select the n nearest points
    [k, idx, _] = pcd_tree.search_knn_vector_3d(point, n)

    #Get the coordinates of the n nearest points
    points = np.asarray(pcd.points)[idx, :]

    #Calculate the centroid of the n nearest points
    centroid = np.mean(points, axis=0)

    #Covariance matrix of the points
    c = points - centroid
    cov = np.dot(c.T, c)

    #Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    #Get the eigenvector with the smallest eigenvalue
    sorted_indx = np.argsort(eigenvalues)
    normal = eigenvectors[:, sorted_indx[0]]

    #Calculate the distance from the centroid to the plane
    d = -np.dot(normal.T, centroid)

    #calculate the dot product between the normal vector and the z-axis
    angle = np.dot(normal.T, np.array([0, 0, 1]))

    return points, centroid, normal, d, angle


#main function
if __name__ == '__main__':
    #create a kdtree from the pointcloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pointcloud = []
    for i in range(len(pcd.points)):
        points, centroid, normal, d, angle = plane_fitting(pcd.points[i], 32)
        if angle > 0:
            continue
        pointcloud.append([pcd.points[i][0], pcd.points[i][1], pcd.points[i][2], angle])

    #visualize the pointcloud with angle as the color using patplotlib
    pointcloud = np.asarray(pointcloud)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c=pointcloud[:, 3], cmap='viridis', marker='o')
    plt.show()





    #create a kdtree from the pointcloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    #choose a point from the pointcloud
    point = pcd.points[5000]
    print("point: ", point)

    #call the plane_fitting function
    points, centroid, normal, d, angle = plane_fitting(point, 32)

    #plot all together the points, the centroid, the normal vector, and the resulting plane from the normal vector
    dist = 0.1
    fig = plt.figure()
    #set the axis range for x,y,z
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(centroid[0]-0.1, centroid[0]+dist)
    ax.set_ylim(centroid[1]-0.1, centroid[1]+dist)
    ax.set_zlim(centroid[2]-0.1, centroid[2]+dist)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    ax.scatter(centroid[0], centroid[1], centroid[2], c='r', marker='o')
    ax.quiver(centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], length=0.05, normalize=False)

    #creat a meshgrid to plot the plane withthe range centered at the centroid
    xx, yy = np.meshgrid(np.arange(centroid[0]-dist, centroid[0]+dist, 0.01), np.arange(centroid[1]-dist, centroid[1]+dist, 0.01))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    ax.plot_surface(xx, yy, z, alpha=0.5)
    plt.show()