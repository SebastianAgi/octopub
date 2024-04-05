#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h> // For loading point cloud data from PCD files
#include "pcl/conversions.h"
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/filters/passthrough.h> // For filtering point clouds
#include <nav_msgs/Odometry.h>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h> // For filtering point clouds
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
// #include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <omp.h> // Include this for OpenMP
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap/OcTreeBase.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <pcl/point_cloud.h>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/OccupancyGrid.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr slam_cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);

pcl::PointCloud<pcl::PointXYZ>::Ptr occupancy_grid(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZI>::Ptr intensity_cloud_no_norm(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZINormal>::Ptr intensity_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
// pcl::PointCloud<pcl::PointXYZINormal>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);

std::vector<float> object_location;

double maxheight = 0.0;
double minheight = 0.0;
double pos_x = 0.0;
double pos_y = 0.0;
bool flag = false;
bool filter_flag = false;
bool filter_flag2 = false;
//declare transform variable
Eigen::Matrix4f livox_angle_transform;


octomap::OcTree *octree;  // Define an OctoMap object
octomap::ColorOcTree *color_octree;  // Define an color OctoMap object


// VoxelGrid filtering to downsample the point cloud
pcl::VoxelGrid<pcl::PointXYZ> sor;
// Delete points that have a z value more and less than 5 and -5 respectively
pcl::PassThrough<pcl::PointXYZ> pass;
// Create the filtering object
pcl::StatisticalOutlierRemoval<pcl::PointXYZ> out;

void Odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
    pos_x = msg->pose.pose.position.x;
    pos_y = msg->pose.pose.position.y;

    if (maxheight == 0.0)
    {
        // Assign the value of z to maxheight and minheight
        maxheight = msg->pose.pose.position.z + 0.5;
        minheight = msg->pose.pose.position.z - 0.6;
        std::cout << "maxheight: " << maxheight << std::endl;
        std::cout << "minheight: " << minheight << std::endl;
        std::cout << "pos_x: " << pos_x << std::endl;
        std::cout << "pos_y: " << pos_y << std::endl;
    }
}

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    pcl::fromROSMsg(*msg, *temp);
    // std::cout << "call back active" << std::endl;
}

void object_location_callback(const geometry_msgs::PointStamped::ConstPtr& msg)
{
  object_location.push_back(msg->point.x);
  object_location.push_back(msg->point.y);
  object_location.push_back(msg->point.z);
  std::cout << "object_location: " << object_location[0] << " " << object_location[1] << " " << object_location[2] << std::endl;
}

// Function to filter the point cloud
void filtering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(pos_x - 10, pos_x + 10);
    pass.filter(*cloud);

    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(pos_y - 10, pos_y + 10);
    pass.filter(*cloud);

    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-2, maxheight);
    pass.filter(*cloud);

    sor.setInputCloud(cloud);
    sor.setLeafSize(0.05f, 0.05f, 0.05f);
    sor.filter(*cloud);

}

// Least Squares Plane Fitting from point cloud, also checks if there are enough points within a certain radius
void planeFitting(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float radius_threshold, int min_points) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    // Clear the intensity_cloud
    intensity_cloud->clear();
    occupancy_grid->clear();


    for (size_t i = 0; i < cloud->size(); ++i) {
        std::vector<int> pointIndices;
        std::vector<float> pointDistances;

        pcl::PointXYZ searchPoint = cloud->points[i];

        // From given point, select all points within the specified radius
        if (kdtree.radiusSearch(searchPoint, radius_threshold, pointIndices, pointDistances) >= min_points) {
            pcl::PointCloud<pcl::PointXYZ> nearestPoints;
            for (size_t idx : pointIndices) {
                // Add the point to the nearestPoints local cloud
                nearestPoints.points.push_back(cloud->points[idx]);
            }
            
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(nearestPoints, centroid);

            Eigen::Matrix3f cov;
            pcl::computeCovarianceMatrixNormalized(nearestPoints, centroid, cov);

            // Calculate the eigenvalues and eigenvectors of the covariance matrix
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(cov);
            Eigen::Vector3f normal = eigenSolver.eigenvectors().col(0); // Eigenvector with the smallest eigenvalue

            // Calculate the dot product between the normal vector and the z-axis
            float angle = normal.dot(Eigen::Vector3f::UnitZ());

            // pcl::PointXYZINormal point;
            // point.x = cloud->points[i].x;
            // point.y = cloud->points[i].y;
            // point.z = cloud->points[i].z;
            // point.intensity = abs(angle);
            // point.normal_x = abs(normal[0]);
            // point.normal_y = abs(normal[1]);
            // point.normal_z = abs(normal[2]);
            // intensity_cloud->points.push_back(point);

            //push point of abs(angle) > 0.5 to occupancy grid
            if (abs(angle) < 0.5 && cloud->points[i].x < abs(5.0) && cloud->points[i].y < abs(5.0))
            {
              pcl::PointXYZ point;
              point.x = cloud->points[i].x;
              point.y = cloud->points[i].y;
              point.z = 0.0;
              occupancy_grid->points.push_back(point);

              if (abs(point.y) > 5.0)
              {
                std::cout << "point not in range" << std::endl;
              }
            }
        }
    }
}


//function to see the change in normal vectors across intensity_cloud.normal vectors
void normal_variation(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double radius)
{
  std::vector<int> pointIndices;
  std::vector<float> pointDistances;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud);
  //go through the intensity_cloud
  for (size_t i = 0; i < cloud->size(); ++i)
  {
    pointIndices.clear();
    pointDistances.clear();
    float dot_product = 0.0;
    float avg_dot_product = 0.0;
    //for each point, check the normal vectors in a radius of the current point
    kdtree.radiusSearchT(cloud->points[i], radius, pointIndices, pointDistances);
    // //for each point, check the normal vectors for n nearest points
    // kdtree.nearestKSearch(cloud->points[i], 24, pointIndices, pointDistances);

    //for each point in the radius, calculate the deviation in normal vectors
    for (size_t idx : pointIndices)
    {
      //calculate the dot product between the normal vectors
      dot_product = intensity_cloud->points[i].normal_x * intensity_cloud->points[idx].normal_x + intensity_cloud->points[i].normal_y * intensity_cloud->points[idx].normal_y + intensity_cloud->points[i].normal_z * intensity_cloud->points[idx].normal_z;
      //average the dot product
      avg_dot_product = avg_dot_product + dot_product;
    }
    //calculate the average dot product
    avg_dot_product = avg_dot_product / pointIndices.size();
    //calculate the deviation in normal vectors
    float deviation = 1 - avg_dot_product;
    //assign the deviation to the intensity of the point
    intensity_cloud->points[i].intensity = deviation;
  }
}

void createAndPublishOccupancyGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, ros::Publisher& pub) {
    nav_msgs::OccupancyGrid grid;

    // Set header
    grid.header.stamp = ros::Time::now();
    grid.header.frame_id = "camera_init"; // or your relevant frame
    
    // Set grid info
    grid.info.resolution = 0.05; // 5cm per cell
    grid.info.width = 200; // 5m wide
    grid.info.height = 200; // 5m long
    grid.info.origin.position.x = -5.0; // Start at -25m
    grid.info.origin.position.y = -5.0; // Start at -25m
    grid.info.origin.orientation.w = 1.0; // No rotation
    
    // Initialize grid data
    std::vector<int8_t> data(grid.info.width * grid.info.height, -1); // Start with all cells as unknown
    
    // Convert points to occupancy
    for (const auto& point : cloud->points) {
        int x_idx = static_cast<int>((point.x - grid.info.origin.position.x) / grid.info.resolution);
        int y_idx = static_cast<int>((point.y - grid.info.origin.position.y) / grid.info.resolution);
        
        if (x_idx >= 0 && x_idx < grid.info.width && y_idx >= 0 && y_idx < grid.info.height) {
            data[y_idx * grid.info.width + x_idx] = 100; // Mark as occupied
        }
    }
    
    // Assign data to the OccupancyGrid message
    grid.data = data;
    
    // Publish the grid
    pub.publish(grid);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_publisher");
    ros::NodeHandle nh;

    // Subscribe to the point cloud data
    ros::Subscriber pc_sub = nh.subscribe("/cloud_registered", 1, pointCloudCallback);
    // ros::Subscriber pc_sub1 = nh.subscribe("/ouster/points", 1, pointCloudCallback);
    ros::Subscriber pc_sub2 = nh.subscribe("/Odometry", 1, Odom_callback);
    ros::Subscriber pc_sub3 = nh.subscribe("/object_point", 1, object_location_callback);
    ros::Publisher pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/occupancy_grid", 1);
    ros::Publisher pc_pub2 = nh.advertise<sensor_msgs::PointCloud2>("/cloud_aligned", 1);
    ros::Publisher map_pub = nh.advertise<nav_msgs::OccupancyGrid>("/occupancy_map", 1);
    //publisher for the octomap
    ros::Publisher octomap_pub = nh.advertise<octomap_msgs::Octomap>("/octomap_flat", 1);
    ros::Publisher octomap_pub1 = nh.advertise<octomap_msgs::Octomap>("/octomap", 1);

    //create octomap object
    double resolution = 0.05;  // OctoMap resolutiona
    octree = new octomap::OcTree(resolution);
    color_octree = new octomap::ColorOcTree(resolution);


    //---------------------Preloaded map---------------------
    //load a pointcloud form a pcd file
    // pcl::io::loadPCDFile<pcl::PointXYZ>("/home/sebastian/catkin_ws/test_pcd.pcd", *cloud);    
    //-------------------------------------------------------
    
    //populate the transformation matrix with the rotation and translation values from the camera
    // transform << 0.03710035, -0.99924117,  0.01185934, -0.02849777,
    //           -0.00811149, -0.01216825, -0.99989306, -0.04824431,
    //            0.99927862,  0.03700018, -0.00855678, -0.0824864,
    //            0.0        ,  0.0        ,  0.0,          1.0;

    livox_angle_transform << 0.9816, 0.0, 0.1908, 0.0,
                             0.0,    1.0, 0.0,    0.0,
                            -0.1908, 0.0, 0.9816, 0.0,
                             0.0,    0.0, 0.0,    1.0;
 
    // transform << -0.0665298 , -0.99774493,  0.00887949,  0.25425081,
    //              -0.01777641, -0.00771256, -0.99981224, -0.02287156,
    //               0.99762607, -0.06667515, -0.0172232 , -0.11442667,
    //               0.        ,  0.        ,  0.        ,  1.        ;


    // ros::spin();
    while (ros::ok()) {
      // Add the point clouds together
      *cloud += *temp;

      //set cloud to temp as temp updates
      // *cloud = *temp;

      //check if the point cloud is empty
      if (cloud->empty()) {
        if (flag == false)
        {
          std::cout << "cloud is empty" << std::endl;
          flag = true;
        }       
        
        ros::spinOnce();
        continue;
      }
      
      pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

      //transform the pointcloud
      pcl::transformPointCloud (*cloud, *transformed_cloud, livox_angle_transform);

      //publsih the transformed pointcloud
      sensor_msgs::PointCloud2 output;
      pcl::toROSMsg(*transformed_cloud, output);
      output.header.frame_id = "camera_init";
      pc_pub2.publish(output);


      filtering(transformed_cloud);

      planeFitting(transformed_cloud, 0.25, 24);

      // std::cout << "intensity_cloud size: " << intensity_cloud->size() << std::endl;

      // Publish the traverasbility point cloud
      // sensor_msgs::PointCloud2 output1;
      // pcl::toROSMsg(*intensity_cloud, output1);
      // output1.header.frame_id = "camera_init";
      // pc_pub.publish(output1);

      // Publish the occupancy grid
      sensor_msgs::PointCloud2 output2;
      pcl::toROSMsg(*occupancy_grid, output2);
      output2.header.frame_id = "camera_init";
      pc_pub.publish(output2);

      // Publish the occupancy map
      createAndPublishOccupancyGrid(occupancy_grid, map_pub);
      
      ros::spinOnce();

    }

    // //save the point cloud to a pcd file
    // pcl::io::savePCDFileASCII("/home/sebastian/catkin_ws/cone_pcd.pcd", *cloud);

    return 0;
}