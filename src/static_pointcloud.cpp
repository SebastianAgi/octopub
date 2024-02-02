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
#include <pcl/visualization/pcl_visualizer.h>
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

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr occupancy_grid(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZI>::Ptr intensity_cloud_no_norm(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZINormal>::Ptr intensity_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
double maxheight = 0.0;
double minheight = 0.0;
double pos_x = 0.0;
double pos_y = 0.0;
bool flag = false;
bool filter_flag = false;
bool filter_flag2 = false;

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
        maxheight = msg->pose.pose.position.z + 1.0;
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

// Least Squares Plane Fitting from point cloud
void planeFitting(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int n) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    //clear the intensity_cloud
    intensity_cloud->clear();

    for (size_t i = 0; i < cloud->size(); ++i) {
        std::vector<int> pointIndices(n);
        std::vector<float> pointDistances(n);

        pcl::PointXYZ searchPoint = cloud->points[i];

        // From given point, select the n nearest points
        kdtree.nearestKSearch(searchPoint, n, pointIndices, pointDistances);

        pcl::PointCloud<pcl::PointXYZ> nearestPoints;
        for (size_t idx : pointIndices) {// Add the point to the nearestPoints local cloud
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

        pcl::PointXYZINormal point;
        point.x = cloud->points[i].x;
        point.y = cloud->points[i].y;
        point.z = cloud->points[i].z;
        point.intensity = abs(angle);
        point.normal_x = abs(normal[0]);
        point.normal_y = abs(normal[1]);
        point.normal_z = abs(normal[2]);
        intensity_cloud->points.push_back(point);

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

//function to convert the intensity_cloud to an occupancy grid
void convert_to_occupancy_grid(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& intensity_cloud)
{
  //clear the occupancy_grid
  occupancy_grid->clear();
  //go through the intensity_cloud
  for (size_t i = 0; i < intensity_cloud->size(); ++i)
  {
    //if the intensity is more than 0.5 then add the point to the occupancy_grid
    if (intensity_cloud->points[i].intensity < 0.75)
    {
      pcl::PointXYZ point;
      point.x = intensity_cloud->points[i].x;
      point.y = intensity_cloud->points[i].y;
      point.z = 0.0;
      occupancy_grid->points.push_back(point);
    }
  }

  //denoise a pointcloud using statistical outlier removal
  out.setInputCloud(occupancy_grid);
  out.setMeanK(10);
  out.setStddevMulThresh(1.5);
  out.filter(*occupancy_grid);

}

void octo_convert(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud)
{
  // Update the OctoMap with the point cloud data
  for (const pcl::PointXYZINormal& point : *cloud) {
    octomap::point3d endpoint(point.x, point.y, point.z);
    //create a uint8_t variable for the color
    octomap::OcTreeKey key = color_octree->coordToKey(endpoint);
    color_octree->updateNode(key, true);
    color_octree->setNodeColor(key, (1-point.intensity) * 255, (1-point.intensity) * 0, (point.intensity) * 160);
    // octree->setNodeValue(endpoint, point.intensity); 
  }
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_publisher");
    ros::NodeHandle nh;

    // Subscribe to the point cloud data
    ros::Subscriber pc_sub = nh.subscribe("/cloud_registered", 1, pointCloudCallback);
    ros::Subscriber pc_sub2 = nh.subscribe("/Odometry", 1, Odom_callback);
    ros::Publisher pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud_downsampled", 1);
    ros::Publisher pc_pub2 = nh.advertise<sensor_msgs::PointCloud2>("/cloud_downsampled2", 1);
    //publisher for the octomap
    ros::Publisher octomap_pub = nh.advertise<octomap_msgs::Octomap>("/octomap", 1);

    //create octomap object
    double resolution = 0.05;  // OctoMap resolution
    octree = new octomap::OcTree(resolution);
    color_octree = new octomap::ColorOcTree(resolution);


    //---------------------Preloaded map---------------------
    //load a pointcloud form a pcd file
    // pcl::io::loadPCDFile<pcl::PointXYZ>("/home/sebastian/catkin_ws/test_pcd.pcd", *cloud);    
    //-------------------------------------------------------


    while (ros::ok()) {
      // Add the point clouds together
      // *cloud += *temp;

      //check if the point cloud is empty
      if (cloud->empty()) {
        std::cout << "cloud is empty" << std::endl;
        ros::spinOnce();
        continue;
      }

      filtering(cloud);

      planeFitting(cloud, 24);

      std::cout << "intensity_cloud size: " << intensity_cloud->size() << std::endl;

      // Publish the traverasbility point cloud
      sensor_msgs::PointCloud2 output;
      pcl::toROSMsg(*intensity_cloud, output);
      output.header.frame_id = "camera_init";
      pc_pub.publish(output);

      //function to calculate the normal variation in intensity_cloud.intensity
      // normal_variation(cloud, 0.1);

      // Convert the point cloud to an octomap
      octo_convert(intensity_cloud);

      //convert to ocupancy grid
      // convert_to_occupancy_grid(intensity_cloud);

      // Publish the downsampled2 point cloud
      sensor_msgs::PointCloud2 output2;
      pcl::toROSMsg(*intensity_cloud, output2);
      output2.header.frame_id = "camera_init";
      pc_pub2.publish(output2);

      // Publish the octree as a ROS message
      octomap_msgs::Octomap octomap_msg;
      // octomap_msgs::binaryMapToMsg(*octree, octomap_msg);
      octomap_msgs::fullMapToMsg(*color_octree, octomap_msg);
      octomap_msg.header.frame_id = "camera_init";
      octomap_pub.publish(octomap_msg);
      
      ros::spinOnce();

    }

    // //save the point cloud to a pcd file
    // pcl::io::savePCDFileASCII("/home/sebastian/catkin_ws/cone_pcd.pcd", *cloud);

    return 0;
}