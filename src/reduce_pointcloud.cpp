#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h> // For filtering point clouds

int
main ()
{
    pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
    pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());
    // Fill in the cloud data
    pcl::PCDReader reader;
    // Replace the path below with the path where you saved your file
    reader.read ("/home/sebastian/catkin_ws/src/octopub/pc_data/scans.pcd", *cloud); // Remember to download the file first!
    std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height 
        << " data points (" << pcl::getFieldsList (*cloud) << ")." << std::endl;

    //Delete points that have a z value more and less than 5 and -5 respectively
    pcl::PassThrough<pcl::PCLPointCloud2> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-5, 5);    

    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (0.05f, 0.05f, 0.05f);
    sor.filter (*cloud_filtered);

    //print the number of points in the pointclouds
    int original_points = cloud->width * cloud->height;
    int filtered_points = cloud_filtered->width * cloud_filtered->height;
    float reduction = (float)filtered_points / (float)original_points;

    std::cerr << "PointCloud after filtering: " << filtered_points 
        << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")." << std::endl;
        //print the percentage dercrease in points
    std::cerr << "Pointcloud reduced to " << reduction * 100 << "% of original size" << std::endl;
    pcl::PCDWriter writer;
    writer.write ("lab_pc_downsampled.pcd", *cloud_filtered, 
          Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);
    return (0);
}