#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/PointIndices.h>


ros::Publisher pub;

void 
cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  pcl::PCLPointCloud2::Ptr cloud_blob (new pcl::PCLPointCloud2); 
  pcl::PCLPointCloud2ConstPtr cloudPtr(cloud_blob);
  pcl::PCLPointCloud2::Ptr cloud_filtered_blob (new pcl::PCLPointCloud2);
  pcl::PCLPointCloud2::Ptr output_blob (new pcl::PCLPointCloud2);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  
  sensor_msgs::PointCloud2::Ptr output;

  pcl_conversions::toPCL(*input, *cloud_blob);

  ROS_INFO("PointCloud initial size [%d]", cloud_blob->width * cloud_blob->height);
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud(cloud_blob);
  sor.setLeafSize(0.01f, 0.01f, 0.01f);
  sor.filter(*cloud_filtered_blob);

  ROS_INFO("PointCloud filtered size [%d]", cloud_filtered_blob->width * cloud_filtered_blob->height);
  pcl::fromPCLPointCloud2(*cloud_filtered_blob, *cloud_filtered);

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients());

  pcl::SACSegmentation<pcl::PointXYZ> seg;

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(1000);
  seg.setDistanceThreshold(0.01);

  pcl::ExtractIndices<pcl::PointXYZ> extract;

  int i = 0, nr_points = (int) cloud_filtered->points.size ();
  
  while (cloud_filtered->points.size () > 0.3 * nr_points)
  {
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices());
    
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      break;
    }

    // Extract the inliers
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cloud_p);
    ROS_INFO("Pointcloud representing the planar compnent size [%d]", cloud_p->width * cloud_p->height);

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_filtered.swap (cloud_f);
    i++;
  }

  pcl::toPCLPointCloud2(*cloud_f, *output_blob);
  pcl_conversions::fromPCL(*output_blob, *output);
  
  // Publish the data.
  pub.publish (output);
}

int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "segment");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/hsrb/head_rgbd_sensor/depth_registered/rectified_points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("segment", 1);

  // Spin
  ros::spin ();
}
