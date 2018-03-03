#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
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
#include <pcl/filters/passthrough.h>
#include <pcl/PointIndices.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#define filter_voxel_size 0.01f //voxel size
#define lower_cutoff -2.0 //ceiling
#define upper_cutoff 0.90 //floor
#define ransac_distance_thresh 0.01 //deviation from plane
#define plane_degree_tolerance 5.0 //deviation from horizontal
#define euclid_cluster_tolerance 0.05 //5cm

ros::Publisher pub;

void visualize(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, char* name) {
  pcl::visualization::CloudViewer viewer (name);
  viewer.showCloud(cloud);
  while (!viewer.wasStopped ()) {};
}

pcl::PointCloud<pcl::PointXYZ>::Ptr toFilteredPointCloud(const sensor_msgs::PointCloud2ConstPtr& input) {
  pcl::PCLPointCloud2::Ptr cloud_blob (new pcl::PCLPointCloud2);
  pcl::PCLPointCloud2::Ptr cloud_filtered_blob (new pcl::PCLPointCloud2);
  pcl::PointCloud<pcl::PointXYZ>::Ptr out (new pcl::PointCloud<pcl::PointXYZ>);

  pcl_conversions::toPCL(*input, *cloud_blob);

  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud(cloud_blob);
  sor.setLeafSize(filter_voxel_size, filter_voxel_size, filter_voxel_size);
  sor.filter(*cloud_filtered_blob);

  pcl::fromPCLPointCloud2(*cloud_filtered_blob, *out);
  return out;
}

void passThrough(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, char* field_name, double lower_bound, double upper_bound) {
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName (field_name);
  pass.setFilterLimits (lower_bound, upper_bound);
  pass.filter (*cloud);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr largestHorizontalPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr h_plane (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices());
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients());

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
 /*  Eigen::Vector3f axis = Eigen::Vector3f(0.0, 1.0, 0.0);
  seg.setAxis(axis);
  seg.setEpsAngle(  5.0f * (M_PI/180.0f) ); */
  seg.setMaxIterations(1000);
  seg.setDistanceThreshold(ransac_distance_thresh);

  int nr_points = (int) cloud->points.size ();
  int i = 0;

  while (1) {
    ROS_INFO("Iteration: [%d]", i);

    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);
    ROS_INFO("Model Coefficients:   ([%f], [%f], [%f], [%f])", coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);

    if (inliers->indices.size () == 0) {
      break;
    }

    extract.setInputCloud (cloud);
    extract.setIndices (inliers);

    if ((coefficients->values[1] > (1 - (plane_degree_tolerance * M_PI / 180))) || (coefficients->values[1] < (-1 + (plane_degree_tolerance * M_PI / 180)))) {
      extract.setNegative (false);
      extract.filter (*h_plane);
      //visualize(h_plane, "Horizontal Plane");
      extract.setNegative (true);
      extract.filter (*cloud_f);
      cloud.swap (cloud_f);
      return h_plane;
    }

    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud.swap (cloud_f);
    //visualize(cloud, "Filtered Cloud");
    i++;
  }
  return h_plane;
}

void euclidExtract(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);

  /*pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud (cloud);*/
  
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (euclid_cluster_tolerance);
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (50000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud);
  ec.extract (cluster_indices);

  ROS_INFO("Extraction Done: [%d] clusters", cluster_indices.size());
  if (cluster_indices.size() == 0) {
    *cloud = *cloud_cluster;
    return;
  }

  std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin ();
  for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
     cloud_cluster->points.push_back (cloud->points[*pit]);
  }
  cloud_cluster->width = cloud_cluster->points.size ();
  cloud_cluster->height = 1;
  cloud_cluster->is_dense = true;

  /*pcl::PointIndices::Ptr indices (new pcl::PointIndices);
  indices = (cluster_indices.at(0));
  extract.setIndices(indices);
  extract.setNegative (false);
  extract.filter (*cloud_cluster);*/

  ROS_INFO("Point Cloud Made");
  *cloud = *cloud_cluster;
}

void 
cloud_cb  (const sensor_msgs::PointCloud2ConstPtr& input) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = toFilteredPointCloud(input);
  pcl::PCLPointCloud2::Ptr output_pcl (new pcl::PCLPointCloud2);
  sensor_msgs::PointCloud2 output;
  
  //Remove the floor through a pass through filter
  passThrough(filtered_cloud, "y", lower_cutoff, upper_cutoff);
  //visualize(filtered_cloud, "Y cutoff");

  //isolate table
  pcl::PointCloud<pcl::PointXYZ>::Ptr h_plane = largestHorizontalPlane(filtered_cloud);
  euclidExtract(h_plane);
  //visualize(h_plane, "Horizontal Plane");

  pcl::PointXYZ min = pcl::PointXYZ();
  pcl::PointXYZ max = pcl::PointXYZ();
  pcl::getMinMax3D(*h_plane, min, max);
  //visualize(filtered_cloud, "Post Extraction");
  passThrough(filtered_cloud, "x", min.x, max.x);
  passThrough(filtered_cloud, "z", min.z, max.z);
  passThrough(filtered_cloud, "y", lower_cutoff, max.y);
  //visualize(filtered_cloud, "Objects");
  
  pcl::toPCLPointCloud2(*filtered_cloud, *output_pcl);
  pcl_conversions::fromPCL(*output_pcl, output);
  pub.publish (output);
}

int
main (int argc, char** argv) {
  // Initialize ROS
  ros::init (argc, argv, "objects");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/hsrb/head_rgbd_sensor/depth_registered/rectified_points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("objects", 1);

  // Spin
  ros::spin ();
}
