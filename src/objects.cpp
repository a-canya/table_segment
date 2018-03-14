#include <ros/ros.h>
#include <iostream>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
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

// All distences in m
#define filter_voxel_size 0.01f //voxel size
#define lower_cutoff 0.1 // floor
#define upper_cutoff 1.8 // ceiling or too high
#define ransac_distance_thresh 0.02 //deviation from plane
#define plane_degree_tolerance 5.0 //deviation from horizontal
#define euclid_cluster_tolerance 0.05 //5cm
#define y_table_buffer 0.025
#define x_table_buffer 0.025
#define z_table_buffer 0.025
#define max_item_height 0.30

#define FIXED_FRAME "map"
#define SENSOR_FRAME "head_rgbd_sensor_rgb_frame"

ros::Publisher external;
ros::Publisher table;
ros::Publisher objects;
ros::Publisher transformed;

tf::TransformListener *tf_listener;

/*For debugging, does nothing else*/
void visualize(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string name) {
  pcl::visualization::CloudViewer viewer (name);
  viewer.showCloud(cloud);
  while (!viewer.wasStopped ()) {};
}

// Mutex
// boost::mutex service_mutex;

//fnc to print components of a transform
void printTf(tf::Transform tf) {
    tf::Vector3 tfVec;
    tf::Matrix3x3 tfR;
    tf::Quaternion quat;
    tfVec = tf.getOrigin();
    cout<<"vector from reference frame to to child frame: "<<tfVec.getX()<<","<<tfVec.getY()<<","<<tfVec.getZ()<<endl;
    tfR = tf.getBasis();
    cout<<"orientation of child frame w/rt reference frame: "<<endl;
    tfVec = tfR.getRow(0);
    cout<<tfVec.getX()<<","<<tfVec.getY()<<","<<tfVec.getZ()<<endl;
    tfVec = tfR.getRow(1);
    cout<<tfVec.getX()<<","<<tfVec.getY()<<","<<tfVec.getZ()<<endl;
    tfVec = tfR.getRow(2);
    cout<<tfVec.getX()<<","<<tfVec.getY()<<","<<tfVec.getZ()<<endl;
    quat = tf.getRotation();
    cout<<"quaternion: " <<quat.x()<<", "<<quat.y()<<", "
            <<quat.z()<<", "<<quat.w()<<endl;
}

/* Voxel downsize the input cloud to reduce the computational load and also convert to PointCloud<PointXYZ> type
 * To do: implement outlier removal and/or other noise reduction methods
 * Also to do: handle transformations of the PointCloud so that it aligns with the axis that we want (y is vertical)
 */
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

void passThrough(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string field_name, double lower_bound, double upper_bound) {
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName (field_name);
  pass.setFilterLimits (lower_bound, upper_bound);
  pass.filter (*cloud);
}

/* Extract the most represented horizontal (+- deviation) plane from the scene*/
pcl::PointCloud<pcl::PointXYZ>::Ptr largestHorizontalPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr h_plane (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices());
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients());

  // Calculate vertical axis
  // tf::StampedTransform transform;
  // tf_listener->lookupTransform(SENSOR_FRAME, FIXED_FRAME, ros::Time(0), transform);
  // // transform: from FIXED_FRAME (origin) to SENSOR_FRAME (target)
  // printTf(transform);
  // tf::Vector3 v = transform.getBasis() * tf::Vector3(0.0, 0.0, 1.0);
  // Eigen::Vector3f axis(v.getX(), v.getY(), v.getZ());
  // ROS_INFO("Looking for a plane perpendicular to (%f, %f, %f) in camera reference", v.getX(), v.getY(), v.getZ());
  Eigen::Vector3f axis(0.0, 0.0, 1.0);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setAxis(axis);
  seg.setEpsAngle(  5.0f * (M_PI/180.0f) );
  seg.setMaxIterations(1000);
  seg.setDistanceThreshold(ransac_distance_thresh);

  int nr_points = (int) cloud->points.size ();
  int i = 0;

  // while (1) {
    ROS_INFO("Iteration: [%d]", i);

    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);
    ROS_INFO("Model Coefficients:   ([%f], [%f], [%f], [%f])", coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);

    // if (inliers->indices.size () == 0) {
    //   break;
    // }

    extract.setInputCloud (cloud);
    extract.setIndices (inliers);

    // if ((coefficients->values[1] > (1 - (plane_degree_tolerance * M_PI / 180))) || (coefficients->values[1] < (-1 + (plane_degree_tolerance * M_PI / 180)))) {
      extract.setNegative (false);
      extract.filter (*h_plane);
      //visualize(h_plane, "Horizontal Plane");
      extract.setNegative (true);
      extract.filter (*cloud_f);
      *cloud = *cloud_f;
      return h_plane;
    // }
    //visualize(cloud, "before");
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud = *cloud_f;
    //visualize(cloud, "Filtered Cloud");
    i++;
  // }
  return h_plane;
}

/* Extracts the table (the largest euclidean cluster from the extracted plane)*/
void tableExtract(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr extract_from) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr extract_out (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud (cloud);

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

  pcl::PointIndices::Ptr indices (new pcl::PointIndices);

  //visualize(cloud, "Before");

  indices->indices = (cluster_indices.at(0)).indices;
  extract.setIndices(indices);
  extract.setNegative (false);
  extract.filter (*cloud_cluster);
  extract.setInputCloud(extract_from);
  extract.setNegative (true);
  extract.filter (*extract_out);

  //visualize(cloud, "After");

  ROS_INFO("Point Cloud Made");
  *extract_from = *extract_out;
  *cloud = *cloud_cluster;
}

/* To do: store extracted object point clouds seperately in a vector of PointCloud<PointXYZ>::Ptr (objects)
 * also define what the minimum # of points euclidean clustered together it takes to constitute an "object"
 */
void objectsExtract(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr extract_from) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr extract_out (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::ExtractIndices<pcl::PointXYZ> extract;

  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (euclid_cluster_tolerance);
  ec.setMinClusterSize (10);
  ec.setMaxClusterSize (2000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud);
  ec.extract (cluster_indices);

  ROS_INFO("Extraction Done: [%d] clusters", cluster_indices.size());
  if (cluster_indices.size() == 0) {
    *cloud = *cloud_cluster;
    return;
  }

  pcl::PointIndices::Ptr indices (new pcl::PointIndices);

  for (int i = 0; i < cluster_indices.size(); i++) {
    indices->indices = (cluster_indices.at(0)).indices;
    extract.setInputCloud (cloud);
    extract.setIndices(indices);
    extract.setNegative (false);
    extract.filter (*cloud_cluster);
    extract.setNegative (true);
    extract.setInputCloud (extract_from);
    extract.filter (*extract_out);
    *extract_from = *extract_out;
  }

  ROS_INFO("Point Cloud Made");
  *cloud = *cloud_cluster;
}

void
cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input) {

  // service_mutex.lock();

  //Filter a point cloud for processing and initialize a cloud for environment (not table or objects)
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = toFilteredPointCloud(input);
  // Calculate vertical axis
  tf::StampedTransform transform;
  tf_listener->lookupTransform(FIXED_FRAME, SENSOR_FRAME, ros::Time(0), transform);
  // transform: from SENSOR_FRAME (target) to FIXED_FRAME (origin)
  printTf(transform);
  pcl::PointCloud<pcl::PointXYZ>::Ptr world_frame_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl_ros::transformPointCloud(*filtered_cloud, *world_frame_cloud, transform);
  world_frame_cloud->header.frame_id = FIXED_FRAME;

  pcl::PointCloud<pcl::PointXYZ>::Ptr remainders (new pcl::PointCloud<pcl::PointXYZ>);
  *remainders = *world_frame_cloud;

  // ROS_INFO("tf_listener frames:");
  // std::cout << tf_listener->allFramesAsString() << '\n';
  // while (ros::ok()){     // keep trying until we get the transform
  //   tf::StampedTransform transform;
  //   try{
  //     // Look up transform
  //     ROS_INFO("lookupTransform");
  //     tf_listener->lookupTransform(FIXED_FRAME, SENSOR_FRAME, ros::Time(0), transform);
  //     printTf(transform);
  //     ROS_INFO("transformPointCloud");
  //     pcl_ros::transformPointCloud((*filtered_cloud), (*remainders), transform);
  //     pcl_ros::transformPointCloud((*filtered_cloud), (*filtered_cloud), transform);
  //     ROS_INFO("transformed");
  //     filtered_cloud->header.frame_id = FIXED_FRAME;
  //     remainders->header.frame_id = FIXED_FRAME;
  //     break;
  //   } catch (tf::TransformException &ex){
  //     ROS_ERROR_THROTTLE(2,"%s",ex.what());
  //     ROS_WARN_THROTTLE(2, "   Waiting for tf to transform desired SAC axis to point cloud frame. trying again");
  //   }
  // }

  //intermediate clouds for publishing back into sensor_msgs
  pcl::PCLPointCloud2::Ptr objects_pcl (new pcl::PCLPointCloud2);
  sensor_msgs::PointCloud2 objects_msg;
  pcl::PCLPointCloud2::Ptr table_pcl (new pcl::PCLPointCloud2);
  sensor_msgs::PointCloud2 table_msg;
  pcl::PCLPointCloud2::Ptr external_pcl (new pcl::PCLPointCloud2);
  sensor_msgs::PointCloud2 external_msg;

  //Remove the floor and the ceiling through a pass through filter
  passThrough(world_frame_cloud, "z", lower_cutoff, upper_cutoff);

  // Output for debug
  pcl::PCLPointCloud2::Ptr ss (new pcl::PCLPointCloud2);
  sensor_msgs::PointCloud2 sss;
  pcl::toPCLPointCloud2(*world_frame_cloud, *ss);
  pcl_conversions::fromPCL(*ss, sss);
  transformed.publish (sss);

  //Isolate plane of the table
  pcl::PointCloud<pcl::PointXYZ>::Ptr h_plane = largestHorizontalPlane(world_frame_cloud);
  //Extract the table from its plane and put that in h_plane, also extract the table from the environment
  tableExtract(h_plane, remainders);

  /*Tried removing plane of table in euclidean clustering, was not working, done in ransac instead*/
  //*world_frame_cloud = *remainders;

  //Get the approximate XYZ boundaries of the table
  //To do: replace with something better
  pcl::PointXYZ min = pcl::PointXYZ();
  pcl::PointXYZ max = pcl::PointXYZ();
  pcl::getMinMax3D(*h_plane, min, max);

  //Isolate only the area above the boundaries of the table upwards (where objects would be)
  passThrough(world_frame_cloud, "x", min.x + x_table_buffer, max.x - x_table_buffer);
  passThrough(world_frame_cloud, "y", min.y + y_table_buffer, max.y - y_table_buffer);
  passThrough(world_frame_cloud, "z", min.z + z_table_buffer, min.z + max_item_height); // vertical

  /* Here is where the all objects in one cloud stuff would be extracted into
   * their own individual pointclouds in a vector that we could output, not done yet
   */


  //Publish the outputs
  pcl::toPCLPointCloud2(*world_frame_cloud, *objects_pcl);
  pcl_conversions::fromPCL(*objects_pcl, objects_msg);
  objects.publish (objects_msg);

  pcl::toPCLPointCloud2(*h_plane, *table_pcl);
  pcl_conversions::fromPCL(*table_pcl, table_msg);
  table.publish (table_msg);

  objectsExtract(world_frame_cloud, remainders);
  pcl::toPCLPointCloud2(*remainders, *external_pcl);
  pcl_conversions::fromPCL(*external_pcl, external_msg);
  external.publish (external_msg);

  // service_mutex.unlock();
}

int
main (int argc, char** argv) {
  // Initialize ROS
  ros::init (argc, argv, "objects");
  ros::NodeHandle nh;

  // Create a Transform Listener
  tf_listener = new tf::TransformListener;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/hsrb/head_rgbd_sensor/depth_registered/rectified_points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  external = nh.advertise<sensor_msgs::PointCloud2> ("external", 1);
  table = nh.advertise<sensor_msgs::PointCloud2> ("table", 1);
  objects = nh.advertise<sensor_msgs::PointCloud2> ("objects", 1);
  transformed = nh.advertise<sensor_msgs::PointCloud2> ("transformed", 1);

  // Spin
  ros::spin ();
}
