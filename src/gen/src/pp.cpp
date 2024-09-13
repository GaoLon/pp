#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>
#include <pcl/filters/extract_indices.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Eigen>
#include <random>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_ros/se2_grid_ros.hpp>
#include <fstream>

using namespace std;
using namespace Eigen;

bool map_ok_ = false;
ros::Publisher _all_map_pub;
ros::Publisher  raw_pub;

sensor_msgs::PointCloud2 globalMap_pcd;
se2_grid_msgs::SE2Grid map_msg;

pcl::PointCloud<pcl::PointXYZ> cloudMap;
se2_grid::SE2Grid fused_map({"elevation", "inpainted", "smooth", "ground"}, {false, false, false, false});

double normal_radius;
double max_curvature;
double min_cosxi;

void processMap(int& point_num,
                float* points,
                MatrixXf& map,
                MatrixXf& inpainted,
                MatrixXf& smooth,
                MatrixXf& ground);

void gpuInit(const Vector2i& size_pos_,
            const Vector2f& length_pos_,
            float resolution_pos_,
            const Array2i& start_i,
            const Vector2f& position_,
            float normal_radius,
            float max_curvature,
            float min_cosxi);

void pubPoints()
{
  // if (pcl::io::loadPCDFile<pcl::PointXYZ>(ros::package::getPath("pp")+"/env/e2e4.pcd", cloudMap) == -1)
  // {
  //   PCL_ERROR("Failed to read PCD file.\n");
  //   return;
  // }
  if (pcl::io::loadPLYFile<pcl::PointXYZ>(ros::package::getPath("pp")+"/env/e2e4.ply", cloudMap) == -1)
  {
    PCL_ERROR("Failed to read PCD file.\n");
    return;
  }

  //! process pcd
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudMapPtr(new pcl::PointCloud<pcl::PointXYZ>(cloudMap));

  // 创建直通滤波器
  ROS_WARN("Start passthrough filtering");
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(cloudMapPtr);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0.0, 2.5);
  pass.filter(*cloudFiltered);
  

  //! init gpu
  //! e2e3
  // Array2d length_pos_(140.0, 90.0);
  // Vector2d position_(30.0, 15.0);

  //! e2e4
  Array2d length_pos_(82.0, 60.0);
  Vector2d position_(16.0, 30.0);

  double resolution_pos_ = 0.1;
  double resolution_yaw_ = 8.2;
  fused_map.initGeometry(length_pos_, resolution_pos_, resolution_yaw_, position_);
  Array2i size_pos_ = fused_map.getSizePos();
  gpuInit(size_pos_.matrix(), length_pos_.matrix().cast<float>(), (float)resolution_pos_, 
          fused_map.getStartIndex(), fused_map.getPosition().cast<float>(), normal_radius, max_curvature, min_cosxi);

  // if (false)
  {
    // points
    int point_num = cloudFiltered->size();
    ROS_INFO("point_num: %d", point_num);
    float* points_world = new float[point_num*3];
    for (int i=0; i<point_num; i++)
    {
        points_world[i*3] = cloudFiltered->points[i].x;
        points_world[i*3+1] = cloudFiltered->points[i].y;
        points_world[i*3+2] = cloudFiltered->points[i].z;
    }
    Eigen::MatrixXf& ele = fused_map["elevation"][0];
    Eigen::MatrixXf& inpainted = fused_map["inpainted"][0];
    Eigen::MatrixXf& smooth = fused_map["smooth"][0];
    Eigen::MatrixXf& ground = fused_map["ground"][0];
    processMap(point_num, points_world, ele, inpainted, smooth, ground);
    delete[] points_world;
    ROS_INFO("Finish processing map");
    se2_grid::SE2GridRosConverter::toMessage(fused_map, map_msg);
  }

  // for (double x=-40.0; x<100.0; x+=0.1)
  // {for (double y=-30.0; y<60.0; y+=0.1)
  for (double x=-25.0; x<57.0; x+=0.1)
  {for (double y=0.0; y<60.0; y+=0.1)
    {
      Vector3d pos(x, y, 0.0);
      Array3i idx;
      fused_map.boundPos(pos);
      if (fused_map.pos2Index(pos, idx))
      {
         float height = fused_map["ground"][0](idx[0], idx[1]);
        if (!isnan(height))
          cloudMap.push_back(pcl::PointXYZ(x, y, height));
      }
    }
  }

  // 创建体素滤波器对象
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudMapPtr2(new pcl::PointCloud<pcl::PointXYZ>(cloudMap));
  ROS_WARN("Start voxel grid filtering");
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloudMapPtr2);
  vg.setLeafSize(0.1f, 0.1f, 0.1f); // 设置体素网格大小
  vg.filter(cloudMap);
  ROS_WARN("voxel grid filtering done.");

  // 创建统计滤波器对象
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudMapPtr3(new pcl::PointCloud<pcl::PointXYZ>(cloudMap));
  ROS_WARN("Start statistical filtering");
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
  outrem.setInputCloud(cloudMapPtr3);  // 将点云输入到滤波器
  outrem.setRadiusSearch(0.5);  // 设置近邻点搜索半径
  outrem.setMinNeighborsInRadius(5);  // 设置查询点最小近邻点数
  outrem.filter (cloudMap);

  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ>(ros::package::getPath("pp")+"/env/real_pp.pcd", cloudMap);

  fused_map.convertToDefaultStartIndex();
  ofstream map_writer;
  map_writer.open(ros::package::getPath("pp")+"/env/real_pp.param");    
  map_writer.clear();
  Eigen::MatrixXf ff = fused_map["ground"][0];
  for (int i=0; i<ff.rows(); i++)
  {
    for (int j=0; j<ff.cols(); j++)
    {
      if (isnan(ff(i,j)))
        ff(i,j) = -10000.0;
    }
  }
  map_writer << (position_ + length_pos_.matrix()/2.0).transpose() << endl;
  map_writer << ff.rows() << " " << ff.cols() << endl;
  map_writer << ff;
  map_writer.close();
  
  pcl::toROSMsg(cloudMap, globalMap_pcd); 
  globalMap_pcd.header.frame_id = "world";
  map_ok_ = true;
  return;
}

void PubTimer(const ros::TimerEvent &event)
{
  if (map_ok_)
  {
    raw_pub.publish(map_msg);
    _all_map_pub.publish(globalMap_pcd);
  }
 
  return;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "random_map_sensing");
  ros::NodeHandle n("~");

  _all_map_pub = n.advertise<sensor_msgs::PointCloud2>("global_cloud", 1);
  raw_pub = n.advertise<se2_grid_msgs::SE2Grid>("map", 1);

  n.getParam("/pp/normal_radius", normal_radius);
  n.getParam("/pp/max_curvature", max_curvature);
  n.getParam("/pp/min_cosxi", min_cosxi);

  pubPoints();

  ros::Timer timer = n.createTimer(ros::Duration(2.0), PubTimer);
  ros::Rate loop_rate(1.0);

  int tt = 0.0;
  while (ros::ok())
  {
    tt ++;
    if (tt > 10)
    {
      timer.stop();
    }
    ros::spinOnce();
    loop_rate.sleep();
  }
}
