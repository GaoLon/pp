#pragma once

#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_msgs/SE2Grid.h>

// STL
#include <vector>
#include <unordered_map>

// Eigen
#include <Eigen/Core>

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/GridCells.h>

namespace se2_grid {

/*!
 * ROS interface for the Grid Map library.
 */
class SE2GridRosConverter
{
 public:
  /*!
   * Default constructor.
   */
  SE2GridRosConverter();

  /*!
   * Destructor.
   */
  virtual ~SE2GridRosConverter();

  /*!
   * Converts a ROS grid map message to a grid map object.
   * @param[in] message the grid map message.
   * @param[in] layers the layers to be copied.
   * @param[in] copyBasicLayers if true, basic layers are copied.
   * @param[in] copyAllNonBasicLayers if true, all basic layers are copied, otherwise only that one specified in layers.
   * @param[out] SE2Grid the grid map object to be initialized.
   * @return true if successful, false otherwise.
   */
  static bool fromMessage(const se2_grid_msgs::SE2Grid& message, se2_grid::SE2Grid& SE2Grid, const std::vector<std::string>& layers);

  /*!
   * Converts a ROS grid map message to a grid map object.
   * @param[in] message the grid map message.
   * @param[out] SE2Grid the grid map object to be initialized.
   * @return true if successful, false otherwise.
   */
  static bool fromMessage(const se2_grid_msgs::SE2Grid& message, se2_grid::SE2Grid& SE2Grid);

  /*!
   * Converts all layers of a grid map object to a ROS grid map message.
   * @param[in] SE2Grid the grid map object.
   * @param[out] message the grid map message to be populated.
   */
  static void toMessage(const se2_grid::SE2Grid& SE2Grid, se2_grid_msgs::SE2Grid& message);

  /*!
   * Converts requested layers of a grid map object to a ROS grid map message.
   * @param[in] SE2Grid the grid map object.
   * @param[in] layers the layers to be added to the message.
   * @param[out] message the grid map message to be populated.
   */
  static void toMessage(const se2_grid::SE2Grid& SE2Grid, const std::vector<std::string>& layers,
                        se2_grid_msgs::SE2Grid& message);

  /*!
   * Saves a grid map into a ROS bag. The timestamp of the grid map
   * is used as time for storing the message in the ROS bag. The time
   * value 0.0 is not a valid bag time and will be replaced by the
   * current time.
   * @param[in] SE2Grid the grid map object to be saved in the ROS bag.
   * @param[in] pathToBag the path to the ROS bag file.
   * @param[in] topic the name of the topic in the ROS bag.
   * @return true if successful, false otherwise.
   */
  static bool saveToBag(const se2_grid::SE2Grid& SE2Grid, const std::string& pathToBag,
                        const std::string& topic);

  /*!
   * Loads a SE2Grid from a ROS bag.
   * @param[in] pathToBag the path to the ROS bag file.
   * @param[in] topic the topic name of the grid map in the ROS bag.
   * @param[out] SE2Grid the grid map object to be initialized.
   * @return true if successful, false otherwise.
   */
  static bool loadFromBag(const std::string& pathToBag, const std::string& topic,
                          se2_grid::SE2Grid& SE2Grid);

};

} /* namespace */
