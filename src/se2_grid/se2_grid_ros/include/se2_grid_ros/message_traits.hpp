#pragma once

#include <se2_grid_msgs/SE2Grid.h>
#include <se2_grid_msgs/SE2Grid.h>

namespace ros {
namespace message_traits {

template<>
struct HasHeader<se2_grid_msgs::SE2Grid> : public TrueType {};

template <>
struct Header<se2_grid_msgs::SE2Grid, typename boost::enable_if<HasHeader<se2_grid_msgs::SE2Grid>>::type>
{
  static std_msgs::Header* pointer(se2_grid_msgs::SE2Grid& m)
  {
    return &m.info.header;
  }

  static std_msgs::Header const* pointer(const se2_grid_msgs::SE2Grid& m)
  {
    return &m.info.header;
  }
};

template<>
struct FrameId<se2_grid_msgs::SE2Grid, typename boost::enable_if<HasHeader<se2_grid_msgs::SE2Grid>>::type>
{
  static std::string* pointer(se2_grid_msgs::SE2Grid& m)
  {
    return &m.info.header.frame_id;
  }

  static std::string const* pointer(const se2_grid_msgs::SE2Grid& m)
  {
    return &m.info.header.frame_id;
  }

  static std::string value(const se2_grid_msgs::SE2Grid& m)
  {
    return m.info.header.frame_id;
  }
};

template<>
struct TimeStamp<se2_grid_msgs::SE2Grid, typename boost::enable_if<HasHeader<se2_grid_msgs::SE2Grid>>::type>
{
  static ros::Time* pointer(se2_grid_msgs::SE2Grid& m)
  {
    return &m.info.header.stamp;
  }

  static ros::Time const* pointer(const se2_grid_msgs::SE2Grid& m)
  {
    return &m.info.header.stamp;
  }

  static ros::Time value(const se2_grid_msgs::SE2Grid& m)
  {
    return m.info.header.stamp;
  }
};

}
}