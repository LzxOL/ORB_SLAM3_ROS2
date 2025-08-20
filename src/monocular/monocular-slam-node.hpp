#ifndef __MONOCULAR_SLAM_NODE_HPP__
#define __MONOCULAR_SLAM_NODE_HPP__

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Context.hpp"
#include "libobsensor/hpp/Device.hpp"
#include "libobsensor/hpp/StreamProfile.hpp"
#include "libobsensor/hpp/Sensor.hpp"
#include "libobsensor/ObSensor.hpp"

#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "utility.hpp"

#define camera_width 1280
#define camera_height 720
#define camera_fps 30

class MonocularSlamNode : public rclcpp::Node
{
public:
    MonocularSlamNode(ORB_SLAM3::System* pSLAM);

    ~MonocularSlamNode();

private:
    using ImageMsg = sensor_msgs::msg::Image;

    void GrabImage();

    ORB_SLAM3::System* m_SLAM;

    cv_bridge::CvImagePtr m_cvImPtr;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_image_subscriber;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::string map_frame_id_;
    std::string base_frame_id_;

    std::shared_ptr<ob::Pipeline> color_pipe_;
    std::shared_ptr<ob::Pipeline> depth_pipe_;
    std::shared_ptr<ob::Pipeline> pipeline;
    ob::PointCloudFilter point_cloud_filter;
    int color_frame_count_;
    int depth_frame_count_;
    rclcpp::Time last_fps_time_;
    rclcpp::TimerBase::SharedPtr fps_timer_;

    std::thread slam_thread_;
    std::atomic<bool> running_{false};
};

#endif
