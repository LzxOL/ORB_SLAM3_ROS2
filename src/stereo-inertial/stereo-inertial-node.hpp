#ifndef __STEREO_INERTIAL_NODE_HPP__
#define __STEREO_INERTIAL_NODE_HPP__

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"


#include <cv_bridge/cv_bridge.h>

#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "utility.hpp"

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp> 
#include "nav_msgs/msg/odometry.hpp"

using ImuMsg = sensor_msgs::msg::Imu;
using ImageMsg = sensor_msgs::msg::Image;

class StereoInertialNode : public rclcpp::Node
{
public:
    StereoInertialNode(ORB_SLAM3::System* pSLAM, const string &strSettingsFile, const string &strDoRectify, const string &strDoEqual);
    ~StereoInertialNode();

private:
    void GrabImu(const ImuMsg::SharedPtr msg);
    void GrabImageLeft(const ImageMsg::SharedPtr msgLeft);
    void GrabImageRight(const ImageMsg::SharedPtr msgRight);
    cv::Mat GetImage(const ImageMsg::SharedPtr msg);
    void SyncWithImu();

    rclcpp::Subscription<ImuMsg>::SharedPtr   subImu_;
    rclcpp::Subscription<ImageMsg>::SharedPtr subImgLeft_;
    rclcpp::Subscription<ImageMsg>::SharedPtr subImgRight_;

    ORB_SLAM3::System *SLAM_;
    std::thread *syncThread_;

    // IMU
    queue<ImuMsg::SharedPtr> imuBuf_;
    std::mutex bufMutex_;

    // Image
    queue<ImageMsg::SharedPtr> imgLeftBuf_, imgRightBuf_;
    std::mutex bufMutexLeft_, bufMutexRight_;

    bool doRectify_;
    bool doEqual_;
    cv::Mat M1l_, M2l_, M1r_, M2r_;

    bool bClahe_;
    cv::Ptr<cv::CLAHE> clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));

       // 新增成员变量
    std::string map_frame_id_;
    std::string base_frame_id_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // 用于跟踪零位置状态和最后有效pose的成员变量
    bool has_printed_zero_warning_;
    bool last_pose_was_zero_;
    Eigen::Vector3f last_valid_translation_;
    Eigen::Quaternionf last_valid_rotation_;
    bool has_valid_pose_;

    void PublishTransform(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp, int tracking_state = -1);
    cv::Mat PreprocessImage(const ImageMsg::SharedPtr msg);
    std::vector<ORB_SLAM3::IMU::Point> ConvertImuMessages(
        const std::vector<ImuMsg::SharedPtr>& imu_msgs);
};

#endif
