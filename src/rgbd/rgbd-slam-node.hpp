#ifndef __RGBD_SLAM_NODE_HPP__
#define __RGBD_SLAM_NODE_HPP__

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <memory>
#include <string>
#include <mutex>
#include <deque>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

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

class RgbdSlamNode : public rclcpp::Node
{
public:
    explicit RgbdSlamNode(ORB_SLAM3::System* pSLAM);
    ~RgbdSlamNode();

private:
    using ImageMsg = sensor_msgs::msg::Image;
    using approximate_sync_policy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;

    void GrabRGBD(const ImageMsg::SharedPtr msgRGB, const ImageMsg::SharedPtr msgD);
    void GrabOdom(nav_msgs::msg::Odometry::SharedPtr msg);
    void GrabIMU(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
    void publishTransform(const Sophus::SE3f& Tcw, const builtin_interfaces::msg::Time& timestamp);

    std::deque<ORB_SLAM3::IMU::Point> imu_queue_;
    std::mutex imu_queue_mutex_;
    ORB_SLAM3::System* m_SLAM;

    cv_bridge::CvImageConstPtr cv_ptrRGB;
    cv_bridge::CvImageConstPtr cv_ptrD;

    std::shared_ptr<message_filters::Subscriber<ImageMsg>> rgb_sub;
    std::shared_ptr<message_filters::Subscriber<ImageMsg>> depth_sub;
    std::shared_ptr<message_filters::Synchronizer<approximate_sync_policy>> syncApproximate;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // 坐标系名称由外部参数动态加载
    std::string map_frame_id_;
    std::string base_frame_id_;
    
    // IMU和里程计订阅器
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    
    // 里程计数据
    nav_msgs::msg::Odometry latest_odom_msg_;
    std::mutex odom_mutex_;
    
    // 位姿连续性保持相关
    Sophus::SE3f last_valid_pose_;           // 最后一个有效的位姿
    bool has_valid_pose_;                    // 是否有有效位姿
    double last_valid_pose_time_;            // 最后有效位姿的时间戳
    std::mutex pose_mutex_;                  // 位姿数据保护锁
    
    // 位姿预测相关
    Eigen::Vector3f velocity_;               // 当前速度估计
    double velocity_update_time_;            // 速度更新时间
    bool use_pose_prediction_;               // 是否使用位姿预测
    double max_prediction_time_;             // 最大预测时间
    
    // 位姿连续性相关方法
    void updatePoseContinuity(const Sophus::SE3f& current_pose, double timestamp);
    Sophus::SE3f predictPose(double current_time);
    bool isPoseValid(const Sophus::SE3f& pose);
};

#endif // __RGBD_SLAM_NODE_HPP__