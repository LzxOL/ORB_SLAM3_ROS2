#ifndef __STEREO_SLAM_NODE_HPP__
#define __STEREO_SLAM_NODE_HPP__

#include <string>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "utility.hpp"

class StereoSlamNode : public rclcpp::Node
{
public:
    StereoSlamNode(ORB_SLAM3::System* pSLAM, const std::string &strSettingsFile, const std::string &strDoRectify);

    ~StereoSlamNode();

private:
    using ImageMsg = sensor_msgs::msg::Image;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> approximate_sync_policy;

    void GrabStereo(const sensor_msgs::msg::Image::SharedPtr msgRGB, const sensor_msgs::msg::Image::SharedPtr msgD);

    ORB_SLAM3::System* m_SLAM;


      // 图像订阅者（显式设置 QoS）
    using ImageSubType = message_filters::Subscriber<ImageMsg>;
    std::shared_ptr<ImageSubType> left_sub, right_sub;
    std::shared_ptr<message_filters::Synchronizer<approximate_sync_policy>> syncApproximate;
    
    bool doRectify;
    cv::Mat M1l,M2l,M1r,M2r;

    cv_bridge::CvImageConstPtr cv_ptrLeft;
    cv_bridge::CvImageConstPtr cv_ptrRight;

    // std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image> > left_sub;
    // std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image> > right_sub;

    // std::shared_ptr<message_filters::Synchronizer<approximate_sync_policy> > syncApproximate;

    std::string map_frame_id_;
    std::string base_frame_id_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
     // 用于跟踪零位置状态和最后有效pose的成员变量
    bool has_printed_zero_warning_;
    bool last_pose_was_zero_;
    Eigen::Vector3f last_valid_translation_;
    Eigen::Quaternionf last_valid_rotation_;
    bool has_valid_pose_;

    void PublishTransform(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp);
};

#endif
