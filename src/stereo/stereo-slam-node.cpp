#include "stereo-slam-node.hpp"
#include <opencv2/core/core.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rmw/qos_profiles.h>

using std::placeholders::_1;
using std::placeholders::_2;

// 定义与相机发布者匹配的 QoS（message_filters::Subscriber 需要 rmw_qos_profile_t）
static rmw_qos_profile_t IMAGE_QOS = rmw_qos_profile_sensor_data;
// 强制使用 RELIABLE 策略
static_assert(std::is_same<decltype(IMAGE_QOS), rmw_qos_profile_t>::value, "IMAGE_QOS must be rmw_qos_profile_t");
  // 强制使用 RELIABLE 策略

StereoSlamNode::StereoSlamNode(ORB_SLAM3::System* pSLAM, const std::string& strSettingsFile, const std::string& strDoRectify)
    : Node("ORB_SLAM3_ROS2"), m_SLAM(pSLAM) {
    // 参数声明
    this->declare_parameter<std::string>("map_frame_id", "map");
    this->declare_parameter<std::string>("base_frame_id", "tita4264886/base_link");
    this->get_parameter("map_frame_id", map_frame_id_);
    this->get_parameter("base_frame_id", base_frame_id_);

    // 初始化 TF 广播器
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    // 解析是否进行图像校正
    std::stringstream ss(strDoRectify);
    ss >> std::boolalpha >> doRectify;

    // 加载相机参数
    if (doRectify) {
        cv::FileStorage fsSettings(strSettingsFile, cv::FileStorage::READ);
        if (!fsSettings.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open settings file: %s", strSettingsFile.c_str());
            throw std::runtime_error("Settings file not found");
        }

        // 加载双目参数
        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;
        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;
        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;
        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        // 初始化校正映射
        cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0,3).colRange(0,3),
                                   cv::Size(fsSettings["LEFT.width"], fsSettings["LEFT.height"]),
                                   CV_32F, M1l, M2l);
        cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0,3).colRange(0,3),
                                   cv::Size(fsSettings["RIGHT.width"], fsSettings["RIGHT.height"]),
                                   CV_32F, M1r, M2r);
    }

    // 初始化订阅者（显式指定 QoS）
    left_sub = std::make_shared<ImageSubType>(this, "camera/left/image_raw", IMAGE_QOS);
    right_sub = std::make_shared<ImageSubType>(this, "camera/right/image_raw", IMAGE_QOS);

    // 初始化同步策略
    syncApproximate = std::make_shared<message_filters::Synchronizer<approximate_sync_policy>>(
        approximate_sync_policy(10), *left_sub, *right_sub
    );
    syncApproximate->registerCallback(&StereoSlamNode::GrabStereo, this);
}

void StereoSlamNode::GrabStereo(const ImageMsg::SharedPtr msgLeft, const ImageMsg::SharedPtr msgRight) {
    try {
        cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
        cv_ptrRight = cv_bridge::toCvShare(msgRight);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    Sophus::SE3f Tcw;
    if (doRectify) {
        cv::Mat imLeft, imRight;
        cv::remap(cv_ptrLeft->image, imLeft, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(cv_ptrRight->image, imRight, M1r, M2r, cv::INTER_LINEAR);
        Tcw = m_SLAM->TrackStereo(imLeft, imRight, msgLeft->header.stamp.sec);
    } else {
        Tcw = m_SLAM->TrackStereo(cv_ptrLeft->image, cv_ptrRight->image, msgLeft->header.stamp.sec);
    }

    // 发布 TF
    if (Tcw.matrix().allFinite()) {
        PublishTransform(Tcw, msgLeft->header.stamp);
    }
}

void StereoSlamNode::PublishTransform(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp) {
    // 转换到世界坐标系
    Sophus::SE3f Twc = Tcw.inverse();
    Eigen::Vector3f t = Twc.translation();
    Eigen::Matrix3f R = Twc.rotationMatrix();
    Eigen::Quaternionf q(R);

    // 发布 TF
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = map_frame_id_;
    tf_msg.child_frame_id = base_frame_id_;
    tf_msg.transform.translation.x = t.x();
    tf_msg.transform.translation.y = t.y();
    tf_msg.transform.translation.z = t.z();
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(tf_msg);

    // 输出调试信息
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                         "Published TF: t=[%.3f, %.3f, %.3f]", t.x(), t.y(), t.z());
}