#include "monocular-slam-node.hpp"

#include <opencv2/core/core.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

using std::placeholders::_1;

MonocularSlamNode::MonocularSlamNode(ORB_SLAM3::System* pSLAM)
:   Node("ORB_SLAM3_ROS2")
{
    m_SLAM = pSLAM;
    // 声明参数
    this->declare_parameter<std::string>("map_frame_id", "map");
    this->declare_parameter<std::string>("base_frame_id", "tita4264886/base_link");
    // 获取参数
    this->get_parameter("map_frame_id", map_frame_id_);
    this->get_parameter("base_frame_id", base_frame_id_);
    // 初始化TF广播器
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    m_image_subscriber = this->create_subscription<ImageMsg>(
        "/camera/color/image_raw",
        10,
        std::bind(&MonocularSlamNode::GrabImage, this, std::placeholders::_1));
    std::cout << "slam changed" << std::endl;
}

MonocularSlamNode::~MonocularSlamNode()
{
    // Stop all threads
    m_SLAM->Shutdown();

    // Save camera trajectory
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void MonocularSlamNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    // Copy the ros image message to cv::Mat.
    try
    {
        m_cvImPtr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    if (m_cvImPtr->image.empty()) {
        RCLCPP_WARN(this->get_logger(), "Empty image, skipping frame.");
        return;
    }
    Sophus::SE3f Tcw = m_SLAM->TrackMonocular(m_cvImPtr->image, Utility::StampToSec(msg->header.stamp));
    if (!Tcw.matrix().allFinite() || Tcw.matrix().isApprox(Eigen::Matrix4f::Identity())) {
        RCLCPP_WARN(this->get_logger(), "Invalid or identity pose from SLAM, skipping frame.");
        return;
    }
    Sophus::SE3f Twc = Tcw.inverse();
    Eigen::Vector3f tw = Twc.translation();
    RCLCPP_INFO(this->get_logger(), "TF: t = [%.3f, %.3f, %.3f]", tw.x(), tw.y(), tw.z());
    // 相机→机体外参（如无外参，单位矩阵）
    Eigen::Matrix4f Tbc_matrix = Eigen::Matrix4f::Identity();
    Sophus::SE3f Tbc(Tbc_matrix);
    Sophus::SE3f Twb = Twc * Tbc.inverse();
    Eigen::Matrix3f R = Twb.rotationMatrix();
    Eigen::Vector3f t = Twb.translation();
    Eigen::Quaternionf q_eigen(R);
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = msg->header.stamp;
    tf_msg.header.frame_id = map_frame_id_;
    tf_msg.child_frame_id = base_frame_id_;
    tf_msg.transform.translation.x = t.x();
    tf_msg.transform.translation.y = t.y();
    tf_msg.transform.translation.z = t.z();
    tf_msg.transform.rotation.x = q_eigen.x();
    tf_msg.transform.rotation.y = q_eigen.y();
    tf_msg.transform.rotation.z = q_eigen.z();
    tf_msg.transform.rotation.w = q_eigen.w();
    tf_broadcaster_->sendTransform(tf_msg);
    // RCLCPP_INFO(this->get_logger(), "TF: t = [%.3f, %.3f, %.3f], q = [%.3f, %.3f, %.3f, %.3f]", t.x(), t.y(), t.z(), q_eigen.x(), q_eigen.y(), q_eigen.z(), q_eigen.w());
}
