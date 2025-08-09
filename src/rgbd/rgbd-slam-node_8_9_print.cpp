#include "rgbd-slam-node.hpp"

#include <opencv2/core/core.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <sensor_msgs/msg/imu.hpp>    
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h> 
#include <cmath>
#include <algorithm>
#include <vector>
using std::placeholders::_1;

rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

sensor_msgs::msg::Imu latest_imu_msg_;
std::mutex imu_mutex_;
nav_msgs::msg::Odometry latest_odom_msg_;
std::mutex odom_mutex_;
// 定义 IMU 队列
std::deque<ORB_SLAM3::IMU::Point> imu_queue_;
RgbdSlamNode::RgbdSlamNode(ORB_SLAM3::System* pSLAM)
: Node("ORB_SLAM3_ROS2"),
  m_SLAM(pSLAM)
{
    // 声明参数
    this->declare_parameter<std::string>("map_frame_id", "map");
    this->declare_parameter<std::string>("base_frame_id", "tita4264886/base_link");
    this->declare_parameter<bool>("use_pose_prediction", true);
    this->declare_parameter<double>("max_prediction_time", 1.0);

    // 获取参数
    this->get_parameter("map_frame_id", map_frame_id_);
    this->get_parameter("base_frame_id", base_frame_id_);
    this->get_parameter("use_pose_prediction", use_pose_prediction_);
    
    double max_prediction_time;
    this->get_parameter("max_prediction_time", max_prediction_time);
    max_prediction_time_ = max_prediction_time;

    // 初始化订阅器
    rgb_sub = std::make_shared<message_filters::Subscriber<ImageMsg>>(this, "/camera/color/image_raw");
    depth_sub = std::make_shared<message_filters::Subscriber<ImageMsg>>(this, "/camera/depth/image_raw");

    syncApproximate = std::make_shared<message_filters::Synchronizer<approximate_sync_policy>>(
        approximate_sync_policy(10), *rgb_sub, *depth_sub);
    syncApproximate->registerCallback(&RgbdSlamNode::GrabRGBD, this);

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/tita4264886/imu_sensor_broadcaster/imu",
        rclcpp::SensorDataQoS(),
        std::bind(&RgbdSlamNode::GrabIMU, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/tita4264886/chassis/odometry", 10, std::bind(&RgbdSlamNode::GrabOdom, this, _1));
        
    // 初始化 TF 广播器
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    
    // 初始化位姿连续性相关变量
    has_valid_pose_ = false;
    last_valid_pose_time_ = 0.0;
    velocity_update_time_ = 0.0;
    use_pose_prediction_ = true;  // 启用位姿预测
    velocity_ = Eigen::Vector3f::Zero();
    last_valid_pose_ = Sophus::SE3f();  // 初始化为单位变换
}

RgbdSlamNode::~RgbdSlamNode()
{
    m_SLAM->Shutdown();
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

// void RgbdSlamNode::GrabRGBD(const ImageMsg::SharedPtr msgRGB, const ImageMsg::SharedPtr msgD)
// {
//     try {
//         cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
//         cv_ptrD = cv_bridge::toCvShare(msgD);
//     } catch (cv_bridge::Exception& e) {
//         RCLCPP_DEBUG(this->get_logger(), "cv_bridge exception: %s", e.what());
//         return;
//     }

//     if (cv_ptrRGB->image.empty() || cv_ptrD->image.empty()) {
//         RCLCPP_DEBUG(this->get_logger(), "Empty RGB or Depth image, skipping frame.");
//         return;
//     }

//     Sophus::SE3f Tcw = m_SLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, Utility::StampToSec(msgRGB->header.stamp));
//     if (!Tcw.matrix().allFinite() || Tcw.matrix().isApprox(Eigen::Matrix4f::Identity())) {
//         RCLCPP_DEBUG(this->get_logger(), "Invalid or identity pose from SLAM, skipping frame.");
//         return;
//     }
//     Eigen::Vector3f Tcwt = Tcw.translation();
//     RCLCPP_DEBUG(this->get_logger(), "Tcwt Current pose: [%.3f, %.3f, %.3f]", Tcwt.x(), Tcwt.y(), Tcwt.z());
//     Sophus::SE3f Twc = Tcw.inverse();
//     Eigen::Vector3f Twct = Twc.translation();
//     RCLCPP_DEBUG(this->get_logger(), "Twct Current pose: [%.3f, %.3f, %.3f]", Twct.x(), Twct.y(), Twct.z());
//     // 相机 → 机体外参（Tbc：OpenCV 相机坐标系 → ROS base_link）
//     Eigen::Matrix4f Tbc_matrix = Eigen::Matrix4f::Identity();
    
//     // 平移部分（根据实际相机安装位置调整）
//     Tbc_matrix(0, 3) = 0.0f;   // X 方向平移（前）
//     Tbc_matrix(1, 3) = 0.0f;   // Y 方向平移（左）
//     Tbc_matrix(2, 3) = 0.1f;  // Z 方向平移（上）

//     // 旋转部分：OpenCV 相机坐标系 → ROS base_link
//     // X_camera = Y_body (right -> left, so -Y)
//     // Y_camera = Z_body (down -> up, so -Z)
//     // Z_camera = X_body (forward -> forward)
//     Tbc_matrix(0, 0) =  0.0f;  Tbc_matrix(0, 1) = -1.0f;  Tbc_matrix(0, 2) =  0.0f;
//     Tbc_matrix(1, 0) =  0.0f;  Tbc_matrix(1, 1) =  0.0f;  Tbc_matrix(1, 2) = -1.0f;
//     Tbc_matrix(2, 0) =  1.0f;  Tbc_matrix(2, 1) =  0.0f;  Tbc_matrix(2, 2) =  0.0f;

//     Sophus::SE3f Tbc(Tbc_matrix);
//     Sophus::SE3f Twb = Twc * Tbc;  // 正确的变换顺序：Twb = Twc * Tbc

//     Eigen::Matrix3f R = Twb.rotationMatrix();
//     Eigen::Vector3f t = Twb.translation();
//     Eigen::Quaternionf q_eigen(R);

//     geometry_msgs::msg::TransformStamped tf_msg;
//     tf_msg.header.stamp = msgRGB->header.stamp;
//     tf_msg.header.frame_id = map_frame_id_;
//     tf_msg.child_frame_id = base_frame_id_;

//     tf_msg.transform.translation.x = t.x();
//     tf_msg.transform.translation.y = t.y();
//     tf_msg.transform.translation.z = t.z();
//     tf_msg.transform.rotation.x = q_eigen.x();
//     tf_msg.transform.rotation.y = q_eigen.y();
//     tf_msg.transform.rotation.z = q_eigen.z();
//     tf_msg.transform.rotation.w = q_eigen.w();

//     tf_broadcaster_->sendTransform(tf_msg);
// }

// ori  
// void RgbdSlamNode::GrabRGBD(const ImageMsg::SharedPtr msgRGB, const ImageMsg::SharedPtr msgD)
// {
//     try {
//         cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
//         cv_ptrD = cv_bridge::toCvShare(msgD);
//     } catch (cv_bridge::Exception& e) {
//         RCLCPP_DEBUG(this->get_logger(), "cv_bridge exception: %s", e.what());
//         return;
//     }

//     if (cv_ptrRGB->image.empty() || cv_ptrD->image.empty()) {
//         RCLCPP_DEBUG(this->get_logger(), "Empty RGB or Depth image, skipping frame.");
//         return;
//     }
    
//     Sophus::SE3f Tcw = m_SLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, Utility::StampToSec(msgRGB->header.stamp));
//     if (!Tcw.matrix().allFinite() || Tcw.matrix().isApprox(Eigen::Matrix4f::Identity())) {
//         RCLCPP_DEBUG(this->get_logger(), "Invalid or identity pose from SLAM, skipping frame.");
//         return;
//     }
//     Eigen::Vector3f Tcwt = Tcw.translation();
//     RCLCPP_DEBUG(this->get_logger(), "Tcwt Current pose: [%.3f, %.3f, %.3f]", Tcwt.x(), Tcwt.y(), Tcwt.z());
//     Sophus::SE3f Twc = Tcw.inverse();

//     // // 相机 → 机体外参（添加180度旋转修正朝向）
//     Eigen::Matrix4f Tbc_matrix = Eigen::Matrix4f::Identity();
//     // Tbc_matrix(0, 3) = 0.1f;
//     // Tbc_matrix(2, 3) = 0.05f;

//     // // 添加180度旋转（绕Z轴）
//     // Tbc_matrix(0, 0) = -1.0f;  // cos(180°) = -1
//     // Tbc_matrix(0, 1) =  0.0f;  // -sin(180°) = 0
//     // Tbc_matrix(1, 0) =  0.0f;  // sin(180°) = 0
//     // Tbc_matrix(1, 1) = -1.0f;  // cos(180°) = -1

//     Sophus::SE3f Tbc(Tbc_matrix);

//     Sophus::SE3f Twb = Twc * Tbc.inverse();

//     Eigen::Matrix3f R = Twb.rotationMatrix();
//     Eigen::Vector3f t = Twb.translation();
//     Eigen::Quaternionf q_eigen(R);

//     geometry_msgs::msg::TransformStamped tf_msg;
//     tf_msg.header.stamp = msgRGB->header.stamp;
//     tf_msg.header.frame_id = map_frame_id_;
//     tf_msg.child_frame_id = base_frame_id_;

//     tf_msg.transform.translation.x = t.x();
//     tf_msg.transform.translation.y = t.y();
//     tf_msg.transform.translation.z = t.z();
//     tf_msg.transform.rotation.x = q_eigen.x();
//     tf_msg.transform.rotation.y = q_eigen.y();
//     tf_msg.transform.rotation.z = q_eigen.z();
//     tf_msg.transform.rotation.w = q_eigen.w();

//     tf_broadcaster_->sendTransform(tf_msg);
// }


void RgbdSlamNode::GrabRGBD(const ImageMsg::SharedPtr msgRGB, const ImageMsg::SharedPtr msgD)
{
    try {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
        cv_ptrD = cv_bridge::toCvShare(msgD);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_DEBUG(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    if (cv_ptrRGB->image.empty() || cv_ptrD->image.empty()) {
        RCLCPP_DEBUG(this->get_logger(), "Empty RGB or Depth image, skipping frame.");
        return;
    }

    // 提取 IMU 数据用于 SLAM 融合
    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    double t_frame = Utility::StampToSec(msgRGB->header.stamp);
    static double last_frame_time = -1.0;  // 初始化为-1，确保第一帧能正常工作
    static bool first_frame = true;
    
    {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        
        // 调试信息：显示队列状态
        if (imu_queue_.size() > 0) {
            double oldest_imu_time = imu_queue_.front().t;
            double newest_imu_time = imu_queue_.back().t;
            RCLCPP_DEBUG(this->get_logger(), "IMU queue size: %lu, oldest: %.6f, newest: %.6f, frame: %.6f", 
                        imu_queue_.size(), oldest_imu_time, newest_imu_time, t_frame);
        }
        
        if (first_frame) {
            // 第一帧：收集当前时间戳之前的所有IMU数据（用于初始化）
            for (const auto& imu_point : imu_queue_) {
                if (imu_point.t <= t_frame) {
                    vImuMeas.push_back(imu_point);
                }
            }
            first_frame = false;
            RCLCPP_DEBUG(this->get_logger(), "First frame: collected %lu IMU measurements", vImuMeas.size());
        } else {
            // 后续帧：收集从上一帧到当前帧之间的IMU数据
            for (const auto& imu_point : imu_queue_) {
                if (imu_point.t > last_frame_time && imu_point.t <= t_frame) {
                    vImuMeas.push_back(imu_point);
                }
            }
            RCLCPP_DEBUG(this->get_logger(), "Frame: collected %lu IMU measurements between %.6f and %.6f", 
                       vImuMeas.size(), last_frame_time, t_frame);
            
            // 检查时间跨度是否足够
            if (!vImuMeas.empty()) {
                double time_span = vImuMeas.back().t - vImuMeas.front().t;
                RCLCPP_DEBUG(this->get_logger(), "IMU time span: %.3f ms", time_span * 1000);
                
                // 如果时间跨度太短，可能预积分会有问题
                if (time_span < 0.005) {  // 少于5ms
                    RCLCPP_DEBUG(this->get_logger(), "IMU time span too short (%.1f ms), may cause issues", time_span * 1000);
                }
            }
        }
        
        // 非常保守的队列清理：只移除非常旧的数据（保留最近2秒）
        double keep_threshold = t_frame - 2.0;
        size_t removed_count = 0;
        while (!imu_queue_.empty() && imu_queue_.front().t < keep_threshold) {
            imu_queue_.pop_front();
            removed_count++;
        }
        if (removed_count > 0) {
            RCLCPP_DEBUG(this->get_logger(), "Removed %lu old IMU measurements", removed_count);
        }
    }
    
    last_frame_time = t_frame;

    // 检查IMU数据质量和数量
    if (vImuMeas.empty()) {
        RCLCPP_DEBUG(this->get_logger(), "No IMU data available, skipping frame to avoid crash");
        // 在IMU-RGBD模式下，不能调用不带IMU参数的TrackRGBD
        // 这会导致内部状态不一致和段错误
        // 直接跳过这一帧，等待IMU数据
        return;
    }
    
    // 检查IMU数据的最小要求（需要至少2个测量值进行预积分）
    if (vImuMeas.size() < 2) {
        RCLCPP_DEBUG(this->get_logger(), "Insufficient IMU data (%lu measurements, need at least 2), skipping frame", vImuMeas.size());
        // IMU预积分需要至少2个测量值
        return;
    }

    // 验证IMU数据质量
    for (const auto& imu : vImuMeas) {
        if (!std::isfinite(imu.a(0)) || !std::isfinite(imu.a(1)) || !std::isfinite(imu.a(2)) ||
            !std::isfinite(imu.w(0)) || !std::isfinite(imu.w(1)) || !std::isfinite(imu.w(2))) {
            RCLCPP_DEBUG(this->get_logger(), "Invalid IMU data in measurements, skipping frame to avoid crash");
            // 不能在IMU-RGBD模式下调用不带IMU的TrackRGBD
            return;
        }
    }

    RCLCPP_DEBUG(this->get_logger(), "Processing frame with %lu IMU measurements", vImuMeas.size());

    // 使用 IMU 融合的位置估计
    Sophus::SE3f Tcw;
    try {
        RCLCPP_DEBUG(this->get_logger(), "Calling TrackRGBD with IMU data...");
        Tcw = m_SLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, t_frame, vImuMeas);
        RCLCPP_DEBUG(this->get_logger(), "TrackRGBD completed successfully");
    } catch (const std::exception& e) {
        RCLCPP_DEBUG(this->get_logger(), "IMU-Visual SLAM failed: %s", e.what());
        // 在IMU-RGBD模式下不能降级，直接跳过
        return;
    } catch (...) {
        RCLCPP_DEBUG(this->get_logger(), "IMU-Visual SLAM failed with unknown exception");
        return;
    }
    // 使用新的位姿有效性检查和连续性保持逻辑
    Sophus::SE3f final_pose;
    bool slam_pose_valid = isPoseValid(Tcw);
    
    if (slam_pose_valid) {
        // SLAM位姿有效，使用SLAM结果
        final_pose = Tcw;
        updatePoseContinuity(Tcw, t_frame);
        
        Eigen::Vector3f Tcwt = Tcw.translation();
        RCLCPP_DEBUG(this->get_logger(), "[SLAM Valid] Tcwt: [%.3f, %.3f, %.3f]",
                    Tcwt.x(), Tcwt.y(), Tcwt.z());
    } else {
        // SLAM位姿无效，使用预测位姿保持连续性
        if (has_valid_pose_) {
            final_pose = predictPose(t_frame);
            
            Eigen::Vector3f predicted_t = final_pose.translation();
            RCLCPP_DEBUG(this->get_logger(), "[SLAM Lost] Using predicted pose: [%.3f, %.3f, %.3f]",
                       predicted_t.x(), predicted_t.y(), predicted_t.z());
        } else {
            // 没有历史位姿，跳过这一帧
            RCLCPP_DEBUG(this->get_logger(), "No valid SLAM pose and no history, skipping frame.");
            return;
        }
    }

    // 始终输出连续的世界坐标位姿
    Eigen::Vector3f world_position = final_pose.translation();
    RCLCPP_INFO(this->get_logger(), "[World Continuous] Position: [%.3f, %.3f, %.3f]",
                world_position.x(), world_position.y(), world_position.z());

    // 发布变换
    publishTransform(final_pose, msgRGB->header.stamp);
}

void RgbdSlamNode::publishTransform(const Sophus::SE3f& Tcw, const builtin_interfaces::msg::Time& timestamp)
{
    Sophus::SE3f Twc = Tcw.inverse();

    // 相机→机体外参（Tbc）
    Eigen::Matrix4f Tbc_matrix = Eigen::Matrix4f::Identity();
    Sophus::SE3f Tbc(Tbc_matrix);
    Sophus::SE3f Twb = Twc * Tbc.inverse();

    // 提取变换并发布 TF
    Eigen::Matrix3f R = Twb.rotationMatrix();
    Eigen::Vector3f t = Twb.translation();
    Eigen::Quaternionf q_eigen(R);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = timestamp;
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
}
// Odometry 接收器，更新最新位姿
void RgbdSlamNode::GrabOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(odom_mutex_);
    latest_odom_msg_ = *msg;
}

void RgbdSlamNode::GrabIMU(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
    double imu_timestamp = Utility::StampToSec(msg->header.stamp);
    
    // 检查IMU数据有效性
    if (!std::isfinite(msg->linear_acceleration.x) || !std::isfinite(msg->linear_acceleration.y) || !std::isfinite(msg->linear_acceleration.z) ||
        !std::isfinite(msg->angular_velocity.x) || !std::isfinite(msg->angular_velocity.y) || !std::isfinite(msg->angular_velocity.z)) {
        RCLCPP_DEBUG(this->get_logger(), "Invalid IMU data detected, skipping");
        return;
    }
    
    // 正确的IMU::Point构造：加速度、角速度、时间戳
    cv::Point3f acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    cv::Point3f gyr(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    ORB_SLAM3::IMU::Point imu_point(acc, gyr, imu_timestamp);
    
    {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        
        // 检查时间戳顺序
        if (!imu_queue_.empty() && imu_timestamp <= imu_queue_.back().t) {
            RCLCPP_DEBUG(this->get_logger(), "IMU timestamp out of order: current=%.6f, last=%.6f", 
                       imu_timestamp, imu_queue_.back().t);
        }
        
        imu_queue_.push_back(imu_point);
        
        // 增加队列大小限制，确保有足够的历史数据
        if (imu_queue_.size() > 2000) {  // 从2000增加到5000
            imu_queue_.pop_front();
        }
        
        // 定期输出IMU接收状态（每100个IMU消息输出一次）
        static int imu_count = 0;
        if (++imu_count % 100 == 0) {
            RCLCPP_DEBUG(this->get_logger(), "IMU queue size: %lu, latest timestamp: %.6f", 
                        imu_queue_.size(), imu_timestamp);
        }
    }
}

// 检查位姿是否有效
bool RgbdSlamNode::isPoseValid(const Sophus::SE3f& pose)
{
    // 检查矩阵是否有限
    if (!pose.matrix().allFinite()) {
        return false;
    }
    
    // 检查是否为单位矩阵（表示无效位姿）
    if (pose.matrix().isApprox(Eigen::Matrix4f::Identity(), 1e-6f)) {
        return false;
    }
    
    // 检查平移部分是否合理（避免异常大的值）
    Eigen::Vector3f translation = pose.translation();
    float max_translation = 1000.0f;  // 最大平移距离（米）
    if (translation.norm() > max_translation) {
        return false;
    }
    
    // 检查旋转矩阵是否正交
    Eigen::Matrix3f R = pose.rotationMatrix();
    if (std::abs(R.determinant() - 1.0f) > 0.1f) {
        return false;
    }
    
    return true;
}

// 更新位姿连续性
void RgbdSlamNode::updatePoseContinuity(const Sophus::SE3f& current_pose, double timestamp)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    
    if (has_valid_pose_) {
        // 计算速度
        double dt = timestamp - last_valid_pose_time_;
        if (dt > 0.001) {  // 避免除零
            Eigen::Vector3f current_translation = current_pose.translation();
            Eigen::Vector3f last_translation = last_valid_pose_.translation();
            velocity_ = (current_translation - last_translation) / dt;
            velocity_update_time_ = timestamp;
            
            // 限制速度的合理范围（避免异常值）
            float max_velocity = 10.0f;  // 最大速度 10 m/s
            if (velocity_.norm() > max_velocity) {
                velocity_ = velocity_.normalized() * max_velocity;
            }
        }
    }
    
    // 更新最后有效位姿
    last_valid_pose_ = current_pose;
    last_valid_pose_time_ = timestamp;
    has_valid_pose_ = true;
}

// 预测当前位姿
Sophus::SE3f RgbdSlamNode::predictPose(double current_time)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    
    if (!has_valid_pose_ || !use_pose_prediction_) {
        return last_valid_pose_;
    }
    
    double dt = current_time - last_valid_pose_time_;
    
    // 如果时间差太大，不进行预测
    if (dt > max_prediction_time_) {
        return last_valid_pose_;
    }
    
    // 使用恒速模型预测位置
    Eigen::Vector3f predicted_translation = last_valid_pose_.translation() + velocity_ * dt;
    
    // 保持旋转不变（可以根据需要添加角速度预测）
    Eigen::Matrix3f predicted_rotation = last_valid_pose_.rotationMatrix();
    
    // 构造预测位姿
    Eigen::Matrix4f predicted_matrix = Eigen::Matrix4f::Identity();
    predicted_matrix.block<3,3>(0,0) = predicted_rotation;
    predicted_matrix.block<3,1>(0,3) = predicted_translation;
    
    return Sophus::SE3f(predicted_matrix);
}