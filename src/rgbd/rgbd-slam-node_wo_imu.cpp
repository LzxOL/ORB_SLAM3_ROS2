#include "rgbd-slam-node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>

using namespace std::chrono_literals;

RgbdSlamNode::RgbdSlamNode(ORB_SLAM3::System* pSLAM)
: Node("ORB_SLAM3_ROS2"),
  m_SLAM(pSLAM),
  running_(false)
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

    // 初始化订阅器（保留IMU和Odometry订阅）
    auto qos = rclcpp::QoS(rclcpp::KeepLast(200))
                .reliable()
                .durability_volatile();

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/tita4264886/imu_sensor_broadcaster/imu",
        qos,
        std::bind(&RgbdSlamNode::GrabIMU, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/tita4264886/chassis/odometry", 10, std::bind(&RgbdSlamNode::GrabOdom, this, std::placeholders::_1));
    
    // 初始化 TF 广播器
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    
    // 初始化位姿连续性相关变量
    has_valid_pose_ = false;
    last_valid_pose_time_ = 0.0;
    velocity_update_time_ = 0.0;
    use_pose_prediction_ = true;
    velocity_ = Eigen::Vector3f::Zero();
    last_valid_pose_ = Sophus::SE3f();

    // 初始化Orbbec相机管道
    try {
        ob::Context ctx;
        auto devices = ctx.queryDeviceList();
        if (devices->deviceCount() == 0) {
            RCLCPP_ERROR(this->get_logger(), "No Orbbec devices found!");
            throw std::runtime_error("No Orbbec devices found");
        }

        auto device = devices->getDevice(0);
        
        // 彩色图管道
        color_pipe_ = std::make_shared<ob::Pipeline>(device);
        auto color_profiles = color_pipe_->getStreamProfileList(OB_SENSOR_COLOR);
        auto color_profile = color_profiles->getVideoStreamProfile(640, 480, OB_FORMAT_RGB, 30);
        auto color_config = std::make_shared<ob::Config>();
        color_config->enableStream(color_profile);
        color_pipe_->start(color_config);

        // 深度图管道
        depth_pipe_ = std::make_shared<ob::Pipeline>(device);
        auto depth_profiles = depth_pipe_->getStreamProfileList(OB_SENSOR_DEPTH);
        auto depth_profile = depth_profiles->getVideoStreamProfile(640, 480, OB_FORMAT_Y16, 30);
        auto depth_config = std::make_shared<ob::Config>();
        depth_config->enableStream(depth_profile);
        depth_pipe_->start(depth_config);

        // 启动SLAM处理线程
        running_ = true;
        slam_thread_ = std::thread(&RgbdSlamNode::ProcessFrame, this);

        // 启动帧获取线程（可以在这里添加，或者像之前示例那样单独的线程函数）
    } catch (const ob::Error &e) {
        RCLCPP_ERROR(this->get_logger(), "Orbbec initialization failed: %s", e.what());
        throw;
    }
}

RgbdSlamNode::~RgbdSlamNode()
{
    // 停止线程
    if (running_) {
        running_ = false;
        frame_cv_.notify_all();
        if (slam_thread_.joinable()) {
            slam_thread_.join();
        }
    }

    // 停止相机管道
    if (color_pipe_) {
        color_pipe_->stop();
    }
    if (depth_pipe_) {
        depth_pipe_->stop();
    }

    // 关闭SLAM系统
    if (m_SLAM) {
        m_SLAM->Shutdown();
        m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    }
}

void RgbdSlamNode::ProcessFrame()
{
    while (running_) {
        try {
            // 获取彩色帧
            auto color_frame_set = color_pipe_->waitForFrames(100);
            if (!color_frame_set) continue;
            auto color_frame = color_frame_set->colorFrame();
            if (!color_frame) continue;

            // 获取深度帧
            auto depth_frame_set = depth_pipe_->waitForFrames(100);
            if (!depth_frame_set) continue;
            auto depth_frame = depth_frame_set->depthFrame();
            if (!depth_frame) continue;

            // 转换为OpenCV格式
            cv::Mat color(cv::Size(640, 480), CV_8UC3, (void*)color_frame->data(), cv::Mat::AUTO_STEP);
            cv::Mat depth(cv::Size(640, 480), CV_16UC1, (void*)depth_frame->data(), cv::Mat::AUTO_STEP);

            // 复制到共享数据
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                color.copyTo(latest_color_frame_);
                depth.copyTo(latest_depth_frame_);
                latest_frame_timestamp_ = Utility::StampToSec(this->now());
                has_new_frame_ = true;
            }
            frame_cv_.notify_one();

            // 处理IMU数据（与帧同步）
            std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
            double t_frame = latest_frame_timestamp_;
            
            {
                std::lock_guard<std::mutex> lock(imu_mutex_);
                // 这里可以添加IMU数据同步逻辑
                // 例如收集从上一帧到当前帧之间的IMU数据
            }

            // 等待SLAM处理完成或新帧到达
            std::unique_lock<std::mutex> lock(frame_mutex_);
            frame_cv_.wait(lock, [this] { return !running_ || has_new_frame_; });
            
            if (!running_) break;

            // 处理最新帧
            if (has_new_frame_) {
                has_new_frame_ = false;
                
                // 确保帧有效
                if (latest_color_frame_.empty() || latest_depth_frame_.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Empty RGB or Depth frame!");
                    continue;
                }

                // 调用SLAM处理
                Sophus::SE3f Tcw;
                if (vImuMeas.empty()) {
                    // 纯视觉模式
                    Tcw = m_SLAM->TrackRGBD(latest_color_frame_, latest_depth_frame_, t_frame);
                } else {
                    // 视觉-惯性模式
                    Tcw = m_SLAM->TrackRGBD(latest_color_frame_, latest_depth_frame_, t_frame, vImuMeas);
                }

                // 检查位姿有效性
                if (!Tcw.matrix().allFinite() || Tcw.matrix().isApprox(Eigen::Matrix4f::Identity())) {
                    RCLCPP_WARN(this->get_logger(), "Invalid or identity pose from SLAM");
                    continue;
                }

                // 发布TF变换
                Sophus::SE3f Twc = Tcw.inverse();
                Eigen::Matrix4f Tbc_matrix = Eigen::Matrix4f::Identity();
                Sophus::SE3f Tbc(Tbc_matrix);
                Sophus::SE3f Twb = Twc * Tbc.inverse();

                Eigen::Matrix3f R = Twb.rotationMatrix();
                Eigen::Vector3f t = Twb.translation();
                Eigen::Quaternionf q_eigen(R);

                geometry_msgs::msg::TransformStamped tf_msg;
                tf_msg.header.stamp = rclcpp::Clock().now(); // 使用当前ROS时间
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

                // 打印位姿信息
                Eigen::Vector3f Tcwt = Tcw.translation();
                RCLCPP_INFO(this->get_logger(), "Current pose: [%.3f, %.3f, %.3f]", 
                           Tcwt.x(), Tcwt.y(), Tcwt.z());
            }
        }  catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in ProcessFrame: %s", e.what());
            std::this_thread::sleep_for(100ms);
        }
    }
}

// 保留原有的GrabOdom和GrabIMU实现
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
        RCLCPP_WARN(this->get_logger(), "Invalid IMU data detected, skipping");
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
            RCLCPP_WARN(this->get_logger(), "IMU timestamp out of order: current=%.6f, last=%.6f", 
                       imu_timestamp, imu_queue_.back().t);
        }
        
        imu_queue_.push_back(imu_point);
        
        // 增加队列大小限制
        if (imu_queue_.size() > 2000) {
            imu_queue_.pop_front();
        }
    }
}