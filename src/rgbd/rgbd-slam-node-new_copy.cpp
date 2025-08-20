#include "rgbd-slam-node.hpp"

#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/imu.hpp>  

#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Context.hpp"
#include "libobsensor/hpp/Device.hpp"
#include "libobsensor/hpp/StreamProfile.hpp"
#include "libobsensor/hpp/Sensor.hpp"
#include "libobsensor/ObSensor.hpp"
using namespace std::chrono_literals;

// 在 RgbdSlamNode 类中添加成员变量
std::atomic<bool> time_initialized_{false};
std::mutex time_mutex_;
double last_processed_timestamp_ = 0;
rclcpp::Time ros_start_time_;

std::atomic<bool> color_time_initialized_{false};
std::atomic<bool> depth_time_initialized_{false};
uint64_t imu_start_time_ns_ = 0;
uint64_t color_start_time_ns_ = 0;
uint64_t depth_start_time_ns_ = 0;
bool use_imu = true;
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

    // 初始化 TF 广播器
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    
    // 初始化位姿连续性相关变量
    has_valid_pose_ = false;
    last_valid_pose_time_ = 0.0;
    velocity_update_time_ = 0.0;
    use_pose_prediction_ = true;
    velocity_ = Eigen::Vector3f::Zero();
    last_valid_pose_ = Sophus::SE3f();

    ros_start_time_ = this->now();

    // 初始化Orbbec相机管道
    try {
        ob::Context ctx;
        std::shared_ptr<ob::DeviceList> devices = ctx.queryDeviceList();
        if (devices->deviceCount() == 0) {
            RCLCPP_ERROR(this->get_logger(), "No Orbbec devices found!");
            throw std::runtime_error("No Orbbec devices found");
        }

        auto device = devices->getDevice(0);

        // 配置相机参数
        if(device->isPropertySupported(OB_PROP_DEPTH_SOFT_FILTER_BOOL, OB_PERMISSION_WRITE)) {
            device->setBoolProperty(OB_PROP_DEPTH_SOFT_FILTER_BOOL, true);
        }
        device->switchDepthWorkMode("High Accuracy");
        if(device->isPropertySupported(OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, OB_PERMISSION_READ)) {
            bool isOpen = device->getBoolProperty(OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL);
            if(device->isPropertySupported(OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, OB_PERMISSION_WRITE)) {
                device->setBoolProperty(OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, !isOpen);
            }
        }
        if(device->isPropertySupported(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, OB_PERMISSION_READ)) {
            if(device->isPropertySupported(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, OB_PERMISSION_WRITE)) {
                device->setBoolProperty(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, true);
            }
        }
        if(device->isPropertySupported(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, OB_PERMISSION_READ)) {
            bool isOpen = device->getBoolProperty(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL);
            if(device->isPropertySupported(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, OB_PERMISSION_WRITE)) {
                device->setBoolProperty(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, !isOpen);
            }
        }

        // 初始化彩色图管道
        color_pipe_ = std::make_shared<ob::Pipeline>(device);
        auto color_profiles = color_pipe_->getStreamProfileList(OB_SENSOR_COLOR);
        auto color_profile = color_profiles->getVideoStreamProfile(848, 480, OB_FORMAT_RGB, 60);
        auto color_config = std::make_shared<ob::Config>();
        color_config->enableStream(color_profile);
        color_pipe_->start(color_config);

        // 初始化深度图管道
        depth_pipe_ = std::make_shared<ob::Pipeline>(device);
        auto depth_profiles = depth_pipe_->getStreamProfileList(OB_SENSOR_DEPTH);
        std::shared_ptr<ob::VideoStreamProfile> depth_profile = depth_profiles->getVideoStreamProfile(848, 480, OB_FORMAT_Y16, 60);
        auto depth_config = std::make_shared<ob::Config>();
        depth_config->enableStream(depth_profile);
        depth_pipe_->start(depth_config);

        auto sensorList = device->getSensorList();
        // 初始化IMU传感器并注册回调
        gyroSensor_ = sensorList->getSensor(OB_SENSOR_GYRO);
        if (gyroSensor_) {
        auto gyroProfiles = gyroSensor_->getStreamProfileList();
        if (gyroProfiles) {
            auto gyroProfile = gyroProfiles->getGyroStreamProfile(OB_GYRO_FS_1000dps,OB_SAMPLE_RATE_200_HZ);
            if (gyroProfile) {
                gyroSensor_->start(gyroProfile, [this](std::shared_ptr<ob::Frame> frame) {
                   ProcessGyroData(frame);   
                });
                RCLCPP_INFO(this->get_logger(), "Gyro sensor started");
            } else {
                RCLCPP_ERROR(this->get_logger(), "No valid gyro profile");
            }
        }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Gyro sensor not found");
        }
        // 注册IMU时间同步回调
        accelSensor_ = sensorList->getSensor(OB_SENSOR_ACCEL);
        if (accelSensor_) {
        auto accelProfiles = accelSensor_->getStreamProfileList();
        if (accelProfiles) {
            auto accelProfile = accelProfiles->getAccelStreamProfile(OB_ACCEL_FS_4g,OB_SAMPLE_RATE_200_HZ);//OB_SAMPLE_RATE_200_HZ OB_SAMPLE_RATE_1_KHZ
            if (accelProfile) {
                accelSensor_->start(accelProfile, [this](std::shared_ptr<ob::Frame> frame) {
                    ProcessAccelData(frame);
                });
                RCLCPP_INFO(this->get_logger(), "Accel sensor started");
            } else {
                RCLCPP_ERROR(this->get_logger(), "No valid accel profile");
            }
        }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Accel sensor not found");
        }
       
        RCLCPP_INFO(this->get_logger(), "Orbbec camera initialized successfully");

        // 启动SLAM处理线程
        running_ = true;
        slam_thread_ = std::thread(&RgbdSlamNode::ProcessFrame, this);

    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Orbbec initialization failed: %s", e.what());
        throw;
    }
}

RgbdSlamNode::~RgbdSlamNode()
{
    running_ = false;
    if (slam_thread_.joinable()) slam_thread_.join();
    
    if (color_pipe_) color_pipe_->stop();
    if (depth_pipe_) depth_pipe_->stop();
    
    if (gyroSensor_) gyroSensor_->stop();
    if (accelSensor_) accelSensor_->stop();

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
            if (!color_time_initialized_.load()) {
                if (!color_time_initialized_.load()) {
                    color_start_time_ns_ = color_frame->systemTimeStamp();
                    color_time_initialized_.store(true);
                    RCLCPP_INFO(this->get_logger(), "COLOR time synchronization initialized");
                }
            }
            // 获取深度帧
            auto depth_frame_set = depth_pipe_->waitForFrames(100);
            if (!depth_frame_set) continue;
            auto depth_frame = depth_frame_set->depthFrame();
            if (!depth_frame) continue;
            if (!depth_time_initialized_.load()) {
                if (!depth_time_initialized_.load()) {
                    depth_start_time_ns_ = depth_frame->systemTimeStamp();
                    depth_time_initialized_.store(true);
                    RCLCPP_INFO(this->get_logger(), "DEPTH time synchronization initialized");
                }
            }
            // 检查时间初始化
            if (!time_initialized_.load()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                    "Waiting for IMU data to initialize time...");
                continue;
            }

            // 计算时间戳
            double color_timestamp, depth_timestamp;
            {
                std::lock_guard<std::mutex> lock(time_mutex_);
                double color_ns = color_frame->systemTimeStamp() - color_start_time_ns_;
                color_timestamp = ros_start_time_.seconds() + color_ns * 1e-9;

                uint64_t depth_ns = depth_frame->systemTimeStamp() - depth_start_time_ns_;
                depth_timestamp = ros_start_time_.seconds() + depth_ns * 1e-9;
            }

            // 检查时间戳差异
            const double max_time_diff = 0.01; // 10ms
            if (std::abs(depth_timestamp - color_timestamp) > max_time_diff) {
                RCLCPP_INFO(this->get_logger(), "Timestamp mismatch - Color: %.6f, Depth: %.6f",
                           color_timestamp, depth_timestamp);
                continue;
            }

            // 使用深度图时间戳作为帧时间戳
            double frame_timestamp = max(depth_timestamp, color_timestamp);

            // 转换为OpenCV格式
            cv::Mat color(cv::Size(848, 480), CV_8UC3, (void*)color_frame->data(), cv::Mat::AUTO_STEP);
            cv::Mat depth(cv::Size(848, 480), CV_16UC1, (void*)depth_frame->data(), cv::Mat::AUTO_STEP);
            if (color.empty() || depth.empty()) {
                RCLCPP_WARN(this->get_logger(), "Empty image or depth frame!");
                continue;
            }
            Sophus::SE3f Tcw;
            if(use_imu){
                std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
                // 在ProcessFrame中提取IMU数据
                {
                    std::lock_guard<std::mutex> lock(imu_mutex_);
                    // 找到第一个时间戳 > last_processed_timestamp_ 的数据
                    auto it_start = std::lower_bound(
                        imu_buffer_.begin(), imu_buffer_.end(),
                        last_processed_timestamp_,
                        [](const ORB_SLAM3::IMU::Point& p, double t) { return p.t < t; }
                    );
                    // 找到最后一个时间戳 <= frame_timestamp 的数据
                    auto it_end = std::upper_bound(
                        it_start, imu_buffer_.end(),
                        frame_timestamp,
                        [](double t, const ORB_SLAM3::IMU::Point& p) { return t < p.t; }
                    );
                    // 提取有效范围内的数据
                    vImuMeas.assign(it_start, it_end);
                    // 记录原始大小
                    // size_t original_size = vImuMeas.size();
                    // // 扩大两倍：在末尾插入原始数据的副本
                    // vImuMeas.insert(vImuMeas.end(), vImuMeas.begin(), vImuMeas.begin() + original_size);
                    // original_size = vImuMeas.size();
                    // vImuMeas.insert(vImuMeas.end(), vImuMeas.begin(), vImuMeas.begin() + original_size);
                    // 从缓冲区中移除已处理的数据
                    imu_buffer_.erase(it_start, it_end);
                }
                if (vImuMeas.empty()) {
                RCLCPP_INFO(this->get_logger(), "No valid IMU measurements for frame at %.6f", frame_timestamp);
                continue;
                }
                for(auto tt : vImuMeas) {
                    RCLCPP_DEBUG(this->get_logger(), "IMU measurement: t=%.6f, a=[%.3f, %.3f, %.3f], w=[%.3f, %.3f, %.3f]",
                                tt.t, tt.a.x(), tt.a.y(), tt.a.z(), tt.w.x(), tt.w.y(), tt.w.z());
                }
                Tcw = m_SLAM->TrackRGBD(color, depth, frame_timestamp+0.001, vImuMeas);
            }
            //https://zhaoxuhui.top/blog/2021/11/18/orb-slam3-note-7-imu-error-state.html
            // 调用SLAM处理
            // RCLCPP_INFO(this->get_logger(), "Processing frame at %6.f %6.f %.6f with %zu IMU measurements and [%6.f %6.f]", 
                        // color_timestamp,depth_timestamp,frame_timestamp, vImuMeas.size(),vImuMeas.empty()?0:vImuMeas.front().t,vImuMeas.empty()?0:vImuMeas.back().t);
            else            
                Tcw = m_SLAM->TrackRGBD(color, depth, frame_timestamp+0.001);
            // 发布TF变换
            if (Tcw.matrix().allFinite() && !Tcw.matrix().isApprox(Eigen::Matrix4f::Identity())) {
                Sophus::SE3f Twc = Tcw.inverse();
                Eigen::Matrix3f R = Twc.rotationMatrix();
                Eigen::Vector3f t = Twc.translation();
                Eigen::Quaternionf q(R);

                geometry_msgs::msg::TransformStamped tf_msg;
                tf_msg.header.stamp = this->now();
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
                RCLCPP_INFO(this->get_logger(),"Position:[%.6f,%.6f,%.6f]",
                            t.x(), t.y(), t.z());
            }

        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in ProcessFrame: %s", e.what());
            std::this_thread::sleep_for(100ms);
        }
    }
}
// 修改ProcessGyroData
inline void RgbdSlamNode:: ProcessGyroData(std::shared_ptr<ob::Frame> frame) {
    if (!time_initialized_.load()) {
        std::lock_guard<std::mutex> lock(time_mutex_);
        if (!time_initialized_.load()) {
            imu_start_time_ns_ = frame->systemTimeStamp();
            ros_start_time_ = this->now();
            time_initialized_.store(true);
            RCLCPP_INFO(this->get_logger(), "IMU time synchronization initialized");
        }
    }
    auto gyroFrame = frame->as<ob::GyroFrame>();
    auto gyro = gyroFrame->value();
    double timestamp = ros_start_time_.seconds() + (frame->systemTimeStamp() - imu_start_time_ns_) * 1e-9;/* 计算ROS时间戳 */
    // 添加时间戳有效性检查
    if (timestamp <= 0) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "Invalid gyro timestamp: %.6f", timestamp);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(gyro_mutex_);
        cv::Point3f empty_accel(0.0f, 0.0f, 0.0f);
        cv::Point3f w(gyro.x, gyro.y, gyro.z);
        gyro_buffer_.emplace_back(empty_accel, w, timestamp); // 临时存储陀螺仪数据
    }
    if(gyro_buffer_.size() > 300) {
         PairIMUData(); // 尝试配对数据
    }
}
 
// 修改ProcessAccelData（类似）
inline void RgbdSlamNode:: ProcessAccelData(std::shared_ptr<ob::Frame> frame) {
    if (!time_initialized_.load()) {
        std::lock_guard<std::mutex> lock(time_mutex_);
        if (!time_initialized_.load()) {
            imu_start_time_ns_ = frame->systemTimeStamp();
            ros_start_time_ = this->now();
            time_initialized_.store(true);
            RCLCPP_INFO(this->get_logger(), "IMU time synchronization initialized");
        }
    }
    auto accelFrame = frame->as<ob::AccelFrame>();
    auto accel = accelFrame->value();
    double timestamp = ros_start_time_.seconds() + (frame->systemTimeStamp() - imu_start_time_ns_) * 1e-9;/* 计算ROS时间戳 */
    // 添加时间戳有效性检查
    if (timestamp <= 0) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "Invalid gyro timestamp: %.6f", timestamp);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(accel_mutex_);
        cv::Point3f a(accel.x, accel.y, accel.z);
        cv::Point3f empty_gyro(0.0f, 0.0f, 0.0f);
        accel_buffer_.emplace_back(a, empty_gyro, timestamp); // 临时存储加速度数据
    }
    if(accel_buffer_.size() > 300) {
         PairIMUData(); // 尝试配对数据
    }

}
 
void RgbdSlamNode::PairIMUData() {
    std::unique_lock<std::mutex> lock_gyro(gyro_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> lock_accel(accel_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> lock_imu(imu_mutex_, std::defer_lock);
    std::lock(lock_gyro, lock_accel, lock_imu);
    
    // 使用临时缓冲区收集有效数据
    std::deque<ORB_SLAM3::IMU::Point> valid_gyro;
    std::deque<ORB_SLAM3::IMU::Point> valid_accel;
    
    // 收集有效的陀螺仪数据
    for (const auto& gyro : gyro_buffer_) {
        if (gyro.t > 0 && !std::isnan(gyro.w.norm())) {
            valid_gyro.push_back(gyro);
        }
    }
    
    // 收集有效的加速度数据
    for (const auto& accel : accel_buffer_) {
        if (accel.t > 0 && !std::isnan(accel.a.norm())) {
            valid_accel.push_back(accel);
        }
    }
    
    // 清空原始缓冲区
    gyro_buffer_.clear();
    accel_buffer_.clear();
    
    // 时间窗口匹配
    constexpr double MAX_TIME_DIFF = 0.1; 
    
    for (auto& gyro : valid_gyro) {
        // 查找最佳匹配的加速度数据
        auto best_match = valid_accel.end();
        double best_time_diff = std::numeric_limits<double>::max();
        
        for (auto it = valid_accel.begin(); it != valid_accel.end(); ++it) {
            double time_diff = std::abs(it->t - gyro.t);
            if (time_diff < MAX_TIME_DIFF && time_diff < best_time_diff) {
                best_match = it;
                best_time_diff = time_diff;
            }
        }
        
        // 如果找到匹配
        if (best_match != valid_accel.end()) {
            // 创建合并的IMU点
            cv::Point3f best_accel(best_match->a.x(), best_match->a.y(), best_match->a.z());
            cv::Point3f best_gyro(gyro.w.x(), gyro.w.y(), gyro.w.z());
            ORB_SLAM3::IMU::Point merged(
                best_accel,
                best_gyro,
                (gyro.t + best_match->t) / 2.0 // 使用平均时间戳
            );
            
            // 添加到主缓冲区
            imu_buffer_.push_back(merged);
            
            // 移除已匹配的加速度数据
            valid_accel.erase(best_match);
        }
    }
    
    // 将未匹配的有效数据放回原始缓冲区
    gyro_buffer_ = std::move(valid_gyro);
    accel_buffer_ = std::move(valid_accel);
}