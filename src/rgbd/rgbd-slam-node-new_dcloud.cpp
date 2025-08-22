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

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Context.hpp"
#include "libobsensor/hpp/Device.hpp"
#include "libobsensor/hpp/StreamProfile.hpp"
#include "libobsensor/hpp/Sensor.hpp"
#include "libobsensor/ObSensor.hpp"

#include <Eigen/Geometry>

using namespace std::chrono_literals;

#define camera_width 1280
#define camera_height 720
#define camera_fps 30

// 全局变量定义
std::atomic<bool> time_initialized_{false};
std::mutex time_mutex_;
double last_processed_timestamp_ = 0;
rclcpp::Time ros_start_time_;

std::atomic<bool> color_time_initialized_{false};
std::atomic<bool> depth_time_initialized_{false};
uint64_t imu_start_time_ns_ = 0;
uint64_t color_start_time_ns_ = 0;
uint64_t depth_start_time_ns_ = 0;
bool use_imu = false;
bool use_RGBD = true;
bool use_MONO = false;
bool use_RGBDimu = false;

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
    this->declare_parameter<std::string>("pointcloud_frame_id", "camera_link");
    this->declare_parameter<double>("pointcloud_interval", 0.05);  // 改为0.05秒(20Hz)

    this->declare_parameter<bool>("publish_pointcloud", true);
    this->declare_parameter<bool>("debug_pointcloud", true);
    this->declare_parameter<std::string>("pointcloud_output_frame", "tita4264886/base_link");
    
    this->declare_parameter<std::string>("my_map_frame_id", "my_map");
    this->declare_parameter<bool>("use_openmp", false);  // 禁用OpenMP参数
    this->declare_parameter<int>("pointcloud_downsample_ratio", 4);  // 下采样比例
    
    // 获取参数
    this->get_parameter("my_map_frame_id", my_map_frame_id_);
    this->get_parameter("publish_pointcloud", publish_pointcloud_);
    this->get_parameter("debug_pointcloud", debug_pointcloud_);
    this->get_parameter("pointcloud_output_frame", pointcloud_output_frame_);
    this->get_parameter("pointcloud_interval", pointcloud_publish_interval_);
    this->get_parameter("map_frame_id", map_frame_id_);
    this->get_parameter("base_frame_id", base_frame_id_);
    this->get_parameter("use_pose_prediction", use_pose_prediction_);
    this->get_parameter("pointcloud_frame_id", pointcloud_frame_id_);
    this->get_parameter("use_openmp", use_openmp_);
    this->get_parameter("pointcloud_downsample_ratio", downsample_ratio_);

    double max_prediction_time;
    this->get_parameter("max_prediction_time", max_prediction_time);
    max_prediction_time_ = max_prediction_time;
    
    // 初始化发布器
    pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/camera/depth_registered/points", 
        rclcpp::SensorDataQoS().keep_last(10).reliable()
    );
    
    depth_pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/registered_scan", 
        rclcpp::SensorDataQoS().keep_last(10).reliable()
    );
    
    odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(
        "/state_estimation", 
        10
    );
    
    // 初始化TF相关
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    // 发布静态TF
    geometry_msgs::msg::TransformStamped static_tf;
    static_tf.header.stamp = this->now();
    static_tf.header.frame_id = "tita4264886/base_link";
    static_tf.child_frame_id = "camera_link";
    static_tf.transform.translation.x = 0.1;
    static_tf.transform.translation.y = 0.0;
    static_tf.transform.translation.z = 0.2;
    static_tf.transform.rotation.x = 0.0;
    static_tf.transform.rotation.y = 0.0;
    static_tf.transform.rotation.z = 0.0;
    static_tf.transform.rotation.w = 1.0;
    static_tf_broadcaster_->sendTransform(static_tf);

    // 预计算相机到base_link的静态变换
    try {
        auto transform = tf_buffer_->lookupTransform(
            pointcloud_output_frame_, 
            pointcloud_frame_id_,
            tf2::TimePointZero,
            tf2::durationFromSec(5.0)
        );
        
        Eigen::Translation3f translation(
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        );
        
        Eigen::Quaternionf rotation(
            transform.transform.rotation.w,
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z
        );
        
        T_camera_to_base_ = translation * rotation;
        
        RCLCPP_INFO(this->get_logger(), 
                   "Precomputed static transform from %s to %s",
                   pointcloud_frame_id_.c_str(), 
                   pointcloud_output_frame_.c_str());
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), 
                   "Failed to get static transform: %s. Using identity transform.", 
                   ex.what());
        T_camera_to_base_ = Eigen::Isometry3f::Identity();
    }

    // 定义从 map(OpenCV) 到 my_map(机器人) 的静态变换
    Eigen::Matrix3f R_map_my_map;
    R_map_my_map << 0, 0, 1,
                   -1, 0, 0,
                    0,-1, 0;
    
    T_map_my_map_ = Sophus::SE3f(R_map_my_map, Eigen::Vector3f::Zero());

    // 发布 map -> my_map 的静态变换
    geometry_msgs::msg::TransformStamped static_tf_map_my_map;
    static_tf_map_my_map.header.stamp = this->now();
    static_tf_map_my_map.header.frame_id = map_frame_id_;
    static_tf_map_my_map.child_frame_id = my_map_frame_id_;
    
    Eigen::Quaternionf q(T_map_my_map_.rotationMatrix());
    static_tf_map_my_map.transform.translation.x = 0;
    static_tf_map_my_map.transform.translation.y = 0;
    static_tf_map_my_map.transform.translation.z = 0;
    static_tf_map_my_map.transform.rotation.x = q.x();
    static_tf_map_my_map.transform.rotation.y = q.y();
    static_tf_map_my_map.transform.rotation.z = q.z();
    static_tf_map_my_map.transform.rotation.w = q.w();
    
    static_tf_broadcaster_->sendTransform(static_tf_map_my_map);
    
    RCLCPP_INFO(this->get_logger(), "Published static transform from %s to %s",
                map_frame_id_.c_str(), my_map_frame_id_.c_str());
    
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
        
        device->switchDepthWorkMode("Default");
        
        // 配置自动曝光和白平衡
        if(device->isPropertySupported(OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, OB_PERMISSION_WRITE)) {
            device->setBoolProperty(OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, true);
        }
        
        if(device->isPropertySupported(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, OB_PERMISSION_WRITE)) {
            device->setBoolProperty(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, true);
        }
        
        if(device->isPropertySupported(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, OB_PERMISSION_WRITE)) {
            device->setBoolProperty(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, true);
        }
        
        // 初始化Pipeline
        pipeline = std::make_shared<ob::Pipeline>(device);
        
        // 配置彩色流和深度流 - 保持1280x720分辨率
        auto config = std::make_shared<ob::Config>();
        
        auto color_profiles = pipeline->getStreamProfileList(OB_SENSOR_COLOR);
        auto depth_profiles = pipeline->getStreamProfileList(OB_SENSOR_DEPTH);
        
        // 使用1280x720分辨率
        auto color_profile = color_profiles->getVideoStreamProfile(camera_width, camera_height, OB_FORMAT_RGB, camera_fps);
        auto depth_profile = depth_profiles->getVideoStreamProfile(camera_width, camera_height, OB_FORMAT_Y16, camera_fps);
        
        if (!color_profile || !depth_profile) {
            RCLCPP_ERROR(this->get_logger(), "Failed to find compatible stream profiles");
            throw std::runtime_error("No compatible stream profiles");
        }
        
        config->enableStream(color_profile);
        config->enableStream(depth_profile);
        //config->setAlignMode(ALIGN_D2C_SW_MODE);
        
        pipeline->start(config);
        auto camera_param = pipeline->getCameraParam();
        pointCloud.setCameraParam(camera_param);
        
        RCLCPP_INFO(this->get_logger(), "Orbbec camera initialized successfully with resolution %dx%d", 
                   camera_width, camera_height);
        last_pointcloud_time_ = this->now();
        
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
    
    if (pipeline) pipeline->stop();
    
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
        auto frameset = pipeline->waitForFrames(50);  // 减少等待时间
        if (!frameset) continue;
        
        // 提取对齐后的帧
        auto color_frame = frameset->colorFrame();
        auto depth_frame = frameset->depthFrame();
        auto depthValueScale = depth_frame->getValueScale();
        pointCloud.setPositionDataScaled(depthValueScale);
        pointCloud.setCreatePointFormat(OB_FORMAT_POINT); // 设置为深度点云格式
        if (!color_frame || !depth_frame){
            RCLCPP_WARN(this->get_logger(), "Failed to get color or depth frame");
            continue;
        }
        
        // 转换为OpenCV格式
        cv::Mat color(cv::Size(camera_width, camera_height), CV_8UC3, 
              (void*)color_frame->data(), cv::Mat::AUTO_STEP);
        cv::Mat depth(cv::Size(camera_width, camera_height), CV_16UC1, 
                  (void*)depth_frame->data(), cv::Mat::AUTO_STEP);
        
        if (color.empty() || depth.empty()) {
            RCLCPP_WARN(this->get_logger(), "Empty image or depth frame!");
            continue;
        }

        // 处理点云（异步）
        auto now = this->now();
        if (publish_pointcloud_ && (now - last_pointcloud_time_).seconds() >= pointcloud_publish_interval_) {
            if (!pointcloud_processing_.load()) {
                pointcloud_processing_.store(true);
                
                // 创建深度点云的副本（确保线程安全）
                auto frame_copy = pointCloud.process(frameset);
                
                // 启动异步处理线程
                std::thread([this, frame_copy]() {
                    try {
                        auto cloud_msg = convertToDepthPointCloudInMapFrame(
                            frame_copy, 
                            pointcloud_frame_id_, 
                            pointcloud_output_frame_,
                            debug_pointcloud_
                        );
                        
                        if (!cloud_msg.data.empty()) {
                            depth_pointcloud_publisher_->publish(cloud_msg);
                            
                            if (debug_pointcloud_) {
                                size_t point_count = cloud_msg.width * cloud_msg.height;
                                RCLCPP_INFO(this->get_logger(), 
                                    "Published depth point cloud: %zu points to frame %s", 
                                    point_count, cloud_msg.header.frame_id.c_str());
                            }
                        }
                    } catch (const std::exception& e) {
                        RCLCPP_ERROR(this->get_logger(), "Point cloud processing error: %s", e.what());
                    }
                    
                    pointcloud_processing_.store(false);
                }).detach();
                
                last_pointcloud_time_ = now;
            }
        }

        // SLAM处理
        double timestamp = this->now().seconds() + this->now().nanoseconds() * 1e-9;
        Sophus::SE3f Tcw;
        
        if(use_RGBD) {
            cv::Mat gray;
            cv::cvtColor(color, gray, cv::COLOR_RGB2GRAY);
            Tcw = m_SLAM->TrackRGBD(gray, depth, timestamp);
        } else if(use_MONO) {
            Tcw = m_SLAM->TrackMonocular(color, timestamp);
        }
        
        // 发布TF变换
        if (Tcw.matrix().allFinite() && !Tcw.matrix().isApprox(Eigen::Matrix4f::Identity())) {
            Sophus::SE3f Twc_map = Tcw.inverse();
            Sophus::SE3f Twc_my_map = T_map_my_map_ * Twc_map * T_map_my_map_.inverse();
            
            // 计算base_link位姿
            Eigen::Vector3f t_bc(0.1, 0.0, 0.2);
            Eigen::Quaternionf q_bc(1.0, 0.0, 0.0, 0.0);
            Sophus::SE3f T_bc(q_bc, t_bc);
            Sophus::SE3f T_cb = T_bc.inverse();
            Sophus::SE3f Twb_my_map = Twc_my_map * T_cb;
            
            Eigen::Matrix3f R = Twb_my_map.rotationMatrix();
            Eigen::Vector3f t = Twb_my_map.translation();
            Eigen::Quaternionf q(R);
            
            // 发布TF
            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header.stamp = this->now();
            tf_msg.header.frame_id = my_map_frame_id_;
            tf_msg.child_frame_id = base_frame_id_;
            tf_msg.transform.translation.x = t.x();
            tf_msg.transform.translation.y = t.y();
            tf_msg.transform.translation.z = t.z();
            tf_msg.transform.rotation.x = q.x();
            tf_msg.transform.rotation.y = q.y();
            tf_msg.transform.rotation.z = q.z();
            tf_msg.transform.rotation.w = q.w();
            
            tf_broadcaster_->sendTransform(tf_msg);
            
            // 发布Odometry
            auto odom_msg = std::make_unique<nav_msgs::msg::Odometry>();
            odom_msg->header.stamp = this->now();
            odom_msg->header.frame_id = my_map_frame_id_;
            odom_msg->child_frame_id = base_frame_id_;
            
            odom_msg->pose.pose.position.x = t.x();
            odom_msg->pose.pose.position.y = t.y();
            odom_msg->pose.pose.position.z = t.z();
            odom_msg->pose.pose.orientation.x = q.x();
            odom_msg->pose.pose.orientation.y = q.y();
            odom_msg->pose.pose.orientation.z = q.z();
            odom_msg->pose.pose.orientation.w = q.w();
            
            odom_publisher_->publish(std::move(odom_msg));
            // 打印base_link的位姿（便于调试）
            RCLCPP_INFO(this->get_logger(), "base_link Position in my_map frame: [%.2f, %.2f, %.2f]", 
                    t.x(), t.y(), t.z());
        }
    }
}

sensor_msgs::msg::PointCloud2 RgbdSlamNode::convertToDepthPointCloudInMapFrame(
    std::shared_ptr<ob::Frame> frame,
    const std::string& input_frame,
    const std::string& output_frame,
    bool debug)
{
    // auto start_time = std::chrono::high_resolution_clock::now();
    // 1. 类型检查
    if (!frame->is<ob::PointsFrame>()) {
        RCLCPP_ERROR(this->get_logger(), "Frame is not a PointsFrame!");
        return sensor_msgs::msg::PointCloud2();
    }
    // 2. 直接获取点云数据指针
    auto points_frame = frame->as<ob::PointsFrame>();
    const OBPoint* points = static_cast<OBPoint*>(points_frame->data());
    const size_t total_points = points_frame->dataSize() / sizeof(OBPoint);
    // 3. 创建ROS点云消息（直接内存映射）
    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = this->now();
    cloud_msg.header.frame_id = output_frame;
    cloud_msg.height = 1;
    cloud_msg.width = total_points;
    cloud_msg.is_dense = false;
    // 4. 设置点云字段（直接使用原始数据格式）
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2Fields(3,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32
    );
    // 5. 直接内存拷贝（零拷贝技术）
    cloud_msg.data.resize(total_points * sizeof(OBPoint));
    memcpy(cloud_msg.data.data(), 
           points, 
           total_points * sizeof(OBPoint));
    // 6. 强制类型转换（跳过单位转换）
    auto* raw_data = reinterpret_cast<OBPoint*>(cloud_msg.data.data());
    for (size_t i = 0; i < total_points; ++i) {
        // 仅调整坐标轴方向，不改变数值大小
        std::swap(raw_data[i].x, raw_data[i].z);  // Z -> X
        raw_data[i].y *= -1;                      // Y -> -Y
        raw_data[i].z *= -1;                      // X -> -Z
    }
    // 7. 性能统计
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // if (debug) {

    //     RCLCPP_INFO(this->get_logger(),

    //         "Direct point cloud published in %.2f ms - %zu points",

    //         duration.count() / 1000.0,

    //         total_points);

    // }

 

    return cloud_msg;

}