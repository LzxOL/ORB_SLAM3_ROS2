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
using namespace std::chrono_literals;

#define camera_width 1280
#define camera_height 720
#define camera_fps 30
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
bool use_imu = false; // 是否使用IMU数据

bool use_RGBD = true; // 是否使用RGBD模式
bool use_MONO = false; // 是否使用单目模式
bool use_RGBDimu = false; // 是否使用RGBD+IMU模式
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
    this->declare_parameter<double>("pointcloud_interval", 0.5);

    this->declare_parameter<bool>("publish_pointcloud", true);
    this->declare_parameter<bool>("debug_pointcloud", true);
    this->declare_parameter<std::string>("pointcloud_output_frame", "tita4264886/base_link");
    
    this->declare_parameter<std::string>("my_map_frame_id", "my_map");
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

    double max_prediction_time;
    this->get_parameter("max_prediction_time", max_prediction_time);
    max_prediction_time_ = max_prediction_time;
    // 初始化彩色点云发布器
    pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/camera/depth_registered/points", 
        rclcpp::SensorDataQoS().keep_last(10).reliable()
    );
    // 初始化强度点云发布器
    intensity_pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/registered_scan", 
        rclcpp::SensorDataQoS().keep_last(10).reliable()
    );
    // 初始化 Odometry 发布器
    odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(
        "/state_estimation", 
        10
    );
    // 初始化 TF 广播器
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    // 初始化静态TF广播器
    static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    // 初始化 TF 缓冲区和监听器
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    // 静态 TF 发布
    geometry_msgs::msg::TransformStamped static_tf;
    static_tf.header.stamp = this->now();
    static_tf.header.frame_id = "tita4264886/base_link";
    static_tf.child_frame_id = "camera_link";
    static_tf.transform.translation.x = 0.1;    //相机到机器人中心的x偏移
    static_tf.transform.translation.y = 0.0;
    static_tf.transform.translation.z = 0.2;    //相机到机器人中心的z偏移
    static_tf.transform.rotation.x = 0.0;
    static_tf.transform.rotation.y = 0.0;
    static_tf.transform.rotation.z = 0.0;
    static_tf.transform.rotation.w = 1.0;
    static_tf_broadcaster_->sendTransform(static_tf);

    // 定义从 map(OpenCV) 到 my_map(机器人) 的静态变换,将 OpenCV 坐标系 (X右, Y下, Z前) 转换为 ROS 标准坐标系 (X前, Y左, Z上)
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
        /// 初始化Pipeline
        pipeline = std::make_shared<ob::Pipeline>(device);
        // 配置彩色流和深度流
        auto color_profiles = pipeline->getStreamProfileList(OB_SENSOR_COLOR);
        auto depth_profiles = pipeline->getStreamProfileList(OB_SENSOR_DEPTH);
        // 创建配置对象
        auto config = std::make_shared<ob::Config>();
        // 配置彩色流
        if (color_profiles) {
            auto color_profile = color_profiles->getVideoStreamProfile(camera_width, camera_height, OB_FORMAT_RGB, camera_fps);
            config->enableStream(color_profile);
        }
        // 配置深度流
        if (depth_profiles) {
            auto depth_profile = depth_profiles->getVideoStreamProfile(camera_width, camera_height, OB_FORMAT_Y16, camera_fps);
            config->enableStream(depth_profile);
        }
        config->setAlignMode(ALIGN_D2C_SW_MODE);// 开启软件D2C对齐, 生成RGBD点云时需要开启
        pipeline->start(config);
        auto camera_param = pipeline->getCameraParam();
        pointCloud.setCameraParam(camera_param);
        
        RCLCPP_INFO(this->get_logger(), "Orbbec camera initialized successfully");
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
        auto frameset = pipeline->waitForFrames(100);
        if (!frameset) continue;
        // 提取对齐后的帧
        auto color_frame = frameset->colorFrame();
        auto depth_frame = frameset->depthFrame();
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

        auto now = this->now();
        if (publish_pointcloud_ && (now - last_pointcloud_time_).seconds() >= pointcloud_publish_interval_) {
            // 检查是否已有处理线程在运行
            if (!pointcloud_processing_.load()) {
                // 标记处理开始
                pointcloud_processing_.store(true);
                
                // 创建帧的副本（确保线程安全）
                auto frame_copy = pointCloud.process(frameset);
                
                // 启动异步处理线程
                std::thread([this, frame_copy]() {
                    // 转换并发布点云
                    auto cloud_msg = convertToIntensityPointCloudInMapFrame(
                        frame_copy, 
                        pointcloud_frame_id_, 
                        pointcloud_output_frame_,
                        debug_pointcloud_
                    );
                    
                    if (!cloud_msg.data.empty()) {
                        intensity_pointcloud_publisher_->publish(cloud_msg);
                        
                        if (debug_pointcloud_) {
                            size_t point_count = cloud_msg.width * cloud_msg.height;
                            RCLCPP_INFO(this->get_logger(), 
                                "Published intensity point cloud: %zu points to frame %s", 
                                point_count, cloud_msg.header.frame_id.c_str());
                        }
                    }
                    
                    // 标记处理完成
                    pointcloud_processing_.store(false);
                }).detach();  // 分离线程，自动销毁
                    
                    last_pointcloud_time_ = now;
                } else {
                    RCLCPP_WARN(this->get_logger(), "Skipping point cloud processing: previous task still running");
                }
        }

    
        double timestamp = this->now().seconds()+this->now().nanoseconds() * 1e-9; // 获取当前时间戳
        Sophus::SE3f Tcw;
        if(use_RGBD) {
            cv::Mat gray;
            cv::cvtColor(color, gray, cv::COLOR_RGB2GRAY);
            // cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
            Tcw = m_SLAM->TrackRGBD(gray, depth, timestamp);
        }
        else if(use_MONO){
            Tcw = m_SLAM->TrackMonocular(color, timestamp);
        }
        // 发布TF变换
        if (Tcw.matrix().allFinite() && !Tcw.matrix().isApprox(Eigen::Matrix4f::Identity())) {
            // 1. 原始位姿：map → camera_link（SLAM输出的相机位姿）
            Sophus::SE3f Twc_map = Tcw.inverse();  // Tcw是camera→map，逆后是map→camera
            
            // 2. 将相机位姿转换到my_map坐标系（原有逻辑保留）
            Sophus::SE3f Twc_my_map = T_map_my_map_ * Twc_map * T_map_my_map_.inverse();
            
            // 3. 定义：base_link → camera_link 的固定变换 T_bc（从静态TF获取）
            // 平移：camera在base_link的x+0.1m、z+0.2m处；旋转：无旋转（单位四元数）
            Eigen::Vector3f t_bc(0.1, 0.0, 0.2);  // base_link到camera_link的平移
            Eigen::Quaternionf q_bc(1.0, 0.0, 0.0, 0.0);  // 无旋转
            Sophus::SE3f T_bc(q_bc, t_bc);
            
            // 4. 计算：camera_link → base_link 的逆变换 T_cb（T_bc的逆）
            // 因T_bc无旋转，逆变换的旋转仍为单位四元数，平移取反
            Sophus::SE3f T_cb = T_bc.inverse();
            
            // 5. 关键：计算 map → base_link 的位姿 Twb_my_map
            // 公式：Twb = Twc（map→camera） × T_cb（camera→base）
            Sophus::SE3f Twb_my_map = Twc_my_map * T_cb;
            
            // 6. 提取base_link的位姿（平移t和旋转q）
            Eigen::Matrix3f R = Twb_my_map.rotationMatrix();  // base_link的旋转矩阵
            Eigen::Vector3f t = Twb_my_map.translation();    // base_link的平移（x,y,z）
            Eigen::Quaternionf q(R);                         // 转换为四元数
            
            // 7. 发布 my_map → base_link 的TF变换（修正为base_link的位姿）
            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header.stamp = this->now();
            tf_msg.header.frame_id = my_map_frame_id_;          // 父坐标系：my_map
            tf_msg.child_frame_id = base_frame_id_;          // 子坐标系：base_link
            tf_msg.transform.translation.x = t.x();          // base_link的x（修正后）
            tf_msg.transform.translation.y = t.y();          // base_link的y（修正后）
            tf_msg.transform.translation.z = t.z();          // base_link的z（修正后）
            tf_msg.transform.rotation.x = q.x();             // base_link的旋转x
            tf_msg.transform.rotation.y = q.y();             // base_link的旋转y
            tf_msg.transform.rotation.z = q.z();             // base_link的旋转z
            tf_msg.transform.rotation.w = q.w();             // base_link的旋转w
            
            tf_broadcaster_->sendTransform(tf_msg);
            
            // 打印base_link的位姿（便于调试）
            RCLCPP_INFO(this->get_logger(), "base_link Position in my_map frame: [%.2f, %.2f, %.2f]", 
                    t.x(), t.y(), t.z());

            // 8. 发布 base_link 的Odometry消息（同样使用修正后的位姿）
            auto odom_msg = std::make_unique<nav_msgs::msg::Odometry>();
            odom_msg->header.stamp = this->now();
            odom_msg->header.frame_id = my_map_frame_id_;       // 父坐标系：my_map
            odom_msg->child_frame_id = base_frame_id_;       // 子坐标系：base_link
            
            // 填充base_link的位置（修正后）
            odom_msg->pose.pose.position.x = t.x();
            odom_msg->pose.pose.position.y = t.y();
            odom_msg->pose.pose.position.z = t.z();
            // 填充base_link的姿态（修正后）
            odom_msg->pose.pose.orientation.x = q.x();
            odom_msg->pose.pose.orientation.y = q.y();
            odom_msg->pose.pose.orientation.z = q.z();
            odom_msg->pose.pose.orientation.w = q.w();
            
            // 发布修正后的Odometry
            odom_publisher_->publish(std::move(odom_msg));
        }
    }
}


sensor_msgs::msg::PointCloud2 RgbdSlamNode::convertToRosPointCloud(std::shared_ptr<ob::Frame> frame) {
    if (!frame->is<ob::PointsFrame>()) {
        RCLCPP_ERROR(this->get_logger(), "Frame is not a PointsFrame!");
        return sensor_msgs::msg::PointCloud2(); // 返回空点云
    }
    auto points_frame = frame->as<ob::PointsFrame>();
 
    sensor_msgs::msg::PointCloud2 cloud;
    
    // 使用传感器时间戳
    uint64_t sensor_time_us = points_frame->systemTimeStampUs();
    cloud.header.stamp = rclcpp::Time(sensor_time_us);
    cloud.header.frame_id = pointcloud_frame_id_; // 使用参数化的frame_id
 
    // 定义字段
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(4,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "rgb", 1, sensor_msgs::msg::PointField::UINT32
    );
 
    auto* points = static_cast<OBColorPoint*>(points_frame->data());
    size_t total_points = points_frame->dataSize() / sizeof(OBColorPoint);
    size_t valid_count = 0;
    const float min_distance = 20.0f; // 20mm 有效距离阈值
    
    // 第一遍：统计有效点
    for (size_t i = 0; i < total_points; ++i) {
        if (points[i].z > min_distance) { // 只保留大于20mm的点
            valid_count++;
        }
    }
 
    // 设置云属性
    modifier.resize(valid_count);
    cloud.width = valid_count;
    cloud.height = 1;
    cloud.is_dense = false;
 
    // 第二遍：填充有效点（毫米转米）
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<uint32_t> iter_rgb(cloud, "rgb");
 
    for (size_t i = 0; i < total_points; ++i) {
        if (points[i].z > min_distance) {
            // 毫米转换为米
            *iter_x = points[i].x * 0.001f;
            *iter_y = points[i].y * 0.001f;
            *iter_z = points[i].z * 0.001f;
            *iter_rgb = (static_cast<uint32_t>(points[i].r) << 16) |
                        (static_cast<uint32_t>(points[i].g) << 8) |
                        static_cast<uint32_t>(points[i].b);
            ++iter_x; ++iter_y; ++iter_z; ++iter_rgb;
        }
    }
 
    RCLCPP_INFO(this->get_logger(), "Published point cloud with %zu/%zu valid points", 
                valid_count, total_points);
    return cloud;
}

//将点云转换为强度点云并变换坐标系
sensor_msgs::msg::PointCloud2 RgbdSlamNode::convertToIntensityPointCloudInMapFrame(
    std::shared_ptr<ob::Frame> frame, 
    const std::string& input_frame,
    const std::string& output_frame,
    bool debug)
{
    sensor_msgs::msg::PointCloud2 output;
    
    if (!frame->is<ob::PointsFrame>()) {
        RCLCPP_ERROR(this->get_logger(), "Frame is not a PointsFrame!");
        return output;
    }
    
    auto points_frame = frame->as<ob::PointsFrame>();
    auto* points = static_cast<OBColorPoint*>(points_frame->data());
    size_t total_points = points_frame->dataSize() / sizeof(OBColorPoint);
    
    // 创建中间点云（带RGB）
    sensor_msgs::msg::PointCloud2 rgb_cloud;
    rgb_cloud.header.stamp = this->now();
    rgb_cloud.header.frame_id = input_frame;
    rgb_cloud.height = 1;
    rgb_cloud.width = total_points;
    rgb_cloud.is_dense = false;
    
    sensor_msgs::PointCloud2Modifier modifier(rgb_cloud);
    modifier.setPointCloud2Fields(4,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "rgb", 1, sensor_msgs::msg::PointField::UINT32
    );
    
    sensor_msgs::PointCloud2Iterator<float> iter_x(rgb_cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(rgb_cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(rgb_cloud, "z");
    sensor_msgs::PointCloud2Iterator<uint32_t> iter_rgb(rgb_cloud, "rgb");
    
    // 填充RGB点云数据（毫米转米）
    for (size_t i = 0; i < total_points; ++i) {
        *iter_x = points[i].x * 0.001f;
        *iter_y = points[i].y * 0.001f;
        *iter_z = points[i].z * 0.001f;
        
        uint32_t r = points[i].r;
        uint32_t g = points[i].g;
        uint32_t b = points[i].b;
        *iter_rgb = (r << 16) | (g << 8) | b;
        
        ++iter_x; ++iter_y; ++iter_z; ++iter_rgb;
    }
    
    // 转换为强度点云
    sensor_msgs::msg::PointCloud2 intensity_cloud;
    intensity_cloud.header = rgb_cloud.header;
    intensity_cloud.height = rgb_cloud.height;
    intensity_cloud.width = rgb_cloud.width;
    intensity_cloud.is_dense = rgb_cloud.is_dense;
    
    sensor_msgs::PointCloud2Modifier intensity_modifier(intensity_cloud);
    intensity_modifier.setPointCloud2Fields(4,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::msg::PointField::FLOAT32
    );
    
    sensor_msgs::PointCloud2ConstIterator<float> in_x(rgb_cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> in_y(rgb_cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> in_z(rgb_cloud, "z");
    sensor_msgs::PointCloud2ConstIterator<uint32_t> in_rgb(rgb_cloud, "rgb");
    
    sensor_msgs::PointCloud2Iterator<float> out_x(intensity_cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(intensity_cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(intensity_cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> out_intensity(intensity_cloud, "intensity");
    
    // 转换为强度并过滤无效点
    size_t valid_count = 0;
    const float min_distance = 0.02f; // 20mm in meters
    
    for (size_t i = 0; i < total_points; ++i) {
        // 跳过无效点
        if (*in_z < min_distance || std::isnan(*in_z)) {
            ++in_x; ++in_y; ++in_z; ++in_rgb;
            continue;
        }
        
        // 提取RGB分量
        uint32_t rgb_val = *in_rgb;
        uint8_t r = (rgb_val >> 16) & 0xFF;
        uint8_t g = (rgb_val >> 8) & 0xFF;
        uint8_t b = rgb_val & 0xFF;
        
        // 计算强度
        float intensity = 0.299f * r + 0.587f * g + 0.114f * b;
        
        // 填充输出点云
        *out_x = *in_z;
        *out_y = -(*in_x);
        *out_z = -(*in_y);
        *out_intensity = intensity;
        
        ++valid_count;
        ++in_x; ++in_y; ++in_z; ++in_rgb;
        ++out_x; ++out_y; ++out_z; ++out_intensity;
    }
    
    // 调整点云大小以匹配有效点数
    intensity_cloud.width = valid_count;
    intensity_cloud.row_step = intensity_cloud.width * intensity_cloud.point_step;
    intensity_cloud.data.resize(intensity_cloud.row_step * intensity_cloud.height);
    
    // 坐标系转换 - 使用 tf_buffer_->transform()
    try {
        // 使用 transform() 方法替代 doTransform()
        output = tf_buffer_->transform(
            intensity_cloud,
            output_frame,
            tf2::durationFromSec(0.1)  // 超时时间
        );
        
        if (debug) {
            RCLCPP_INFO(this->get_logger(), 
                "Transformed point cloud from %s to %s (%zu points)",
                input_frame.c_str(), output_frame.c_str(), valid_count);
        }
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), 
            "Point cloud transform failed: %s. Using original frame.", 
            ex.what());
        output = intensity_cloud;
        output.header.frame_id = output_frame;
    }
    
    return output;
}