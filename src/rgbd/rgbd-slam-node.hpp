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
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp> 
#include "nav_msgs/msg/odometry.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Context.hpp"
#include "libobsensor/hpp/Device.hpp"
#include "libobsensor/hpp/StreamProfile.hpp"
#include "libobsensor/hpp/Sensor.hpp" 
#include "libobsensor/ObSensor.hpp"

#include <Eigen/Geometry> // 添加Eigen几何头文件

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
    void ProcessFrame();
    void CalculateFPS();
    void GyroCallback(std::shared_ptr<ob::Frame> frame);
    void AccelCallback(std::shared_ptr<ob::Frame> frame);
    void IMUDataCollector();
    inline void ProcessGyroData(std::shared_ptr<ob::Frame> frame);
    inline void ProcessAccelData(std::shared_ptr<ob::Frame> frame);
    void PairIMUData();
    std::deque<ORB_SLAM3::IMU::Point>::iterator FindClosestIMUPoint(double timestamp);

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
    std::string pointcloud_frame_id_;
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

    // 新增：多线程相关

    std::thread slam_thread_;
    std::atomic<bool> running_{false};
    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    // 新增：存储最新帧
    cv::Mat latest_color_frame_;
    cv::Mat latest_depth_frame_;
    double latest_frame_timestamp_;
    bool has_new_frame_{false};
    // 新增：Orbbec相机管道
    std::shared_ptr<ob::Pipeline> color_pipe_;
    std::shared_ptr<ob::Pipeline> depth_pipe_;
    std::shared_ptr<ob::Pipeline> pipeline;
    ob::PointCloudFilter point_cloud_filter;
    int color_frame_count_;
    int depth_frame_count_;
    rclcpp::Time last_fps_time_;
    rclcpp::TimerBase::SharedPtr fps_timer_;

    // IMU 相关成员变量
    std::shared_ptr<ob::Sensor> gyroSensor_;
    std::shared_ptr<ob::Sensor> accelSensor_;

    std::deque<ORB_SLAM3::IMU::Point> imu_buffer_;
    // 在类中新增成员变量
    std::deque<ORB_SLAM3::IMU::Point> gyro_buffer_;
    std::deque<ORB_SLAM3::IMU::Point> accel_buffer_;
    const double IMU_PAIRING_WINDOW = 0.01; // 5ms
    
    std::mutex gyro_mutex_;
    std::mutex accel_mutex_;
    std::mutex imu_mutex_;

    ob::PointCloudFilter pointCloud;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
    sensor_msgs::msg::PointCloud2 convertToRosPointCloud(std::shared_ptr<ob::Frame> frame);
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    rclcpp::Time last_pointcloud_time_;
    double pointcloud_publish_interval_ = 0.05; // 10Hz
    double depth_scale = 0.001; // 默认值，需替换为实际值

    sensor_msgs::msg::PointCloud2 convertToIntensityPointCloudInMapFrame(
                        std::shared_ptr<ob::Frame> frame, 
                        const std::string& input_frame,
                        const std::string& output_frame,
                        bool debug);
    bool publish_pointcloud_;
    bool debug_pointcloud_;
    std::string pointcloud_output_frame_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr intensity_pointcloud_publisher_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::atomic<bool> pointcloud_processing_{false};
    std::mutex pointcloud_mutex_;

    std::string my_map_frame_id_;  // 新的机器人坐标系
    Sophus::SE3f T_map_my_map_;    // 从 map(OpenCV) 到 my_map(机器人) 的变换
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;  // Odometry坐标系，机器人的位姿 

    std::shared_ptr<rclcpp::TimerBase> pointcloud_timer_;
    std::shared_ptr<ob::Frame> latest_pointcloud_frame_;
    
    // 添加预计算的静态变换矩阵
    Eigen::Isometry3f T_camera_to_base_; // 相机坐标系到base_link坐标系的变换
    int downsample_ratio_ = 1; // 点云下采样比例
    bool use_openmp_ = true;

};

#endif // __RGBD_SLAM_NODE_HPP__