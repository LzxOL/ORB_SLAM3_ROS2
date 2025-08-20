#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <iomanip>

#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Context.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rgbd-slam-node.hpp"

#include "System.h"

bool use_RGBDimu_mode = false; // 是否使用RGBD+IMU模式
bool use_RGBD_mode =  true; // 是否使用RGBD模式
bool use_MONO_mode = false; // 是否使用单目模式
int main(int argc, char **argv)
{
    if(argc < 3)
    {
        std::cerr << "\nUsage: ros2 run orbslam rgbd path_to_vocabulary path_to_settings" << std::endl;
        return 1;
    }

    rclcpp::init(argc, argv);

    // 初始化SLAM系统
    bool visualization = false;
    if(use_RGBD_mode){
        ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, visualization); //IMU_RGBD=5,RGBD=2 MONOCULAR
        // 创建节点
        auto node = std::make_shared<RgbdSlamNode>(&SLAM);
        // 主循环现在由RgbdSlamNode内部的线程处理
        // 只需要保持ROS运行
        rclcpp::spin(node);
        // 清理工作在节点析构函数中完成
        rclcpp::shutdown();
    }
    else if(use_RGBDimu_mode){
        ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_RGBD, visualization);
        // 创建节点
        auto node = std::make_shared<RgbdSlamNode>(&SLAM);
        // 主循环现在由RgbdSlamNode内部的线程处理
        // 只需要保持ROS运行
        rclcpp::spin(node);
        // 清理工作在节点析构函数中完成
        rclcpp::shutdown();
    }
    else if(use_MONO_mode){
        ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, visualization);
        // 创建节点
        auto node = std::make_shared<RgbdSlamNode>(&SLAM);
        // 主循环现在由RgbdSlamNode内部的线程处理
        // 只需要保持ROS运行
        rclcpp::spin(node);
        // 清理工作在节点析构函数中完成
        rclcpp::shutdown();
    }


    return 0;
}