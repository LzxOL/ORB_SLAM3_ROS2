#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "rgbd-slam-node.hpp"

#include "System.h"


struct FrameData {
    ob::Frame depthFrame;
    ob::Frame colorFrame;
    uint64_t timestamp;
};
 
// 全局变量用于线程同步
std::mutex mtx;
std::condition_variable cv;
FrameData sharedFrameData;
std::atomic<bool> running(true);
 
// 深度图获取线程函数
void depthThreadFunc() {
    try {
        // 创建深度图管道
        ob::Pipeline depthPipe;
        auto depthProfiles = depthPipe.getStreamProfileList(OB_SENSOR_DEPTH);
        auto depthProfile = depthProfiles->getVideoStreamProfile(640, 480, OB_FORMAT_Y16, 30);
        
        auto depthConfig = std::make_shared<ob::Config>();
        depthConfig->enableStream(depthProfile);
        depthPipe.start(depthConfig);
 
        // FPS计算相关
        using clock = std::chrono::steady_clock;
        std::deque<clock::time_point> depthTimeQueue;
        const auto ONE_SECOND = std::chrono::seconds(1);
 
        while (running) {
            auto frameSet = depthPipe.waitForFrames(100);
            if (!frameSet) continue;
 
            auto depthFrame = frameSet->depthFrame();
            if (!depthFrame) continue;
 
            // 计算深度图FPS
            auto now = clock::now();
            depthTimeQueue.push_back(now);
            while (!depthTimeQueue.empty() && (now - depthTimeQueue.front()) > ONE_SECOND) {
                depthTimeQueue.pop_front();
            }
            double depthFps = static_cast<double>(depthTimeQueue.size());
 
            // 每30帧打印一次中心点深度
            if (depthFrame->index() % 30 == 0) {
                uint32_t w = depthFrame->width();
                uint32_t h = depthFrame->height();
                float scale = depthFrame->getValueScale();
                uint16_t* data = (uint16_t*)depthFrame->data();
                float dist = data[w * h / 2 + w / 2] * scale;
                
                std::unique_lock<std::mutex> lock(mtx);
                std::cout << "Depth FPS: " << std::fixed << std::setprecision(1) << depthFps 
                          << ", Center distance: " << dist << " mm\n";
            }
 
            // 更新共享数据
            {
                std::lock_guard<std::mutex> lock(mtx);
                sharedFrameData.depthFrame = depthFrame;
                sharedFrameData.timestamp = depthFrame->systemTimeStamp();
                cv.notify_one();
            }
        }
 
        depthPipe.stop();
    }
    catch (ob::Error &e) {
        std::cerr << "Depth thread error:\nfunction:" << e.getName()
                  << "\nargs:" << e.getArgs()
                  << "\nmessage:" << e.getMessage()
                  << "\ntype:" << e.getExceptionType() << std::endl;
        running = false;
    }
}
 
// 彩色图获取线程函数
void colorThreadFunc() {
    try {
        // 创建彩色图管道
        ob::Pipeline colorPipe;
        auto colorProfiles = colorPipe.getStreamProfileList(OB_SENSOR_COLOR);
        auto colorProfile = colorProfiles->getVideoStreamProfile(640, 480, OB_FORMAT_RGB, 30);
        
        auto colorConfig = std::make_shared<ob::Config>();
        colorConfig->enableStream(colorProfile);
        colorPipe.start(colorConfig);
 
        // FPS计算相关
        using clock = std::chrono::steady_clock;
        std::deque<clock::time_point> colorTimeQueue;
        const auto ONE_SECOND = std::chrono::seconds(1);
 
        while (running) {
            auto frameSet = colorPipe.waitForFrames(100);
            if (!frameSet) continue;
 
            auto colorFrame = frameSet->colorFrame();
            if (!colorFrame) continue;
 
            // 计算彩色图FPS
            auto now = clock::now();
            colorTimeQueue.push_back(now);
            while (!colorTimeQueue.empty() && (now - colorTimeQueue.front()) > ONE_SECOND) {
                colorTimeQueue.pop_front();
            }
            double colorFps = static_cast<double>(colorTimeQueue.size());
 
            // 每30帧打印一次元数据
            if (colorFrame->index() % 30 == 0) {
                std::unique_lock<std::mutex> lock(mtx);
                std::cout << "Color FPS: " << std::fixed << std::setprecision(1) << colorFps << "\n";
                
                // 打印部分元数据
                std::cout << "Color Frame Metadata (sample):\n";
                std::cout << "  Timestamp: " << colorFrame->systemTimeStamp() << "\n";
                std::cout << "  Frame Index: " << colorFrame->index() << "\n";
                if (colorFrame->hasMetadata(OB_FRAME_METADATA_EXPOSURE)) {
                    std::cout << "  Exposure: " << colorFrame->getMetadataValue(OB_FRAME_METADATA_EXPOSURE) << "\n";
                }
                if (colorFrame->hasMetadata(OB_FRAME_METADATA_GAIN)) {
                    std::cout << "  Gain: " << colorFrame->getMetadataValue(OB_FRAME_METADATA_GAIN) << "\n";
                }
                std::cout << std::endl;
            }
 
            // 更新共享数据
            {
                std::lock_guard<std::mutex> lock(mtx);
                sharedFrameData.colorFrame = colorFrame;
                sharedFrameData.timestamp = colorFrame->systemTimeStamp();
                cv.notify_one();
            }
        }
 
        colorPipe.stop();
    }
    catch (ob::Error &e) {
        std::cerr << "Color thread error:\nfunction:" << e.getName()
                  << "\nargs:" << e.getArgs()
                  << "\nmessage:" << e.getMessage()
                  << "\ntype:" << e.getExceptionType() << std::endl;
        running = false;
    }
}

int main(int argc, char **argv)
{
    if(argc < 3)
    {
        std::cerr << "\nUsage: ros2 run orbslam rgbd path_to_vocabulary path_to_settings" << std::endl;
        return 1;
    }

    rclcpp::init(argc, argv);

    // malloc error using new.. try shared ptr
    // Create SLAM system. It initializes all system threads and gets ready to process frames.

    bool visualization = false;
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, visualization);

    auto node = std::make_shared<RgbdSlamNode>(&SLAM);
    std::cout << "============================ " << std::endl;

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}