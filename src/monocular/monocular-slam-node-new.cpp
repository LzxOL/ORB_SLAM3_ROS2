#include "monocular-slam-node.hpp"

#include <opencv2/core/core.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

//AddNew
#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Context.hpp"
#include "libobsensor/hpp/Device.hpp"
#include "libobsensor/hpp/StreamProfile.hpp"
#include "libobsensor/hpp/Sensor.hpp"
#include "libobsensor/ObSensor.hpp"
//

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
//DeleteNew
    // m_image_subscriber = this->create_subscription<ImageMsg>(
    //     "/camera/color/image_raw",
    //     10,
    //     std::bind(&MonocularSlamNode::GrabImage, this, std::placeholders::_1));
//
    std::cout << "slam changed" << std::endl;

//AddNew
    //初始化Orbbec相机管道
    try{
        ob::Context ctx;
        std::shared_ptr<ob::DeviceList> devices = ctx.queryDeviceList();
        if (devices->deviceCount() == 0) {
            RCLCPP_ERROR(this->get_logger(), "No Orbbec devices found!");
            throw std::runtime_error("No Orbbec devices found");
        }

        auto device = devices->getDevice(0);

        // 初始化Pipeline
        pipeline = std::make_shared<ob::Pipeline>(device);

        // 配置彩色流
        auto color_profiles = pipeline->getStreamProfileList(OB_SENSOR_COLOR);

        // 创建配置对象
        auto config = std::make_shared<ob::Config>();
        // 配置彩色流
        if (color_profiles) {
            auto color_profile = color_profiles->getVideoStreamProfile(camera_width, camera_height, OB_FORMAT_BGR, camera_fps);
            config->enableStream(color_profile);
        }

        pipeline->start(config);
        auto camera_param = pipeline->getCameraParam();
        point_cloud_filter.setCameraParam(camera_param);

        RCLCPP_INFO(this->get_logger(), "Orbbec camera initialized successfully");

        // 启动SLAM处理线程
        running_ = true;
        slam_thread_ = std::thread(&MonocularSlamNode::GrabImage, this);

    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Orbbec initialization failed: %s", e.what());
        throw;
        }

//
}

MonocularSlamNode::~MonocularSlamNode()
{
//ChangeNew
    running_ = false;
    if (slam_thread_.joinable()) slam_thread_.join();
    
    if (pipeline) pipeline->stop();
    
    // if (gyroSensor_) gyroSensor_->stop();
    // if (accelSensor_) accelSensor_->stop();

    if (m_SLAM) {
        m_SLAM->Shutdown();
        m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    }
//
}

void MonocularSlamNode::GrabImage()//const ImageMsg::SharedPtr msg
{
    //ChangeNew
    while(running_)
    {
        auto msg = pipeline->waitForFrames(100);//此处把函数传参的msg修改为从pipeline获取
        if (!msg) continue;
        // Copy the image message to cv::Mat.
        try {
            cv_bridge::CvImagePtr cv_ptr;
            if (msg->image.format == OB_FORMAT_BGR) {
                cv_ptr = cv_bridge::toCvCopy(msg);
            } else {
                // 如果图像格式不是BGR格式，需要先转换为BGR格式
                ob::FrameSet::SharedPtr frame_set = pipeline->getFrames();
                if (frame_set) {
                    cv_ptr = cv_bridge::toCvCopy(cv_bridge::toCvMat(frame_set->getFrame(0)->image));
                }
            }
        }
        catch (cv_bridge::Exception& e) {
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
    //
}

//Done