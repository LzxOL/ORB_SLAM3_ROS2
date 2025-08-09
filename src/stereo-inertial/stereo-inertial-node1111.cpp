#include "stereo-inertial-node.hpp"

#include <opencv2/core/core.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <signal.h>
#include <cmath>

using std::placeholders::_1;

StereoInertialNode::StereoInertialNode(ORB_SLAM3::System *SLAM, const string &strSettingsFile, const string &strDoRectify, const string &strDoEqual) :
    Node("ORB_SLAM3_ROS2"),
    SLAM_(SLAM)
{
    // 声明参数
    this->declare_parameter<std::string>("map_frame_id", "map");
    this->declare_parameter<std::string>("base_frame_id", "tita4264886/base_link");

    // 获取参数
    this->get_parameter("map_frame_id", map_frame_id_);
    this->get_parameter("base_frame_id", base_frame_id_);

    // 初始化 TF 广播器
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    // 初始化位姿跟踪变量
    has_printed_zero_warning_ = false;
    last_pose_was_zero_ = false;
    has_valid_pose_ = false;

    stringstream ss_rec(strDoRectify);
    ss_rec >> boolalpha >> doRectify_;

    stringstream ss_eq(strDoEqual);
    ss_eq >> boolalpha >> doEqual_;

    bClahe_ = doEqual_;
    std::cout << "Rectify: " << doRectify_ << std::endl;
    std::cout << "Equal: " << doEqual_ << std::endl;

    if (doRectify_)
    {
        // Load settings related to stereo calibration
        cv::FileStorage fsSettings(strSettingsFile, cv::FileStorage::READ);
        if (!fsSettings.isOpened())
        {
            cerr << "ERROR: Wrong path to settings" << endl;
            assert(0);
        }

        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            assert(0);
        }

        cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, M1l_, M2l_);
        cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, M1r_, M2r_);
    }

    subImu_ = this->create_subscription<ImuMsg>("/tita4264886/imu_sensor_broadcaster/imu", 1000, std::bind(&StereoInertialNode::GrabImu, this, _1));
    subImgLeft_ = this->create_subscription<ImageMsg>("/tita4264886/perception/camera/image/left",  rclcpp::SensorDataQoS().keep_last(10),   std::bind(&StereoInertialNode::GrabImageLeft, this, _1));
    subImgRight_ = this->create_subscription<ImageMsg>("/tita4264886/perception/camera/image/right",  rclcpp::SensorDataQoS().keep_last(10),   std::bind(&StereoInertialNode::GrabImageRight, this, _1));

    syncThread_ = new std::thread(&StereoInertialNode::SyncWithImu, this);
}

StereoInertialNode::~StereoInertialNode()
{
    // Delete sync thread
    syncThread_->join();
    delete syncThread_;

    // Stop all threads
    SLAM_->Shutdown();

    // Save camera trajectory
    SLAM_->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void StereoInertialNode::GrabImu(const ImuMsg::SharedPtr msg)
{
    if (!msg) {
        RCLCPP_WARN(this->get_logger(), "Received null IMU message");
        return;
    }
    
    static int imu_count = 0;
    if (++imu_count % 100 == 0) {  // 每100个IMU消息输出一次
        RCLCPP_INFO(this->get_logger(), "Received %d IMU messages, buffer size: %lu", 
                   imu_count, imuBuf_.size());
    }
    
    bufMutex_.lock();
    imuBuf_.push(msg);
    
    // 限制缓冲区大小，防止内存无限增长
    while (imuBuf_.size() > 2000) {
        imuBuf_.pop();
    }
    bufMutex_.unlock();
}

void StereoInertialNode::GrabImageLeft(const ImageMsg::SharedPtr msgLeft)
{
    if (!msgLeft) {
        RCLCPP_WARN(this->get_logger(), "Received null left image message");
        return;
    }
    
    static int left_count = 0;
    if (++left_count % 50 == 0) {  // 每50帧输出一次
        RCLCPP_INFO(this->get_logger(), "Received %d left images, buffer size: %lu", 
                   left_count, imgLeftBuf_.size());
    }
    
    bufMutexLeft_.lock();

    if (!imgLeftBuf_.empty())
        imgLeftBuf_.pop();
    imgLeftBuf_.push(msgLeft);

    bufMutexLeft_.unlock();
}

void StereoInertialNode::GrabImageRight(const ImageMsg::SharedPtr msgRight)
{
    if (!msgRight) {
        RCLCPP_WARN(this->get_logger(), "Received null right image message");
        return;
    }
    
    static int right_count = 0;
    if (++right_count % 50 == 0) {  // 每50帧输出一次
        RCLCPP_INFO(this->get_logger(), "Received %d right images, buffer size: %lu", 
                   right_count, imgRightBuf_.size());
    }
    
    bufMutexRight_.lock();

    if (!imgRightBuf_.empty())
        imgRightBuf_.pop();
    imgRightBuf_.push(msgRight);

    bufMutexRight_.unlock();
}

cv::Mat StereoInertialNode::GetImage(const ImageMsg::SharedPtr msg)
{
    // 检查消息指针有效性
    if (!msg) {
        RCLCPP_ERROR(this->get_logger(), "Null image message pointer");
        return cv::Mat();
    }

    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return cv::Mat();
    }

    // 检查转换结果
    if (!cv_ptr || cv_ptr->image.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to convert image or empty image");
        return cv::Mat();
    }

    // 检查图像尺寸是否合理
    if (cv_ptr->image.rows < 10 || cv_ptr->image.cols < 10) {
        RCLCPP_ERROR(this->get_logger(), "Image too small: %dx%d", cv_ptr->image.cols, cv_ptr->image.rows);
        return cv::Mat();
    }

    if (cv_ptr->image.type() == 0)
    {
        return cv_ptr->image.clone();
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Unexpected image type: %d", cv_ptr->image.type());
        return cv_ptr->image.clone();
    }
}

void StereoInertialNode::SyncWithImu()
{
    const double maxTimeDiff = 0.02;
    
    RCLCPP_INFO(this->get_logger(), "SyncWithImu thread started");

    while (1)
    {
        cv::Mat imLeft, imRight;
        double tImLeft = 0, tImRight = 0;
        
        // 添加调试信息
        static int debug_count = 0;
        if (++debug_count % 1000 == 0) {  // 每1000次循环输出一次状态
            RCLCPP_INFO(this->get_logger(), "Buffer status - Left: %lu, Right: %lu, IMU: %lu", 
                       imgLeftBuf_.size(), imgRightBuf_.size(), imuBuf_.size());
        }
        
        if (!imgLeftBuf_.empty() && !imgRightBuf_.empty() && !imuBuf_.empty())
        {
            tImLeft = Utility::StampToSec(imgLeftBuf_.front()->header.stamp);
            tImRight = Utility::StampToSec(imgRightBuf_.front()->header.stamp);

            bufMutexRight_.lock();
            while ((tImLeft - tImRight) > maxTimeDiff && imgRightBuf_.size() > 1)
            {
                imgRightBuf_.pop();
                tImRight = Utility::StampToSec(imgRightBuf_.front()->header.stamp);
            }
            bufMutexRight_.unlock();

            bufMutexLeft_.lock();
            while ((tImRight - tImLeft) > maxTimeDiff && imgLeftBuf_.size() > 1)
            {
                imgLeftBuf_.pop();
                tImLeft = Utility::StampToSec(imgLeftBuf_.front()->header.stamp);
            }
            bufMutexLeft_.unlock();

            if ((tImLeft - tImRight) > maxTimeDiff || (tImRight - tImLeft) > maxTimeDiff)
            {
                // RCLCPP_WARN(this->get_logger(), "Big time difference: Left=%.6f, Right=%.6f, diff=%.6f", 
                //            tImLeft, tImRight, abs(tImLeft - tImRight));
                continue;
            }
            if (tImLeft > Utility::StampToSec(imuBuf_.back()->header.stamp))
            {
                // RCLCPP_WARN(this->get_logger(), "Image timestamp newer than latest IMU: img=%.6f, imu=%.6f", 
                //            tImLeft, Utility::StampToSec(imuBuf_.back()->header.stamp));
                continue;
            }

            // RCLCPP_INFO(this->get_logger(), "Processing stereo frame at time %.6f", tImLeft);
            
            bufMutexLeft_.lock();
            if (!imgLeftBuf_.empty()) {
                imLeft = GetImage(imgLeftBuf_.front());
                imgLeftBuf_.pop();
            }
            bufMutexLeft_.unlock();

            bufMutexRight_.lock();
            if (!imgRightBuf_.empty()) {
                imRight = GetImage(imgRightBuf_.front());
                imgRightBuf_.pop();
            }
            bufMutexRight_.unlock();
            
            // 检查图像是否有效
            if (imLeft.empty() || imRight.empty()) {
                RCLCPP_WARN(this->get_logger(), "Empty image detected, skipping frame");
                continue;
            }

            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            bufMutex_.lock();
            if (!imuBuf_.empty())
            {
                // Load imu measurements from buffer
                vImuMeas.clear();
                while (!imuBuf_.empty() && Utility::StampToSec(imuBuf_.front()->header.stamp) <= tImLeft)
                {
                    auto imu_msg = imuBuf_.front();
                    if (imu_msg) {  // 检查指针有效性
                        double t = Utility::StampToSec(imu_msg->header.stamp);
                        
                        // 检查 IMU 数据是否有效
                        if (std::isfinite(imu_msg->linear_acceleration.x) && 
                            std::isfinite(imu_msg->linear_acceleration.y) && 
                            std::isfinite(imu_msg->linear_acceleration.z) &&
                            std::isfinite(imu_msg->angular_velocity.x) && 
                            std::isfinite(imu_msg->angular_velocity.y) && 
                            std::isfinite(imu_msg->angular_velocity.z)) {
                            
                            cv::Point3f acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
                            cv::Point3f gyr(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
                            vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                        } else {
                            RCLCPP_WARN(this->get_logger(), "Invalid IMU data detected, skipping");
                        }
                    }
                    imuBuf_.pop();
                }
            }
            bufMutex_.unlock();
            
            // RCLCPP_INFO(this->get_logger(), "Collected %lu IMU measurements for frame", vImuMeas.size());

            if (bClahe_)
            {
                clahe_->apply(imLeft, imLeft);
                clahe_->apply(imRight, imRight);
            }

            if (doRectify_)
            {
                cv::remap(imLeft, imLeft, M1l_, M2l_, cv::INTER_LINEAR);
                cv::remap(imRight, imRight, M1r_, M2r_, cv::INTER_LINEAR);
            }

            // RCLCPP_INFO(this->get_logger(), "Calling SLAM TrackStereo...");
            Sophus::SE3f Tcw;
            try {
                RCLCPP_INFO(this->get_logger(), "111");
                Tcw = SLAM_->TrackStereo(imLeft, imRight, tImLeft, vImuMeas);
                RCLCPP_INFO(this->get_logger(), "222");
                // RCLCPP_INFO(this->get_logger(), "SLAM TrackStereo completed");
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "SLAM TrackStereo exception: %s", e.what());
                continue;
            } catch (...) {
                RCLCPP_ERROR(this->get_logger(), "SLAM TrackStereo unknown exception");
                continue;
            }
            
            // 获取 SLAM 跟踪状态
            int tracking_state = SLAM_->GetTrackingState();
            static int last_state = -1;
            if (tracking_state != last_state) {
                std::string state_str;
                switch(tracking_state) {
                    case -1: state_str = "SYSTEM_NOT_READY"; break;
                    case 0: state_str = "NO_IMAGES_YET"; break;
                    case 1: state_str = "NOT_INITIALIZED"; break;
                    case 2: state_str = "OK"; break;
                    case 3: state_str = "RECENTLY_LOST"; break;
                    case 4: state_str = "LOST"; break;
                    case 5: state_str = "OK_KLT"; break;
                    default: state_str = "UNKNOWN"; break;
                }
                RCLCPP_INFO(this->get_logger(), "SLAM tracking state changed: %s (%d)", state_str.c_str(), tracking_state);
                last_state = tracking_state;
            }
            
            // 发布 TF 变换
            PublishTransform(Tcw, rclcpp::Time(static_cast<int64_t>(tImLeft * 1e9)), tracking_state);

            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
        else
        {
            // 如果缓冲区为空，添加短暂延迟避免CPU占用过高
            std::chrono::milliseconds tSleep(5);
            std::this_thread::sleep_for(tSleep);
        }
    }
}

void StereoInertialNode::PublishTransform(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp, int tracking_state)
{
    try {
        // 输出 Tcw 矩阵的平移部分用于调试
        Eigen::Vector3f translation = Tcw.translation();
    
    // 检查位姿是否有效 - 放宽检查条件
    if (!Tcw.matrix().allFinite()) {
        static int nan_count = 0;
        if (++nan_count % 100 == 0) {
            RCLCPP_WARN(this->get_logger(), "NaN/Inf pose from SLAM (%d times)", nan_count);
        }
        last_pose_was_zero_ = true;
        return;
    }
    
    // 检查是否为单位矩阵（未初始化状态）- 使用更宽松的阈值
    if (Tcw.matrix().isApprox(Eigen::Matrix4f::Identity(), 1e-6)) {
        static int identity_count = 0;
        if (++identity_count % 100 == 0) {
            RCLCPP_WARN(this->get_logger(), "Identity pose from SLAM (%d times) - SLAM not initialized", identity_count);
        }
        last_pose_was_zero_ = true;
        return;
    }
    
    // 检查平移是否过大（可能的异常值）
    if (translation.norm() > 1000.0f) {  // 1000米的阈值
        static int large_count = 0;
        if (++large_count % 10 == 0) {
            RCLCPP_WARN(this->get_logger(), "Abnormally large translation (%d times): [%.3f, %.3f, %.3f]", 
                       large_count, translation.x(), translation.y(), translation.z());
        }
        return;
    }

    // 重置警告标志，因为我们有了有效的位姿
    if (last_pose_was_zero_) {
        RCLCPP_INFO(this->get_logger(), "SLAM tracking recovered, resuming TF publication.");
        has_printed_zero_warning_ = false;
        last_pose_was_zero_ = false;
    }

    // 输出 SLAM 位姿信息
    Eigen::Vector3f Tcwt = Tcw.translation();
    static int valid_count = 0;
    std::string state_str = (tracking_state == 2) ? "OK" : 
                           (tracking_state == 5) ? "OK_KLT" : 
                           (tracking_state == 1) ? "NOT_INIT" : "OTHER";
    RCLCPP_INFO(this->get_logger(), "[Stereo-Inertial] Valid pose #%d (%s) - Tcwt: [%.3f, %.3f, %.3f]",
                ++valid_count, state_str.c_str(), Tcwt.x(), Tcwt.y(), Tcwt.z());

    // 计算世界到相机的变换
    Sophus::SE3f Twc = Tcw.inverse();

    // 相机→机体外参（Tbc）- 根据实际安装情况调整
    Eigen::Matrix4f Tbc_matrix = Eigen::Matrix4f::Identity();
    
    // 如果需要，可以在这里添加相机到机体的平移和旋转
    // 例如：
    // Tbc_matrix(0, 3) = 0.0f;   // X 方向平移
    // Tbc_matrix(1, 3) = 0.0f;   // Y 方向平移  
    // Tbc_matrix(2, 3) = 0.1f;   // Z 方向平移
    
    Sophus::SE3f Tbc(Tbc_matrix);
    Sophus::SE3f Twb = Twc * Tbc.inverse();

    // 提取变换并发布 TF
    Eigen::Matrix3f R = Twb.rotationMatrix();
    Eigen::Vector3f t = Twb.translation();
    Eigen::Quaternionf q_eigen(R);

    // 存储有效位姿
    last_valid_translation_ = t;
    last_valid_rotation_ = q_eigen;
    has_valid_pose_ = true;

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
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
        
        RCLCPP_INFO(this->get_logger(), "Published TF: [%.3f, %.3f, %.3f]",
                    t.x(), t.y(), t.z());
                    
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "PublishTransform exception: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "PublishTransform unknown exception");
    }
}
