#include "stereo-inertial-node.hpp"

#include <opencv2/core/core.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <sensor_msgs/msg/imu.hpp>    
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h> 
using std::placeholders::_1;

StereoInertialNode::StereoInertialNode(ORB_SLAM3::System *SLAM, const string &strSettingsFile, const string &strDoRectify, const string &strDoEqual) :
    Node("ORB_SLAM3_ROS2"),
    SLAM_(SLAM),
    has_printed_zero_warning_(false),
    last_pose_was_zero_(false),
    has_valid_pose_(false)
{
    this->declare_parameter<std::string>("map_frame_id", "map");
    this->declare_parameter<std::string>("base_frame_id", "tita4264886/base_link");
    this->get_parameter("map_frame_id", map_frame_id_);
    this->get_parameter("base_frame_id", base_frame_id_);

    // 初始化TF广播器
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    
    // 初始化最后有效的pose为单位四元数
    last_valid_translation_ = Eigen::Vector3f::Zero();
    last_valid_rotation_ = Eigen::Quaternionf::Identity();

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
    // RCLCPP_INFO(this->get_logger(), "GrabImu: %s", msg->header.frame_id.c_str());
    bufMutex_.lock();
    imuBuf_.push(msg);
    // 在GrabImu中增加缓冲上限，避免内存爆炸
    const size_t MAX_IMU_BUFFER = 2000;
    if (imuBuf_.size() > MAX_IMU_BUFFER) {
        imuBuf_.pop(); // 丢弃最旧数据
    }
    bufMutex_.unlock();
}

void StereoInertialNode::GrabImageLeft(const ImageMsg::SharedPtr msgLeft)
{
    // RCLCPP_INFO(this->get_logger(), "GrabImageLeft: %s", msgLeft->header.frame_id.c_str());
    bufMutexLeft_.lock();

    if (!imgLeftBuf_.empty())
        imgLeftBuf_.pop();
    imgLeftBuf_.push(msgLeft);

    bufMutexLeft_.unlock();
}

void StereoInertialNode::GrabImageRight(const ImageMsg::SharedPtr msgRight)
{
    // RCLCPP_INFO(this->get_logger(), "GrabImageRight: %s", msgRight->header.frame_id.c_str());
    bufMutexRight_.lock();

    if (!imgRightBuf_.empty())
        imgRightBuf_.pop();
    imgRightBuf_.push(msgRight);

    bufMutexRight_.unlock();
}

cv::Mat StereoInertialNode::GetImage(const ImageMsg::SharedPtr msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }

    if (cv_ptr->image.type() == 0)
    {
        return cv_ptr->image.clone();
    }
    else
    {
        std::cerr << "Error image type" << std::endl;
        return cv_ptr->image.clone();
    }
}

void StereoInertialNode::SyncWithImu()
{
    const double maxTimeDiff = 0.08;

    while (1)
    {
        
        cv::Mat imLeft, imRight;
        double tImLeft = 0, tImRight = 0;
        
        if (!imgLeftBuf_.empty() && !imgRightBuf_.empty() && !imuBuf_.empty())
        {
            // RCLCPP_INFO(this->get_logger(), "SyncWithImu: %zu left images, %zu right images, %zu IMU messages", 
            //                imgLeftBuf_.size(), imgRightBuf_.size(), imuBuf_.size());
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
                std::queue<std::shared_ptr<sensor_msgs::msg::Imu>> emptyQueue;

                std::swap(imuBuf_, emptyQueue);
                std::cout << "big time difference" << std::endl;
                continue;
            }
            if (tImLeft > Utility::StampToSec(imuBuf_.back()->header.stamp)){
                std::queue<std::shared_ptr<sensor_msgs::msg::Imu>> emptyQueue;
                std::swap(imuBuf_, emptyQueue);
                continue;
            }

            bufMutexLeft_.lock();
            imLeft = GetImage(imgLeftBuf_.front());
            imgLeftBuf_.pop();
            bufMutexLeft_.unlock();

            bufMutexRight_.lock();
            imRight = GetImage(imgRightBuf_.front());
            imgRightBuf_.pop();
            bufMutexRight_.unlock();
            //RCLCPP_INFO(this->get_logger(), "Grabbed images at time: %.3f", tImLeft);
            vector<ORB_SLAM3::IMU::Point> vImuMeas;

            // 在SyncWithImu线程中修改IMU处理逻辑
            bufMutex_.lock();
            if (imuBuf_.size() < 30) {  // 确保有足够IMU数据（约10ms@200Hz）
                bufMutex_.unlock();
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                                "Waiting for more IMU data (current: %zu)", imuBuf_.size());
                std::this_thread::sleep_for(5ms);
                continue;
            }

            vImuMeas.clear();
            const double imu_end_time = tImLeft + 0.05;  // 比图像时间稍延后

            double min_acc_magnitude = std::numeric_limits<double>::max();
            double max_acc_magnitude = 0.0;
            
            while (!imuBuf_.empty() && Utility::StampToSec(imuBuf_.front()->header.stamp) <= tImLeft)
            {
                double t = Utility::StampToSec(imuBuf_.front()->header.stamp);
                auto msg = imuBuf_.front();
    
                // 检查NaN/inf
                if (!std::isfinite(msg->linear_acceleration.x) || 
                    !std::isfinite(msg->linear_acceleration.y) || 
                    !std::isfinite(msg->linear_acceleration.z) ||
                    !std::isfinite(msg->angular_velocity.x) || 
                    !std::isfinite(msg->angular_velocity.y) || 
                    !std::isfinite(msg->angular_velocity.z)) {
                    RCLCPP_WARN(this->get_logger(), "Invalid IMU data (NaN/inf) at time %.3f, skipping...", t);
                    imuBuf_.pop();
                    continue;
                }

                cv::Point3f acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);

                cv::Point3f gyr(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

            

                vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                imuBuf_.pop();
                if(vImuMeas.size() == 60)
                    break;
            }
   
            bufMutex_.unlock();

            if (bClahe_)
            {
                RCLCPP_INFO(this->get_logger(), "Applying CLAHE to images");
                clahe_->apply(imLeft, imLeft);
                clahe_->apply(imRight, imRight);
            }

            if (doRectify_)
            {
                RCLCPP_INFO(this->get_logger(), "Applying stereo rectification");
                cv::remap(imLeft, imLeft, M1l_, M2l_, cv::INTER_LINEAR);
                cv::remap(imRight, imRight, M1r_, M2r_, cv::INTER_LINEAR);
            }

            // RCLCPP_INFO(this->get_logger(), "IMU measurements: %zu, time range: [%.3f, %.3f]", 
            //                 vImuMeas.size(), vImuMeas.front().t, vImuMeas.back().t);
    
            Sophus::SE3f Tcw = SLAM_->TrackStereo(imLeft, imRight, tImLeft, vImuMeas);
          
            // 检查SLAM跟踪状态
            int tracking_state = SLAM_->GetTrackingState();
            // RCLCPP_INFO(this->get_logger(), "SLAM Tracking State: %d", tracking_state);
            std::string state_str;
            switch(tracking_state) {
                case 0: state_str = "SYSTEM_NOT_READY"; break;
                case 1: state_str = "NO_IMAGES_YET"; break;
                case 2: state_str = "NOT_INITIALIZED"; break;
                case 3: state_str = "OK"; break;
                case 4: state_str = "RECENTLY_LOST"; break;
                case 5: state_str = "LOST"; break;
                default: state_str = "UNKNOWN"; break;
            }
            
      
            rclcpp::Time stamp = rclcpp::Time(static_cast<int64_t>(tImLeft * 1e9));// 安全时间戳
            PublishTransform(Tcw, stamp, tracking_state);
            
            
            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
    }
}

void StereoInertialNode::PublishTransform(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp, int tracking_state)
{
    // 1. Check for invalid pose
    if (!Tcw.matrix().allFinite()) {
        RCLCPP_WARN(this->get_logger(), "Invalid pose from SLAM, skipping TF publish.");
        return;
    }
   
    // 2. Convert camera pose to world frame
    Eigen::Vector3f tc = Tcw.translation();
    Sophus::SE3f Twc = Tcw.inverse();
    // RCLCPP_INFO(this->get_logger(), "Twc: [%.3f, %.3f, %.3f]", 
    //            Twc.translation().x(), Twc.translation().y(), Twc.translation().z());
    // 3. Define camera-to-base transform (Tbc)
    // Note: Adjust these values according to your actual camera mounting position
    Eigen::Matrix4f Tbc_matrix = Eigen::Matrix4f::Identity();
    Tbc_matrix(0, 3) = 0.0f;   // X translation
    Tbc_matrix(1, 3) = 0.0f;   // Y translation 
    Tbc_matrix(2, 3) = 0.0f;   // Z translation

    // 4. Convert to base frame in world coordinates
    Sophus::SE3f Tbc(Tbc_matrix);
    Sophus::SE3f Twb = Twc * Tbc.inverse();

    // 5. Extract translation and rotation
    Eigen::Vector3f t = Twb.translation();
    Eigen::Matrix3f R = Twb.rotationMatrix();
    Eigen::Quaternionf q(R);

    // 6. 检查是否为零位置 (阈值设为很小的值)
    const float zero_threshold = 1e-6f;
    bool is_zero_pose = (std::abs(t.x()) < zero_threshold && 
                        std::abs(t.y()) < zero_threshold && 
                        std::abs(t.z()) < zero_threshold);

    // 7. 处理零位置的情况
    if (is_zero_pose) {
        // 如果这是第一次检测到零位置，打印警告
        if (!has_printed_zero_warning_) {
            RCLCPP_WARN(this->get_logger(), "Transform is zero [0.000, 0.000, 0.000] - filtering output. Will use last valid pose if available.");
            has_printed_zero_warning_ = true;
        }
        
        // 如果有有效的pose，使用最后的有效pose
        if (has_valid_pose_) {
            t = last_valid_translation_;
            q = last_valid_rotation_;
            
            // 只在状态从非零变为零时打印一次
            if (!last_pose_was_zero_) {
                RCLCPP_INFO(this->get_logger(), "Using last valid pose: [%.3f, %.3f, %.3f]", 
                           t.x(), t.y(), t.z());
            }
        } else {
            // 没有有效pose，跳过发布
            last_pose_was_zero_ = true;
            return;
        }
        last_pose_was_zero_ = true;
    } else {
        // 非零位置，保存为最后有效的pose
        last_valid_translation_ = t;
        last_valid_rotation_ = q;
        has_valid_pose_ = true;
        
        // 如果之前是零位置，现在恢复了，打印信息
        if (last_pose_was_zero_) {
            // RCLCPP_INFO(this->get_logger(), "Transform recovered from zero to valid pose: [%.3f, %.3f, %.3f]", 
            //            t.x(), t.y(), t.z());
            has_printed_zero_warning_ = false; // 重置警告标志，以便下次零位置时能再次警告
        }
        last_pose_was_zero_ = false;
        
      
    }

    // 8. Create and send TF message (使用有效的pose，无论是当前的还是最后有效的)
    if (tracking_state == 3 || has_valid_pose_) { // 如果tracking OK 或者有有效pose
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = stamp;
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
        RCLCPP_INFO(this->get_logger(), "t: [%.3f, %.3f, %.3f]", t.x(), t.y(), t.z());
        RCLCPP_DEBUG(this->get_logger(), "Published transform from %s to %s",
                    map_frame_id_.c_str(), base_frame_id_.c_str());
    }
}

