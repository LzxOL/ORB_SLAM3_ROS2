cmake_minimum_required(VERSION 3.5)
project(orbslam3)

# ====================================================================
# ============== [手动添加的路径配置 - 开始] ===========================
# 
# 为了确保CMake能找到原生的ORB_SLAM3库，我们在这里直接指定它的路径。
# 这会告诉 find_package(ORB_SLAM3) 在这个路径下进行搜索。
# 请确保下面的路径是你自己系统中原生ORB_SLAM3的真实根目录。
#
set(ORB_SLAM3_DIR "/home/robot/orbslam3/ORB_SLAM3")
#
# ================= [手动添加的路径配置 - 结束] ============================
# ====================================================================


# You should set the PYTHONPATH to your own python site-packages path
# (下面这行通常不需要修改，它指向ROS2自带的Python环境)
set(ENV{PYTHONPATH} "/opt/ros/humble/lib/python3.10/site-packages")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ROS2 and other required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
# Find external non-ROS dependencies
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
# 现在，由于我们设置了 ORB_SLAM3_DIR, find_package 会在那里搜索
find_package(ORB_SLAM3 REQUIRED)
# Explicitly find OpenCV and its required components to solve linking issues
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d features2d)
find_package(nav_msgs REQUIRED)

# Include directories
include_directories(
  include
  ${ORB_SLAM3_ROOT_DIR}/include
  ${ORB_SLAM3_ROOT_DIR}/include/CameraModels
  ${OpenCV_INCLUDE_DIRS}
)

# This is usually not needed if include_directories is used, but kept for safety
link_directories(
  include
)

# --- Define Executables and Dependencies ---

# Monocular Node
add_executable(mono
  src/monocular/mono.cpp
  src/monocular/monocular-slam-node.cpp
)
ament_target_dependencies(mono rclcpp sensor_msgs cv_bridge ORB_SLAM3 Pangolin)
target_link_libraries(mono ${OpenCV_LIBRARIES})

# RGB-D Node
add_executable(rgbd
  src/rgbd/rgbd.cpp
  src/rgbd/rgbd-slam-node.cpp
)
ament_target_dependencies(rgbd rclcpp sensor_msgs tf2_ros cv_bridge message_filters ORB_SLAM3 Pangolin nav_msgs)
target_link_libraries(rgbd ${OpenCV_LIBRARIES})

# # Stereo Node
# add_executable(stereo
#   src/stereo/stereo.cpp
#   src/stereo/stereo-slam-node.cpp
# )
# ament_target_dependencies(stereo rclcpp sensor_msgs cv_bridge message_filters ORB_SLAM3 Pangolin tf2_ros nav_msgs)
# target_link_libraries(ster`eo tf2_ros::tf2_ros ${OpenCV_LIBRARIES})

# Stereo-Inertial Node
add_executable(stereo-inertial
  src/stereo-inertial/stereo-inertial.cpp
  src/stereo-inertial/stereo-inertial-node.cpp
)
ament_target_dependencies(stereo-inertial rclcpp sensor_msgs cv_bridge ORB_SLAM3 tf2_ros message_filters Pangolin nav_msgs)
target_link_libraries(stereo-inertial ${OpenCV_LIBRARIES})

# --- Installation ---

install(TARGETS mono rgbd 
# stereo 
stereo-inertial
  DESTINATION lib/${PROJECT_NAME})

# ====================================================================
# ============== [手动添加的安装指令 - 开始] ===========================
# 
# 将我们创建的 launch 文件和 config 文件安装到ROS2的share目录，
# 这样 ros2 launch 和 ament_index_python 才能找到它们。
#
install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)
#
# ================= [手动添加的安装指令 - 结束] ============================
# ====================================================================

ament_package()