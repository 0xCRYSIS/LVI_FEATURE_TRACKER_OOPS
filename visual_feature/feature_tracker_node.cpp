#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0


// // mtx lock for two threads
// std::mutex mtx_lidar;

// // global variable for saving the depthCloud shared between two threads
// pcl::PointCloud<PointType>::Ptr depthCloud(new pcl::PointCloud<PointType>());

// // global variables saving the lidar point cloud
// deque<pcl::PointCloud<PointType>> cloudQueue;
// deque<double> timeQueue;

// // global depth register for obtaining depth of a feature
// DepthRegister *depthRegister;

// // feature publisher for VINS estimator
// // ros::Publisher pub_feature;
// // ros::Publisher pub_match;
// // ros::Publisher pub_restart;

// rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_feature;
// rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match;
// rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_restart;

// // feature tracker variables
// FeatureTracker trackerData[NUM_OF_CAM];
// double first_image_time;
// int pub_count = 1;
// bool first_image_flag = true;
// double last_image_time = 0;
// bool init_pub = 0;

// void img_callback(const sensor_msgs::msg::Image::SharedPtr img_msg)
// {
//     double cur_img_time = rclcpp::Time(img_msg->header.stamp).seconds();

//     if(first_image_flag)
//     {
//         first_image_flag = false;
//         first_image_time = cur_img_time;
//         last_image_time = cur_img_time;
//         return;
//     }
//     // detect unstable camera stream
//     if (cur_img_time - last_image_time > 1.0 || cur_img_time < last_image_time)
//     {
//         RCLCPP_WARN(rclcpp::get_logger("feature_tracker_node"),"image discontinue! reset the feature tracker!");
//         first_image_flag = true; 
//         last_image_time = 0;
//         pub_count = 1;
//         std_msgs::msg::Bool restart_flag;
//         restart_flag.data = true;
//         pub_restart->publish(restart_flag);
//         return;
//     }
//     last_image_time = cur_img_time;
//     // frequency control
//     if (round(1.0 * pub_count / (cur_img_time - first_image_time)) <= FREQ)
//     {
//         PUB_THIS_FRAME = true;
//         // reset the frequency control
//         if (abs(1.0 * pub_count / (cur_img_time - first_image_time) - FREQ) < 0.01 * FREQ)
//         {
//             first_image_time = cur_img_time;
//             pub_count = 0;
//         }
//     }
//     else
//     {
//         PUB_THIS_FRAME = false;
//     }

//     cv_bridge::CvImageConstPtr ptr;
//     if (img_msg->encoding == "8UC1")
//     {
//         sensor_msgs::msg::Image img;
//         img.header = img_msg->header;
//         img.height = img_msg->height;
//         img.width = img_msg->width;
//         img.is_bigendian = img_msg->is_bigendian;
//         img.step = img_msg->step;
//         img.data = img_msg->data;
//         img.encoding = "mono8";
//         ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
//     }
//     else
//         ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

//     cv::Mat show_img = ptr->image;
//     TicToc t_r;
//     for (int i = 0; i < NUM_OF_CAM; i++)
//     {
//         RCLCPP_DEBUG(rclcpp::get_logger("feature_tracker_node"),"processing camera %d", i);
//         if (i != 1 || !STEREO_TRACK)
//             trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), cur_img_time);
//         else
//         {
//             if (EQUALIZE)
//             {
//                 cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
//                 clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
//             }
//             else
//                 trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
//         }

//         #if SHOW_UNDISTORTION
//             trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
//         #endif
//     }

//     for (unsigned int i = 0;; i++)
//     {
//         bool completed = false;
//         for (int j = 0; j < NUM_OF_CAM; j++)
//             if (j != 1 || !STEREO_TRACK)
//                 completed |= trackerData[j].updateID(i);
//         if (!completed)
//             break;
//     }

//    if (PUB_THIS_FRAME)
//    {
//         pub_count++;
//         sensor_msgs::msg::PointCloud feature_points;
//         sensor_msgs::msg::ChannelFloat32 id_of_point;
//         sensor_msgs::msg::ChannelFloat32 u_of_point;
//         sensor_msgs::msg::ChannelFloat32 v_of_point;
//         sensor_msgs::msg::ChannelFloat32 velocity_x_of_point;
//         sensor_msgs::msg::ChannelFloat32 velocity_y_of_point;

//         feature_points.header.stamp = img_msg->header.stamp;
//         feature_points.header.frame_id = "vins_body";

//         vector<set<int>> hash_ids(NUM_OF_CAM);
//         for (int i = 0; i < NUM_OF_CAM; i++)
//         {
//             auto &un_pts = trackerData[i].cur_un_pts;
//             auto &cur_pts = trackerData[i].cur_pts;
//             auto &ids = trackerData[i].ids;
//             auto &pts_velocity = trackerData[i].pts_velocity;
//             for (unsigned int j = 0; j < ids.size(); j++)
//             {
//                 if (trackerData[i].track_cnt[j] > 1)
//                 {
//                     int p_id = ids[j];
//                     hash_ids[i].insert(p_id);
//                     geometry_msgs::msg::Point32 p;
//                     p.x = un_pts[j].x;
//                     p.y = un_pts[j].y;
//                     p.z = 1;

//                     feature_points.points.push_back(p);
//                     id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
//                     u_of_point.values.push_back(cur_pts[j].x);
//                     v_of_point.values.push_back(cur_pts[j].y);
//                     velocity_x_of_point.values.push_back(pts_velocity[j].x);
//                     velocity_y_of_point.values.push_back(pts_velocity[j].y);
//                 }
//             }
//         }

//         feature_points.channels.push_back(id_of_point);
//         feature_points.channels.push_back(u_of_point);
//         feature_points.channels.push_back(v_of_point);
//         feature_points.channels.push_back(velocity_x_of_point);
//         feature_points.channels.push_back(velocity_y_of_point);

//         // get feature depth from lidar point cloud
//         pcl::PointCloud<PointType>::Ptr depth_cloud_temp(new pcl::PointCloud<PointType>());
//         mtx_lidar.lock();
//         *depth_cloud_temp = *depthCloud;
//         mtx_lidar.unlock();

//         sensor_msgs::msg::ChannelFloat32 depth_of_points = depthRegister->get_depth(img_msg->header.stamp, show_img, depth_cloud_temp, trackerData[0].m_camera, feature_points.points);
//         feature_points.channels.push_back(depth_of_points);
        
//         // skip the first image; since no optical speed on frist image
//         if (!init_pub)
//         {
//             init_pub = 1;
//         }
//         else
//             pub_feature->publish(feature_points);

//         // publish features in image
//         if (pub_match->get_subscription_count() != 0)
//         {
//             ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::RGB8);
//             //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
//             cv::Mat stereo_img = ptr->image;

//             for (int i = 0; i < NUM_OF_CAM; i++)
//             {
//                 cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
//                 cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

//                 for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
//                 {
//                     if (SHOW_TRACK)
//                     {
//                         // track count
//                         double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
//                         cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(255 * (1 - len), 255 * len, 0), 4);
//                     } else {
//                         // depth 
//                         if(j < depth_of_points.values.size())
//                         {
//                             if (depth_of_points.values[j] > 0)
//                                 cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 255, 0), 4);
//                             else
//                                 cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 0, 255), 4);
//                         }
//                     }
//                 }
//             }

//             pub_match->publish(*ptr->toImageMsg().get());
//         }
//     }
// }

// void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr laser_msg)
// {
//     static int lidar_count = -1;
//     if (++lidar_count % (LIDAR_SKIP+1) != 0)
//         return;

//     // 0. listen to transform
//     // static tf::TransformListener listener;
//     // static tf::StampedTransform transform;

//     rclcpp::Node::SharedPtr listener_node = rclcpp::Node::make_shared("feature_tracker_tf_listener");
//     std::unique_ptr<tf2_ros::Buffer> tf_buffer = std::make_unique<tf2_ros::Buffer>(listener_node->get_clock());
//     std::shared_ptr<tf2_ros::TransformListener> listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

//     geometry_msgs::msg::TransformStamped transform;
 
//     try{
//         // listener.waitForTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, ros::Duration(0.01));
//         // listener.lookupTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, transform);

//         tf_buffer->canTransform("vins_world", "vins_body_ros",tf2::TimePointZero,tf2::durationFromSec(1.0));
//         transform = tf_buffer->lookupTransform("vins_world", "vins_body_ros",tf2::TimePointZero);
//     } 
//     catch (tf2::TransformException &ex){
//         // ROS_ERROR("lidar no tf");
//         RCLCPP_ERROR_STREAM(rclcpp::get_logger("feature_tracker_node"),ex.what());
//         return;
//     }

//     double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
//     // xCur = transform.getOrigin().x();
//     // yCur = transform.getOrigin().y();
//     // zCur = transform.getOrigin().z();

//     xCur = transform.transform.translation.x;
//     yCur = transform.transform.translation.y;
//     zCur = transform.transform.translation.x;

//     // tf::Matrix3x3 m(transform.getRotation());
//     tf2::Quaternion q(transform.transform.rotation.x,
//                       transform.transform.rotation.y,
//                       transform.transform.rotation.z,
//                       transform.transform.rotation.w);
//     tf2::Matrix3x3 m(q);

//     m.getRPY(rollCur, pitchCur, yawCur);
//     Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

//     // 1. convert laser cloud message to pcl
//     pcl::PointCloud<PointType>::Ptr laser_cloud_in(new pcl::PointCloud<PointType>());
//     pcl::fromROSMsg(*laser_msg, *laser_cloud_in);

//     // 2. downsample new cloud (save memory)
//     pcl::PointCloud<PointType>::Ptr laser_cloud_in_ds(new pcl::PointCloud<PointType>());
//     static pcl::VoxelGrid<PointType> downSizeFilter;
//     downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
//     downSizeFilter.setInputCloud(laser_cloud_in);
//     downSizeFilter.filter(*laser_cloud_in_ds);
//     *laser_cloud_in = *laser_cloud_in_ds;

//     // 3. filter lidar points (only keep points in camera view)
//     pcl::PointCloud<PointType>::Ptr laser_cloud_in_filter(new pcl::PointCloud<PointType>());
//     for (int i = 0; i < (int)laser_cloud_in->size(); ++i)
//     {
//         PointType p = laser_cloud_in->points[i];
//         if (p.x >= 0 && abs(p.y / p.x) <= 10 && abs(p.z / p.x) <= 10)
//             laser_cloud_in_filter->push_back(p);
//     }
//     *laser_cloud_in = *laser_cloud_in_filter;

//     // TODO: transform to IMU body frame
//     // 4. offset T_lidar -> T_camera 
//     pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());
//     Eigen::Affine3f transOffset = pcl::getTransformation(L_C_TX, L_C_TY, L_C_TZ, L_C_RX, L_C_RY, L_C_RZ);
//     pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_offset, transOffset);
//     *laser_cloud_in = *laser_cloud_offset;

//     // 5. transform new cloud into global odom frame
//     pcl::PointCloud<PointType>::Ptr laser_cloud_global(new pcl::PointCloud<PointType>());
//     pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_global, transNow);

//     // 6. save new cloud
//     double timeScanCur = rclcpp::Time(laser_msg->header.stamp).seconds();
//     cloudQueue.push_back(*laser_cloud_global);
//     timeQueue.push_back(timeScanCur);

//     // 7. pop old cloud
//     while (!timeQueue.empty())
//     {
//         if (timeScanCur - timeQueue.front() > 5.0)
//         {
//             cloudQueue.pop_front();
//             timeQueue.pop_front();
//         } else {
//             break;
//         }
//     }

//     std::lock_guard<std::mutex> lock(mtx_lidar);
//     // 8. fuse global cloud
//     depthCloud->clear();
//     for (int i = 0; i < (int)cloudQueue.size(); ++i)
//         *depthCloud += cloudQueue[i];

//     // 9. downsample global cloud
//     pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
//     downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
//     downSizeFilter.setInputCloud(depthCloud);
//     downSizeFilter.filter(*depthCloudDS);
//     *depthCloud = *depthCloudDS;
// }

// int main(int argc , char **argv)
// {
//     rclcpp::init(argc,argv);

//     rclcpp::Node::SharedPtr n = rclcpp::Node::make_shared("lvi_sam_featureTracker");

//     RCLCPP_INFO(n->get_logger(),"\033[1;32m----> Visual Feature Tracker Started.\033[0m");

//     std::cout << "before read camera params " << std::endl;

//     readParameters(n);

//     // read camera params
//     for (int i = 0; i < NUM_OF_CAM; i++)
//         trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

//     std::cout << "after read camera params " << std::endl;

//     // load fisheye mask to remove features on the boundry
//     if(FISHEYE)
//     {
//         for (int i = 0; i < NUM_OF_CAM; i++)
//         {
//             trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
//             if(!trackerData[i].fisheye_mask.data)
//             {
//                 RCLCPP_ERROR(n->get_logger(),"load fisheye mask fail");
//                 // ROS_BREAK();
//                 break;
//             }
//             else
//                 RCLCPP_INFO(n->get_logger(),"load mask success");
//         }
    
//     }

//     std::cout << "after FISHEYE : " << std::endl;

//     // initialize depthRegister (after readParameters())
//     depthRegister = new DepthRegister(n);

//     rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img = n->create_subscription<sensor_msgs::msg::Image>(IMAGE_TOPIC,5,img_callback);
//     rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar = n->create_subscription<sensor_msgs::msg::PointCloud2>(POINT_CLOUD_TOPIC,5,lidar_callback);

//     // if (!USE_LIDAR)
//     //     sub_lidar.shutdown();

//     pub_feature = n->create_publisher<sensor_msgs::msg::PointCloud>(PROJECT_NAME + "/vins/feature/feature",5);
//     pub_match   = n->create_publisher<sensor_msgs::msg::Image>(PROJECT_NAME + "/vins/feature/feature_img",5);
//     pub_restart = n->create_publisher<std_msgs::msg::Bool>(PROJECT_NAME + "/vins/feature/restart",5);    

//     rclcpp::executors::MultiThreadedExecutor exec;

//     exec.add_node(n);

//     exec.spin();

//     return 0;
// }

class FeatureTrackerNode : public rclcpp::Node
{
    public:

        // from DepthRegister class in feature_tracker.h
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_depth_cloud;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_depth_feature;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_image;

        std::unique_ptr<tf2_ros::Buffer> tf_buffer;
        std::shared_ptr<tf2_ros::TransformListener> listener{nullptr};

        const int num_bins = 360;
        vector<vector<PointType>> pointsArray;

        geometry_msgs::msg::TransformStamped transform;

        //////////// from feature_tracker_node.cpp
        // mtx lock for two threads
        std::mutex mtx_lidar;

        // global variable for saving the depthCloud shared between two threads
        // pcl::PointCloud<PointType>::Ptr depthCloud(new pcl::PointCloud<PointType>());
        // pcl::PointCloud<PointType>::Ptr depthCloud = new pcl::PointCloud<PointType>();
        pcl::PointCloud<PointType>::Ptr depthCloud{new pcl::PointCloud<PointType>()};

        // global variables saving the lidar point cloud
        deque<pcl::PointCloud<PointType>> cloudQueue;
        deque<double> timeQueue;

        rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_feature;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match;
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_restart;

        // feature tracker variables
        FeatureTracker trackerData[NUM_OF_CAM];
        double first_image_time;
        int pub_count = 1;
        bool first_image_flag = true;
        double last_image_time = 0;
        bool init_pub = 0;

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img ;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar ;


        FeatureTrackerNode(std::string NodeName) : Node(NodeName)
        {

            ///// from feature_tracker.cpp

            readParameters();

            // from DepthRegister class in feature_tracker.h
            pub_depth_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>(PROJECT_NAME + "/vins/depth/depth_cloud", 5);
            pub_depth_feature = this->create_publisher<sensor_msgs::msg::PointCloud2>(PROJECT_NAME + "/vins/depth/depth_feature", 5);
            pub_depth_image = this->create_publisher<sensor_msgs::msg::Image>(PROJECT_NAME + "/vins/depth/depth_image",   5);

            tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
            listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

            pointsArray.resize(num_bins);
            for (int i = 0; i < num_bins; ++i)
            {
                pointsArray[i].resize(num_bins);
            }
            
            ///// from feature_tracker.cpp
            
            // read camera params
            for (int i = 0; i < NUM_OF_CAM; i++)
                trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

            // load fisheye mask to remove features on the boundry
            if(FISHEYE)
            {
                for (int i = 0; i < NUM_OF_CAM; i++)
                {
                    trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
                    if(!trackerData[i].fisheye_mask.data)
                    {
                        RCLCPP_ERROR(this->get_logger(),"load fisheye mask fail");
                        break;
                    }
                    else
                        RCLCPP_INFO(this->get_logger(),"load mask success");
                }
            
            }

            sub_img = this->create_subscription<sensor_msgs::msg::Image>(IMAGE_TOPIC,5,std::bind(&FeatureTrackerNode::img_callback,this,std::placeholders::_1));
            sub_lidar = this->create_subscription<sensor_msgs::msg::PointCloud2>(POINT_CLOUD_TOPIC,5,std::bind(&FeatureTrackerNode::lidar_callback,this,std::placeholders::_1));

            pub_feature = this->create_publisher<sensor_msgs::msg::PointCloud>(PROJECT_NAME + "/vins/feature/feature",5);
            pub_match   = this->create_publisher<sensor_msgs::msg::Image>(PROJECT_NAME + "/vins/feature/feature_img",5);
            pub_restart = this->create_publisher<std_msgs::msg::Bool>(PROJECT_NAME + "/vins/feature/restart",5);

        }

        /// from parameters.h and parameters.cpp
        void readParameters()
        {
            std::string config_file;
            
            this->declare_parameter("vf_config_file","");
            this->get_parameter("vf_config_file",config_file);

            RCLCPP_INFO(this->get_logger(),"config file : %s",config_file.c_str());

            cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
            if(!fsSettings.isOpened())
            {
                std::cerr << "ERROR: Wrong path to settings" << std::endl;
            }

            // project name
            fsSettings["project_name"] >> PROJECT_NAME;
            // std::string pkg_path = ros::package::getPath(PROJECT_NAME);
            std::string pkg_path = ament_index_cpp::get_package_share_directory(PROJECT_NAME);

            // sensor topics
            fsSettings["image_topic"]       >> IMAGE_TOPIC;
            fsSettings["imu_topic"]         >> IMU_TOPIC;
            fsSettings["point_cloud_topic"] >> POINT_CLOUD_TOPIC;

            // lidar configurations
            fsSettings["use_lidar"] >> USE_LIDAR;
            fsSettings["lidar_skip"] >> LIDAR_SKIP;

            // feature and image settings
            MAX_CNT = fsSettings["max_cnt"];
            MIN_DIST = fsSettings["min_dist"];
            ROW = fsSettings["image_height"];
            COL = fsSettings["image_width"];
            FREQ = fsSettings["freq"];
            F_THRESHOLD = fsSettings["F_threshold"];
            SHOW_TRACK = fsSettings["show_track"];
            EQUALIZE = fsSettings["equalize"];

            L_C_TX = fsSettings["lidar_to_cam_tx"];
            L_C_TY = fsSettings["lidar_to_cam_ty"];
            L_C_TZ = fsSettings["lidar_to_cam_tz"];
            L_C_RX = fsSettings["lidar_to_cam_rx"];
            L_C_RY = fsSettings["lidar_to_cam_ry"];
            L_C_RZ = fsSettings["lidar_to_cam_rz"];

            // fisheye mask
            FISHEYE = fsSettings["fisheye"];
            if (FISHEYE == 1)
            {
                std::string mask_name;
                fsSettings["fisheye_mask"] >> mask_name;
                FISHEYE_MASK = pkg_path + mask_name;

                // std::cout << "FISHEYE : " << FISHEYE_MASK << std::endl;
                RCLCPP_INFO(this->get_logger(),"FISHEYE MASK : %s",FISHEYE_MASK.c_str());
            }

            // camera config
            CAM_NAMES.push_back(config_file);

            WINDOW_SIZE = 20;
            STEREO_TRACK = false;
            FOCAL_LENGTH = 460;
            PUB_THIS_FRAME = false;

            if (FREQ == 0)
                FREQ = 100;

            fsSettings.release();
            usleep(100);
        }

        float pointDistance(PointType p)
        {
            return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        }

        float pointDistance(PointType p1, PointType p2)
        {
            return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
        }

        void publishCloud(pcl::PointCloud<PointType>::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame)
        {
            if (pub_depth_cloud->get_subscription_count() == 0)
                return;
            sensor_msgs::msg::PointCloud2 tempCloud;
            pcl::toROSMsg(*thisCloud, tempCloud);
            tempCloud.header.stamp = thisStamp;
            tempCloud.header.frame_id = thisFrame;
            pub_depth_cloud->publish(tempCloud); 
        }

        // from DepthRegister class in feature_tracker.h

        sensor_msgs::msg::ChannelFloat32 get_depth(const rclcpp::Time& stamp_cur, const cv::Mat& imageCur, 
                                          const pcl::PointCloud<PointType>::Ptr& depthCloud,
                                          const camodocal::CameraPtr& camera_model ,
                                          const vector<geometry_msgs::msg::Point32>& features_2d)
        {
            // 0.1 initialize depth for return
            sensor_msgs::msg::ChannelFloat32 depth_of_point;
            depth_of_point.name = "depth";
            depth_of_point.values.resize(features_2d.size(), -1);

            // 0.2  check if depthCloud available
            if (depthCloud->size() == 0)
                return depth_of_point;

            // 0.3 look up transform at current image time
            try{

                // tf_buffer->canTransform("vins_world", "vins_body_ros",tf2::TimePointZero);
                transform = tf_buffer->lookupTransform("vins_world", "vins_body_ros",tf2::TimePointZero);
            } 
            catch (tf2::TransformException &ex){
                // ROS_ERROR("image no tf");
                RCLCPP_ERROR_STREAM(this->get_logger(),ex.what());
                return depth_of_point;
            }

            double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
        
            xCur = transform.transform.translation.x;
            yCur = transform.transform.translation.y;
            zCur = transform.transform.translation.z;

            tf2::Quaternion q(transform.transform.rotation.x,
                            transform.transform.rotation.y,
                            transform.transform.rotation.z,
                            transform.transform.rotation.w);
            tf2::Matrix3x3 m(q);
            m.getRPY(rollCur, pitchCur, yawCur);
            Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

            // 0.4 transform cloud from global frame to camera frame
            pcl::PointCloud<PointType>::Ptr depth_cloud_local(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*depthCloud, *depth_cloud_local, transNow.inverse());

            // 0.5 project undistorted normalized (z) 2d features onto a unit sphere
            pcl::PointCloud<PointType>::Ptr features_3d_sphere(new pcl::PointCloud<PointType>());
            for (int i = 0; i < (int)features_2d.size(); ++i)
            {
                // normalize 2d feature to a unit sphere
                Eigen::Vector3f feature_cur(features_2d[i].x, features_2d[i].y, features_2d[i].z); // z always equal to 1
                feature_cur.normalize(); 
                // convert to ROS standard
                PointType p;
                p.x =  feature_cur(2);
                p.y = -feature_cur(0);
                p.z = -feature_cur(1);
                p.intensity = -1; // intensity will be used to save depth
                features_3d_sphere->push_back(p);
            }

            // 3. project depth cloud on a range image, filter points satcked in the same region
            float bin_res = 180.0 / (float)num_bins; // currently only cover the space in front of lidar (-90 ~ 90)
            cv::Mat rangeImage = cv::Mat(num_bins, num_bins, CV_32F, cv::Scalar::all(FLT_MAX));

            for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
            {
                PointType p = depth_cloud_local->points[i];
                // filter points not in camera view
                if (p.x < 0 || abs(p.y / p.x) > 10 || abs(p.z / p.x) > 10)
                    continue;
                // find row id in range image
                float row_angle = atan2(p.z, sqrt(p.x * p.x + p.y * p.y)) * 180.0 / M_PI + 90.0; // degrees, bottom -> up, 0 -> 360
                int row_id = round(row_angle / bin_res);
                // find column id in range image
                float col_angle = atan2(p.x, p.y) * 180.0 / M_PI; // degrees, left -> right, 0 -> 360
                int col_id = round(col_angle / bin_res);
                // id may be out of boundary
                if (row_id < 0 || row_id >= num_bins || col_id < 0 || col_id >= num_bins)
                    continue;
                // only keep points that's closer
                float dist = pointDistance(p);
                if (dist < rangeImage.at<float>(row_id, col_id))
                {
                    rangeImage.at<float>(row_id, col_id) = dist;
                    pointsArray[row_id][col_id] = p;
                }
            }

            // 4. extract downsampled depth cloud from range image
            pcl::PointCloud<PointType>::Ptr depth_cloud_local_filter2(new pcl::PointCloud<PointType>());
            for (int i = 0; i < num_bins; ++i)
            {
                for (int j = 0; j < num_bins; ++j)
                {
                    if (rangeImage.at<float>(i, j) != FLT_MAX)
                        depth_cloud_local_filter2->push_back(pointsArray[i][j]);
                }
            }
            *depth_cloud_local = *depth_cloud_local_filter2;
            publishCloud(depth_cloud_local, stamp_cur, "vins_body_ros");

            // 5. project depth cloud onto a unit sphere
            pcl::PointCloud<PointType>::Ptr depth_cloud_unit_sphere(new pcl::PointCloud<PointType>());
            for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
            {
                PointType p = depth_cloud_local->points[i];
                float range = pointDistance(p);
                p.x /= range;
                p.y /= range;
                p.z /= range;
                p.intensity = range;
                depth_cloud_unit_sphere->push_back(p);
            }
            if (depth_cloud_unit_sphere->size() < 10)
                return depth_of_point;

            // 6. create a kd-tree using the spherical depth cloud
            pcl::KdTreeFLANN<PointType>::Ptr kdtree(new pcl::KdTreeFLANN<PointType>());
            kdtree->setInputCloud(depth_cloud_unit_sphere);

            // 7. find the feature depth using kd-tree
            vector<int> pointSearchInd;
            vector<float> pointSearchSqDis;
            float dist_sq_threshold = pow(sin(bin_res / 180.0 * M_PI) * 5.0, 2);
            for (int i = 0; i < (int)features_3d_sphere->size(); ++i)
            {
                kdtree->nearestKSearch(features_3d_sphere->points[i], 3, pointSearchInd, pointSearchSqDis);
                if (pointSearchInd.size() == 3 && pointSearchSqDis[2] < dist_sq_threshold)
                {
                    float r1 = depth_cloud_unit_sphere->points[pointSearchInd[0]].intensity;
                    Eigen::Vector3f A(depth_cloud_unit_sphere->points[pointSearchInd[0]].x * r1,
                                    depth_cloud_unit_sphere->points[pointSearchInd[0]].y * r1,
                                    depth_cloud_unit_sphere->points[pointSearchInd[0]].z * r1);

                    float r2 = depth_cloud_unit_sphere->points[pointSearchInd[1]].intensity;
                    Eigen::Vector3f B(depth_cloud_unit_sphere->points[pointSearchInd[1]].x * r2,
                                    depth_cloud_unit_sphere->points[pointSearchInd[1]].y * r2,
                                    depth_cloud_unit_sphere->points[pointSearchInd[1]].z * r2);

                    float r3 = depth_cloud_unit_sphere->points[pointSearchInd[2]].intensity;
                    Eigen::Vector3f C(depth_cloud_unit_sphere->points[pointSearchInd[2]].x * r3,
                                    depth_cloud_unit_sphere->points[pointSearchInd[2]].y * r3,
                                    depth_cloud_unit_sphere->points[pointSearchInd[2]].z * r3);

                    // https://math.stackexchange.com/questions/100439/determine-where-a-vector-will-intersect-a-plane
                    Eigen::Vector3f V(features_3d_sphere->points[i].x,
                                    features_3d_sphere->points[i].y,
                                    features_3d_sphere->points[i].z);

                    Eigen::Vector3f N = (A - B).cross(B - C);
                    float s = (N(0) * A(0) + N(1) * A(1) + N(2) * A(2)) 
                            / (N(0) * V(0) + N(1) * V(1) + N(2) * V(2));

                    float min_depth = min(r1, min(r2, r3));
                    float max_depth = max(r1, max(r2, r3));
                    if (max_depth - min_depth > 2 || s <= 0.5)
                    {
                        continue;
                    } else if (s - max_depth > 0) {
                        s = max_depth;
                    } else if (s - min_depth < 0) {
                        s = min_depth;
                    }
                    // convert feature into cartesian space if depth is available
                    features_3d_sphere->points[i].x *= s;
                    features_3d_sphere->points[i].y *= s;
                    features_3d_sphere->points[i].z *= s;
                    // the obtained depth here is for unit sphere, VINS estimator needs depth for normalized feature (by value z), (lidar x = camera z)
                    features_3d_sphere->points[i].intensity = features_3d_sphere->points[i].x;
                }
            }

            // visualize features in cartesian 3d space (including the feature without depth (default 1))
            publishCloud(features_3d_sphere, stamp_cur, "vins_body_ros");
            
            // update depth value for return
            for (int i = 0; i < (int)features_3d_sphere->size(); ++i)
            {
                if (features_3d_sphere->points[i].intensity > 3.0)
                    depth_of_point.values[i] = features_3d_sphere->points[i].intensity;
            }

            // visualization project points on image for visualization (it's slow!)
            if (pub_depth_image->get_subscription_count() != 0)
            {
                vector<cv::Point2f> points_2d;
                vector<float> points_distance;

                for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
                {
                    // convert points from 3D to 2D
                    Eigen::Vector3d p_3d(-depth_cloud_local->points[i].y,
                                        -depth_cloud_local->points[i].z,
                                        depth_cloud_local->points[i].x);
                    Eigen::Vector2d p_2d;
                    camera_model->spaceToPlane(p_3d, p_2d);
                    
                    points_2d.push_back(cv::Point2f(p_2d(0), p_2d(1)));
                    points_distance.push_back(pointDistance(depth_cloud_local->points[i]));
                }

                cv::Mat showImage, circleImage;
                cv::cvtColor(imageCur, showImage, cv::COLOR_GRAY2RGB);
                circleImage = showImage.clone();
                for (int i = 0; i < (int)points_2d.size(); ++i)
                {
                    float r, g, b;
                    getColor(points_distance[i], 50.0, r, g, b);
                    cv::circle(circleImage, points_2d[i], 0, cv::Scalar(r, g, b), 5);
                }
                cv::addWeighted(showImage, 1.0, circleImage, 0.7, 0, showImage); // blend camera image and circle image

                cv_bridge::CvImage bridge;
                bridge.image = showImage;
                bridge.encoding = "rgb8";
                sensor_msgs::msg::Image::SharedPtr imageShowPointer = bridge.toImageMsg();
                imageShowPointer->header.stamp = stamp_cur;
                pub_depth_image->publish(*imageShowPointer.get());
            }

            return depth_of_point;
        }

        void getColor(float p, float np, float&r, float&g, float&b) 
        {
            float inc = 6.0 / np;
            float x = p * inc;
            r = 0.0f; g = 0.0f; b = 0.0f;
            if ((0 <= x && x <= 1) || (5 <= x && x <= 6)) r = 1.0f;
            else if (4 <= x && x <= 5) r = x - 4;
            else if (1 <= x && x <= 2) r = 1.0f - (x - 1);

            if (1 <= x && x <= 3) g = 1.0f;
            else if (0 <= x && x <= 1) g = x - 0;
            else if (3 <= x && x <= 4) g = 1.0f - (x - 3);

            if (3 <= x && x <= 5) b = 1.0f;
            else if (2 <= x && x <= 3) b = x - 2;
            else if (5 <= x && x <= 6) b = 1.0f - (x - 5);
            r *= 255.0;
            g *= 255.0;
            b *= 255.0;
        }

        ////// form feature_tracker.cpp
        void img_callback(const sensor_msgs::msg::Image::SharedPtr img_msg)
        {
            double cur_img_time = rclcpp::Time(img_msg->header.stamp).seconds();

            if(first_image_flag)
            {
                first_image_flag = false;
                first_image_time = cur_img_time;
                last_image_time = cur_img_time;
                return;
            }
            // detect unstable camera stream
            if (cur_img_time - last_image_time > 1.0 || cur_img_time < last_image_time)
            {
                RCLCPP_WARN(this->get_logger(),"image discontinue! reset the feature tracker!");
                first_image_flag = true; 
                last_image_time = 0;
                pub_count = 1;
                std_msgs::msg::Bool restart_flag;
                restart_flag.data = true;
                pub_restart->publish(restart_flag);
                return;
            }
            last_image_time = cur_img_time;
            // frequency control
            if (round(1.0 * pub_count / (cur_img_time - first_image_time)) <= FREQ)
            {
                PUB_THIS_FRAME = true;
                // reset the frequency control
                if (abs(1.0 * pub_count / (cur_img_time - first_image_time) - FREQ) < 0.01 * FREQ)
                {
                    first_image_time = cur_img_time;
                    pub_count = 0;
                }
            }
            else
            {
                PUB_THIS_FRAME = false;
            }

            cv_bridge::CvImageConstPtr ptr;
            if (img_msg->encoding == "8UC1")
            {
                sensor_msgs::msg::Image img;
                img.header = img_msg->header;
                img.height = img_msg->height;
                img.width = img_msg->width;
                img.is_bigendian = img_msg->is_bigendian;
                img.step = img_msg->step;
                img.data = img_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

            cv::Mat show_img = ptr->image;
            TicToc t_r;
            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                RCLCPP_DEBUG(this->get_logger(),"processing camera %d", i);
                if (i != 1 || !STEREO_TRACK)
                    trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), cur_img_time);
                else
                {
                    if (EQUALIZE)
                    {
                        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                        clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
                    }
                    else
                        trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
                }

                #if SHOW_UNDISTORTION
                    trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
                #endif
            }

            for (unsigned int i = 0;; i++)
            {
                bool completed = false;
                for (int j = 0; j < NUM_OF_CAM; j++)
                    if (j != 1 || !STEREO_TRACK)
                        completed |= trackerData[j].updateID(i);
                if (!completed)
                    break;
            }

        if (PUB_THIS_FRAME)
        {
                pub_count++;
                sensor_msgs::msg::PointCloud feature_points;
                sensor_msgs::msg::ChannelFloat32 id_of_point;
                sensor_msgs::msg::ChannelFloat32 u_of_point;
                sensor_msgs::msg::ChannelFloat32 v_of_point;
                sensor_msgs::msg::ChannelFloat32 velocity_x_of_point;
                sensor_msgs::msg::ChannelFloat32 velocity_y_of_point;

                feature_points.header.stamp = img_msg->header.stamp;
                feature_points.header.frame_id = "vins_body";

                vector<set<int>> hash_ids(NUM_OF_CAM);
                for (int i = 0; i < NUM_OF_CAM; i++)
                {
                    auto &un_pts = trackerData[i].cur_un_pts;
                    auto &cur_pts = trackerData[i].cur_pts;
                    auto &ids = trackerData[i].ids;
                    auto &pts_velocity = trackerData[i].pts_velocity;
                    for (unsigned int j = 0; j < ids.size(); j++)
                    {
                        if (trackerData[i].track_cnt[j] > 1)
                        {
                            int p_id = ids[j];
                            hash_ids[i].insert(p_id);
                            geometry_msgs::msg::Point32 p;
                            p.x = un_pts[j].x;
                            p.y = un_pts[j].y;
                            p.z = 1;

                            feature_points.points.push_back(p);
                            id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                            u_of_point.values.push_back(cur_pts[j].x);
                            v_of_point.values.push_back(cur_pts[j].y);
                            velocity_x_of_point.values.push_back(pts_velocity[j].x);
                            velocity_y_of_point.values.push_back(pts_velocity[j].y);
                        }
                    }
                }

                feature_points.channels.push_back(id_of_point);
                feature_points.channels.push_back(u_of_point);
                feature_points.channels.push_back(v_of_point);
                feature_points.channels.push_back(velocity_x_of_point);
                feature_points.channels.push_back(velocity_y_of_point);

                // get feature depth from lidar point cloud
                pcl::PointCloud<PointType>::Ptr depth_cloud_temp(new pcl::PointCloud<PointType>());
                mtx_lidar.lock();
                *depth_cloud_temp = *depthCloud;
                mtx_lidar.unlock();

                sensor_msgs::msg::ChannelFloat32 depth_of_points = get_depth(img_msg->header.stamp, show_img, depth_cloud_temp, trackerData[0].m_camera, feature_points.points);
                feature_points.channels.push_back(depth_of_points);
                
                // skip the first image; since no optical speed on frist image
                if (!init_pub)
                {
                    init_pub = 1;
                }
                else
                    pub_feature->publish(feature_points);

                // publish features in image
                if (pub_match->get_subscription_count() != 0)
                {
                    ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::RGB8);
                    //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
                    cv::Mat stereo_img = ptr->image;

                    for (int i = 0; i < NUM_OF_CAM; i++)
                    {
                        cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                        cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                        for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                        {
                            if (SHOW_TRACK)
                            {
                                // track count
                                double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(255 * (1 - len), 255 * len, 0), 4);
                            } else {
                                // depth 
                                if(j < depth_of_points.values.size())
                                {
                                    if (depth_of_points.values[j] > 0)
                                        cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 255, 0), 4);
                                    else
                                        cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 0, 255), 4);
                                }
                            }
                        }
                    }

                    pub_match->publish(*ptr->toImageMsg().get());
                }
            }
        }

        void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr laser_msg)
        {
            static int lidar_count = -1;
            if (++lidar_count % (LIDAR_SKIP+1) != 0)
                return;

            geometry_msgs::msg::TransformStamped transform;
        
            try{
                
                // tf_buffer->canTransform("vins_world", "vins_body_ros",tf2::TimePointZero);
                transform = tf_buffer->lookupTransform("vins_world", "vins_body_ros",tf2::TimePointZero);
            } 
            catch (tf2::TransformException &ex){
                // ROS_ERROR("lidar no tf");
                RCLCPP_ERROR_STREAM(this->get_logger(),ex.what());
                return;
            }

            double xCur, yCur, zCur, rollCur, pitchCur, yawCur;

            xCur = transform.transform.translation.x;
            yCur = transform.transform.translation.y;
            zCur = transform.transform.translation.x;

            tf2::Quaternion q(transform.transform.rotation.x,
                            transform.transform.rotation.y,
                            transform.transform.rotation.z,
                            transform.transform.rotation.w);
            tf2::Matrix3x3 m(q);

            m.getRPY(rollCur, pitchCur, yawCur);
            Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

            // 1. convert laser cloud message to pcl
            pcl::PointCloud<PointType>::Ptr laser_cloud_in(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*laser_msg, *laser_cloud_in);

            // 2. downsample new cloud (save memory)
            pcl::PointCloud<PointType>::Ptr laser_cloud_in_ds(new pcl::PointCloud<PointType>());
            static pcl::VoxelGrid<PointType> downSizeFilter;
            downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
            downSizeFilter.setInputCloud(laser_cloud_in);
            downSizeFilter.filter(*laser_cloud_in_ds);
            *laser_cloud_in = *laser_cloud_in_ds;

            // 3. filter lidar points (only keep points in camera view)
            pcl::PointCloud<PointType>::Ptr laser_cloud_in_filter(new pcl::PointCloud<PointType>());
            for (int i = 0; i < (int)laser_cloud_in->size(); ++i)
            {
                PointType p = laser_cloud_in->points[i];
                if (p.x >= 0 && abs(p.y / p.x) <= 10 && abs(p.z / p.x) <= 10)
                    laser_cloud_in_filter->push_back(p);
            }
            *laser_cloud_in = *laser_cloud_in_filter;

            // TODO: transform to IMU body frame
            // 4. offset T_lidar -> T_camera 
            pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());
            Eigen::Affine3f transOffset = pcl::getTransformation(L_C_TX, L_C_TY, L_C_TZ, L_C_RX, L_C_RY, L_C_RZ);
            pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_offset, transOffset);
            *laser_cloud_in = *laser_cloud_offset;

            // 5. transform new cloud into global odom frame
            pcl::PointCloud<PointType>::Ptr laser_cloud_global(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_global, transNow);

            // 6. save new cloud
            double timeScanCur = rclcpp::Time(laser_msg->header.stamp).seconds();
            cloudQueue.push_back(*laser_cloud_global);
            timeQueue.push_back(timeScanCur);

            // 7. pop old cloud
            while (!timeQueue.empty())
            {
                if (timeScanCur - timeQueue.front() > 5.0)
                {
                    cloudQueue.pop_front();
                    timeQueue.pop_front();
                } else {
                    break;
                }
            }

            std::lock_guard<std::mutex> lock(mtx_lidar);
            // 8. fuse global cloud
            depthCloud->clear();
            for (int i = 0; i < (int)cloudQueue.size(); ++i)
                *depthCloud += cloudQueue[i];

            // 9. downsample global cloud
            pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
            downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
            downSizeFilter.setInputCloud(depthCloud);
            downSizeFilter.filter(*depthCloudDS);
            *depthCloud = *depthCloudDS;
        }
        


};

int main(int argc,char **argv)
{
    rclcpp::init(argc,argv);

    rclcpp::spin(std::make_shared<FeatureTrackerNode>("visual_feature_tracker"));

    rclcpp::shutdown();


    return 0;
}