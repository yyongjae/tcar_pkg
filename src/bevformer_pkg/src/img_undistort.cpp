#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp> // Fisheye 함수를 위해 필수!
#include <string>
#include <vector>
#include <memory> // for std::shared_ptr

/**
 * @class FastUndistorter
 * @brief cv::remap을 사용하여 고속으로 왜곡 보정을 수행하는 클래스.
 * 표준 모델과 Fisheye 모델을 모두 지원.
 */
class FastUndistorter {
public:
    /**
     * @brief 생성자
     * @param nh ROS 노드 핸들
     * @param camera_id 카메라 ID (예: "1", "2")
     * @param K Intrinsic 행렬 (3x3)
     * @param D Distortion 벡터 (Fisheye: 1x4, Standard: 1x5)
     * @param is_fisheye 이것이 Fisheye 모델인지 여부
     */
    FastUndistorter(ros::NodeHandle& nh, 
                    const std::string& camera_id, 
                    const cv::Mat& K, 
                    const cv::Mat& D, 
                    bool is_fisheye)
        : K_(K), D_(D), is_fisheye_(is_fisheye), maps_initialized_(false)
    {
        // 토픽 이름 동적 생성
        std::string sub_topic = "/camera_" + camera_id + "/compressed";
        std::string pub_topic = "/camera_" + camera_id + "_undistorted/compressed";

        // Subscriber 및 Publisher 초기화
        sub_ = nh.subscribe(sub_topic, 1, &FastUndistorter::imageCallback, this);
        pub_ = nh.advertise<sensor_msgs::CompressedImage>(pub_topic, 1);

        ROS_INFO("FastUndistorter: Initialized for /camera_%s. Fisheye mode: %s", 
                 camera_id.c_str(), is_fisheye ? "true" : "false");
    }

    /**
     * @brief 첫 프레임에서 고속 Remap을 위한 맵을 계산
     */
    void precomputeMaps(const cv::Size& image_size)
    {
        cv::Mat newK; // 보정 후 사용할 새 카메라 행렬
        
        if (is_fisheye_) {
            // Fisheye 모델: cv::fisheye::initUndistortRectifyMap
            // balance=0.0: 검은 영역 없이 꽉 채우도록 자름 (alpha=0과 유사)
            cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K_, D_, image_size, cv::Matx33d::eye(), newK, 0.0);
            cv::fisheye::initUndistortRectifyMap(K_, D_, cv::Matx33d::eye(), newK, image_size, CV_16SC2, map1_, map2_);
        } else {
            // 표준 모델: cv::initUndistortRectifyMap
            // alpha=0.0: 검은 영역 없이 꽉 채우도록 자름
            newK = cv::getOptimalNewCameraMatrix(K_, D_, image_size, 0.0);
            cv::initUndistortRectifyMap(K_, D_, cv::Mat(), newK, image_size, CV_16SC2, map1_, map2_);
        }
        
        maps_initialized_ = true;
        ROS_INFO("Initialized undistortion maps for %s (size: %d x %d)", 
                 sub_.getTopic().c_str(), image_size.width, image_size.height);
    }

    /**
     * @brief 이미지 메시지 콜백 함수
     */
    void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg)
    {
        try {
            // 1. CompressedImage -> cv::Mat 디코딩
            cv::Mat raw_data = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
            if (raw_data.empty()) {
                ROS_WARN_THROTTLE(1.0, "Failed to decode compressed image for topic %s", sub_.getTopic().c_str());
                return;
            }

            // 2. 첫 프레임인 경우, 맵(map)을 미리 계산
            if (!maps_initialized_) {
                precomputeMaps(raw_data.size());
            }

            // 3. 고속 왜곡 보정 (cv::remap 사용)
            cv::Mat undistorted;
            cv::remap(raw_data, undistorted, map1_, map2_, cv::INTER_LINEAR);

            // 4. cv::Mat -> CompressedImage 인코딩
            sensor_msgs::CompressedImage out_msg;
            out_msg.header = msg->header;
            out_msg.format = "jpeg";
            cv::imencode(".jpg", undistorted, out_msg.data);

            // 5. 발행
            pub_.publish(out_msg);
        }
        catch (cv::Exception& e) {
            ROS_ERROR("cv::Exception in callback for %s: %s", sub_.getTopic().c_str(), e.what());
        }
    }

private:
    ros::Subscriber sub_;
    ros::Publisher pub_;
    cv::Mat K_, D_;           // 원본 카메라 파라미터
    cv::Mat map1_, map2_;     // 고속 Remap을 위한 맵 (매우 중요)
    bool is_fisheye_;         // Fisheye 모델 여부
    bool maps_initialized_;   // 맵 초기화 여부
};


// --- Main ---
int main(int argc, char** argv)
{
    ros::init(argc, argv, "fast_undistort_node");
    ros::NodeHandle nh;

    // --- 1. Intrinsic (K) 행렬 정의 ---
    cv::Mat K1 = (cv::Mat_<double>(3, 3) << 864.499894, 0.0, 965.124919,
                                             0.0, 863.736716, 561.331455,
                                             0.0, 0.0, 1.0);
    cv::Mat K2 = (cv::Mat_<double>(3, 3) << 861.727905, 0.0, 969.340548,
                                             0.0, 860.320827, 514.296749,
                                             0.0, 0.0, 1.0);
    cv::Mat K3 = (cv::Mat_<double>(3, 3) << 1781.66124076, 0.0, 975.79031276,
                                             0.0, 1779.88894723, 558.55655286,
                                             0.0, 0.0, 1.0);
    cv::Mat K4 = (cv::Mat_<double>(3, 3) << 860.006404, 0.0, 944.868592,
                                             0.0, 851.338721, 547.489153,
                                             0.0, 0.0, 1.0);
    cv::Mat K5 = (cv::Mat_<double>(3, 3) << 852.68419, 0.0, 945.350571,
                                             0.0, 853.94132, 552.390304,
                                             0.0, 0.0, 1.0);
    cv::Mat K6 = (cv::Mat_<double>(3, 3) << 852.467418, 0.0, 989.195971,
                                             0.0, 840.372397, 547.152682,
                                             0.0, 0.0, 1.0);

    
    // --- 2. Distortion (D) 벡터 정의 ---
    // Fisheye 모델 (1, 2, 4, 5, 6)은 파라미터 4개 [k1, k2, k3, k4]
    cv::Mat D1 = (cv::Mat_<double>(1, 4) << -0.0338228, -0.01653735, 0.02970194, -0.01606049);
    cv::Mat D2 = (cv::Mat_<double>(1, 4) << -0.01486526, -0.03042306, 0.02531966, -0.01106005);
    cv::Mat D4 = (cv::Mat_<double>(1, 4) << 0.00289335, 0.01472757, -0.06029724, 0.02531067);
    cv::Mat D5 = (cv::Mat_<double>(1, 4) << -0.04224633, 0.06512146, -0.09313782, 0.0380891);
    cv::Mat D6 = (cv::Mat_<double>(1, 4) << 0.0, 0.0, 0.0, 0.0); // Fisheye지만 계수가 0

    // 표준 모델 (3)은 파라미터 5개 [k1, k2, p1, p2, k3]
    cv::Mat D3 = (cv::Mat_<double>(1, 5) << -0.30581, 0.07952, 0.00289, -0.00145, 0.12274);


    // --- 3. 6개의 Undistorter 객체 생성 ---
    std::vector<std::shared_ptr<FastUndistorter>> undistorters;
    
    // Fisheye (true)
    undistorters.push_back(std::make_shared<FastUndistorter>(nh, "1", K1, D1, true));
    undistorters.push_back(std::make_shared<FastUndistorter>(nh, "2", K2, D2, true));
    // Standard (false)
    undistorters.push_back(std::make_shared<FastUndistorter>(nh, "3", K3, D3, false));
    // Fisheye (true)
    undistorters.push_back(std::make_shared<FastUndistorter>(nh, "4", K4, D4, true));
    undistorters.push_back(std::make_shared<FastUndistorter>(nh, "5", K5, D5, true));
    undistorters.push_back(std::make_shared<FastUndistorter>(nh, "6", K6, D6, true));


    // 콜백 처리 시작 (AsyncSpinner를 사용하면 멀티코어 활용으로 더 빨라질 수 있음)
    // ros::spin(); // 싱글 스레드
    
    // 6개 카메라를 병렬로 처리하기 위해 AsyncSpinner 사용 (권장)
    ros::AsyncSpinner spinner(6); // 6개의 스레드 사용
    spinner.start();
    ros::waitForShutdown();
    
    return 0;
}