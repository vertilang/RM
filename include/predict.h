#pragma once
#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/time.h>
#include <thread>
#include <termio.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/src/Core/DenseBase.h>
class DenseBase;
using namespace std;
using namespace cv;
namespace Horizon{
    const static string yaml = "../param/camera_info.yaml";
    static const Mat k;//相机内参
    static const float RealHeight=5.7;
    static const float RealWidth=13.5;
    static const float z_c2w=-0.108;
    static const float x_c2w=0;
    static const float y_c2w=-0.0715;

enum class  CameraMoode
    {
        MonnocularCamera,//单目相机
        DriveFreeMonnocularCamera,//单目深度相机
        BinocularCamera,//双目相机
        DriveFreeBinocularCamera//双目深度相机
    };
enum class DECTORSTATE
    {
        CONTINUE,
        LOST,
        SWITCH,
        FIRSTFIND,
        BUFFERING
    };
enum class PREDICTORMODE{
    Directradiation,
    FOLLOW,
    ANTIGYRO,
    NONEPREDICT
    };
    static long now()
    {
        timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec*1000 + tv.tv_usec/1000;
    }
    class GimbalPose//相机位姿    左手系  上为y 前为z
    {
    public:
        float  pitch;
        float  yaw;
        float  roll;
        double timestamp;
        // 初始化函数
        GimbalPose(float pitch = 0.0, float yaw = 0.0, float roll = 0.0,double timestamp=0.0)
        {
            this->pitch     = pitch;
            this->yaw       = yaw;
            this->roll      = roll;
            this->timestamp=timestamp;

        }
        // 左值
        GimbalPose operator=(const GimbalPose& gm)
        {
            this->pitch     = gm.pitch;
            this->yaw       = gm.yaw;
            this->roll      = gm.roll;
            this->timestamp = gm.timestamp;
            return *this;
        }

        GimbalPose operator=(const float init_value)
        {
            this->pitch     = init_value;
            this->yaw       = init_value;
            this->roll      = init_value;
            this->timestamp = now();
            return *this;
        }

        friend GimbalPose operator-(const GimbalPose& gm1, const GimbalPose gm2)
        {
            GimbalPose temp{};
            temp.pitch     = gm1.pitch - gm2.pitch;
            temp.yaw       = gm1.yaw   - gm2.yaw;
            temp.roll      = gm1.roll  - gm2.roll;
            temp.timestamp = now();
            return temp;
        }

        friend GimbalPose operator+(const GimbalPose& gm1, const GimbalPose gm2)
        {
            GimbalPose temp{};
            temp.pitch     = gm1.pitch + gm2.pitch;
            temp.yaw       = gm1.yaw   + gm2.yaw;
            temp.roll      = gm1.roll  + gm2.roll;
            temp.timestamp = now();
            return temp;
        }

        friend GimbalPose operator*(const GimbalPose& gm, const float k)
        {
            GimbalPose temp{};
            temp.pitch     = gm.pitch * k;
            temp.yaw       = gm.yaw   * k;
            temp.roll      = gm.roll  * k;
            temp.timestamp = now();
            return temp;
        }

        friend GimbalPose operator*(const float k, const GimbalPose& gm)
        {
            GimbalPose temp{};
            temp.pitch     = gm.pitch * k;
            temp.yaw       = gm.yaw   * k;
            temp.roll      = gm.roll  * k;
            temp.timestamp = now();
            return temp ;
        }

        friend GimbalPose operator/(const GimbalPose& gm, const float k)
        {
            GimbalPose temp{};
            temp.pitch     = gm.pitch / k;
            temp.yaw       = gm.yaw   / k;
            temp.roll      = gm.roll  / k;
            temp.timestamp = now();
            return temp;
        }

        friend std::ostream& operator<<(std::ostream& out, const GimbalPose& gm)
        {
            out << "[pitch : " << gm.pitch << ", yaw : " << gm.yaw << "]" << endl;
            return out;
        }
    };
    class Armor
    {
    public:
        Eigen::Vector3f   center3d_;//三维中心点
        float     distance_;//装甲板距离
        GimbalPose  cur_pose_;//云台坐标ee
        long h_time_stamp_;//时间戳
        cv::Point2f pts[4];//四个点
    };
    enum THRESHOLD{
        LOST_TARGET = 10,
        SEEM_ARMOR_MIN_SPACE_DISTANCE = 30,     //相同装甲前后两帧相差的可允许最小空间距离差
        SEEM_ARMOR_MIN_PIEX_DISTANCE = 30       //相同装甲前后两帧相差的可允许最小像素距离差
    };
    class PnpSolver
    {
    public:
	    PnpSolver() = delete;					// 删除默认构造函数
	    PnpSolver(const string yaml);

	    Eigen::Vector3f poseCalculation(Armor &obj);
    public:
	    cv::Mat K_;								// 内参
	    cv::Mat distCoeffs_;					// 畸变系数
    public:
	    cv::Mat rotate_world_cam_;				// 从世界系到相机系的旋转矩阵
        bool is_large_;

    };
    class predictor
    {
    
    private:
        DECTORSTATE  current_dector_state_;     // 这一帧识别的状态
        PREDICTORMODE current_predict_mode_;    // 这一帧预测状态
        GimbalPose predictLocation();           // 预测位置，返回相机要转到装甲板的角度
        
        GimbalPose point_to_armor(Eigen::Vector3f point);
        
        //void JudgeState();                      // 判断状态
        //Point3f Iteration(cv::Point3f coordinate_m, float shootSpeed,float vx, float vz);
        
    public:
        std::shared_ptr<PnpSolver> pnp_solve_ = std::make_shared<PnpSolver>(yaml); 
        float v0;
        GimbalPose previous_gimbalpose_;        //上一次位姿
        GimbalPose current_gimbalpose_;         //当前位姿
        GimbalPose shot;
        //单张图片最多装甲板数量
        Armor target1;
        Armor target2;
        void init();
        Armor best_target_;                     //装甲板
        Armor previous_target_;                 //上一帧目标坐标
        bool is_have_data_;
        float delta_t_;                         //时间
    };

    /*
    * @brief:  相机坐标系转云台坐标系
    *
    * @parma:  gm是当前云台位姿，pos是要转换到云台的坐标
    *
    * @author: 李黔琦
    */
    Eigen::Vector3f cam3ptz(GimbalPose gm,Eigen::Vector3f &pos);

}