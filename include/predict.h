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
#include <ceres/jet.h>
#include <ceres/ceres.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/src/Core/DenseBase.h>
class DenseBase;
using namespace std;
using namespace cv;
namespace Horizon{
    static  struct timeval Time_all;
    const static string yaml = "../param/camera_info.yaml";
    static const Mat k;//相机内参
    static const float RealHeight=5.7;
    static const float RealWidth=13.5;
    //哨兵
    // static const float z_c2w=0.108;
    // static const float x_c2w=0;
    // static const float y_c2w=0.0715;
    //英雄
    static const float z_c2w=0.065;
    static const float x_c2w=0;
    static const float y_c2w=0.09;
    
    static const float velocities_deque_size_ = 15;

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
        GimbalPose  cur_pose_;//云台坐标
        Eigen::Vector3f velocitie;
        long time;//时间戳
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
	    GimbalPose rotate_world_cam_;				// 从世界系到相机系的旋转矩阵
        bool is_large_;

    };
    /**
    * @brief  自适应扩展卡尔曼滤波, 花山甲老师写的自适应扩展卡尔曼滤波实在优雅
    * @author 上交:唐欣阳
    */
    template <int N_X, int N_Y>
    class AdaptiveEKF
    {   
        using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
        using MatrixYX = Eigen::Matrix<double, N_Y, N_X>;
        using MatrixXY = Eigen::Matrix<double, N_X, N_Y>;
        using MatrixYY = Eigen::Matrix<double, N_Y, N_Y>;
        using VectorX = Eigen::Matrix<double, N_X, 1>;
        using VectorY = Eigen::Matrix<double, N_Y, 1>;

    public:
        explicit AdaptiveEKF(const VectorX &X0 = VectorX::Zero())
            : Xe(X0), P(MatrixXX::Identity())
        {
            std::cout << P << std::endl;
            Q << 0.15 ,0,    0,     0,  0,  0,
                0,    1,  0,     0,  0,  0,
                0,    0,    0.1,   0,  0,  0,
                0,    0,    0,    0.3, 0,  0,
                0,    0,    0,     0,  0.5,0,
                0,    0,    0,     0,  0,  1;

            R << 4,0,0,
                0,1,0,
                0,0,0.5;
        }
    
        void init(const VectorX &X0 = VectorX::Zero()) 
        {
            Xe = X0;
        }

        template <class Func>
        VectorX predict(Func &&func)
        {
            ceres::Jet<double, N_X> Xe_auto_jet[N_X];

            for (int i = 0; i < N_X; i++)
            {
                Xe_auto_jet[i].a = Xe[i];
                Xe_auto_jet[i].v[i] = 1;
            }

            ceres::Jet<double, N_X> Xp_auto_jet[N_X];
            func(Xe_auto_jet, Xp_auto_jet);
            for (int i = 0; i < N_X; i++)
            {
                Xp[i] = Xp_auto_jet[i].a;
                F.block(i, 0, 1, N_X) = Xp_auto_jet[i].v.transpose();
            }
            std::cout << F * P * F.transpose() << std::endl;
            P = F * P * F.transpose() + Q;
            std::cout << "predict variables is x: " << Xp[0] << " y: " << Xp[2] << " z: " << Xp[4] << std::endl; 

            return Xp;
        }

        template <class Func>
        VectorX update(Func &&func, const VectorY &Y)
        {
            ceres::Jet<double, N_X> Xp_auto_jet[N_X];
            for (int i = 0; i < N_X; i++)
            {
                Xp_auto_jet[i].a = Xp[i];
                Xp_auto_jet[i].v[i] = 1;
            }
            ceres::Jet<double, N_X> Yp_auto_jet[N_Y];
            func(Xp_auto_jet, Yp_auto_jet);
            for (int i = 0; i < N_Y; i++)
            {
                Yp[i] = Yp_auto_jet[i].a;
                H.block(i, 0, 1, N_X) = Yp_auto_jet[i].v.transpose();
            }
            K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

            Xe = Xp + K * (Y - Yp);
            P = (MatrixXX::Identity() - K * H) * P;

            std::cout << "update variables is x: " << Xe[0] << " y: " << Xe[2] << " z: " << Xe[4] << std::endl; 
            return Xe;
        }

        VectorX Xe; // 估计状态变量
        VectorX Xp; // 预测状态变量
        MatrixXX F; // 预测雅克比
        MatrixYX H; // 观测雅克比
        MatrixXX P; // 状态协方差
        MatrixXX Q; // 预测过程协方差
        MatrixYY R; // 观测过程协方差
        MatrixXY K; // 卡尔曼增益
        VectorY Yp; // 预测观测量
    };

    class predictor
    {
    
    private:
        DECTORSTATE  current_dector_state_;     // 这一帧识别的状态
        //PREDICTORMODE current_predict_mode_;    // 这一帧预测状态
        GimbalPose predictLocation();           // 预测位置，返回相机要转到装甲板的角度
        GimbalPose point_to_armor(Eigen::Vector3f point);

        GimbalPose antigyro_Armor(Armor target);
        std::deque<Armor> velocities_;// 速度的循环队列，方便做拟合，装甲板切换初始化
        Eigen::Vector3f CeresVelocity(std::deque<Armor> target);
        Armor ArmorSelect(std::vector<Armor> &objects);
    public:
        std::shared_ptr<PnpSolver> pnp_solve_ = std::make_shared<PnpSolver>(yaml); 
        float v0;
        GimbalPose previous_gimbalpose_;        //上一次位姿
        GimbalPose current_gimbalpose_;         //当前位姿
        //单张图片最多装甲板数量
        void init();
        Armor best_target_;                     //装甲板
        Armor previous_target_;                 //上一帧目标坐标
        bool is_have_data_;
        float delta_t_;                         //时间
        std::vector<Armor> objects;
        PREDICTORMODE current_predict_mode_;    // 这一帧预测状态
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