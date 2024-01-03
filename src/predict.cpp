#include"../include/predict.h"
#define g 9.80665
#define w 1.5
namespace Horizon{
void predictor::init()
{
    
    best_target_.center3d_=pnp_solve_->poseCalculation(best_target_);
    best_target_.cur_pose_=predictLocation();
    
}
    /*
    @brief  预测敌方控制方式并跟踪
    @author liqianqi
    @return 相机位姿
    */
GimbalPose predictor::predictLocation()
{
    GimbalPose return_gimbalpose;
    current_predict_mode_ = PREDICTORMODE::Directradiation;

    switch(current_predict_mode_)
    {
        case PREDICTORMODE::Directradiation:
        {
            return_gimbalpose=point_to_armor(best_target_.center3d_);
            break;
        }
        case PREDICTORMODE::FOLLOW:
        {   
            //return_gimbalpose = followTarget();
            break;
        }
        case PREDICTORMODE::ANTIGYRO:
        {

            //return_gimbalpose = antiGyroTarget();
            break;
        }
        case PREDICTORMODE::NONEPREDICT :{
            return_gimbalpose.yaw = 0.0;
            return_gimbalpose.pitch = 0.0;
            best_target_.h_time_stamp_=0.0;
            break;
        }
    }
    return return_gimbalpose;

}

GimbalPose predictor::point_to_armor(Eigen::Vector3f point) //将相机转向目标 没有空气阻力  
{
    //yaw
    GimbalPose point_to_armor;
    float dis;
    dis=std::pow(point[0]*point[0]+point[2]*point[2],0.5);
    point_to_armor.yaw = std::asin(point[0]/dis)*180/CV_PI;
    if (point[2] < 0 && point[0] > 0)
	{
		point_to_armor.yaw=180-point_to_armor.yaw;
	}
	else if (point[2] < 0 && point[0] < 0)
	{
		point_to_armor.yaw=-180+point_to_armor.yaw;
	}
    //std::cout<<point_to_armor.yaw<<std::endl;
    //pitch   //斜抛运动求角度
    float a = -0.5*g*(std::pow(dis,2)/std::pow(v0,2));
    float b = dis;
    float c = a - point[1];
    float Discriminant = (float)(std::pow(b,2) - 4*a*c);  //判别式
    //cout<<Discriminant<<endl;
    if(Discriminant < 0) 
    {
        return -1;
    }
    float tan_angle1 = (-b + std::pow(Discriminant,0.5))/(2*a);
    float tan_angle2 = (-b - std::pow(Discriminant,0.5))/(2*a);

    float angle1 = std::atan(tan_angle1)*180/CV_PI;
    float angle2 = std::atan(tan_angle2)*180/CV_PI;

	if (tan_angle1 >=-3  && tan_angle1 <=3 ) 
    {   
        //std::cout << "pitch1     " <<tan_angle1 << std::endl;
        point_to_armor.pitch = angle1;
	}
	else
    {
        //std::cout << "pitch2     " << tan_angle2 << std::endl;
        point_to_armor.pitch = angle2;
	}
    
    return point_to_armor;
}
Eigen::Vector3f cam3ptz(GimbalPose gm,Eigen::Vector3f &pos)
    {
        pos[0]=pos[0]+x_c2w;
        pos[1]=pos[1]+y_c2w;
        pos[2]=pos[2]+z_c2w;
        pos = pos.transpose();//转置
        //cout<<gm.pitch<<"   "<<gm.yaw<<endl;
        Eigen::Matrix3f pitch_rotation_matrix_;
        Eigen::Matrix3f yaw_rotation_matrix_;

        gm.pitch = (gm.pitch*CV_PI)/180;
        gm.yaw = (gm.yaw*CV_PI)/180;

        pitch_rotation_matrix_
        <<
        1.0,              0.0,              0.0,
        0.0,  std::cos(gm.pitch),  std::sin(gm.pitch),
        0.0,  -std::sin(gm.pitch), std::cos(gm.pitch);

        yaw_rotation_matrix_
        <<
        std::cos(gm.yaw),  0.0,  std::sin(gm.yaw),
        0.0,               1.0,            0.0,
        -std::sin(gm.yaw),  0.0,  std::cos(gm.yaw);
        Eigen::Vector3f t_pos_;
        t_pos_ = yaw_rotation_matrix_ * pitch_rotation_matrix_ * pos;
        cout<<"w"<<t_pos_[0]<<"   "<<t_pos_[1]<<"   "<<t_pos_[2]<<"   "<<endl;
        return t_pos_;
    }
    PnpSolver::PnpSolver(const string yaml)
    {
	    cv::FileStorage fs(yaml, cv::FileStorage::READ);
	    fs["M1"] >> K_;
	    fs["D1"] >> distCoeffs_;
	    fs.release();
    }
    /**
    * @brief:  位姿解算器
    *
    * @author: liqianqi
    *
    * @param:  obj: 装甲板信息，主要用四点
    *
    * @return: 装甲板在相机系的位置和姿态
    */
    Eigen::Vector3f PnpSolver::poseCalculation(Armor &obj)
    {   
	
	    std::vector<cv::Point3f> point_in_world; // 装甲板世界坐标系
	    std::vector<cv::Point2f> point_in_pixe; // 像素坐标点
	    //float id = obj.label;
	    point_in_pixe.push_back(obj.pts[0]);
	    point_in_pixe.push_back(obj.pts[1]);
	    point_in_pixe.push_back(obj.pts[2]);
	    point_in_pixe.push_back(obj.pts[3]);
		
		float fHalfX = RealWidth * 0.5f;	 // 将装甲板的宽的一半作为原点的x
		float fHalfY = RealHeight * 0.5f; //将装甲板的宽的一半作为原点的y
		point_in_world.emplace_back(cv::Point3f(-fHalfX, fHalfY, 0));
		point_in_world.emplace_back(cv::Point3f( fHalfX, fHalfY, 0));
		point_in_world.emplace_back(cv::Point3f(-fHalfX,-fHalfY, 0));
		point_in_world.emplace_back(cv::Point3f( fHalfX,-fHalfY, 0));

        cv::Mat rvecs = cv::Mat::zeros(3, 1, CV_64FC1);
	    cv::Mat tvecs = cv::Mat::zeros(3, 1, CV_64FC1);

	    if (point_in_world.size() == 4 && point_in_pixe.size() == 4)
	    {
	        // 世界坐标系到相机坐标系的变换
	        // tvecs 表示从相机系到世界系的平移向量并在相机系下的坐标
	        // rvecs 表示从相机系到世界系的旋转向量，需要做进一步的转换
	        // 默认迭代法: SOLVEPNP_ITERATIVE，最常规的用法，精度较高，速度适中，一次解算一次耗时不到1ms
            cv::solvePnP(point_in_world, point_in_pixe, K_, distCoeffs_, rvecs, tvecs, false, cv::SOLVEPNP_SQPNP);
	    }
	    else
	    {
		    std::cout << "[world] size " << point_in_world.size() << std::endl;
		    std::cout << "[pixe] size " << point_in_pixe.size() << std::endl;
	    }

	    cv::Mat rotM = cv::Mat::zeros(3, 3, CV_64FC1); // 解算出来的旋转矩阵

	    // 将旋转矩阵分解为三个轴的欧拉角（roll、pitch、yaw）
	    //Eigen::Vector3d euler_angles = get_euler_angle(rvecs);
        //obj.cur_pose_<<euler_angles[0],euler_angles[1],euler_angles[2];

	    Eigen::Vector3f coord;
	    coord << tvecs.ptr<double>(0)[0] / 100, -tvecs.ptr<double>(0)[1] / 100, tvecs.ptr<double>(0)[2] / 100;
        cout<<"c"<<coord[0]<<"   "<<coord[1]<<"   "<<coord[2]<<endl;
        coord=cam3ptz(obj.cur_pose_,coord);
        //cout<<obj.cur_pose_.pitch<<"   "<<obj.cur_pose_.yaw<<endl;
	    return coord;
    }


}