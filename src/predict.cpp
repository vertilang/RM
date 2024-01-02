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
            return_gimbalpose = followTarget();
            break;
        }
        case PREDICTORMODE::ANTIGYRO:
        {

            return_gimbalpose = antiGyroTarget();
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
    point_to_armor.yaw = std::atan(point[0]/point[2])*180/CV_PI;
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
GimbalPose predictor::followTarget() //预测下一帧目标位置   世界坐标系不变
{   
    // Eigen::Vector3f point1;
    // point1 << previous_target_.center3d_.x,previous_target_.center3d_.y,previous_target_.center3d_.z;
    // point1 = cam3ptz(previous_gimbalpose_,point1);
    Eigen::Vector3f point2,point_shot;
    // point2 << best_target_.center3d_.x,best_target_.center3d_.y,best_target_.center3d_.z;
    // point2 = cam3ptz(current_gimbalpose_,point2);
    // point_shot<<2*point2[0]-point1[0],2*point2[1]-point1[1],2*point2[2]-point1[2];
    // point_shot=cam3ptz(current_gimbalpose_,point_shot);
    return point_to_armor(point_shot);
}
GimbalPose predictor::antiGyroTarget()//选取一帧图片有两块装甲板的图片
{   
    // Eigen::Vector3f p1l;
    // p1l << target1.light[0].x, target1.light[0].y, target1.light[0].z;
    // p1l= cam3ptz(current_gimbalpose_,p1l);
    // Eigen::Vector3f p1r;
    // p1r<<  target1.light[1].x, target1.light[1].y, target1.light[1].z;
    // p1r= cam3ptz(current_gimbalpose_,p1r);
    // Eigen::Vector3f p2r;
    // p2r<<  target2.light[1].x, target2.light[1].y, target2.light[1].z;
    // p2r= cam3ptz(current_gimbalpose_,p2r);
    // Eigen::Matrix2f den, ater1,ater2;
    // den<<
    // p1l[0]-p1r[0],p1l[2]-p1r[2],
    // p1l[0]-p2r[0],p1l[2]-p2r[2];
    // ater1<<
    // p1r[0]*p1r[0]-p1l[0]*p1l[0]+p1r[2]*p1r[2]-p1l[2]*p1l[2],p1l[2]-p1r[2],
    // p2r[0]*p2r[0]-p1l[0]*p1l[0]+p2r[2]*p2r[2]-p1l[2]*p1l[2],p1l[2]-p2r[2];
    // ater2<<
    // p1l[0]-p1r[0],p1r[0]*p1r[0]-p1l[0]*p1l[0]+p1r[2]*p1r[2]-p1l[2]*p1l[2],
    // p1l[0]-p2r[0],p2r[0]*p2r[0]-p1l[0]*p1l[0]+p2r[2]*p2r[2]-p1l[2]*p1l[2];
    // float d=ater1.determinant()/den.determinant();
    // float e=ater2.determinant()/den.determinant();
    // Point3f cercle;
    // cercle.x=-d/2;
    // cercle.y=-e/2;
    // cercle.z=p1l[2];
    // p1r[0]=p1r[0]-cercle.x;
    // p1r[2]=p1r[2]-cercle.z;
    // p2r[0]=p2r[0]-cercle.x;
    // p2r[2]=p2r[2]-cercle.x;
    // float cosx=(p1r[0]*p2r[0]+p1r[2]*p2r[2])/(pow(p1r[0]*p1r[0]+p1r[2]*p1r[2],0.5)*pow(p2r[0]*p2r[0]+p2r[2]*p2r[2],0.5));
    // float x=acos(cosx);

    
    // Eigen::AngleAxis<float> rotation_vector(x,Eigen::Vector3f(cercle.x,cercle.y,cercle.z));
    // Eigen::Matrix3f rotation_matrix=Eigen::Matrix3f::Identity();
    // rotation_matrix=rotation_vector.toRotationMatrix();
    Eigen::Vector3f point;
    // best_target_=best_target();
    // point<<best_target_.center3d_.x,best_target_.center3d_.y,best_target_.center3d_.z;
    // point=rotation_matrix*point;
    return point_to_armor(point);
}
Armor predictor::best_target()
{
    Armor to_target;
    // Eigen::Vector3f point1l;
    // point1l << target1.light[0].x, target1.light[0].y, target1.light[0].z;
    // point1l= cam3ptz(current_gimbalpose_,point1l);
    // Eigen::Vector3f point1r;
    // point1r<<  target1.light[1].x, target1.light[1].y, target1.light[1].z;
    // point1r= cam3ptz(current_gimbalpose_,point1r);
    // float angle=sin(point1l[0]/pow(pow(point1l[0],2)+pow(point1r[2],2),0.5));
    // if(angle>pow(3,0.5)/2&&angle<0.5)
    // {
    //     to_target=target1;
    // }
    // else to_target=target2;
    return to_target;
}
Eigen::Vector3f cam3ptz(GimbalPose gm,Eigen::Vector3f &pos)
    {
        pos[0]+=x_c2w;
        pos[1]+=y_c2w;
        pos[2]+=z_c2w;
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
        //cout<<"w"<<t_pos_[0]<<"   "<<t_pos_[1]<<"   "<<t_pos_[2]<<"   "<<endl;
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
		float fHalfY = RealHeight * 0.5f; // 将装甲板的宽的一半作为原点的y
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
        //cout<<"c"<<coord[0]<<"   "<<coord[1]<<"   "<<coord[2]<<endl;
        coord=cam3ptz(obj.cur_pose_,coord);
        //cout<<obj.cur_pose_.pitch<<"   "<<obj.cur_pose_.yaw<<endl;
	    return coord;
    }


}