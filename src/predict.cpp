#include"../include/predict.h"
#define g 9.80665
#define w 1.5
namespace Horizon{
void predictor::init()
{
	
	gettimeofday(&Time_all, NULL);	
    best_target_=ArmorSelect(objects);
	current_predict_mode_ = PREDICTORMODE::Directradiation;
	if(best_target_.center3d_[0]-previous_target_.center3d_[0]>30)
	{
		current_predict_mode_ = PREDICTORMODE::ANTIGYRO;
	}
	else if(previous_target_.time-Time_all.tv_usec>1000)
	{
		current_predict_mode_ = PREDICTORMODE::NONEPREDICT;
	}
	else
	{
		current_predict_mode_ = PREDICTORMODE::Directradiation;
	}
    best_target_.cur_pose_=predictLocation();
	objects.clear();
	previous_target_=best_target_;
    
}
    /*
    @brief  预测敌方控制方式并跟踪
    @author liqianqi
    @return 相机位姿
    */
GimbalPose predictor::predictLocation()
{
    GimbalPose return_gimbalpose;


    switch(current_predict_mode_)
    {
        case PREDICTORMODE::Directradiation:
        {
            return_gimbalpose=point_to_armor(best_target_.center3d_);
            break;
        }
        case PREDICTORMODE::FOLLOW:
        {
			break;
        }
        case PREDICTORMODE::ANTIGYRO:
        {
			if(velocities_.size()<velocities_deque_size_)
			{
				velocities_.push_back(best_target_);
			}
			else
			{
				velocities_.pop_front();
				velocities_.push_back(best_target_);
			}
			best_target_.velocitie=CeresVelocity(velocities_);
			return_gimbalpose=antigyro_Armor(best_target_);
            break;
        }
        case PREDICTORMODE::NONEPREDICT :{
            return_gimbalpose.yaw = previous_target_.cur_pose_.yaw;
            return_gimbalpose.pitch = previous_target_.cur_pose_.pitch;
            best_target_.time=0.0;
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

	    // cv::Mat rotM = cv::Mat::zeros(3, 3, CV_64FC1); // 解算出来的旋转矩阵
		// Eigen::Matrix3d rotM_eigen;
		// cv::cv2eigen(rotM, rotM_eigen);
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
	Eigen::Vector3f predictor::CeresVelocity(std::deque<Armor> target) // 最小二乘法拟合速度
	{
		int N = target.size();
	if (N < 4)
	{
		return previous_target_.velocitie;
	}

	double avg_x = 0;
	double avg_x2 = 0;
	double avg_f = 0;
	double avg_xf = 0;

	double time_first = target.front().time;

	for (int i = 0; i < N; i++)
	{
		avg_x +=target [i].time- time_first;
		avg_x2 += std::pow(target[i].time - time_first, 2);
		avg_f += target[i].center3d_[0];
		avg_xf += (target[i].time - time_first) * target[i].center3d_[0];
	}

	double vx = (avg_xf - N * (avg_x / N) * (avg_f / N)) / (avg_x2 - N * std::pow(avg_x / N, 2));

	
	avg_f = 0;
	avg_xf = 0;
	for (int i = 0; i < N; i++)
	{
		avg_f += target[i].center3d_[1];
		avg_xf += (target[i].time - time_first) * target[i].center3d_[1];
	}
	double vy = (avg_xf - N * (avg_x / N) * (avg_f / N)) / (avg_x2 - N * std::pow(avg_x / N, 2));

	double avg_f_ = 0;
	double avg_xf_ = 0;
	for (int i = 0; i < N; i++)
	{

		avg_f_ += target[i].center3d_[2];
		avg_xf_ += (target[i].time - time_first) * target[i].center3d_[2];
	}
	double vz = (avg_xf_ - N * (avg_x / N) * (avg_f_ / N)) / (avg_x2 - N * std::pow(avg_x / N, 2));

	// Vector3d ave_v_;
	// ave_v_[0] = vx;
	// ave_v_[1] = vy;
	// ave_v_[2] = vz;

	// if (ave_v.size() != 0)
	// {
	// double sum_vx, sum_vy, sum_vz;
	// for (int u = 0; u < ave_v.size(); u++)
	// {
	// 	sum_vx += ave_v[u][0];
	// 	sum_vy += ave_v[u][1];
	// 	sum_vz += ave_v[u][2];
	// }
	// double aver_vx = sum_vx / ave_v.size();
	// double aver_vy = sum_vy / ave_v.size();
	// double aver_vz = sum_vz / ave_v.size();

	if (vx * previous_target_.velocitie[0] < 0)
	{
		vx = previous_target_.velocitie[0];
		//v_count++;
	}
	else
	{
		//v_count = 0;
	}

	return {vx, vy, vz};

	}
	Armor predictor::ArmorSelect(std::vector<Armor> &objects)
	{
		if(!objects.size())
		{
			return previous_target_;
		}
		
		for (int i = 0; i < objects.size(); i++)
		{
			
			objects[i].center3d_ = pnp_solve_->poseCalculation(objects[i]);
		}
		
		float distances[objects.size()];
		for (int i = 0; i < objects.size(); i++)
		{
			distances[i] = std::sqrt(std::pow(objects[i].center3d_[0], 2) + std::pow(objects[i].center3d_[1], 2) + std::pow(objects[i].center3d_[2], 2));
		}

		float last_pose = std::sqrt(std::pow(previous_target_.center3d_[0], 2) + std::pow(previous_target_.center3d_[1], 2) + std::pow(previous_target_.center3d_[2], 2));

		float distances_residual[objects.size()];

		for (int i = 1; i < objects.size(); i++)
		{
			distances_residual[i] = std::abs(distances[i] - last_pose);
		}

		int index = 0;
		for (int i = 1; i < objects.size(); i++)
		{
			if (distances_residual[i] < distances_residual[index])
				index = i;
		}
		return objects[index];
	}
	GimbalPose predictor::antigyro_Armor(Armor target)
	{
		gettimeofday(&Time_all, NULL);
		float T_fly=(target.time-Time_all.tv_usec)/1000+
		pow((target.center3d_[0]*target.center3d_[0]+target.center3d_[2]*target.center3d_[2]),0.5)
		/(target.cur_pose_.pitch/180)*CV_PI;
		target.center3d_[0]=target.velocitie[0]*T_fly;
		target.center3d_[1]=target.velocitie[1]*T_fly;
		target.center3d_[2]=target.velocitie[2]*T_fly;

		return  point_to_armor(target.center3d_);
	}

}
