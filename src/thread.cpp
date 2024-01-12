#include "../include/thread.h"
#define MIDVISION
namespace GxCamera
{
int GX_exp_time = 10000;

int GX_gain = 10;
DaHengCamera* camera_ptr_ = nullptr;
int GX_blance_r = 50;//rbg颜色通道
int GX_blance_g = 32;
int GX_blance_b = 44;


int GX_gamma = 1;

//DaHengCamera* camera_ptr_ = nullptr;

void DaHengSetExpTime(int,void* ){
    camera_ptr_->SetExposureTime(GX_exp_time);
}

void DaHengSetGain(int,void* ){
    camera_ptr_->SetGain(3,GX_gain);
}

}
namespace MidCamera
{
	int MV_exp_value = 10000;
	MVCamera *camera_ptr_ = nullptr;
	void MVSetExpTime(int, void *)
	{
		camera_ptr_->SetExpose(MV_exp_value);
	}
}
void Factory::producer()
{
    #ifdef DAHENG
    while(true)
    {
        if(GxCamera::camera_ptr_ != nullptr)//打印图片
        {
            while(image_buffer_front_ - image_buffer_rear_ > IMGAE_BUFFER){};
            if(GxCamera::camera_ptr_->GetMat(image_buffer_[image_buffer_front_%IMGAE_BUFFER]))
            {
                // 调整后，把这段注释掉
                ++image_buffer_front_;
                // 收数开始
            }
            else{
                delete GxCamera::camera_ptr_;
                GxCamera::camera_ptr_ = nullptr;
            }

        }
        else
        {
            GxCamera::camera_ptr_ = new DaHengCamera;
            while(!GxCamera::camera_ptr_->StartDevice());
            GxCamera::camera_ptr_->SetResolution();
            while(!GxCamera::camera_ptr_->StreamOn());
            // 设置是否自动白平衡
            GxCamera::camera_ptr_->Set_BALANCE_AUTO(1);
            // 手动设置白平衡通道及系数，此之前需关闭自动白平衡

            GxCamera::camera_ptr_->SetExposureTime(GxCamera::GX_exp_time);
            GxCamera::camera_ptr_->SetGain(3, GxCamera::GX_gain);

            double GX_Gamma = 2.85;
            GxCamera::camera_ptr_->setGamma(GX_Gamma);

            cv::namedWindow("DaHengCameraDebug", cv::WINDOW_AUTOSIZE);
            cv::createTrackbar("DaHengExpTime", "DaHengCameraDebug", &GxCamera::GX_exp_time, 10000,GxCamera::DaHengSetExpTime);
            GxCamera::DaHengSetExpTime(0,nullptr);
            cv::createTrackbar("DaHengGain", "DaHengCameraDebug", &GxCamera::GX_gain, 10,GxCamera::DaHengSetGain);
            GxCamera::DaHengSetGain(0,nullptr);
           //GxCamera::DaHengSetGain(0,nullptr);

            image_buffer_front_ = 0;
            image_buffer_rear_ = 0;
        }    
    }
    #endif
    #ifdef MIDVISION

	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	while (true)
	{
		if (MidCamera::camera_ptr_ != nullptr)
		{

			std::cout << "enter producer" << std::endl;
			while (image_buffer_front_ - image_buffer_rear_ > IMGAE_BUFFER - 1)
			{
			};
			//bool is = image_buffer_.try_lock();
			std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

			if (MidCamera::camera_ptr_->GetMat(image_buffer_[image_buffer_front_ % IMGAE_BUFFER]))
			{
				std::chrono::duration<double> time_run = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
				// std::cout << "time :" << time_run.count() << std::endl;

				MidCamera::camera_ptr_->SetExpose(MidCamera::MV_exp_value);

				// if (!is)
				// {
				// 	std::cout << "try lock failed!!" << std::endl;
				// }
				// std::cout << "enter producer lock" << std::endl;
				timer_buffer_[image_buffer_front_ % IMGAE_BUFFER] = time_run.count();
			    //cv::imshow("windowName", image_buffer_[image_buffer_front_ % IMGAE_BUFFER]);
				++image_buffer_front_;

				// std::cout << "out producer lock" << std::endl;
			}
			else
			{
				delete MidCamera::camera_ptr_;
				MidCamera::camera_ptr_ = nullptr;
			}
			//image_mutex_.unlock();
            
		}
		else
		{
			MidCamera::camera_ptr_ = new MVCamera;

			MidCamera::camera_ptr_->SetExpose(5000);

			cv::namedWindow("MVCameraDebug", cv::WINDOW_AUTOSIZE);
			cv::createTrackbar("MVExpTime", "MVCameraDebug", &MidCamera::MV_exp_value, 15000, MidCamera::MVSetExpTime);
			MidCamera::MVSetExpTime(0,nullptr);

			image_buffer_front_ = 0;
			image_buffer_rear_ = 0;
		}
	}
    #endif
    
}


void Factory::consumer()
{
    TRTModule trtmodel("../armor-best.engine");
    while(true)
    {
        // 若满足这个条件，则让这个函数一只停在这里
        while(image_buffer_front_ <= image_buffer_rear_);
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // 读取最新的图片
        image_buffer_rear_ = image_buffer_front_ - 1;
        // 直接获取引用
        cv::Mat &img = image_buffer_[image_buffer_rear_%IMGAE_BUFFER];
        predict->v0=20.0;
        int i=0;
        auto detectors = trtmodel(img);
        predict->best_target_.cur_pose_.yaw=0;//stm32data.yaw_data_.f;
        predict->best_target_.cur_pose_.pitch=0;//stm32data.pitch_data_.f;
        gettimeofday(&Time_all, NULL);
        predict->best_target_.time=Time_all.tv_usec/1000;
        for(auto detector : detectors)
        {
            predict->objects[i].pts[0].x=detector.rect.x;
            predict->objects[i].pts[0].y=detector.rect.y;
            predict->objects[i].pts[1].x=detector.rect.x+detector.rect.width;
            predict->objects[i].pts[1].y=detector.rect.y;
            predict->objects[i].pts[2].x=detector.rect.x;
            predict->objects[i].pts[2].y=detector.rect.y+detector.rect.height;
            predict->objects[i].pts[3].x=detector.rect.x+detector.rect.width;
            predict->objects[i].pts[3].y=detector.rect.y+detector.rect.height;
            visiondata.is_have_armor=1;
        }
        if(!detectors.size())
        {
            visiondata.is_have_armor=0;
        }
        predict->init();
        visiondata.yaw_data_.f=predict->best_target_.cur_pose_.yaw;
        visiondata.pitch_data_.f=predict->best_target_.cur_pose_.pitch;
        data_controler_.sentData(fd,visiondata);
        
        char test[100];
        sprintf(test, "tz:%0.4f", predict->best_target_.center3d_[2]);
        cv::putText(img, test, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 0), 1, 8);

        sprintf(test, "tx:%0.4f", predict->best_target_.center3d_[0]);
        cv::putText(img, test, cv::Point(img.cols/3, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 1, 8);

        sprintf(test, "ty:%0.4f", predict->best_target_.center3d_[1]);
        cv::putText(img, test, cv::Point(2*img.cols/3, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1, 8);

        sprintf(test, "send yaw:%0.4f ", predict->best_target_.cur_pose_.yaw);
        cv::putText(img, test, cv::Point(10, 160), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1, 8);

        sprintf(test, "send pitch:%0.4f ", predict->best_target_.cur_pose_.pitch);
        cv::putText(img, test, cv::Point(img.cols/2, 160), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1, 8);

        sprintf(test, "get yaw:%0.4f ", stm32data.yaw_data_.f);
        cv::putText(img, test, cv::Point(2, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1, 8);

        sprintf(test, "get pitch:%0.4f ", stm32data.pitch_data_.f);
        cv::putText(img, test, cv::Point(img.cols/2, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1, 8);
        std::string windowName = "show";
        
        cv::namedWindow(windowName, 0);
        cv::imshow(windowName,img);
        cv::waitKey(1);
        
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_run = std::chrono::duration_cast <std::chrono::duration < double>>(t2 - t1);

        float FPS = 1/(time_run.count());

        //std::cout << "                 " << "FPS: " << FPS << std::endl;

    }
}
void Factory::getdata()
    {
	fd = OpenPort("/dev/ttyUSB0");
	if (fd == -1)
	{
		fd = OpenPort("/dev/ttyUSB1");
	}
	configureSerial(fd);
	while (1)
	{
		// cv::waitKey(1);
		// cv::waitKey(2);
		if (fd == -1)
		{
			std::cout << "[the serial dosen`t open!!!]" << std::endl;
			continue;
		}

		serial_mutex_.lock();
		data_controler_.getData(fd, stm32data);
		// if (!stm32data_temp.is_aim)
		// {
		// 	// stm32data_temp.yaw_data_.f = 0;
		// 	// stm32data_temp.pitch_data_.f = 0;
		// 	//is_aim_ = false;
		// 	// imu_data.yaw = stm32data_temp.yaw_data_.f;
		// 	// imu_data.pitch = stm32data_temp.pitch_data_.f;
		// }
		// else
		// {
		// 	//is_aim_ = true;
		// }

		// if (!stm32data_temp.dubug_print)
		// {
		// 	// std::cout << "is_not_receive" << std::endl;
		// 	stm32data = last_stm32_;
		// 	serial_mutex_.unlock();
		// 	continue;
		// }
		// else
		// {
		// 	last_stm32_ = stm32data;
		// 	// std::cout << "is_received" << std::endl;
		// }

		// if (MCU_data_.size() < mcu_size_)
		// {
		// 	MCU_data_.push_back(stm32data_temp);
		// }
		// else
		// {
		// 	MCU_data_.pop_front();
		// 	MCU_data_.push_back(stm32data_temp);
		// }

		serial_mutex_.unlock();
	}
}