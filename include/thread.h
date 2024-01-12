#ifndef _THREAD_H_
#define _THREAD_H_
#include "../DaHeng/DaHengCamera.h"
#include "../autoaim/TRTModule.hpp"
#include "predict.h"
#include "Send_Receive.h"
using namespace Horizon;
enum BufferSize
{
    IMGAE_BUFFER = 5
};

class Factory
{
public:
    Factory(){}
public:
    cv::Mat img;
    Horizon::predictor predict;
    Horizon::GimbalPose get_gim; 

    Horizon::DataControler data_controler_;
    int fd;
    mutex serial_mutex_;
    Horizon::DataControler::VisionData visiondata; // 视觉向电控传数据
    Horizon::DataControler::Stm32Data stm32data;   // 电控向视觉发数据
    Horizon::DataControler::Stm32Data last_stm32_;

    Horizon::DataControler::Stm32Data stm32data_temp;
public:
    cv::Mat image_buffer_[BufferSize::IMGAE_BUFFER];
    volatile unsigned int image_buffer_front_ = 0;   // the produce index
    volatile unsigned int image_buffer_rear_ = 0;    // the comsum index 
    std::vector<Armor> objects;
    void producer();
    void consumer();
    void getdata();

};
#endif

