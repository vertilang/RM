#include "TRTModule.hpp"
#include <fstream>
#include <../common/logging.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fmt/format.h>
#include <fmt/color.h>
#include <opencv2/imgproc.hpp>

const char* INPUT_BLOB_NAME = "input";   // onnx 输入 名字
const char* OUTPUT_BLOB_NAME = "output"; // onnx 输出 名字
static Logger gLogger;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

float TRTModule::intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void TRTModule::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void TRTModule::qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void TRTModule::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

int TRTModule::find_max(float *prob,int num) //找到类别
{
    int max= 0;
    for(int i=1; i<num; i++)
    {
        if (prob[max]<prob[i])
         max = i;
    }

    return max;

}

void TRTModule::generate_yolox_proposals(float *feat_blob, float prob_threshold,std::vector<Object> &objects,int OUTPUT_CANDIDATES)
{
    const int num_class = 8;  	// BMD 类
    const int ckpt=10;			//yolov7 是15， 3*5=15

    const int num_anchors = OUTPUT_CANDIDATES;

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) 
    {
        const int basic_pos = anchor_idx * (num_class + 5 + ckpt); //5代表 x,y,w,h,object_score  ckpt代表5个关键点 每个关键点2个数据
        float box_objectness = feat_blob[basic_pos + 4];

        // int cls_id = find_max(&feat_blob[basic_pos +5+ckpt],num_class);   //找到类别v5
        int cls_id = find_max(&feat_blob[basic_pos + 15],num_class);   //v7
        // float score = feat_blob[basic_pos + 5 +8 + cls_id]; //v5
        float score = feat_blob[basic_pos + 5 + 10 + cls_id];  //v7
        score *= box_objectness; 


        if (score > prob_threshold) 
        {
            // yolox/models/yolo_head.py decode logic
            float x_center = feat_blob[basic_pos + 0];
            float y_center = feat_blob[basic_pos + 1];
            float w = feat_blob[basic_pos + 2];
            float h = feat_blob[basic_pos + 3];
            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;
            
            // float *landmarks = &feat_blob[basic_pos +5]; //v5
            float *landmarks = &feat_blob[basic_pos + 5];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = w;
            obj.rect.height = h;
            obj.label = cls_id;
            obj.prob = score;
            int k = 0;
            //   for (int i = 0; i<ckpt; i++)
            //   {
            //      obj.landmarks[k++]=landmarks[i];
            //   }

            // 人脸五个关键点，总共10个数
           obj.landmarks[0]=landmarks[0];
            obj.landmarks[1]=landmarks[1];
            obj.landmarks[2]=landmarks[2];
            obj.landmarks[3]=landmarks[3];
            obj.landmarks[4]=landmarks[4];
            obj.landmarks[5]=landmarks[5];
            obj.landmarks[6]=landmarks[6];
            obj.landmarks[7]=landmarks[7];
            obj.landmarks[8]=landmarks[8];
            obj.landmarks[9]=landmarks[9]; 
            objects.push_back(obj);
        }
    }
}


void TRTModule::decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h,int OUTPUT_CANDIDATES,int top,int left) {
        std::vector<Object> proposals;
        std::vector<bbox> bboxes;
        generate_yolox_proposals(prob,  BBOX_CONF_THRESH, proposals,OUTPUT_CANDIDATES);
        // generate_proposals(prob,  BBOX_CONF_THRESH, bboxes,OUTPUT_CANDIDATES);
        // std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);
        // std::sort(bboxes.begin(),bboxes.end(),my_func);
        std::vector<int> picked;

        nms_sorted_bboxes(proposals, picked, NMS_THRESH);
        // auto choice =my_nms(bboxes, NMS_THRESH);
        int count = picked.size();

        // std::cout << "num of boxes: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x-left) / scale;
            float y0 = (objects[i].rect.y-top) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width-left) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height-top) / scale;
            
            float *landmarks = objects[i].landmarks;
            for(int i= 0; i<10; i++)
            {
                if(i%2==0)
                landmarks[i]=(landmarks[i]-left)/scale;
                else
                landmarks[i]=(landmarks[i]-top)/scale;
            }
            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}

void TRTModule::doInference_cu(nvinfer1::IExecutionContext& context, cudaStream_t &stream, void **buffers, float* output, int batchSize,int OUTPUT_SIZE) {
    // infer on the batch asynchronously, and DMA output back to host
   // CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

TRTModule::TRTModule(const std::string &trt_file)
{
    cudaSetDevice(DEVICE);
    char *trtModelStreamDet{nullptr};
    size_t size{0};
    const std::string engine_file_path {trt_file};  
    std::ifstream file(engine_file_path, std::ios::binary);

    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStreamDet = new char[size];
        assert(trtModelStreamDet);
        file.read(trtModelStreamDet, size);
        file.close();
            std::cout << "*******************************" << std::endl;
    }

    //det模型trt初始化
    nvinfer1::IRuntime* runtime_det = nvinfer1::createInferRuntime(gLogger);
    assert(runtime_det != nullptr);
    engine = runtime_det->deserializeCudaEngine(trtModelStreamDet, size);
    assert(engine != nullptr); 
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStreamDet;

    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
   

    auto out_dims = engine->getBindingDimensions(1);

    OUTPUT_CANDIDATES = out_dims.d[1];

    for(int j=0;j<out_dims.nbDims;j++) 
    {
        output_size *= out_dims.d[j];
    }

    CHECK(cudaMalloc((void**)&buffers[inputIndex],  3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc((void**)&buffers[outputIndex], output_size * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // prepare input data cache in pinned memory 
    CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    runtime_det->destroy();
}

std::vector<Object> TRTModule::operator()(const cv::Mat &img) 
{
    float *buffer_idx = (float*)buffers[inputIndex];
    size_t size_image = img.cols * img.rows * 3;
    size_t size_image_dst = INPUT_H * INPUT_W * 3;
    memcpy(img_host, img.data, size_image);

    CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);

    float* prob = new float[output_size];

    doInference_cu(*context, stream, (void**)buffers, prob, 1, output_size);

    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    int left = (INPUT_W-unpad_w)/2;
    int top = (INPUT_H-unpad_h)/2; 
    int img_w = img.cols;
    int img_h = img.rows;
    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));

    std::vector<Object> objects;
    decode_outputs(prob, objects, scale, img_w, img_h,OUTPUT_CANDIDATES,top,left);
    for (int i = 0; i<objects.size(); i++)
    {
        //cv::rectangle(img, cv::Point(objects[i].rect.x,objects[i].rect.y),cv::Point(objects[i].rect.x+objects[i].rect.width,objects[i].rect.y+objects[i].rect.height), cv::Scalar(0,255,0), 2);
        cv::line(img, cv::Point2f(objects[i].landmarks[0],objects[i].landmarks[1]), cv::Point2f(objects[i].landmarks[2],objects[i].landmarks[3]),cv::Scalar(193, 182, 255), 1, 8);
        cv::line(img, cv::Point2f(objects[i].landmarks[2],objects[i].landmarks[3]), cv::Point2f(objects[i].landmarks[4],objects[i].landmarks[5]),cv::Scalar(193, 182, 255), 1, 8);
        cv::line(img, cv::Point2f(objects[i].landmarks[4],objects[i].landmarks[5]), cv::Point2f(objects[i].landmarks[6],objects[i].landmarks[7]),cv::Scalar(193, 182, 255), 1, 8);
        cv::line(img, cv::Point2f(objects[i].landmarks[6],objects[i].landmarks[7]), cv::Point2f(objects[i].landmarks[0],objects[i].landmarks[1]),cv::Scalar(193, 182, 255), 1, 8);

    }
    delete prob;
    return objects;
}

TRTModule::~TRTModule()
{
    cudaStreamDestroy(stream);
    //CHECK(cudaFree(img_device));
    //CHECK(cudaFreeHost(img_host));
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    engine->destroy();
    context->destroy();
    CHECK(cudaFree(img_device));
    CHECK(cudaFreeHost(img_host));
    //delete prob;
}
