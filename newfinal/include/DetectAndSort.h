//
// Created by xiao on 20/10/9.
//

#ifndef YOLOV5_TMP_H
#define YOLOV5_TMP_H
#include <opencv2/opencv.hpp>
#include "deepsort.h"
#include "cuda_runtime_api.h"
#include <chrono>
#include "NvInfer.h"
#include "common.hpp"
#include "logging.h"


static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

class DetectAndSort{
public:
    DetectAndSort() = default;
    ~DetectAndSort();
    DetectAndSort(const DetectAndSort&) = delete;
    DetectAndSort &operator = (const DetectAndSort&) = delete;

    bool loadModel(char *engine_name);
    bool detectAndTrack(unsigned char *imgBuffer, char *detect_result, int h, int w,int camera_id);
    bool destroyModel();

    void doInference(float* input, float* output);
private:
    char *trtModelStream{nullptr};
    cudaStream_t stream;
    void* buffers[2];

    Logger gLogger;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    Deep_sort Tracker[4]{};

    std::deque<cv::Point> line_point;
    std::vector<float> scores = {0.3};
    std::vector<std::string> classes = {"person"};
    float data[3 * INPUT_H * INPUT_W];
    float prob[OUTPUT_SIZE];

};



#endif //YOLOV5_TMP_H
