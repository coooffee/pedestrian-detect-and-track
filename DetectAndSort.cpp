#include <iostream>
#include <fstream>
#include <time.h>
#include <string.h>

#include "DetectAndSort.h"

#include <net/if.h>
#include <sys/ioctl.h>
#include <stdlib.h>

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.3
#define BATCH_SIZE 1
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)

#define ETH_NAME "eth0"
#define KEY 0x59

//#define SHOW

// stuff we know about the network and the input/output blobs

REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

void DetectAndSort::doInference(float* input, float* output) {
//    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
//    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
//    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
//
//    // Create GPU buffers on device
//    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float)));
//    CHECK(cudaMalloc(&buffers[outputIndex],  OUTPUT_SIZE * sizeof(float)));
//
//    // Create stream
//    CHECK(cudaStreamCreate(&stream));
//
//    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
//    context.enqueue(1, buffers, stream, nullptr);
//    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
//    cudaStreamSynchronize(stream);
//
//    // Release stream and buffers
//    cudaStreamDestroy(stream);
//    CHECK(cudaFree(buffers[inputIndex]));
//    CHECK(cudaFree(buffers[outputIndex]));
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host


    CHECK(cudaMemcpyAsync(buffers[0], input, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

bool isTimeAvailable(){
    time_t now;
    time (&now);
    char curyear[8];
    char curmonth[8];
    strftime(curyear, 8, "%Y",localtime(&now));
    strftime(curmonth, 8, "%m",localtime(&now));

    if(atoi(curyear)<2026 || (atoi(curyear)==2025&&atoi(curmonth)<12))
        return true;
    std::cout<<"Authorization expired"<<std::endl;
    return false;
}

void get_mac(char* mac_a)
{
    int sockfd;
    struct ifreq ifr;

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd == -1) {
        exit(1);
    }
    strncpy(ifr.ifr_name, ETH_NAME, IFNAMSIZ);      //Interface name

    if (ioctl(sockfd, SIOCGIFHWADDR, &ifr) == 0) {  //SIOCGIFHWADDR 获取hardware address
        memcpy(mac_a, ifr.ifr_hwaddr.sa_data, 6);
    }
}

bool isMacAvailable() {
    char * this_mac = new char[6];
    get_mac(this_mac);
    std::string macAddress = std::to_string(this_mac[0]&0xff)+std::to_string(this_mac[1]&0xff)+
            std::to_string(this_mac[2]&0xff)+std::to_string(this_mac[3]&0xff)+
            std::to_string(this_mac[4]&0xff)+std::to_string(this_mac[5]&0xff);
//    std::cout<<macAddress<<std::endl;
    delete [] this_mac;
    if(macAddress=="7217645710119")
        return true;
    std::cout<<"Unauthorized Device"<<std::endl;
    return false;

}

DetectAndSort::~DetectAndSort(){
};

//bool DetectAndSort::encrypModel(char* engine_name){
//    size_t size{0};
//    std::ifstream file(engine_name, std::ios::binary);
//
//    if (file.good()) {
//        file.seekg(0, file.end);
//        size = file.tellg();
//        file.seekg(0, file.beg);
//        trtModelStream = new char[size];
//        if(!trtModelStream)return false;
//        file.read(trtModelStream, size);
//        file.close();
//    }else{
//        std::cout<<"No model found...Please confirm the model location..."<<std::endl;
//        return false;}
//
//    for (int loopi=0;loopi<500;loopi++){
//        trtModelStream[loopi] ^= KEY;
//    }
//
//    std::ofstream ofs;
//    ofs.open("/home/nvidia/final/model.engine", std::ios::binary);
//    ofs.write(trtModelStream,size);
//    ofs.close();
//    return true;
//}

bool DetectAndSort::loadModel(char *engine_name) {
    if (!isMacAvailable() || !isTimeAvailable()){
        return false;
    }
    cudaSetDevice(DEVICE);
    size_t size{0};
    std::ifstream file(engine_name, std::ios::binary);

    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        if(!trtModelStream)return false;
        file.read(trtModelStream, size);
        file.close();
    }else{
        std::cout<<"No model found...Please confirm the model location..."<<std::endl;
        return false;}

    for (int loopi=0;loopi<500;loopi++){
        trtModelStream[loopi] ^= KEY;
    }

    runtime = createInferRuntime(gLogger);
    if(runtime == nullptr)return false;
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    if(engine == nullptr)return false;
    context = engine->createExecutionContext();
    if(context == nullptr)return false;
    delete[] trtModelStream;

    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CHECK(cudaStreamCreate(&stream));

    std::cout<<"Load Model Successfully..."<<std::endl;
    return true;
}

bool DetectAndSort::detectAndTrack(unsigned char *imgBuffer, char *detect_result, int h, int w ,int camera_id) {
    if(camera_id<0 || camera_id>3){
        std::cout<<"OUT OF RANGE:camera_id should be 0 or 1 or 2 or 3"<<std::endl;
        return false;
    }
//    auto start_1 = std::chrono::system_clock::now();
    cv::Mat frame = cv::Mat(h,w,CV_8UC3,imgBuffer,0);
    cv::Mat frame_copy_detect = frame.clone();
    cv::Mat frame_copy_track = frame.clone();
    cv::Mat pr_img = preprocess_img(frame_copy_detect); // letterbox BGR to RGB

//    int i = 0;
//    for (int row = 0; row < INPUT_H; ++row) {
//        uchar *uc_pixel = pr_img.data + row * pr_img.step;
//        for (int col = 0; col < INPUT_W; ++col) {
//            data[i] = uc_pixel[2] / 255.0;
//            data[i + INPUT_H * INPUT_W] = uc_pixel[1] / 255.0;
//            data[i + 2 * INPUT_H * INPUT_W] = uc_pixel[0] / 255.0;
//            uc_pixel += 3;
//            ++i;
//        }
//    }
    cv::Mat rgb_img;
    cv::cvtColor(pr_img, rgb_img, cv::COLOR_BGR2RGB);
    cv::Mat rgb_img_new;
    rgb_img.convertTo(rgb_img_new, CV_32FC3, 1/255.0);
    cv::Mat rgb_img_chw;
    hwc_to_chw(rgb_img_new, rgb_img_chw);
    float *ptr = (float*) rgb_img_chw.data;
    memcpy(data, ptr, INPUT_H*INPUT_W*3*sizeof(ptr[0]));
//    auto start_2 = std::chrono::system_clock::now();
    doInference(data, prob);
//    auto start_3 = std::chrono::system_clock::now();

    int fcount = 1;
    std::vector <std::vector<Yolo::Detection>> batch_res(fcount);
    for (int b = 0; b < fcount; b++) {
        auto &res = batch_res[0];
        nms(res, &prob[0], CONF_THRESH, NMS_THRESH);
    }
//    auto start_4 = std::chrono::system_clock::now();

    // tracking
    for (int b = 0; b < fcount; b++) {
        // auto tracking_start = std::chrono::system_clock::now();
        DS_DetectObjects detect_objects;
        auto &res = batch_res[0];
        for (auto &re : res) {
            if (re.conf < scores[(int) re.class_id]) {
                continue;
            }
            cv::Rect r = get_rect(frame_copy_track, re.bbox);
            DS_Rect rec;
            DS_DetectObject obj;
            rec.x = r.x;
            rec.y = r.y;
            rec.width = r.width;
            rec.height = r.height;
            obj.class_id = (int) re.class_id;
            obj.rect = rec;
            obj.confidence = re.conf;
            detect_objects.push_back(obj);
        }
//        Deep_sort *p{nullptr};
//        p = &(Tracker[i]);
//        Tracker.update(detect_objects, line_point, frame_copy_track);
        (Tracker[camera_id]).update(detect_objects, line_point, frame_copy_track);
        DS_TrackObjects track_objects = (Tracker[camera_id]).get_detect_obj();

#ifdef SHOW
        cv::namedWindow("test");
        cv::Size videoSize(1920,1080);

        if(camera_id==0) {
            for (const auto &det_obj : track_objects) {
                cv::Rect rec;
                rec.x = det_obj.rect.x;
                rec.y = det_obj.rect.y;
                rec.width = det_obj.rect.width;
                rec.height = det_obj.rect.height;
                cv::rectangle(frame_copy_track, rec, (0,150,0), 2);
                cv::putText(frame_copy_track, classes[det_obj.class_id] + ":" + std::to_string(det_obj.track_id),
                            cv::Point(rec.x, rec.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, (0,150,0), 2);
            }
            cv::imshow("test", frame_copy_track);
            cv::waitKey(2);
        }
#endif

        std::string finalstring = std::to_string(camera_id)+"#";
        for (const auto& det_obj : track_objects){
            finalstring += std::to_string(det_obj.rect.x)+"-"+std::to_string(det_obj.rect.y)+"-"+
                    std::to_string(det_obj.rect.x+det_obj.rect.width)+"-"+
                    std::to_string(det_obj.rect.y+det_obj.rect.height)+"-"+
                    std::to_string(det_obj.track_id)+"#";
        }
        if (finalstring!=""){
            finalstring = finalstring.substr(0,finalstring.length()-1);
        }
        int tmpi=0;
        for(auto c : finalstring)
        {
            detect_result[tmpi] = c;
            tmpi++;
        }
        detect_result[tmpi] = '\0';

    }
//    auto start_5 = std::chrono::system_clock::now();
//    std::cout<<"process time:"<<std::chrono::duration_cast<std::chrono::microseconds>(start_2 - start_1).count()/1000.0 << "ms" << std::endl;
//    std::cout<<"det time:"<<std::chrono::duration_cast<std::chrono::microseconds>(start_3 - start_2).count()/1000.0 << "ms" << std::endl;
//    std::cout<<"nms time:"<<std::chrono::duration_cast<std::chrono::microseconds>(start_4 - start_3).count()/1000.0 << "ms" << std::endl;
//    std::cout<<"tracking time:"<<std::chrono::duration_cast<std::chrono::microseconds>(start_5 - start_4).count()/1000.0 << "ms" << std::endl;

    return true;
}

bool DetectAndSort::destroyModel() {

    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    context->destroy();
    engine->destroy();
    runtime->destroy();
    std::cout<<"The environment has been released successfully..."<<std::endl;
    return true;
}
