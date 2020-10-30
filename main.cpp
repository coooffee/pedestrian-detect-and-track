//
// Created by xiao on 20/10/9.


#include "DetectAndSort.h"
#include <iostream>
#include <pthread.h>


int main(){

    DetectAndSort d;
    //load model
//    char* engineFile = "/home/nvidia/final/build/model.engine";
//    d.encrypModel(engineFile);

    char* engineFile = "/home/nvidia/final/model.engine";
    d.loadModel(engineFile);
    //detect and track
    cv::Mat frame;

    char *videoPath1 = "/home/nvidia/final/videos/zuanshi.mp4";
    char *videoPath2 = "/home/nvidia/final/videos/zuanshi1.mp4";
    char *videoPath3 = "/home/nvidia/final/videos/zuanshi2.mp4";
    char *videoPath4 = "/home/nvidia/final/videos/zuanshi3.mp4";

    cv::VideoCapture capture[4]{};

    capture[0].open(videoPath1);
    capture[1].open(videoPath2);
    capture[2].open(videoPath3);
    capture[3].open(videoPath4);

    if(!capture[0].isOpened())
        std::cout<<"Video Not Open"<<std::endl;

    char *detection_result = new char[1024];
    while (capture[0].isOpened()) {
        for (int i=0; i<4;i++) {
            capture[i] >> frame;

            if (frame.empty()) {
                goto END;
            }
            unsigned char *imageBuffer = frame.data;
            int h = 1080;
            int w = 1920;

            auto start = std::chrono::system_clock::now();
            d.detectAndTrack(imageBuffer, detection_result, h, w, i);
            auto end = std::chrono::system_clock::now();

            std::cout << "detection_result:" << detection_result << std::endl;
            std::cout << "inf time:"
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms"
                      << std::endl;
        }
    }
    END:
        std::cout << "end of video" << std::endl;
    delete []detection_result;

    //destroy model
    d.destroyModel();
    return 0;
}