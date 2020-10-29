//
// Created by xiao on 20/10/20.
//

#include "DetectAndSort.h"

//#include "detect_and_track.h"
#include "omp.h"
//#define SHOW

int main(int argc, char** argv){
    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[1], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

#ifdef SHOW
    cv::namedWindow("test");
    std::sort(file_names.begin(),file_names.end());
    cv::RNG rng(time(0));
    std::vector<cv::Scalar> color_map;
    for (int i=0;i<200;i++){
        color_map.emplace_back(cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)));
    }
    cv::Size videoSize(1280,1024);
#endif
//#pragma omp parallel for num_threads(4)

    for (auto i=0;i<file_names.size();i++) {
        std::cout << file_names[i] << std::endl;
        DetectAndSort detect_and_track;
        char* engine_name = "/home/nvidia/ped/build/yolov5s_960_fp16_n2n.engine";
        detect_and_track.loadModel(engine_name);
        cv::VideoCapture capture;
        capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('A', 'V', 'C', '1'));
#ifdef SHOW
        std::string video_file_name = "/home/nvidia/dk/v2x_data/video_pred/" + file_name;
        cv::VideoWriter writer(video_file_name, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 10, videoSize, true);
#endif
        cv::Mat frame;
        char *detection_result = new char[1024];

        frame = capture.open(std::string(argv[1]) + "/" + file_names[i]);
        if(!capture.isOpened())
        {
            printf("can not open ...\n");
            continue;
        }
        int frame_id = 0;
        while (capture.read(frame)) {

            unsigned char* imageBuffer = frame.data;
            int h=1080;
            int w=1920;
            auto start = std::chrono::system_clock::now();
            detect_and_track.detectAndTrack(imageBuffer, detection_result, h, w ,i);
            auto end = std::chrono::system_clock::now();
            std::cout<<i<<" detection_result:"<<detection_result<<std::endl;
            std::cout << "inf time:" << std::chrono::duration_cast<std::chrono::microseconds>(
                    end - start).count()/1000.0 << "ms" << std::endl;
#ifdef SHOW
            //            std::cout << "res/" + std::to_string(frame_id) + "_" + file_name << std::endl;

        for (const auto& det_obj : dets){
            std::vector<std::string> classes = {"person", "car", "bicycle", "tricycle", "truck"};
            cv::Rect rec;
            rec.x = det_obj.bbox_x_pixel;
            rec.y = det_obj.bbox_y_pixel;
            rec.width = det_obj.bbox_width;
            rec.height = det_obj.bbox_height;
            int color_id = det_obj.track_id % 99;
            cv::rectangle(frame, rec, color_map[color_id], 2);
            cv::putText(frame, classes[det_obj.type] + ":"+ std::to_string(det_obj.track_id), cv::Point(rec.x, rec.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, color_map[color_id], 2);
        }

        cv::imshow("test", frame);
        cv::waitKey(10);
        writer << frame;
#endif
            frame_id++;
        }
        detect_and_track.destroyModel();

#ifdef SHOW
        writer.release();
#endif
    }
#ifdef SHOW
    cv::destroyWindow("test");
#endif
    std::cout << "work!" << std::endl;
    return 1;
}