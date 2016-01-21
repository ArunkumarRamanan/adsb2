#pragma once
#include <string>
#include <opencv2/opencv.hpp>

namespace heart {

    using std::string;

    class Detector {
    public:
        virtual ~Detector () {}
        virtual void apply (cv::Mat input, cv::Mat *output) = 0;
    };

    Detector *make_caffe_detector (string const &);
}
