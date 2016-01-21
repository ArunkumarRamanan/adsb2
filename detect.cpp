#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/assert.hpp>
#include <glog/logging.h>
#include "heart.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    //Stack stack("sax", "tmp");
    namespace po = boost::program_options; 
    string input_path;
    string output_path;
    string model_dir;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("input,i", po::value(&input_path), "")
    ("output,o", po::value(&output_path)->default_value("/home/wdong/public_html/prob.jpg"), "")
    ("model,m", po::value(&model_dir)->default_value("caffe-model"), "")
    ;


    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);
    //p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || input_path.empty() || output_path.empty() || model_dir.empty()) {
        cerr << desc;
        return 1;
    }

    heart::Detector *det = heart::make_caffe_detector(model_dir);
    BOOST_VERIFY(det);

    Mat image = imread(input_path, -1);
    Mat prob;
    det->apply(image, &prob);

    Mat norm;
    normalize(prob, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    imwrite(output_path, norm);

    delete det;
    return 0;
}

