#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <boost/assert.hpp>
#include <glog/logging.h>
#include "heart.h"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

Mat imreadx (string const &path) {
    Mat v = imread(path, -1);
    if (!v.data) {
        // failed, resort to convert
        ostringstream ss;
        fs::path tmp(fs::unique_path("%%%%-%%%%-%%%%-%%%%.pgm"));
        ss << "convert " << path << " " << tmp.native();
        ::system(ss.str().c_str());
        v = imread(tmp.native(), -1);
        fs::remove(tmp);
    }
    if (!v.data) return v;
    if (v.cols < v.rows) {
        transpose(v, v);
    }
    // TODO! support color image
    if (v.channels() == 3) {
        cv::cvtColor(v, v, CV_BGR2GRAY);
    }
    else CHECK(v.channels() == 1);
    // always to gray
    if (v.type() == CV_16UC1
            || v.type() == CV_32FC1) {
        normalize(v, v, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    }
    else CHECK(v.type() == CV_8UC1);
    return v;
}

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

    Mat image = imreadx(input_path);
    Mat prob;
    det->apply(image, &prob);

    Mat norm;
    normalize(prob, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    BOOST_VERIFY(norm.size() == image.size());

    Mat out;
    hconcat(image, norm, out);
    imwrite(output_path, out);

    delete det;
    return 0;
}

