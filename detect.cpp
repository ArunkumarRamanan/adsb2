#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <boost/assert.hpp>
#include <glog/logging.h>
#include "adsb2.h"

namespace fs = boost::filesystem;

using namespace std;
using namespace cv;
using namespace adsb2;



void EM_binarify (Mat &mat, Mat *out) {
    Mat pixels;
    mat.convertTo(pixels, CV_64F);
    pixels = pixels.reshape(1, pixels.total());
    BOOST_VERIFY(pixels.total() == mat.total());
    Mat logL, labels, probs;
    EM em(2);
    em.train(pixels, noArray(), labels, noArray());
    cerr << pixels.total() << ' ' << pixels.rows << ' ' << labels.total() << endl;
    BOOST_VERIFY(labels.total() == mat.total());
    labels.convertTo(*out, CV_8UC1, 255);
    *out = out->reshape(1, mat.rows);
}

#if 0
void scale_detect (Detector *det, Mat &image, Rect *bb, float scale, float bbth) {
    Mat from;
    if (scale != 1) {
        resize(image, from, round(image.size() * scale));
    }
    else {
        from = image;
    }
    Mat prob;
    det->apply(from, &prob);
    if (from.data != image.data) {
        resize(prob, prob, image.size());
    }
    bound(prob, bb, bbth);
}
#endif

int main(int argc, char **argv) {
    //Stack stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string input_dir;
    string output_dir;
    string gif;
    float th;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    //("output,o", po::value(&output_dir), "")
    ("gif", po::value(&gif), "")
    ("th", po::value(&th)->default_value(0.99), "")
    ;


    po::positional_options_description p;
    p.add("input", 1);
    //p.add("output", 1);
    //p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || input_dir.empty()) {
        cerr << desc;
        return 1;
    }

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);

    Cook cook(config);
    Stack stack(input_dir);
    cook.apply(&stack);

    Detector *det = make_caffe_detector(config.get<string>("adsb2.caffe.model", "model"));
    CHECK(det) << " cannot create detector.";

    float bbth = config.get<float>("adsb2.bound_th", 0.95);
    for (auto &s: stack) {
        Rect bb;
        Mat prob;
        det->apply(s, &prob);
        bound(prob, &bb, bbth);
        cv::rectangle(s.image, bb, cv::Scalar(0xFF));
    }

    delete det;

    if (gif.size()) {
        stack.save_gif(gif);
    }
    return 0;
}

