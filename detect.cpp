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

int main(int argc, char **argv) {
    //Series stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string input_dir;
    string output_dir;
    string gif;
    bool do_prob = false;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    ("gif", po::value(&gif), "")
    ("prob", "")
    ;


    po::positional_options_description p;
    p.add("input", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || input_dir.empty()) {
        cerr << desc;
        return 1;
    }

    if (vm.count("prob")) do_prob = true;

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);

    GlobalInit(argv[0], config);

    Cook cook(config);
    Series stack(input_dir, true, true);
    cook.apply(&stack);

    float bbth = config.get<float>("adsb2.bound_th", 0.95);
#pragma omp parallel
    {
        Detector *det = make_caffe_detector(config);
        CHECK(det) << " cannot create detector.";
#pragma omp for schedule(dynamic, 1)
        for (unsigned i = 0; i < stack.size(); ++i) {
            det->apply(&stack[i]);
        }
        delete det;
    }
    MotionFilter(&stack, config);
#pragma omp parallel for
    for (unsigned i = 0; i < stack.size(); ++i) {
        auto &s = stack[i];
        FindSquare(s.prob, &s.pred, config);
        if (gif.size()) {
            Rect bb = s.pred;
            cv::rectangle(s.image, bb, cv::Scalar(0xFF));
            if (do_prob) {
                cv::normalize(s.prob, s.prob, 0, 255, cv::NORM_MINMAX, CV_32FC1);
                cv::rectangle(s.prob, bb, cv::Scalar(0xFF));
                cv::hconcat(s.image, s.prob, s.image);
            }
        }
    }
    for (auto const &s: stack) {
        Rect bb = s.pred;
        cout << s.path.native() << '\t' << sqrt(bb.area()) * s.meta.spacing << endl;
    }
    if (gif.size()) {
        stack.save_gif(gif);
    }
    return 0;
}


