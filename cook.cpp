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
    //Stack stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string input_dir;
    string output_dir;
    string gif;
    float th;
    int mk;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    //("output,o", po::value(&output_dir), "")
    ("gif", po::value(&gif), "")
    ("th", po::value(&th)->default_value(0.90), "")
    ("mk", po::value(&mk)->default_value(5), "")
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

    /*
    cv::Mat pr;
    cv::Mat vx = stack.front().vimage.clone();
    th = percentile<float>(vx, th);
    cv::threshold(vx, vx, th, 1.0, cv::THRESH_BINARY);
    cv::Mat kernel = cv::Mat::ones(mk, mk, CV_32F);
    cv::morphologyEx(vx, vx, cv::MORPH_OPEN, kernel);
    pr = vx;
    //Var2Prob(vx, &pr);
    cv::normalize(pr, pr, 0, 255, cv::NORM_MINMAX, CV_32F);
    */
    stack.resize(1);
        #if 0
    for (auto &s: stack) {
        cv::Mat v;
        //CaffeAdaptor::apply(s, &v, nullptr, 2);
        v = s.vimage;
        /*
        cv::Mat a = s.image.clone();
        hconcat(a, s.vimage, s.image);
        */
        cv::normalize(v, v, 0, 255, cv::NORM_MINMAX, CV_32F);
        cv::hconcat(v, pr, s.image);
    }
#endif
    if (gif.size()) {
        stack.save_gif(gif);
    }
    return 0;
}

