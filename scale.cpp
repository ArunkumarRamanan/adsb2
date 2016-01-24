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

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    //("output,o", po::value(&output_dir), "")
    ("gif", po::value(&gif), "")
    ("th", po::value(&th)->default_value(0.90), "")
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

    ImageLoader loader(config);
    DcmStack images(input_dir, loader);

    ColorRange cr;
    images.getColorRange(&cr, th);

    cerr << cr.min << ' ' << cr.umin << ' ' << cr.umax << ' ' << cr.max << endl;

    for (auto &image: images) {
        cv::Mat a = image.clone();
        cv::Mat b = image.clone();
        ImageAdaptor::apply(&a, cr);
        ImageAdaptor::apply(&b);
        cv::hconcat(a, b, image);
    }
    if (gif.size()) {
        images.make_gif(gif);
    }
    return 0;
}

