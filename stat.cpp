#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
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

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ;


    po::positional_options_description p;
    p.add("list", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help")) { //
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
    string path;
    while (getline(cin, path)) {
        Meta meta;
        cv::Mat image = ImageLoader::load_raw(path, &meta);
        /*
        cout << meta.pixel_spacing <<
            '\t' << image.rows <<
            '\t' << image.cols <<
            '\t' << image.rows * meta.pixel_spacing <<
            '\t' << image.cols * meta.pixel_spacing << endl;
            */
        CHECK(image.data);
        CHECK(image.type() == CV_16UC1);
        for (unsigned i = 0; i < image.rows; ++i) {
            uint16_t const *p = image.ptr<uint16_t const>(i);
            for (unsigned j = 0; j < image.cols; ++j) {
                cout << p[j] << endl;
            }
        }
        break;
    }
}

