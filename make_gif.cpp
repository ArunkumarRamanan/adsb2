#include <map>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
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
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string input_dir;
    string anno_path;
    string gif;
    int sample;
    float spacing;
    bool do_contour = false;
    bool do_fill = false;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    ("gif", po::value(&gif), "")
    ("sample", po::value(&sample)->default_value(1), "")
    ("spacing", po::value(&spacing), "")
    ("anno", po::value(&anno_path), "")
    ("fill", "")
    ;


    po::positional_options_description p;
    p.add("input", 1);
    p.add("gif", 1);

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
    if (vm.count("spacing")) {
        config.put("adsb2.cook.spacing", spacing);
    }

    GlobalInit(argv[0], config);

    Cook cook(config);
    Series stack(input_dir, true, true);
    if (anno_path.size()) {
        CHECK(sample == 1);
        if (vm.count("fill")) do_fill = true;
        else do_contour = true;
        map<string, Slice> anno;
        ifstream is(anno_path.c_str());
        string line;
        while (getline(is, line)) {
            Slice sl(line);
            anno[sl.path.filename().native()] = sl;
        }
        for (auto &s: stack) {
            auto it = anno.find(s.path.filename().native());
            CHECK(it != anno.end());
            s.anno = it->second.anno;
            s.anno_data = it->second.anno_data;
        }
    }
    cook.apply(&stack);
    if (sample > 1) {
        Series stack2;
        for (unsigned i = 0; i < stack.size(); ++i) {
            if (i % sample == 0) {
                stack2.push_back(stack[i]);
            }
        }
        std::swap(stack, stack2);
    }
    if (do_contour || do_fill) {
        for (auto &s: stack) {
            cv::Mat tmp;
            if (do_contour) {
                s.anno->contour(s, &tmp, cv::Scalar(0xff));
            }
            else {
                s.anno->fill(s, &tmp, cv::Scalar(0xff));
                type_convert(&tmp, CV_32F);
            }
            s.images[IM_IMAGE] -= tmp;
        }
    }
    stack.visualize(false);
    stack.save_gif(gif);
    return 0;
}

