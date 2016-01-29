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
    /*
    string output_dir;
    string gif;
    float th;
    int mk;
    */

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    //("output,o", po::value(&output_dir), "")
    /*
    ("gif", po::value(&gif), "")
    ("th", po::value(&th)->default_value(0.90), "")
    ("mk", po::value(&mk)->default_value(5), "")
    */
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

    GlobalInit(argv[0], config);
    Cook cook(config);

    Study study(input_dir, true, true, true);
    cook.apply(*study);
    vector<Slice *> slices;
    for (auto &s: study) {
        for (auto &ss: s) {
            slices.push_back(&ss);
        }
    }
    LOG(INFO) << "Detecting all slices..." << endl;
#pragma omp parallel
    {
        Detector *det = make_caffe_detector(config);
        CHECK(det) << " cannot create detector.";
#pragma omp for schedule(dynamic, 1)
        for (unsigned i = 0; i < slices.size(); ++i) {
            det->apply(slices[i]);
        }
        delete det;
    }
    LOG(INFO) << "Postprocessing..." << endl;
    for (auto &s: study) {
        MotionFilter(&s, config);
    }
#pragma omp parallel schedule(synamic, 1)
    for (unsigned i = 0; i < slices.size(); ++i) {
        FindSquare(slices[i]->prob,
                  &slices[i]->pred, config);

    }
    for (auto const &series: study) {
        for (auto const &s: series) {
            float r = std::sqrt(s.pred.area())/2 * s.meta.spacing;
            Point_<float> raw_pt((s.pred.x + s.pred.width/2.0) * s.meta.spacing / s.meta.raw_spacing,
                             (s.pred.y + s.pred.height/2.0) * s.meta.spacing / s.meta.raw_spacing);
            float raw_r = r / s.meta.raw_spacing;
            cout << s.path.native() << '\t' << r
                 << '\t' << raw_pt.x << '\t' << raw_pt.y << '\t' << raw_r;
                 << '\t' << s.meta.trigger_time;
            for (unsigned i = 0; i < Meta::SERIES_FIELDS; ++i) {
                cout << '\t' << s.meta[i];
            }
            cout << endl;
        }
    }
    return 0;
}

