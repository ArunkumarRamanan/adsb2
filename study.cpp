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
using namespace boost;
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
    cook.apply(&study);
    vector<Slice *> slices;
    for (auto &s: study) {
        for (auto &ss: s) {
            slices.push_back(&ss);
        }
    }
    {
        cerr << "Detecting " << slices.size() << "  slices..." << endl;
        timer::auto_cpu_timer timer(cerr);
        progress_display progress(slices.size(), cerr);
#pragma omp parallel
        {
            Detector *det = make_caffe_detector(config);
            CHECK(det) << " cannot create detector.";
#pragma omp for schedule(dynamic, 1)
            for (unsigned i = 0; i < slices.size(); ++i) {
                det->apply(slices[i]);
#pragma omp critical
                ++progress;
            }
            delete det;
        }
    }
    {
        cerr << "Filtering..." << endl;
        timer::auto_cpu_timer timer(cerr);
        for (auto &s: study) {
            MotionFilter(&s, config);
        }
    }
    {
        cerr << "Finding squares..." << endl;
        timer::auto_cpu_timer timer(cerr);
        progress_display progress(slices.size(), cerr);
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < slices.size(); ++i) {
            FindSquare(slices[i]->prob,
                      &slices[i]->pred, config);
#pragma omp critical
            ++progress;
        }
    }
    for (auto const &series: study) {
        for (auto const &s: series) {
            report(cout, s);
        }
    }
    return 0;
}

