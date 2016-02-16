#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/program_options.hpp>
#include <boost/assert.hpp>
#include <glog/logging.h>
#include "adsb2.h"

using namespace std;
using namespace boost;
using namespace cv;
using namespace adsb2;

int main(int argc, char **argv) {
    nice(10);
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string input_dir;
    string output_dir;
    int ca;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    ("output,o", po::value(&output_dir), "")
    ("ca", po::value(&ca)->default_value(1), "")
    ("bound", "")
    ;


    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);
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

    bool do_gif = vm.count("no-gif") == 0;

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);

    GlobalInit(argv[0], config);
    Cook cook(config);

    timer::auto_cpu_timer timer(cerr);
    Study study(input_dir, true, true, true);
    cook.apply(&study);
    cv::Rect bound;

    ComputeBoundProb(&study);
    cerr << "Filtering..." << endl;
    ProbFilter(&study, config);
    vector<Slice *> slices;
    study.pool(&slices);
    cerr << "Finding squares..." << endl;
#pragma omp parallel for schedule(dynamic, 1)
    for (unsigned i = 0; i < slices.size(); ++i) {
        FindBox(slices[i], config);
    }

    ComputeContourProb(&study, config);
    study_CA1(&study, config, true);
    
    if (output_dir.size()) {
        fs::path dir(output_dir);
        fs::create_directories(dir);
#pragma omp parallel for
        for (unsigned i = 0; i < study.size(); ++i) {
            auto &sax = study[i];
            int sax_id = lexical_cast<int>(sax.dir().filename().native().substr(4));
            fs::ofstream os(dir/fs::path(fmt::format("{}.ctr", sax_id)));
            for (auto const &s: sax) {
                float rate = s.meta.spacing / s.meta.raw_spacing;
                os << s.path.native()
                   << "\tpred\t"
                   << s.polar_R * rate
                   << '\t' << s.polar_C.x * rate
                   << '\t' << s.polar_C.y * rate
                   << '\t' << s.images[IM_POLAR].cols
                   << '\t' << s.images[IM_POLAR].rows;
                CHECK(s.polar_contour.size() == s.images[IM_POLAR].rows);
                for (int v: s.polar_contour) {
                    os << '\t' << v;
                }
                os << endl;
            }
        }
    }
    return 0;
}

